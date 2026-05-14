"""
YoutuberAgent — a BaseAgent subclass that drives a real Chromium browser via
Playwright to log into YouTube, then subscribe to channels and watch one video
from each, one step at a time.

NOTE: Automating logins to Google services violates Google's Terms of Service
and may cause your account to be flagged. Use a dedicated test account, a
persistent profile (so you only solve 2FA/CAPTCHA once), and run headful so
you can intervene when Google challenges the login.

The agent is driven turn-by-turn by Claude: it calls ONE tool, observes the
result, then decides the next action. Subscribing and watching happen
sequentially per channel, exactly as the operator requested.
"""

from __future__ import annotations

import os
import time
from typing import Any

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from .base import AgentConfig, BaseAgent, tool


SYSTEM_PROMPT = """You are YoutuberAgent — a careful browser-automation agent for YouTube.

Your job: given a list of channel URLs, work through them ONE AT A TIME, fully
completing the workflow for each channel before moving to the next:

  1. login()  (only once, at the very start)
  2. For each channel in the list, in order:
        a. open_channel(url)
        b. subscribe()
        c. watch_latest_video(seconds=30)
        d. report progress to the operator
  3. close_browser() at the end

Rules:
- Call exactly ONE tool per turn. Wait for the tool result before deciding the next step.
- Never batch actions; never call the same tool twice in a single turn.
- If a tool reports an error (e.g. already subscribed, login challenge needed),
  surface it clearly and continue with the next step or channel.
- Do not invent channel URLs. Only use the URLs the operator gave you.
"""


class YoutuberAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            config=AgentConfig(
                system_prompt=SYSTEM_PROMPT,
                max_iterations=60,  # one channel = several tool calls
            )
        )
        self._pw = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_browser(self) -> Page:
        if self._page is not None:
            return self._page
        self._pw = sync_playwright().start()
        headless = os.getenv("YOUTUBER_HEADLESS", "false").lower() == "true"
        user_data_dir = os.getenv("YOUTUBER_USER_DATA_DIR", "./.yt_profile")
        # Persistent context => cookies + 2FA-trusted-device persist across runs
        self._context = self._pw.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=headless,
            viewport={"width": 1280, "height": 800},
        )
        self._page = self._context.new_page()
        return self._page

    def _click_first(self, selectors: list[str], timeout: int = 5000) -> bool:
        page = self._ensure_browser()
        for sel in selectors:
            try:
                el = page.wait_for_selector(sel, timeout=timeout, state="visible")
                if el:
                    el.click()
                    return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------------
    # Tools exposed to the model
    # ------------------------------------------------------------------

    @tool(
        description=(
            "Open YouTube and sign in with the credentials from environment "
            "variables YOUTUBE_EMAIL and YOUTUBE_PASSWORD. Returns 'ok' on "
            "success, or a message describing the challenge (2FA, CAPTCHA) "
            "that the human operator must solve manually in the open browser."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    def login(self) -> str:
        page = self._ensure_browser()
        email = os.environ.get("YOUTUBE_EMAIL")
        password = os.environ.get("YOUTUBE_PASSWORD")
        if not email or not password:
            return "Error: YOUTUBE_EMAIL and YOUTUBE_PASSWORD must be set in .env"

        page.goto("https://www.youtube.com", wait_until="domcontentloaded")
        # Already signed in? (avatar button present)
        try:
            page.wait_for_selector("button#avatar-btn", timeout=3000)
            return "ok (already signed in via persistent profile)"
        except Exception:
            pass

        # Click the Sign in button
        if not self._click_first([
            'a[aria-label="Sign in"]',
            'ytd-button-renderer a[href*="ServiceLogin"]',
        ]):
            return "Error: could not find the Sign in button"

        # Email
        page.wait_for_selector('input[type="email"]', timeout=15000)
        page.fill('input[type="email"]', email)
        page.click("#identifierNext")

        # Password
        page.wait_for_selector('input[type="password"]', timeout=15000, state="visible")
        page.fill('input[type="password"]', password)
        page.click("#passwordNext")

        # Wait for either the avatar (success) or a challenge page
        try:
            page.wait_for_selector("button#avatar-btn", timeout=30000)
            return "ok"
        except Exception:
            return (
                "Login appears to need manual intervention (2FA / device check / "
                "CAPTCHA). Please complete it in the open browser window, then "
                "ask me to retry or proceed."
            )

    @tool(
        description="Navigate the browser to the given YouTube channel URL.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to a YouTube channel page (e.g. https://www.youtube.com/@mkbhd)",
                }
            },
            "required": ["url"],
        },
    )
    def open_channel(self, url: str) -> str:
        page = self._ensure_browser()
        if "youtube.com" not in url:
            return f"Error: '{url}' does not look like a YouTube URL"
        page.goto(url, wait_until="domcontentloaded")
        try:
            page.wait_for_selector("ytd-channel-name, #channel-header", timeout=10000)
        except Exception:
            return f"Error: channel page did not load: {url}"
        return f"ok — on channel page: {url}"

    @tool(
        description=(
            "Click the Subscribe button on the currently open channel page. "
            "Returns 'ok', 'already subscribed', or an error."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    def subscribe(self) -> str:
        page = self._ensure_browser()
        # YouTube uses several variations of the Subscribe button
        sub_selectors = [
            'ytd-subscribe-button-renderer button:has-text("Subscribe")',
            'yt-button-shape button:has-text("Subscribe")',
            'button[aria-label^="Subscribe to"]',
        ]
        already_selectors = [
            'ytd-subscribe-button-renderer button:has-text("Subscribed")',
            'button[aria-label^="Unsubscribe"]',
        ]
        for sel in already_selectors:
            if page.locator(sel).count() > 0:
                return "already subscribed"
        if self._click_first(sub_selectors, timeout=8000):
            time.sleep(1.0)
            return "ok — subscribed"
        return "Error: could not find a Subscribe button on this page"

    @tool(
        description=(
            "Open the latest video on the currently open channel and let it "
            "play for the given number of seconds before returning."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "How long to keep the video playing before continuing. Default 30.",
                    "minimum": 5,
                    "maximum": 600,
                }
            },
            "required": ["seconds"],
        },
    )
    def watch_latest_video(self, seconds: int = 30) -> str:
        page = self._ensure_browser()
        # Go to the channel's Videos tab to pick the most recent upload
        try:
            videos_tab = page.locator('tp-yt-paper-tab:has-text("Videos"), yt-tab-shape:has-text("Videos")').first
            if videos_tab.count() > 0:
                videos_tab.click()
                page.wait_for_load_state("domcontentloaded")
                time.sleep(1.5)
        except Exception:
            pass

        first_video = page.locator(
            'ytd-rich-grid-media a#video-title-link, ytd-grid-video-renderer a#video-title'
        ).first
        try:
            first_video.wait_for(timeout=10000)
            first_video.click()
        except Exception:
            return "Error: could not find a video on this channel"

        # Wait for the player and let it play
        try:
            page.wait_for_selector("video", timeout=15000)
        except Exception:
            return "Error: video player did not load"
        # Ensure not paused
        page.evaluate(
            "() => { const v = document.querySelector('video'); if (v && v.paused) v.play(); }"
        )
        time.sleep(seconds)
        return f"ok — watched ~{seconds}s of the latest video"

    @tool(
        description="Close the browser and release resources. Call this when the workflow is fully complete.",
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    def close_browser(self) -> str:
        try:
            if self._context:
                self._context.close()
            if self._pw:
                self._pw.stop()
        finally:
            self._context = None
            self._page = None
            self._pw = None
        return "ok — browser closed"
