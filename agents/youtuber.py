"""
YoutuberAgent — a BaseAgent subclass that drives a real Chromium browser via
Playwright to run an end-to-end YouTube workflow:

  1. log in once
  2. for each channel query (search term):
        search YouTube  →  open channel  →  subscribe  →  watch latest video
        to the end      →  like the video
  3. open the operator's own channel:
        read bio + links
        if bio looks empty / placeholder-ish → rewrite it
        if no external link is set → shorten the configured "original link"
        via clck.ru and add it with a random label ("Гайд" or "Инструкция")
  4. close the browser

NOTE: Automating Google services violates Google's ToS and can flag accounts.
Use a dedicated profile (the persistent user-data-dir is enabled by default).
Run headful so you can hand-solve any 2FA / CAPTCHA challenge.
"""

from __future__ import annotations

import os
import random
import re
import time

from playwright.sync_api import BrowserContext, Page, sync_playwright

from .base import AgentConfig, BaseAgent, tool


SYSTEM_PROMPT = """You are YoutuberAgent — a careful browser-automation agent.

You receive (a) a list of channel search queries, (b) an "original link" to add
to the operator's own channel, and optionally (c) a desired bio.

Execute this workflow ONE STEP AT A TIME. Call exactly ONE tool per turn,
read its result, then decide the next step. Never batch or skip.

Phase A — Channels loop:
  1. login()
  2. For each channel query in order:
       a. search_channel(query)       — opens the channel page
       b. subscribe()
       c. play_latest_video()         — opens the latest upload
       d. watch_current_video_fully() — blocks until the video ends
       e. like_current_video()
       Then move to the next channel. Never start a new channel until the
       current one is fully done.

Phase B — Own channel maintenance (after ALL channels are processed):
  3. go_to_my_channel()
  4. read_my_channel_about()
     Decide:
       - If the returned bio is empty, very short (< 20 chars), or looks like
         a placeholder ("Welcome to my channel", "No description", default
         language template, etc.) AND the operator provided a desired_bio,
         call edit_bio(new_text=<desired_bio>). Otherwise leave the bio alone.
       - If `links` is empty, take the operator-provided original_link, call
         shorten_link_clck(url=original_link), then call add_channel_link with
         the shortened URL and a name chosen RANDOMLY between "Гайд" and
         "Инструкция" (use pick_random_link_name()). If a link is already
         present, do not add another.

Phase C — Finish:
  5. close_browser()
  6. Reply with a concise final summary of what was done per channel.

Rules:
- Exactly ONE tool call per turn.
- If any tool returns an error, report it, then continue with the next safe step.
- Do not invent URLs, channel handles, or bio text. Use only what the operator gave you.
"""


class YoutuberAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            config=AgentConfig(
                system_prompt=SYSTEM_PROMPT,
                # watching full videos = many turns; raise the safety cap
                max_iterations=200,
            )
        )
        self._pw = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    # ------------------------------------------------------------------
    # Browser lifecycle
    # ------------------------------------------------------------------

    def _ensure_browser(self) -> Page:
        if self._page is not None:
            return self._page
        self._pw = sync_playwright().start()
        headless = os.getenv("YOUTUBER_HEADLESS", "false").lower() == "true"
        user_data_dir = os.getenv("YOUTUBER_USER_DATA_DIR", "./.yt_profile")
        self._context = self._pw.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=headless,
            viewport={"width": 1366, "height": 900},
        )
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
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

    def _studio_goto_basic_info(self) -> Page:
        """Navigate to YouTube Studio → Customization → Basic info.

        Resolves the real channel ID from the Studio redirect so the URL is
        always correct regardless of account.  Falls back to sidebar clicks
        if the URL pattern doesn't match.
        """
        page = self._ensure_browser()
        page.goto("https://studio.youtube.com/", wait_until="domcontentloaded")
        time.sleep(2)

        # Studio redirects to /channel/{REAL_ID}/ — extract it
        m = re.search(r"/channel/(UC[^/?#]+)", page.url)
        if m:
            cid = m.group(1)
            page.goto(
                f"https://studio.youtube.com/channel/{cid}/editing/details",
                wait_until="domcontentloaded",
            )
        else:
            # Fallback: use the sidebar Customization link
            for text in ("Customization", "Настройка канала", "Настройка"):
                try:
                    link = page.get_by_role("link", name=re.compile(text, re.IGNORECASE)).first
                    if link.count() > 0:
                        link.click()
                        page.wait_for_load_state("domcontentloaded")
                        break
                except Exception:
                    continue

        # Make sure we're on the Basic info tab
        for tab_text in ("Basic info", "Основная информация"):
            try:
                tab = page.get_by_role("tab", name=re.compile(tab_text, re.IGNORECASE)).first
                if tab.count() > 0:
                    tab.click()
                    time.sleep(1)
                    break
            except Exception:
                continue

        time.sleep(1.5)
        return page

    # ------------------------------------------------------------------
    # Phase A — login + per-channel loop
    # ------------------------------------------------------------------

    @tool(
        description=(
            "Open YouTube and sign in using YOUTUBE_EMAIL / YOUTUBE_PASSWORD "
            "env vars. Returns 'ok' on success, or a message indicating that "
            "the operator must hand-solve a 2FA / CAPTCHA challenge in the "
            "open browser window."
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
        try:
            page.wait_for_selector("button#avatar-btn", timeout=3000)
            return "ok (already signed in via persistent profile)"
        except Exception:
            pass

        if not self._click_first([
            'a[aria-label="Sign in"]',
            'ytd-button-renderer a[href*="ServiceLogin"]',
        ]):
            return "Error: could not find the Sign in button"

        page.wait_for_selector('input[type="email"]', timeout=15000)
        page.fill('input[type="email"]', email)
        page.click("#identifierNext")

        page.wait_for_selector('input[type="password"]', timeout=15000, state="visible")
        page.fill('input[type="password"]', password)
        page.click("#passwordNext")

        try:
            page.wait_for_selector("button#avatar-btn", timeout=45000)
            return "ok"
        except Exception:
            return (
                "Login needs manual intervention (2FA / device check / CAPTCHA). "
                "Please complete it in the open browser, then ask me to continue."
            )

    @tool(
        description=(
            "Type a query into the YouTube search bar and open the first "
            "channel result. Returns the channel URL that was opened."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Channel name or search term, e.g. 'MKBHD'.",
                }
            },
            "required": ["query"],
        },
    )
    def search_channel(self, query: str) -> str:
        page = self._ensure_browser()
        page.goto("https://www.youtube.com", wait_until="domcontentloaded")
        try:
            search = page.wait_for_selector('input#search', timeout=10000)
            search.click()
            search.fill("")
            search.type(query, delay=40)
            page.keyboard.press("Enter")
            page.wait_for_load_state("domcontentloaded")
        except Exception as exc:
            return f"Error: search failed: {exc}"

        # Prefer a channel-card result; fall back to first ytd-channel-renderer
        time.sleep(1.5)
        channel_link = page.locator(
            'ytd-channel-renderer a#main-link, ytd-channel-renderer a.channel-link'
        ).first
        try:
            channel_link.wait_for(timeout=8000)
            href = channel_link.get_attribute("href") or ""
            channel_link.click()
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_selector("ytd-channel-name, #channel-header", timeout=10000)
            return f"ok — opened channel for '{query}' ({href})"
        except Exception:
            return f"Error: no channel result found for '{query}'"

    @tool(
        description=(
            "Click the Subscribe button on the currently open channel page. "
            "Returns 'ok', 'already subscribed', or an error."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    def subscribe(self) -> str:
        page = self._ensure_browser()
        already_sels = [
            'ytd-subscribe-button-renderer button:has-text("Subscribed")',
            'button[aria-label^="Unsubscribe"]',
            'yt-button-shape button:has-text("Subscribed")',
        ]
        for sel in already_sels:
            if page.locator(sel).count() > 0:
                return "already subscribed"
        # Role-based check (language-independent "Unsubscribe" aria-pressed pattern)
        try:
            if page.get_by_role("button", name=re.compile(r"unsubscribe", re.IGNORECASE)).count() > 0:
                return "already subscribed"
        except Exception:
            pass

        sub_sels = [
            'ytd-subscribe-button-renderer button:has-text("Subscribe")',
            'yt-button-shape button:has-text("Subscribe")',
            'button[aria-label^="Subscribe to"]',
        ]
        if self._click_first(sub_sels, timeout=8000):
            time.sleep(1.0)
            return "ok — subscribed"
        # Final fallback: role-based
        try:
            btn = page.get_by_role("button", name=re.compile(r"^subscribe", re.IGNORECASE)).first
            btn.wait_for(state="visible", timeout=5000)
            btn.click()
            time.sleep(1.0)
            return "ok — subscribed"
        except Exception:
            pass
        return "Error: could not find a Subscribe button"

    @tool(
        description=(
            "Open the latest uploaded video on the currently open channel "
            "page. Returns 'ok' once the video player is visible."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    def play_latest_video(self) -> str:
        page = self._ensure_browser()
        try:
            tab = page.locator(
                'tp-yt-paper-tab:has-text("Videos"), yt-tab-shape:has-text("Videos")'
            ).first
            if tab.count() > 0:
                tab.click()
                page.wait_for_load_state("domcontentloaded")
                time.sleep(1.5)
        except Exception:
            pass

        first = page.locator(
            'ytd-rich-grid-media a#video-title-link, ytd-grid-video-renderer a#video-title'
        ).first
        try:
            first.wait_for(timeout=10000)
            first.click()
            page.wait_for_selector("video", timeout=15000)
            return "ok — video player opened"
        except Exception:
            return "Error: could not open the latest video"

    @tool(
        description=(
            "Watch the currently playing YouTube video until it ends or the "
            "cap is reached. Default cap is 480 seconds (8 minutes); pass "
            "max_seconds to override."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "max_seconds": {
                    "type": "integer",
                    "minimum": 30,
                    "maximum": 480,
                    "description": "Safety cap in seconds. Default and maximum is 480 (8 min).",
                }
            },
            "required": [],
        },
    )
    def watch_current_video_fully(self, max_seconds: int = 480) -> str:
        page = self._ensure_browser()
        # Make sure it's playing and not muted-paused
        page.evaluate(
            "() => { const v = document.querySelector('video'); if (v) { v.muted = false; if (v.paused) v.play(); } }"
        )
        start = time.time()
        while time.time() - start < max_seconds:
            state = page.evaluate(
                """() => {
                    const v = document.querySelector('video');
                    if (!v) return {ok:false};
                    return {ok:true, ended:v.ended, t:v.currentTime, d:v.duration, paused:v.paused};
                }"""
            )
            if not state.get("ok"):
                return "Error: no video element on the page"
            if state.get("ended"):
                return f"ok — video ended (duration {state.get('d')}s)"
            # If paused (e.g. ad gate), nudge it
            if state.get("paused"):
                page.evaluate("() => { const v=document.querySelector('video'); v && v.play(); }")
            d = state.get("d") or 0
            t = state.get("t") or 0
            if d and t >= d - 0.5:
                return f"ok — reached end of video ({t:.1f}/{d:.1f}s)"
            time.sleep(5)
        return f"Stopped — hit max_seconds={max_seconds} cap before the video ended"

    @tool(
        description=(
            "Click the Like button on the currently playing video. Returns "
            "'ok', 'already liked', or an error."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    def like_current_video(self) -> str:
        page = self._ensure_browser()
        # Scroll down a bit so the like bar is in view
        page.evaluate("window.scrollBy(0, 300)")
        time.sleep(0.5)

        already_sels = [
            'button[aria-pressed="true"][aria-label*="like" i]',
            'button[aria-pressed="true"][title*="unlike" i]',
        ]
        for sel in already_sels:
            if page.locator(sel).count() > 0:
                return "already liked"
        try:
            if page.get_by_role("button", name=re.compile(r"unlike", re.IGNORECASE)).count() > 0:
                return "already liked"
        except Exception:
            pass

        like_sels = [
            # Modern segmented like button (2024+)
            'segmented-like-dislike-button-view-model button:first-child',
            'like-button-view-model button',
            # Older renderers
            'ytd-toggle-button-renderer button[aria-label*="like" i]',
            'button[aria-label^="like this video" i]',
            'button[aria-label^="Like"]',
        ]
        if self._click_first(like_sels, timeout=8000):
            time.sleep(0.8)
            return "ok — liked"
        # Role-based fallback
        try:
            btn = page.get_by_role("button", name=re.compile(r"^like", re.IGNORECASE)).first
            btn.wait_for(state="visible", timeout=5000)
            btn.click()
            time.sleep(0.8)
            return "ok — liked"
        except Exception:
            pass
        return "Error: could not find a Like button"

    # ------------------------------------------------------------------
    # Phase B — own-channel bio / link maintenance
    # ------------------------------------------------------------------

    @tool(
        description="Navigate to the operator's own YouTube channel page.",
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    def go_to_my_channel(self) -> str:
        page = self._ensure_browser()
        page.goto("https://www.youtube.com/feed/you", wait_until="domcontentloaded")
        # The "You" feed has a "View channel" button
        try:
            view = page.locator('a:has-text("View channel"), a:has-text("Your channel")').first
            view.wait_for(timeout=8000)
            view.click()
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_selector("ytd-channel-name", timeout=10000)
            return f"ok — on own channel: {page.url}"
        except Exception:
            return f"Error: could not navigate to own channel (current url: {page.url})"

    @tool(
        description=(
            "Read the bio (description) and external links from the operator's "
            "own channel. Returns a JSON-ish string with keys 'bio' and 'links'."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    def read_my_channel_about(self) -> str:
        # Read from Studio Basic info — more stable than the public channel page
        page = self._studio_goto_basic_info()

        bio = ""
        try:
            # The description textarea in Studio Basic info
            desc_box = page.locator(
                "ytcp-form-textarea #textbox, "
                "#description-container #textbox, "
                "textarea[aria-label*='description' i], "
                "textarea[aria-label*='описание' i]"
            ).first
            if desc_box.count() > 0:
                bio = (desc_box.inner_text(timeout=5000) or "").strip()
        except Exception:
            pass

        links: list[str] = []
        try:
            # Links rows in Studio — each external link has a URL input
            url_inputs = page.locator(
                "ytcp-url-endpoint-input input, "
                "input[aria-label*='URL' i][value^='http']"
            )
            for i in range(url_inputs.count()):
                val = url_inputs.nth(i).get_attribute("value") or ""
                if val.startswith("http"):
                    links.append(val)
        except Exception:
            pass

        return f'{{"bio": {bio!r}, "links": {links!r}}}'

    @tool(
        description=(
            "Update the channel bio (description) in YouTube Studio. Pass the "
            "new bio text. Returns 'ok' or an error."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "new_text": {
                    "type": "string",
                    "description": "The new bio/description text to set.",
                }
            },
            "required": ["new_text"],
        },
    )
    def edit_bio(self, new_text: str) -> str:
        page = self._studio_goto_basic_info()
        try:
            desc = page.locator(
                "ytcp-form-textarea #textbox, "
                "#description-container #textbox, "
                "textarea[aria-label*='description' i], "
                "textarea[aria-label*='описание' i]"
            ).first
            desc.wait_for(timeout=15000)
            desc.click()
            # Select-all then replace (works regardless of existing content length)
            page.keyboard.press("Control+A")
            page.keyboard.press("Delete")
            time.sleep(0.3)
            desc.type(new_text, delay=15)
            # Click Publish / Save
            for pub_text in ("Publish", "Опубликовать", "Save", "Сохранить"):
                try:
                    btn = page.get_by_role("button", name=re.compile(pub_text, re.IGNORECASE)).first
                    if btn.count() > 0:
                        btn.wait_for(state="visible", timeout=5000)
                        btn.click()
                        break
                except Exception:
                    continue
            time.sleep(2.5)
            return "ok — bio updated"
        except Exception as exc:
            return f"Error: could not edit bio: {exc}"

    @tool(
        description=(
            "Open clck.ru in a new tab, paste the given URL, click the shorten "
            "button, and return the shortened URL."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The original URL to shorten via clck.ru",
                }
            },
            "required": ["url"],
        },
    )
    def shorten_link_clck(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            return f"Error: '{url}' is not a valid URL"
        page = self._ensure_browser()
        new_tab = self._context.new_page()  # type: ignore[union-attr]
        try:
            new_tab.goto("https://clck.ru/", wait_until="domcontentloaded")
            inp = new_tab.locator('input[name="url"], input#shortener-url, input[type="url"], input[type="text"]').first
            inp.wait_for(timeout=10000)
            inp.fill(url)
            # Submit (Enter or click the button)
            btn = new_tab.locator('button:has-text("Сократить"), button[type="submit"], input[type="submit"]').first
            if btn.count() > 0:
                btn.click()
            else:
                new_tab.keyboard.press("Enter")
            # Result appears in an input / span containing clck.ru/...
            result_el = new_tab.locator('input[value^="https://clck.ru/"], a[href^="https://clck.ru/"]').first
            result_el.wait_for(timeout=15000)
            short = (result_el.get_attribute("value")
                     or result_el.get_attribute("href")
                     or result_el.inner_text())
            short = (short or "").strip()
            return f"ok — shortened: {short}" if short else "Error: no shortened URL captured"
        except Exception as exc:
            return f"Error: clck.ru shortening failed: {exc}"
        finally:
            try:
                new_tab.close()
            except Exception:
                pass
            # Switch focus back to the main YouTube tab
            try:
                page.bring_to_front()
            except Exception:
                pass

    @tool(
        description=(
            "Add an external link to the operator's channel via YouTube Studio. "
            "Pass the URL and a display name."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Link URL (already shortened if needed)."},
                "name": {"type": "string", "description": 'Display title for the link, e.g. "Гайд".'},
            },
            "required": ["url", "name"],
        },
    )
    def add_channel_link(self, url: str, name: str) -> str:
        page = self._studio_goto_basic_info()
        try:
            # Click "Add link" / "Добавить ссылку"
            add_btn = None
            for btn_text in ("Add link", "Добавить ссылку", "Add", "Добавить"):
                try:
                    candidate = page.get_by_role(
                        "button", name=re.compile(btn_text, re.IGNORECASE)
                    ).first
                    if candidate.count() > 0:
                        candidate.wait_for(state="visible", timeout=8000)
                        add_btn = candidate
                        break
                except Exception:
                    continue
            if add_btn is None:
                # CSS fallback
                add_btn = page.locator(
                    "ytcp-button:has-text('Add'), ytcp-button:has-text('Добавить')"
                ).last
            add_btn.click()
            time.sleep(1.0)

            # After click, new row of inputs appears — fill the LAST (newest) ones
            title_field = page.locator(
                "input[aria-label*='title' i], "
                "input[placeholder*='title' i], "
                "input[aria-label*='назван' i], "
                "ytcp-url-endpoint-input input[type='text']:not([type='url'])"
            ).last
            url_field = page.locator(
                "ytcp-url-endpoint-input input[type='url'], "
                "input[aria-label*='url' i], "
                "input[placeholder*='url' i]"
            ).last
            title_field.wait_for(state="visible", timeout=8000)
            title_field.fill(name)
            url_field.wait_for(state="visible", timeout=8000)
            url_field.fill(url)
            time.sleep(0.5)

            # Publish
            for pub_text in ("Publish", "Опубликовать", "Save", "Сохранить"):
                try:
                    btn = page.get_by_role("button", name=re.compile(pub_text, re.IGNORECASE)).first
                    if btn.count() > 0:
                        btn.wait_for(state="visible", timeout=5000)
                        btn.click()
                        break
                except Exception:
                    continue
            time.sleep(2.5)
            return f"ok — link added ({name} -> {url})"
        except Exception as exc:
            return f"Error: could not add channel link: {exc}"

    @tool(
        description='Return one of "Гайд" or "Инструкция" chosen uniformly at random.',
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    def pick_random_link_name(self) -> str:
        return random.choice(["Гайд", "Инструкция"])

    # ------------------------------------------------------------------
    # Phase C — teardown
    # ------------------------------------------------------------------

    @tool(
        description="Close the browser and release resources. Call after the workflow is fully complete.",
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
