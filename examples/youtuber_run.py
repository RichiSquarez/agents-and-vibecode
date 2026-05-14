"""
Run the YoutuberAgent against a list of channels.

The agent will, step by step:
  1. log in once
  2. for each channel, open it, subscribe, watch ~30s of the latest video
  3. close the browser

Make sure .env contains ANTHROPIC_API_KEY, YOUTUBE_EMAIL, YOUTUBE_PASSWORD.
Also run once:  playwright install chromium
"""

from agents import YoutuberAgent


CHANNELS = [
    "https://www.youtube.com/@mkbhd",
    "https://www.youtube.com/@veritasium",
    "https://www.youtube.com/@LinusTechTips",
]


def main() -> None:
    agent = YoutuberAgent()
    task = (
        "Please log in to YouTube, then go through this list of channels ONE "
        "AT A TIME and for each one: subscribe, then watch the latest video "
        "for 30 seconds, then move to the next. Close the browser when done.\n\n"
        f"Channels:\n" + "\n".join(f"- {c}" for c in CHANNELS)
    )
    final = agent.run(task)
    print("\n=== Agent final reply ===\n")
    print(final)


if __name__ == "__main__":
    main()
