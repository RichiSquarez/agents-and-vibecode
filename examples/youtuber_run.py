"""
Run the YoutuberAgent through the full workflow:

  login → for each channel (search → subscribe → watch fully → like)
        → review own channel bio + links (edit bio if placeholder,
          shorten ORIGINAL_LINK via clck.ru and add it as "Гайд"/"Инструкция"
          if no link is present)
        → close browser

Requirements:
  - .env contains ANTHROPIC_API_KEY, YOUTUBE_EMAIL, YOUTUBE_PASSWORD
  - Run once:   playwright install chromium
"""

from agents import YoutuberAgent


CHANNEL_QUERIES = [
    "MKBHD",
    "Veritasium",
    "Linus Tech Tips",
]

# The link the agent will shorten via clck.ru and add to your channel
# if no external link is currently set on your channel.
ORIGINAL_LINK = "https://example.com/my-real-destination"

# Optional — the agent will only replace the bio if it currently looks empty
# or placeholder-ish. Leave as None to never overwrite.
DESIRED_BIO: str | None = (
    "Привет! Здесь я делюсь гайдами и инструкциями по интересным темам. "
    "Подпишись, чтобы ничего не пропустить."
)


def main() -> None:
    agent = YoutuberAgent()

    task = f"""Please run the full YouTube workflow described in your instructions.

Channel queries (in order):
{chr(10).join(f"- {q}" for q in CHANNEL_QUERIES)}

original_link = {ORIGINAL_LINK!r}
desired_bio   = {DESIRED_BIO!r}

For each channel: search_channel → subscribe → play_latest_video →
watch_current_video_fully → like_current_video, then move on.
After all channels, go to my channel, read the about info, and apply the
bio / link maintenance rules from your system prompt. Finally close the
browser and give me a short per-channel summary."""

    final = agent.run(task)
    print("\n=== Agent final reply ===\n")
    print(final)


if __name__ == "__main__":
    main()
