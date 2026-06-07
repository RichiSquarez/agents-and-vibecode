"""
Пример запуска PhotoAgent.

Использование:
    python examples/run_photo_agent.py prompts/example_prompts.txt

Перед запуском:
    1. Запустите Automatic1111 с флагом --api:
           python launch.py --api --listen
       или ComfyUI:
           python main.py --listen

    2. Убедитесь, что в .env заданы переменные:
           ANTHROPIC_API_KEY=sk-ant-...
           PHOTO_BACKEND=automatic1111   # или comfyui
           PHOTO_API_URL=http://127.0.0.1:7860
"""

import logging
import sys
from pathlib import Path

# Чтобы работал импорт из корня проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.photo_agent import PhotoAgent  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python examples/run_photo_agent.py <prompts_file>")
        print("Example: python examples/run_photo_agent.py prompts/example_prompts.txt")
        sys.exit(1)

    prompts_file = sys.argv[1]
    agent = PhotoAgent()

    # Опционально: настройте параметры генерации
    # agent._settings["steps"] = 30
    # agent._settings["width"] = 768
    # agent._settings["height"] = 768
    # agent._settings["cfg_scale"] = 7.5

    print(f"Starting photo generation from: {prompts_file}")
    print(f"Backend: {agent._backend} @ {agent._api_url}")
    print(f"Output dir: {agent._output_dir}\n")

    result = agent.run_batch(prompts_file)
    print("\n=== Agent final response ===")
    print(result)
    print(f"\nTotal generated: {agent._total_done} | Errors: {agent._total_errors}")


if __name__ == "__main__":
    main()
