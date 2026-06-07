"""
Запуск PhotoAgent через Ollama (локальная LLM) + Automatic1111/ComfyUI.

Использование:
    python examples/run_photo_agent.py prompts/example_prompts.txt

Перед запуском:
    1. Запустите Ollama:
           ollama serve
       и потяните модель с поддержкой tool calling:
           ollama pull llama3.1

    2. Запустите Automatic1111 с API:
           python launch.py --api --listen
       или ComfyUI:
           python main.py --listen

    3. Задайте в .env (или переменных окружения):
           OLLAMA_MODEL=llama3.1
           OLLAMA_URL=http://localhost:11434
           PHOTO_BACKEND=automatic1111
           PHOTO_API_URL=http://127.0.0.1:7860
           PHOTO_OUTPUT_DIR=./outputs
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.photo_agent import PhotoAgent  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PhotoAgent — Ollama + локальный AI-генератор")
    p.add_argument("prompts_file", help="Путь к файлу с промптами")
    p.add_argument("--steps",    type=int,   default=25,    help="Шагов сэмплирования (default: 25)")
    p.add_argument("--cfg",      type=float, default=7.0,   help="CFG scale (default: 7.0)")
    p.add_argument("--size",     type=int,   default=512,   help="Ширина=Высота в пикселях (default: 512)")
    p.add_argument("--seed",     type=int,   default=-1,    help="Seed (-1 = случайный)")
    p.add_argument("--negative", type=str,
                   default="ugly, blurry, low quality, watermark",
                   help="Negative prompt")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    agent = PhotoAgent()

    # Применяем CLI-параметры к настройкам генерации
    agent._settings.update({
        "steps": args.steps,
        "cfg_scale": args.cfg,
        "width": args.size,
        "height": args.size,
        "seed": args.seed,
        "negative_prompt": args.negative,
    })

    print(f"LLM     : {agent.config.model} @ {agent.config.base_url}")
    print(f"Бэкенд  : {agent._photo_backend} @ {agent._photo_api_url}")
    print(f"Размер  : {args.size}x{args.size}  шагов: {args.steps}  cfg: {args.cfg}")
    print(f"Выход   : {agent._output_dir}\n")

    result = agent.run_batch(args.prompts_file)

    print("\n=== Ответ агента ===")
    print(result)
    print(f"\nВсего сгенерировано: {agent._total_done} | Ошибок: {agent._total_errors}")


if __name__ == "__main__":
    main()
