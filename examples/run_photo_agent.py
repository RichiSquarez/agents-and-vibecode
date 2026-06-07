"""
Запуск батч-генерации изображений через PhotoRunner (без LLM).

Использование:
    python examples/run_photo_agent.py prompts/example_prompts.txt [--steps 30] [--size 768]

Перед запуском — запустите Automatic1111 с флагом --api:
    python launch.py --api --listen

или ComfyUI:
    python main.py --listen

Переменные окружения (или задать в .env):
    PHOTO_BACKEND=automatic1111   # или comfyui
    PHOTO_API_URL=http://127.0.0.1:7860
    PHOTO_OUTPUT_DIR=./outputs
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.photo_runner import GenerationSettings, PhotoRunner  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Батч-генерация изображений через локальный AI")
    p.add_argument("prompts_file", help="Путь к файлу с промптами")
    p.add_argument("--steps",   type=int,   default=25,    help="Шагов сэмплирования (default: 25)")
    p.add_argument("--cfg",     type=float, default=7.0,   help="CFG scale (default: 7.0)")
    p.add_argument("--size",    type=int,   default=512,   help="Ширина и высота в пикселях (default: 512)")
    p.add_argument("--seed",    type=int,   default=-1,    help="Seed (-1 = случайный)")
    p.add_argument("--sampler", type=str,   default="DPM++ 2M Karras", help="Сэмплер")
    p.add_argument("--negative", type=str,  default="ugly, blurry, low quality, watermark",
                   help="Negative prompt")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    settings = GenerationSettings(
        steps=args.steps,
        cfg_scale=args.cfg,
        width=args.size,
        height=args.size,
        sampler_name=args.sampler,
        negative_prompt=args.negative,
        seed=args.seed,
    )

    runner = PhotoRunner(settings=settings)

    print(f"Бэкенд : {runner.backend} @ {runner.api_url}")
    print(f"Размер : {settings.width}x{settings.height}  шагов: {settings.steps}  cfg: {settings.cfg_scale}")
    print(f"Выход  : {runner.output_dir}\n")

    results = runner.run_batch(args.prompts_file)

    print("\n=== Итог ===")
    ok = [r for r in results if r.success]
    err = [r for r in results if not r.success]
    print(f"Успешно: {len(ok)}/{len(results)}")
    for r in ok:
        print(f"  [OK]  {r.file_path}  seed={r.seed}  {r.elapsed:.1f}s")
    for r in err:
        print(f"  [ERR] {r.prompt[:50]!r}  →  {r.error}")


if __name__ == "__main__":
    main()
