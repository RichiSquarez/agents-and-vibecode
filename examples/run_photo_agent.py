"""
Запуск PhotoAgent: Ollama (LLM) + Stable Diffusion Forge (генерация).

Использование:
    python examples/run_photo_agent.py prompts/example_prompts.txt [опции]

Опции:
    --steps INT        Шагов сэмплирования (default: 20)
    --cfg FLOAT        CFG scale (default: 7.0)
    --width INT        Ширина (default: 512)
    --height INT       Высота (default: 512)
    --seed INT         Seed, -1 = случайный (default: -1)
    --sampler STR      Сэмплер (default: "DPM++ 2M")
    --scheduler STR    Scheduler (default: "Karras")
    --negative STR     Negative prompt
    --checkpoint STR   Имя чекпоинта (.safetensors), если нужно переключить
    --hr               Включить HR Fix (high-res upscale)
    --hr-scale FLOAT   Масштаб HR Fix (default: 2.0)
    --hr-steps INT     Шагов второго прохода (default: 10)
    --denoise FLOAT    Denoising strength для HR Fix (default: 0.4)

Перед запуском:
    1. ollama serve && ollama pull llama3.1
    2. Запустить Forge с флагом --api:
           python launch.py --api
       или через bat-файл с добавленным --api

Переменные в .env:
    OLLAMA_MODEL=llama3.1
    OLLAMA_URL=http://localhost:11434
    PHOTO_BACKEND=forge
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
    p = argparse.ArgumentParser(description="PhotoAgent — Ollama + Stable Diffusion Forge")
    p.add_argument("prompts_file", help="Файл с промптами (по одному на строку)")

    # Базовые
    p.add_argument("--steps",      type=int,   default=20,          help="Шагов (default: 20)")
    p.add_argument("--cfg",        type=float, default=7.0,         help="CFG scale (default: 7.0)")
    p.add_argument("--width",      type=int,   default=512,         help="Ширина (default: 512)")
    p.add_argument("--height",     type=int,   default=512,         help="Высота (default: 512)")
    p.add_argument("--seed",       type=int,   default=-1,          help="Seed (default: -1 = random)")
    p.add_argument("--sampler",    type=str,   default="DPM++ 2M",  help='Сэмплер (default: "DPM++ 2M")')
    p.add_argument("--scheduler",  type=str,   default="Karras",    help='Scheduler (default: "Karras")')
    p.add_argument("--negative",   type=str,
                   default="ugly, blurry, low quality, watermark, deformed",
                   help="Negative prompt")
    p.add_argument("--checkpoint", type=str,   default=None,        help="Имя чекпоинта Forge")

    # HR Fix
    p.add_argument("--hr",         action="store_true",             help="Включить HR Fix")
    p.add_argument("--hr-scale",   type=float, default=2.0,         help="Масштаб HR Fix (default: 2.0)")
    p.add_argument("--hr-steps",   type=int,   default=10,          help="Шагов второго прохода (default: 10)")
    p.add_argument("--denoise",    type=float, default=0.4,         help="Denoising strength (default: 0.4)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    agent = PhotoAgent()

    # Применяем CLI-параметры
    agent._settings.update({
        "steps": args.steps,
        "cfg_scale": args.cfg,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "sampler_name": args.sampler,
        "scheduler": args.scheduler,
        "negative_prompt": args.negative,
    })
    agent._hr_settings.update({
        "enable_hr": args.hr,
        "hr_scale": args.hr_scale,
        "hr_second_pass_steps": args.hr_steps,
        "denoising_strength": args.denoise,
    })
    if args.checkpoint:
        agent._forge_settings["checkpoint"] = args.checkpoint

    hr_info = f"  HR Fix: x{args.hr_scale}" if args.hr else "  HR Fix: выкл"
    ckpt = args.checkpoint or "текущий"

    print(f"LLM       : {agent.config.model} @ {agent.config.base_url}")
    print(f"Forge     : {agent._photo_api_url}")
    print(f"Чекпоинт  : {ckpt}")
    print(f"Размер    : {args.width}x{args.height}  шагов: {args.steps}  cfg: {args.cfg}  seed: {args.seed}")
    print(f"Сэмплер   : {args.sampler} / {args.scheduler}")
    print(hr_info)
    print(f"Выход     : {agent._output_dir}\n")

    result = agent.run_batch(args.prompts_file)

    print("\n=== Ответ агента ===")
    print(result)
    print(f"\nВсего сгенерировано: {agent._total_done} | Ошибок: {agent._total_errors}")


if __name__ == "__main__":
    main()
