"""
PhotoRunner — батч-генератор изображений без LLM.

Читает промпты из файла построчно, отправляет каждый на локальный
AI-сервер (Automatic1111 или ComfyUI), ждёт результата, сохраняет PNG.

Никаких API-ключей не требуется — только локальный AI-бэкенд.

Переменные окружения (или передать явно в конструктор):
    PHOTO_BACKEND    — "automatic1111" (по умолчанию) или "comfyui"
    PHOTO_API_URL    — URL сервера  (по умолчанию http://127.0.0.1:7860)
    PHOTO_OUTPUT_DIR — папка для PNG (по умолчанию ./outputs)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 10  # секунд между попытками при недоступном бэкенде
_GENERATION_TIMEOUT = 300  # секунд ожидания одной генерации


@dataclass
class GenerationSettings:
    steps: int = 25
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    sampler_name: str = "DPM++ 2M Karras"
    negative_prompt: str = "ugly, blurry, low quality, watermark"
    seed: int = -1
    batch_size: int = 1


@dataclass
class GenerationResult:
    prompt: str
    success: bool
    file_path: str = ""
    seed: int = -1
    elapsed: float = 0.0
    error: str = ""


class PhotoRunner:
    """Прямой батч-генератор изображений. Без LLM, без API-ключей."""

    def __init__(
        self,
        backend: str | None = None,
        api_url: str | None = None,
        output_dir: str | None = None,
        settings: GenerationSettings | None = None,
    ) -> None:
        self.backend = (backend or os.getenv("PHOTO_BACKEND", "forge")).lower()
        self.api_url = (api_url or os.getenv("PHOTO_API_URL", "http://127.0.0.1:7860")).rstrip("/")
        self.output_dir = Path(output_dir or os.getenv("PHOTO_OUTPUT_DIR", "./outputs"))
        self.settings = settings or GenerationSettings()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def _load_prompts(self, prompts_file: str) -> list[str]:
        path = Path(prompts_file)
        if not path.exists():
            logger.error("Файл промптов не найден: %s", prompts_file)
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]

    def run_batch(self, prompts_file: str) -> list[GenerationResult]:
        """Обработать все промпты из файла. Возвращает список результатов."""
        prompts = self._load_prompts(prompts_file)
        if not prompts:
            logger.warning("Файл промптов пуст или не найден: %s", prompts_file)
            return []

        total = len(prompts)
        logger.info("Загружено %d промптов из %s", total, prompts_file)

        results: list[GenerationResult] = []
        for idx, prompt in enumerate(prompts, start=1):
            logger.info("[%d/%d] Генерация: %s", idx, total, prompt[:60])
            result = self.generate(prompt)
            results.append(result)
            if result.success:
                logger.info("  Сохранено: %s (%.1fs)", result.file_path, result.elapsed)
            else:
                logger.error("  Ошибка: %s", result.error)

        ok = sum(1 for r in results if r.success)
        logger.info("Готово: %d/%d успешно", ok, total)
        return results

    def generate(self, prompt: str) -> GenerationResult:
        """Сгенерировать одно изображение. Повторяет при ошибке сети."""
        if not prompt.strip():
            return GenerationResult(prompt=prompt, success=False, error="пустой промпт")

        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                t0 = time.time()
                if self.backend == "comfyui":
                    file_path, seed = self._generate_comfyui(prompt)
                else:
                    file_path, seed = self._generate_automatic1111(prompt)
                return GenerationResult(
                    prompt=prompt,
                    success=True,
                    file_path=file_path,
                    seed=seed,
                    elapsed=time.time() - t0,
                )
            except requests.exceptions.ConnectionError as exc:
                if attempt < _RETRY_ATTEMPTS:
                    logger.warning(
                        "Бэкенд недоступен, повтор через %ds (попытка %d/%d)",
                        _RETRY_DELAY, attempt, _RETRY_ATTEMPTS,
                    )
                    time.sleep(_RETRY_DELAY)
                else:
                    return GenerationResult(
                        prompt=prompt, success=False,
                        error=f"нет связи с {self.backend} @ {self.api_url}: {exc}",
                    )
            except Exception as exc:  # noqa: BLE001
                return GenerationResult(prompt=prompt, success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Automatic1111
    # ------------------------------------------------------------------

    def _generate_automatic1111(self, prompt: str) -> tuple[str, int]:
        s = self.settings
        payload = {
            "prompt": prompt,
            "negative_prompt": s.negative_prompt,
            "steps": s.steps,
            "cfg_scale": s.cfg_scale,
            "width": s.width,
            "height": s.height,
            "sampler_name": s.sampler_name,
            "seed": s.seed,
            "batch_size": s.batch_size,
        }
        resp = requests.post(
            f"{self.api_url}/sdapi/v1/txt2img",
            json=payload,
            timeout=_GENERATION_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        images = data.get("images", [])
        if not images:
            raise ValueError("API не вернул изображений")

        info = json.loads(data.get("info", "{}"))
        seed = info.get("seed", -1)

        file_path = self._save_base64(images[0], prompt)
        return file_path, seed

    # ------------------------------------------------------------------
    # ComfyUI
    # ------------------------------------------------------------------

    def _generate_comfyui(self, prompt: str) -> tuple[str, int]:
        client_id = str(uuid.uuid4())
        s = self.settings
        actual_seed = s.seed if s.seed != -1 else int(time.time() * 1000) % (2**32)

        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": actual_seed,
                    "steps": s.steps,
                    "cfg": s.cfg_scale,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
            },
            "4": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "v1-5-pruned-emaonly.ckpt"}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"batch_size": s.batch_size, "height": s.height, "width": s.width}},
            "6": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": prompt, "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": s.negative_prompt, "clip": ["4", 1]}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"filename_prefix": "PhotoRunner", "images": ["8", 0]}},
        }

        resp = requests.post(
            f"{self.api_url}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=30,
        )
        resp.raise_for_status()
        prompt_id: str = resp.json()["prompt_id"]

        # Polling до завершения
        deadline = time.time() + _GENERATION_TIMEOUT
        while time.time() < deadline:
            hist = requests.get(f"{self.api_url}/history/{prompt_id}", timeout=10).json()
            if prompt_id in hist:
                for node_out in hist[prompt_id].get("outputs", {}).values():
                    for img in node_out.get("images", []):
                        params = {"filename": img["filename"]}
                        if img.get("subfolder"):
                            params["subfolder"] = img["subfolder"]
                        img_bytes = requests.get(
                            f"{self.api_url}/view", params=params, timeout=30
                        ).content
                        file_path = self._save_bytes(img_bytes, prompt)
                        return file_path, actual_seed
            time.sleep(2)

        raise TimeoutError("ComfyUI не завершил генерацию за 5 минут")

    # ------------------------------------------------------------------
    # Сохранение файлов
    # ------------------------------------------------------------------

    def _make_path(self, prompt: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        slug = "".join(
            c if c.isalnum() or c in " _-" else ""
            for c in prompt[:40]
        ).strip().replace(" ", "_")
        return self.output_dir / f"{ts}_{slug}.png"

    def _save_base64(self, b64: str, prompt: str) -> str:
        return self._save_bytes(base64.b64decode(b64), prompt)

    def _save_bytes(self, data: bytes, prompt: str) -> str:
        path = self._make_path(prompt)
        path.write_bytes(data)
        return str(path)
