"""
PhotoAgent — генератор изображений на локальной LLM (Ollama) + SD Forge.

Ollama управляет логикой, Stable Diffusion WebUI Forge генерирует изображения
через стандартный API /sdapi/v1/txt2img.

Переменные окружения:
    OLLAMA_MODEL     — LLM (по умолчанию llama3.1)
    OLLAMA_URL       — URL Ollama (по умолчанию http://localhost:11434)
    PHOTO_BACKEND    — "forge" или "automatic1111" (одно и то же API, default: forge)
    PHOTO_API_URL    — URL Forge (по умолчанию http://127.0.0.1:7860)
    PHOTO_OUTPUT_DIR — папка для PNG (по умолчанию ./outputs)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

import requests

from .ollama_base import OllamaAgent, OllamaConfig, tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are PhotoAgent — an automated image generation assistant using Stable Diffusion Forge.

Your job is to process a list of text prompts and generate one image per prompt.

Follow this workflow strictly:

1. Call load_prompts(file_path=<path>) to read the prompt file.
2. Loop:
   a. Call get_next_prompt() — if it returns "DONE", stop.
   b. Call generate_image(prompt=<prompt text>).
   c. Report file path, seed, and time briefly.
3. After all prompts, print a summary (total images, errors).

Rules:
- One tool call per turn.
- Never skip a prompt.
- On error — log it, continue to the next prompt.
- Use prompts exactly as loaded, do not modify them.
"""

_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 10
_GENERATION_TIMEOUT = 600  # Forge с SDXL/FLUX может быть медленнее


class PhotoAgent(OllamaAgent):
    def __init__(self) -> None:
        super().__init__(
            config=OllamaConfig(
                system_prompt=SYSTEM_PROMPT,
                max_iterations=500,
                temperature=0.1,
            )
        )
        self._photo_backend: str = os.getenv("PHOTO_BACKEND", "forge").lower()
        self._photo_api_url: str = os.getenv("PHOTO_API_URL", "http://127.0.0.1:7860").rstrip("/")
        self._output_dir = Path(os.getenv("PHOTO_OUTPUT_DIR", "./outputs"))
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._prompts: deque[str] = deque()
        self._total_done = 0
        self._total_errors = 0

        # Базовые настройки генерации
        self._settings: dict = {
            "steps": 20,
            "cfg_scale": 7.0,
            "width": 512,
            "height": 512,
            "sampler_name": "DPM++ 2M",
            "scheduler": "Karras",
            "negative_prompt": "ugly, blurry, low quality, watermark, deformed",
            "seed": -1,
            "batch_size": 1,
        }

        # HR Fix (high-res upscale после генерации)
        self._hr_settings: dict = {
            "enable_hr": False,
            "hr_scale": 2.0,
            "hr_upscaler": "R-ESRGAN 4x+",
            "hr_second_pass_steps": 10,
            "denoising_strength": 0.4,
        }

        # Forge/SD — дополнительные модули
        self._forge_settings: dict = {
            "checkpoint": None,  # None = текущая модель; строка = имя чекпоинта
            "vae": None,         # None = встроенная VAE модели
        }

    # ------------------------------------------------------------------
    # Инструменты
    # ------------------------------------------------------------------

    @tool(
        description="Load prompts from a text file. Lines starting with # and empty lines are ignored.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to prompts file."}
            },
            "required": ["file_path"],
        },
    )
    def load_prompts(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            return f"Error: file not found: {file_path}"
        lines = path.read_text(encoding="utf-8").splitlines()
        self._prompts = deque(
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        )
        return f"Loaded {len(self._prompts)} prompts from {file_path}"

    @tool(
        description="Return the next prompt from the queue. Returns 'DONE' when queue is empty.",
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    def get_next_prompt(self) -> str:
        if not self._prompts:
            return "DONE"
        return self._prompts.popleft()

    @tool(
        description=(
            "Update Stable Diffusion generation settings. All parameters optional.\n"
            "Basic: steps, cfg_scale, width, height, sampler_name, scheduler, negative_prompt, seed, batch_size.\n"
            "HR Fix: enable_hr (bool), hr_scale (1.5-4.0), hr_upscaler, hr_second_pass_steps, denoising_strength.\n"
            "Model: checkpoint (filename, e.g. 'realisticVisionV60.safetensors'), vae."
        ),
        input_schema={
            "type": "object",
            "properties": {
                # Базовые
                "steps":                 {"type": "integer", "minimum": 1, "maximum": 150},
                "cfg_scale":             {"type": "number",  "minimum": 1, "maximum": 30},
                "width":                 {"type": "integer", "enum": [256, 512, 640, 768, 896, 1024, 1280, 1536]},
                "height":                {"type": "integer", "enum": [256, 512, 640, 768, 896, 1024, 1280, 1536]},
                "sampler_name":          {"type": "string"},
                "scheduler":             {"type": "string"},
                "negative_prompt":       {"type": "string"},
                "seed":                  {"type": "integer", "minimum": -1},
                "batch_size":            {"type": "integer", "minimum": 1, "maximum": 4},
                # HR Fix
                "enable_hr":             {"type": "boolean"},
                "hr_scale":              {"type": "number", "minimum": 1.0, "maximum": 4.0},
                "hr_upscaler":           {"type": "string"},
                "hr_second_pass_steps":  {"type": "integer", "minimum": 1, "maximum": 50},
                "denoising_strength":    {"type": "number", "minimum": 0.0, "maximum": 1.0},
                # Модель
                "checkpoint":            {"type": "string", "description": "Checkpoint filename"},
                "vae":                   {"type": "string"},
            },
            "required": [],
        },
    )
    def set_generation_settings(self, **kwargs) -> str:
        updated = []
        for key, value in kwargs.items():
            if key in self._settings:
                self._settings[key] = value
                updated.append(f"{key}={value}")
            elif key in self._hr_settings:
                self._hr_settings[key] = value
                updated.append(f"{key}={value}")
            elif key in self._forge_settings:
                self._forge_settings[key] = value
                updated.append(f"{key}={value}")
        return f"Updated: {', '.join(updated)}" if updated else "No valid keys provided."

    @tool(
        description=(
            "List available checkpoints (models) loaded in Forge. "
            "Use the returned names with set_generation_settings(checkpoint=...)."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    def list_models(self) -> str:
        try:
            resp = requests.get(f"{self._photo_api_url}/sdapi/v1/sd-models", timeout=10)
            resp.raise_for_status()
            models = [m.get("model_name", m.get("title", "?")) for m in resp.json()]
            return f"Available models ({len(models)}): {', '.join(models)}"
        except Exception as exc:  # noqa: BLE001
            return f"Error listing models: {exc}"

    @tool(
        description=(
            "Generate an image for the given prompt using Stable Diffusion Forge. "
            "Blocks until the image is saved to disk. Returns file path, seed, and elapsed time."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Text prompt. LoRA syntax supported: <lora:name:weight>."}
            },
            "required": ["prompt"],
        },
    )
    def generate_image(self, prompt: str) -> str:
        if not prompt.strip():
            return "Error: empty prompt — skipped"

        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                result = self._generate_forge(prompt)
                self._total_done += 1
                return result
            except requests.exceptions.ConnectionError:
                if attempt < _RETRY_ATTEMPTS:
                    logger.warning("Forge недоступен, повтор через %ds (%d/%d)",
                                   _RETRY_DELAY, attempt, _RETRY_ATTEMPTS)
                    time.sleep(_RETRY_DELAY)
                else:
                    self._total_errors += 1
                    return (f"Error: cannot reach Forge @ {self._photo_api_url} "
                            f"after {_RETRY_ATTEMPTS} attempts. "
                            "Make sure Forge is running with --api flag.")
            except Exception as exc:  # noqa: BLE001
                self._total_errors += 1
                return f"Error: {exc}"

    # ------------------------------------------------------------------
    # Forge / Automatic1111 API
    # ------------------------------------------------------------------

    def _generate_forge(self, prompt: str) -> str:
        s = self._settings
        payload: dict = {
            "prompt": prompt,
            "negative_prompt": s["negative_prompt"],
            "steps": s["steps"],
            "cfg_scale": s["cfg_scale"],
            "width": s["width"],
            "height": s["height"],
            "sampler_name": s["sampler_name"],
            "scheduler": s["scheduler"],
            "seed": s["seed"],
            "batch_size": s["batch_size"],
            # HR Fix
            **{k: v for k, v in self._hr_settings.items()},
        }

        # Выбор чекпоинта через override_settings
        override: dict = {}
        if self._forge_settings.get("checkpoint"):
            override["sd_model_checkpoint"] = self._forge_settings["checkpoint"]
        if self._forge_settings.get("vae"):
            override["sd_vae"] = self._forge_settings["vae"]
        if override:
            payload["override_settings"] = override
            payload["override_settings_restore_afterwards"] = True

        t0 = time.time()
        resp = requests.post(
            f"{self._photo_api_url}/sdapi/v1/txt2img",
            json=payload,
            timeout=_GENERATION_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        images = data.get("images", [])
        if not images:
            raise ValueError("Forge вернул пустой список изображений")

        info = json.loads(data.get("info", "{}"))
        seed = info.get("seed", "?")

        out = self._save_base64(images[0], prompt)
        elapsed = time.time() - t0
        return f"ok — {out} | seed={seed} | {elapsed:.1f}s"

    # ------------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------------

    def _make_path(self, prompt: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = "".join(
            c if c.isalnum() or c in " _-" else "" for c in prompt[:40]
        ).strip().replace(" ", "_")
        return self._output_dir / f"{ts}_{slug}.png"

    def _save_base64(self, b64: str, prompt: str) -> str:
        path = self._make_path(prompt)
        path.write_bytes(base64.b64decode(b64))
        return str(path)

    # ------------------------------------------------------------------
    # Точка входа
    # ------------------------------------------------------------------

    def run_batch(self, prompts_file: str) -> str:
        return self.run(
            f"Generate images for all prompts in: {prompts_file}\n"
            "Load the prompts, then process each one in order."
        )
