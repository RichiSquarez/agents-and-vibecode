"""
PhotoAgent — генератор изображений на локальной LLM (Ollama).

Ollama управляет логикой (читает промпты, решает когда звонить инструментам),
локальный Automatic1111 или ComfyUI генерирует сами изображения.

Переменные окружения:
    OLLAMA_MODEL     — LLM для оркестрации (по умолчанию llama3.1)
    OLLAMA_URL       — URL Ollama (по умолчанию http://localhost:11434)
    PHOTO_BACKEND    — "automatic1111" (по умолчанию) или "comfyui"
    PHOTO_API_URL    — URL генератора (по умолчанию http://127.0.0.1:7860)
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

SYSTEM_PROMPT = """You are PhotoAgent — an automated image generation assistant.

Your job is to process a list of text prompts and generate one image per prompt
using the local AI image generation backend.

Follow this workflow strictly:

1. Call load_prompts(file_path=<path>) to read the prompt file.
2. Loop:
   a. Call get_next_prompt() — if it returns "DONE", stop the loop.
   b. Call generate_image(prompt=<the prompt text>) and wait for the result.
   c. Report the saved file path and generation stats briefly.
3. After all prompts are processed, print a short summary (total images, errors).

Rules:
- Call exactly ONE tool per turn. Never batch tool calls.
- Never skip a prompt.
- If generate_image returns an error, log it and continue to the next prompt.
- Do not modify prompts — use them exactly as loaded.
"""

_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 10
_GENERATION_TIMEOUT = 300


class PhotoAgent(OllamaAgent):
    def __init__(self) -> None:
        super().__init__(
            config=OllamaConfig(
                system_prompt=SYSTEM_PROMPT,
                max_iterations=500,
                temperature=0.1,  # детерминированнее для tool-calling
            )
        )
        self._photo_backend: str = os.getenv("PHOTO_BACKEND", "automatic1111").lower()
        self._photo_api_url: str = os.getenv("PHOTO_API_URL", "http://127.0.0.1:7860").rstrip("/")
        self._output_dir = Path(os.getenv("PHOTO_OUTPUT_DIR", "./outputs"))
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._prompts: deque[str] = deque()
        self._total_done = 0
        self._total_errors = 0

        self._settings: dict = {
            "steps": 25,
            "cfg_scale": 7.0,
            "width": 512,
            "height": 512,
            "sampler_name": "DPM++ 2M Karras",
            "negative_prompt": "ugly, blurry, low quality, watermark",
            "seed": -1,
            "batch_size": 1,
        }

    # ------------------------------------------------------------------
    # Инструменты
    # ------------------------------------------------------------------

    @tool(
        description="Load prompts from a text file. Each non-empty, non-comment line is one prompt.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the prompts file."}
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
        description="Return the next prompt from the queue. Returns 'DONE' when all prompts are processed.",
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    def get_next_prompt(self) -> str:
        if not self._prompts:
            return "DONE"
        return self._prompts.popleft()

    @tool(
        description=(
            "Update image generation settings. All parameters are optional. "
            "Keys: steps, cfg_scale, width, height, sampler_name, negative_prompt, seed, batch_size."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "steps":           {"type": "integer", "minimum": 1, "maximum": 150},
                "cfg_scale":       {"type": "number",  "minimum": 1, "maximum": 30},
                "width":           {"type": "integer", "enum": [256, 512, 768, 1024]},
                "height":          {"type": "integer", "enum": [256, 512, 768, 1024]},
                "sampler_name":    {"type": "string"},
                "negative_prompt": {"type": "string"},
                "seed":            {"type": "integer", "minimum": -1},
                "batch_size":      {"type": "integer", "minimum": 1, "maximum": 4},
            },
            "required": [],
        },
    )
    def set_generation_settings(self, **kwargs) -> str:
        updated = [f"{k}={v}" for k, v in kwargs.items() if k in self._settings and not self._settings.update({k: v})]  # type: ignore[func-returns-value]
        return f"Updated: {', '.join(updated)}" if updated else f"No valid keys. Current: {self._settings}"

    @tool(
        description=(
            "Generate an image for the given prompt using the local AI backend. "
            "Blocks until the image is saved. Returns file path and stats."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Text prompt for image generation."}
            },
            "required": ["prompt"],
        },
    )
    def generate_image(self, prompt: str) -> str:
        if not prompt.strip():
            return "Error: empty prompt — skipped"

        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                if self._photo_backend == "comfyui":
                    result = self._generate_comfyui(prompt)
                else:
                    result = self._generate_automatic1111(prompt)
                self._total_done += 1
                return result
            except requests.exceptions.ConnectionError:
                if attempt < _RETRY_ATTEMPTS:
                    logger.warning("Backend unreachable, retry in %ds (%d/%d)",
                                   _RETRY_DELAY, attempt, _RETRY_ATTEMPTS)
                    time.sleep(_RETRY_DELAY)
                else:
                    self._total_errors += 1
                    return (f"Error: cannot reach {self._photo_backend} @ {self._photo_api_url} "
                            f"after {_RETRY_ATTEMPTS} attempts")
            except Exception as exc:  # noqa: BLE001
                self._total_errors += 1
                return f"Error: {exc}"

    # ------------------------------------------------------------------
    # Бэкенды
    # ------------------------------------------------------------------

    def _generate_automatic1111(self, prompt: str) -> str:
        s = self._settings
        t0 = time.time()
        resp = requests.post(
            f"{self._photo_api_url}/sdapi/v1/txt2img",
            json={"prompt": prompt, **s},
            timeout=_GENERATION_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        images = data.get("images", [])
        if not images:
            raise ValueError("API returned no images")
        info = json.loads(data.get("info", "{}"))
        seed = info.get("seed", "?")
        out = self._save_base64(images[0], prompt)
        return f"ok — {out} | seed={seed} | {time.time()-t0:.1f}s"

    def _generate_comfyui(self, prompt: str) -> str:
        client_id = str(uuid.uuid4())
        s = self._settings
        actual_seed = s["seed"] if s["seed"] != -1 else int(time.time() * 1000) % (2**32)
        workflow = {
            "3": {"class_type": "KSampler", "inputs": {
                "seed": actual_seed, "steps": s["steps"], "cfg": s["cfg_scale"],
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0],
            }},
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.ckpt"}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": s["batch_size"], "height": s["height"], "width": s["width"]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": s["negative_prompt"], "clip": ["4", 1]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "PhotoAgent", "images": ["8", 0]}},
        }
        resp = requests.post(f"{self._photo_api_url}/prompt",
                             json={"prompt": workflow, "client_id": client_id}, timeout=30)
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]

        t0 = time.time()
        deadline = t0 + _GENERATION_TIMEOUT
        while time.time() < deadline:
            hist = requests.get(f"{self._photo_api_url}/history/{prompt_id}", timeout=10).json()
            if prompt_id in hist:
                for node_out in hist[prompt_id].get("outputs", {}).values():
                    for img in node_out.get("images", []):
                        params = {"filename": img["filename"]}
                        if img.get("subfolder"):
                            params["subfolder"] = img["subfolder"]
                        raw = requests.get(f"{self._photo_api_url}/view", params=params, timeout=30).content
                        out = self._save_bytes(raw, prompt)
                        return f"ok — {out} | seed={actual_seed} | {time.time()-t0:.1f}s"
            time.sleep(2)
        raise TimeoutError("ComfyUI did not finish within 5 minutes")

    # ------------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------------

    def _make_path(self, prompt: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt[:40]).strip().replace(" ", "_")
        return self._output_dir / f"{ts}_{slug}.png"

    def _save_base64(self, b64: str, prompt: str) -> str:
        return self._save_bytes(base64.b64decode(b64), prompt)

    def _save_bytes(self, data: bytes, prompt: str) -> str:
        p = self._make_path(prompt)
        p.write_bytes(data)
        return str(p)

    # ------------------------------------------------------------------
    # Точка входа
    # ------------------------------------------------------------------

    def run_batch(self, prompts_file: str) -> str:
        return self.run(
            f"Generate images for all prompts in: {prompts_file}\n"
            "Load the prompts file, then process each prompt one by one."
        )
