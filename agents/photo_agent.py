"""
PhotoAgent — генерирует изображения через локальный AI (Automatic1111 или ComfyUI).

Читает промпты из файла построчно, отправляет каждый на локальный сервер,
ждёт результата, сохраняет PNG и переходит к следующему промпту.

Переменные окружения:
    PHOTO_BACKEND   — "automatic1111" (по умолчанию) или "comfyui"
    PHOTO_API_URL   — URL локального сервера (по умолчанию http://127.0.0.1:7860)
    PHOTO_OUTPUT_DIR — папка для сохранения PNG (по умолчанию ./outputs)
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

from .base import AgentConfig, BaseAgent, tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are PhotoAgent — an automated image generation assistant.

Your job is to process a list of text prompts and generate one image per prompt
using the local AI backend.

Follow this workflow strictly:

1. Call load_prompts(file_path=<path>) to read the prompt file.
2. Loop:
   a. Call get_next_prompt() — if it returns "DONE", stop.
   b. Call generate_image(prompt=<prompt>) and wait for the result.
   c. Report the saved file path and any generation stats.
3. After all prompts are processed, print a summary (total images, any errors).

Rules:
- Call exactly ONE tool per turn.
- Never skip a prompt.
- If generate_image returns an error, log it and continue to the next prompt.
"""

_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 10  # seconds


class PhotoAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            config=AgentConfig(
                system_prompt=SYSTEM_PROMPT,
                max_iterations=500,
            )
        )
        self._backend: str = os.getenv("PHOTO_BACKEND", "automatic1111").lower()
        self._api_url: str = os.getenv("PHOTO_API_URL", "http://127.0.0.1:7860").rstrip("/")
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
    # Tools
    # ------------------------------------------------------------------

    @tool(
        description="Load prompts from a text file. Each non-empty, non-comment line becomes one prompt.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the prompts file (absolute or relative to cwd).",
                }
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
        count = len(self._prompts)
        return f"Loaded {count} prompts from {file_path}"

    @tool(
        description="Return the next prompt from the queue. Returns 'DONE' when the queue is empty.",
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    def get_next_prompt(self) -> str:
        if not self._prompts:
            return "DONE"
        return self._prompts.popleft()

    @tool(
        description=(
            "Update generation settings. All parameters are optional — only provided keys are changed. "
            "Available keys: steps, cfg_scale, width, height, sampler_name, negative_prompt, seed, batch_size."
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
        updated = []
        for key, value in kwargs.items():
            if key in self._settings:
                self._settings[key] = value
                updated.append(f"{key}={value}")
        if not updated:
            return f"No valid keys updated. Current settings: {self._settings}"
        return f"Updated settings: {', '.join(updated)}"

    @tool(
        description=(
            "Generate one image for the given prompt using the configured local AI backend. "
            "Blocks until the image is saved to disk. Returns the output file path and stats."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The text prompt to generate an image for.",
                }
            },
            "required": ["prompt"],
        },
    )
    def generate_image(self, prompt: str) -> str:
        if not prompt.strip():
            return "Error: empty prompt — skipped"

        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                if self._backend == "comfyui":
                    result = self._generate_comfyui(prompt)
                else:
                    result = self._generate_automatic1111(prompt)
                self._total_done += 1
                return result
            except requests.exceptions.ConnectionError:
                if attempt < _RETRY_ATTEMPTS:
                    logger.warning("Backend unreachable, retrying in %ds (attempt %d/%d)",
                                   _RETRY_DELAY, attempt, _RETRY_ATTEMPTS)
                    time.sleep(_RETRY_DELAY)
                else:
                    self._total_errors += 1
                    return f"Error: cannot reach {self._backend} at {self._api_url} after {_RETRY_ATTEMPTS} attempts"
            except Exception as exc:  # noqa: BLE001
                self._total_errors += 1
                return f"Error: generation failed: {exc}"

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _generate_automatic1111(self, prompt: str) -> str:
        payload = {
            "prompt": prompt,
            **self._settings,
        }
        t0 = time.time()
        response = requests.post(
            f"{self._api_url}/sdapi/v1/txt2img",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        images = data.get("images", [])
        if not images:
            raise ValueError("API returned no images")

        info = json.loads(data.get("info", "{}"))
        seed = info.get("seed", "unknown")

        out_path = self._save_png_base64(images[0], prompt)
        elapsed = time.time() - t0
        return f"ok — saved to {out_path} | seed={seed} | time={elapsed:.1f}s"

    def _generate_comfyui(self, prompt: str) -> str:
        client_id = str(uuid.uuid4())
        workflow = self._build_comfyui_workflow(prompt)

        # Queue prompt
        response = requests.post(
            f"{self._api_url}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=30,
        )
        response.raise_for_status()
        prompt_id: str = response.json()["prompt_id"]

        # Poll history until done
        t0 = time.time()
        deadline = t0 + 300
        while time.time() < deadline:
            hist_resp = requests.get(f"{self._api_url}/history/{prompt_id}", timeout=10)
            if hist_resp.status_code == 200:
                hist = hist_resp.json()
                if prompt_id in hist:
                    outputs = hist[prompt_id].get("outputs", {})
                    for node_outputs in outputs.values():
                        for img_info in node_outputs.get("images", []):
                            filename = img_info["filename"]
                            subfolder = img_info.get("subfolder", "")
                            params = {"filename": filename}
                            if subfolder:
                                params["subfolder"] = subfolder
                            img_resp = requests.get(
                                f"{self._api_url}/view",
                                params=params,
                                timeout=30,
                            )
                            img_resp.raise_for_status()
                            out_path = self._save_png_bytes(img_resp.content, prompt)
                            elapsed = time.time() - t0
                            return f"ok — saved to {out_path} | time={elapsed:.1f}s"
            time.sleep(2)

        raise TimeoutError("ComfyUI did not complete generation within 5 minutes")

    def _build_comfyui_workflow(self, prompt: str) -> dict:
        s = self._settings
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": s["seed"] if s["seed"] != -1 else int(time.time()),
                    "steps": s["steps"],
                    "cfg": s["cfg_scale"],
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
            },
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.ckpt"}},
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"batch_size": s["batch_size"], "height": s["height"], "width": s["width"]},
            },
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": s["negative_prompt"], "clip": ["4", 1]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "PhotoAgent", "images": ["8", 0]},
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_output_path(self, prompt: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        slug = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt[:40]).strip().replace(" ", "_")
        return self._output_dir / f"{ts}_{slug}.png"

    def _save_png_base64(self, b64: str, prompt: str) -> str:
        img_bytes = base64.b64decode(b64)
        return self._save_png_bytes(img_bytes, prompt)

    def _save_png_bytes(self, data: bytes, prompt: str) -> str:
        out_path = self._make_output_path(prompt)
        out_path.write_bytes(data)
        return str(out_path)

    # ------------------------------------------------------------------
    # Convenience entry point
    # ------------------------------------------------------------------

    def run_batch(self, prompts_file: str) -> str:
        """Run the full generation loop for all prompts in the file."""
        return self.run(
            f"Generate images for all prompts in the file: {prompts_file}\n"
            "Start by loading the prompts, then process each one in order."
        )
