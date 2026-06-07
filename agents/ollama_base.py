"""
OllamaAgent — базовый класс для агентов на локальной LLM через Ollama.

Зеркало BaseAgent, но вместо Anthropic SDK использует OpenAI-совместимый
API Ollama (http://localhost:11434/v1). Никаких API-ключей не требуется.

Переменные окружения:
    OLLAMA_MODEL  — модель (по умолчанию llama3.1)
    OLLAMA_URL    — URL Ollama (по умолчанию http://localhost:11434)

Модели с поддержкой инструментов (tool calling):
    llama3.1, llama3.2, llama3.3, mistral-nemo,
    qwen2.5, qwen2.5-coder, firefunction-v2, command-r
"""

from __future__ import annotations

import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.1"
DEFAULT_MAX_TOKENS = 4096
MAX_ITERATIONS = 50


def tool(
    description: str,
    input_schema: dict[str, Any],
    name: str | None = None,
) -> Callable:
    """Тот же декоратор @tool, что и в base.py — совместим с обоими базовыми классами."""
    def decorator(fn: Callable) -> Callable:
        fn._tool_meta = {
            "name": name or fn.__name__,
            "description": description,
            "input_schema": input_schema,
        }
        return fn
    return decorator


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable


@dataclass
class OllamaConfig:
    model: str = DEFAULT_MODEL
    system_prompt: str = "You are a helpful AI assistant."
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_iterations: int = MAX_ITERATIONS
    temperature: float = 0.3
    base_url: str = "http://localhost:11434/v1"


class OllamaAgent:
    """Базовый класс агента на Ollama. Использует тот же интерфейс что и BaseAgent."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()
        # Подменяем model/url из env если заданы
        self.config.model = os.getenv("OLLAMA_MODEL", self.config.model)
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.config.base_url = f"{ollama_url.rstrip('/')}/v1"

        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key="ollama",  # Ollama не проверяет ключ, но поле обязательно
        )
        self._tools: dict[str, ToolDef] = {}
        self._history: list[dict[str, Any]] = []

        # Авто-регистрация методов, декорированных @tool
        for _, method in inspect.getmembers(self, predicate=inspect.ismethod):
            meta = getattr(method, "_tool_meta", None)
            if meta:
                self._tools[meta["name"]] = ToolDef(
                    name=meta["name"],
                    description=meta["description"],
                    input_schema=meta["input_schema"],
                    fn=method,
                )

    # ------------------------------------------------------------------
    # Регистрация инструментов
    # ------------------------------------------------------------------

    def register_tool_fn(
        self,
        name: str,
        fn: Callable,
        description: str,
        input_schema: dict[str, Any],
    ) -> None:
        self._tools[name] = ToolDef(
            name=name, description=description, input_schema=input_schema, fn=fn
        )

    # ------------------------------------------------------------------
    # Хуки — переопределить в подклассе
    # ------------------------------------------------------------------

    def on_tool_call(self, tool_name: str, tool_input: dict[str, Any]) -> None:
        pass

    def on_tool_result(self, tool_name: str, result: Any) -> None:
        pass

    def on_assistant_message(self, text: str) -> None:
        pass

    # ------------------------------------------------------------------
    # Конвертация схем инструментов в формат OpenAI
    # ------------------------------------------------------------------

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in self._tools.values()
        ]

    # ------------------------------------------------------------------
    # Вызов инструмента
    # ------------------------------------------------------------------

    def _call_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Error: unknown tool '{name}'"
        self.on_tool_call(name, tool_input)
        try:
            result = self._tools[name].fn(**tool_input)
        except Exception as exc:  # noqa: BLE001
            result = f"Error: {exc}"
        self.on_tool_result(name, result)
        return str(result)

    # ------------------------------------------------------------------
    # Основной агентный цикл
    # ------------------------------------------------------------------

    def _run_loop(self, messages: list[dict[str, Any]]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "system", "content": self.config.system_prompt}] + messages,
        }
        if self._tools:
            kwargs["tools"] = self._tool_schemas()

        final_text = ""

        for _ in range(self.config.max_iterations):
            response = self._client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            stop_reason = response.choices[0].finish_reason

            if msg.content:
                final_text = msg.content
                self.on_assistant_message(final_text)

            if stop_reason != "tool_calls" or not msg.tool_calls:
                break

            # Добавляем ответ ассистента в историю
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Выполняем все tool calls и добавляем результаты
            for tc in msg.tool_calls:
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}
                result_text = self._call_tool(tc.function.name, tool_input)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })

            kwargs["messages"] = (
                [{"role": "system", "content": self.config.system_prompt}] + messages
            )

        return final_text

    # ------------------------------------------------------------------
    # Публичный API (идентичен BaseAgent)
    # ------------------------------------------------------------------

    def run(self, user_message: str) -> str:
        """Однократный запрос без сохранения истории."""
        messages = [{"role": "user", "content": user_message}]
        return self._run_loop(messages)

    def chat(self, user_message: str) -> str:
        """Мульти-тёрн разговор с сохранением истории."""
        self._history.append({"role": "user", "content": user_message})
        messages = list(self._history)
        response = self._run_loop(messages)
        self._history.append({"role": "assistant", "content": response})
        return response

    def reset_history(self) -> None:
        self._history = []
