"""
BaseAgent — a reusable foundation for any Claude-powered agent.

Subclass BaseAgent and register tools with @tool or agent.register_tool().
Call agent.run(message) for a single turn or agent.chat(message) for
multi-turn conversations that preserve history.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 8096
MAX_ITERATIONS = 10  # guard against infinite tool-call loops


# ---------------------------------------------------------------------------
# @tool decorator — attach metadata so BaseAgent can auto-register functions
# ---------------------------------------------------------------------------

def tool(
    description: str,
    input_schema: dict[str, Any],
    name: str | None = None,
) -> Callable:
    """Decorator that marks a function as an agent tool.

    Usage::

        @tool(
            description="Add two numbers together.",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        )
        def add(a: float, b: float) -> float:
            return a + b
    """
    def decorator(fn: Callable) -> Callable:
        fn._tool_meta = {
            "name": name or fn.__name__,
            "description": description,
            "input_schema": input_schema,
        }
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable


@dataclass
class AgentConfig:
    model: str = DEFAULT_MODEL
    system_prompt: str = "You are a helpful AI assistant."
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_iterations: int = MAX_ITERATIONS
    temperature: float | None = None
    # Enable prompt caching for the system prompt (reduces cost on repeated calls)
    cache_system_prompt: bool = True


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent:
    """Foundation class for any Claude-powered agent.

    Subclassing example::

        class MyAgent(BaseAgent):
            def __init__(self):
                super().__init__(
                    config=AgentConfig(system_prompt="You are a data analyst."),
                )
                self.register_tool_fn(
                    name="query_db",
                    fn=self._query_db,
                    description="Run a SQL query.",
                    input_schema={
                        "type": "object",
                        "properties": {"sql": {"type": "string"}},
                        "required": ["sql"],
                    },
                )

            def _query_db(self, sql: str) -> str:
                ...
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._tools: dict[str, ToolDef] = {}
        self._history: list[dict[str, Any]] = []

        # Auto-register methods decorated with @tool
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
    # Tool registration
    # ------------------------------------------------------------------

    def register_tool_fn(
        self,
        name: str,
        fn: Callable,
        description: str,
        input_schema: dict[str, Any],
    ) -> None:
        """Register a callable as a tool available to the model."""
        self._tools[name] = ToolDef(
            name=name,
            description=description,
            input_schema=input_schema,
            fn=fn,
        )

    def register_tool(self, fn: Callable) -> Callable:
        """Register a function that has been decorated with @tool."""
        meta = getattr(fn, "_tool_meta", None)
        if meta is None:
            raise ValueError(f"{fn.__name__} must be decorated with @tool first.")
        self._tools[meta["name"]] = ToolDef(
            name=meta["name"],
            description=meta["description"],
            input_schema=meta["input_schema"],
            fn=fn,
        )
        return fn

    # ------------------------------------------------------------------
    # Hooks — override in subclasses to add custom behaviour
    # ------------------------------------------------------------------

    def on_tool_call(self, tool_name: str, tool_input: dict[str, Any]) -> None:
        """Called just before every tool invocation."""

    def on_tool_result(self, tool_name: str, result: Any) -> None:
        """Called after every tool invocation with its result."""

    def on_assistant_message(self, text: str) -> None:
        """Called when the assistant produces a text response."""

    # ------------------------------------------------------------------
    # Core agentic loop
    # ------------------------------------------------------------------

    def _build_system(self) -> list[dict[str, Any]] | str:
        prompt = self.config.system_prompt
        if self.config.cache_system_prompt:
            return [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]
        return prompt

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def _call_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Error: unknown tool '{name}'"
        self.on_tool_call(name, tool_input)
        try:
            result = self._tools[name].fn(**tool_input)
            # Support coroutines registered as tools
            if asyncio.iscoroutine(result):
                result = asyncio.get_event_loop().run_until_complete(result)
        except Exception as exc:  # noqa: BLE001
            result = f"Error: {exc}"
        self.on_tool_result(name, result)
        return str(result)

    def _run_loop(self, messages: list[dict[str, Any]]) -> str:
        """Core synchronous agentic loop."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": self._build_system(),
            "messages": messages,
        }
        if self._tools:
            kwargs["tools"] = self._tool_schemas()
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        final_text = ""

        for _ in range(self.config.max_iterations):
            response = self._client.messages.create(**kwargs)
            logger.debug("stop_reason=%s", response.stop_reason)

            # Collect text blocks for the hook
            text_parts = [b.text for b in response.content if b.type == "text"]
            if text_parts:
                final_text = "\n".join(text_parts)
                self.on_assistant_message(final_text)

            if response.stop_reason != "tool_use":
                break

            # Append the assistant turn
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result_text = self._call_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

            messages.append({"role": "user", "content": tool_results})
            kwargs["messages"] = messages

        return final_text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_message: str) -> str:
        """Single-turn: send a message and return the final text response.

        Does NOT update the agent's conversation history.
        """
        messages = [{"role": "user", "content": user_message}]
        return self._run_loop(messages)

    def chat(self, user_message: str) -> str:
        """Multi-turn: send a message, persist history, return the response."""
        self._history.append({"role": "user", "content": user_message})
        # Work on a copy so the loop can append tool turns without confusion
        messages = list(self._history)
        response = self._run_loop(messages)
        # Only store the final user + assistant pair in clean history
        self._history.append({"role": "assistant", "content": response})
        return response

    def reset_history(self) -> None:
        """Clear multi-turn conversation history."""
        self._history = []

    # ------------------------------------------------------------------
    # Async variants
    # ------------------------------------------------------------------

    async def arun(self, user_message: str) -> str:
        """Async single-turn wrapper (runs the sync loop in a thread pool)."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.run, user_message
        )

    async def achat(self, user_message: str) -> str:
        """Async multi-turn wrapper."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.chat, user_message
        )
