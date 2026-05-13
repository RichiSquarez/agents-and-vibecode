"""
Minimal example: a math + weather agent built on BaseAgent.

Run with:  python examples/simple_agent.py
"""

import json

from agents import BaseAgent, tool
from agents.base import AgentConfig


# ---------------------------------------------------------------------------
# Option A — standalone tool functions registered after instantiation
# ---------------------------------------------------------------------------

@tool(
    description="Add two numbers and return their sum.",
    input_schema={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First operand"},
            "b": {"type": "number", "description": "Second operand"},
        },
        "required": ["a", "b"],
    },
)
def add(a: float, b: float) -> float:
    return a + b


@tool(
    description="Multiply two numbers and return their product.",
    input_schema={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    },
)
def multiply(a: float, b: float) -> float:
    return a * b


# ---------------------------------------------------------------------------
# Option B — subclass with tools as methods decorated with @tool
# ---------------------------------------------------------------------------

class MathAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            config=AgentConfig(
                system_prompt=(
                    "You are a maths tutor. Use tools for calculations and explain "
                    "your reasoning step by step."
                ),
            )
        )
        # Register standalone functions
        self.register_tool(add)
        self.register_tool(multiply)

    def on_tool_call(self, tool_name, tool_input):
        print(f"  [tool] {tool_name}({json.dumps(tool_input)})")

    def on_tool_result(self, tool_name, result):
        print(f"  [result] {result}")

    def on_assistant_message(self, text):
        print(f"\nAssistant: {text}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    agent = MathAgent()

    # Single-turn
    print("=== Single turn ===")
    agent.run("What is (3 + 7) × 12?")

    # Multi-turn conversation
    print("=== Multi-turn ===")
    agent.chat("Hi! I need help with a maths problem.")
    agent.chat("What is 144 multiplied by 25?")
    agent.chat("And if I add 1000 to that, what do I get?")


if __name__ == "__main__":
    main()
