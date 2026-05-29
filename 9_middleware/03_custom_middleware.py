"""
03_custom_middleware.py
=======================
Demonstrates how to build CUSTOM middleware by subclassing BaseMiddleware
and implementing lifecycle hooks.

Concepts covered:
  - BaseMiddleware and available hook signatures
  - before_agent / after_agent        — Wrap the full agent execution
  - before_model / after_model        — Wrap each LLM call
  - wrap_tool_call                    — Intercept every tool invocation
  - Composing hooks for logging, timing, and cost estimation
  - Tool-specific middleware (apply only to named tools)
"""

import time
import os
from typing import Any, Optional
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import BaseMiddleware
from langchain.tools import tool

load_dotenv()

print("=" * 60)
print("Custom Middleware Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"It's 22°C and sunny in {city}."


@tool
def get_population(country: str) -> str:
    """Get the population of a country."""
    data = {"India": "1.44B", "USA": "335M", "Germany": "84M"}
    return f"Population of {country}: {data.get(country, 'unknown')}."


# ════════════════════════════════════════════════════════════════════
# 1. LOGGING MIDDLEWARE — Logs every hook invocation
# ════════════════════════════════════════════════════════════════════

class LoggingMiddleware(BaseMiddleware):
    """Logs the before/after of every agent, model, and tool event."""

    def before_agent(self, state: dict) -> Optional[dict]:
        print("  [Log] ▶ Agent started")
        return None  # returning None passes through unchanged state

    def after_agent(self, state: dict) -> Optional[dict]:
        print("  [Log] ◀ Agent finished")
        return None

    def before_model(self, state: dict, **kwargs) -> Optional[dict]:
        messages = state.get("messages", [])
        print(f"  [Log] 🤖 LLM call — {len(messages)} messages in context")
        return None

    def after_model(self, state: dict, **kwargs) -> Optional[dict]:
        last_msg = state["messages"][-1] if state.get("messages") else None
        content_preview = (last_msg.content[:60] + "...") if last_msg and last_msg.content else ""
        print(f"  [Log] 🤖 LLM response: {content_preview}")
        return None

    def wrap_tool_call(self, tool_call: dict, call_tool, **kwargs) -> Any:
        name = tool_call.get("name", "unknown")
        args = tool_call.get("args", {})
        print(f"  [Log] 🔧 Tool call: {name}({args})")
        result = call_tool(tool_call, **kwargs)
        print(f"  [Log] 🔧 Tool result: {str(result)[:80]}")
        return result


# ════════════════════════════════════════════════════════════════════
# 2. TIMING MIDDLEWARE — Measures latency at each stage
# ════════════════════════════════════════════════════════════════════

class TimingMiddleware(BaseMiddleware):
    """Measures wall-clock time for model calls and tool invocations."""

    def before_model(self, state: dict, **kwargs) -> Optional[dict]:
        self._model_start = time.perf_counter()
        return None

    def after_model(self, state: dict, **kwargs) -> Optional[dict]:
        elapsed = time.perf_counter() - self._model_start
        print(f"  [Timing] LLM latency: {elapsed:.2f}s")
        return None

    def wrap_tool_call(self, tool_call: dict, call_tool, **kwargs) -> Any:
        start = time.perf_counter()
        result = call_tool(tool_call, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  [Timing] Tool '{tool_call.get('name')}' latency: {elapsed * 1000:.1f}ms")
        return result


# ════════════════════════════════════════════════════════════════════
# 3. COST ESTIMATION MIDDLEWARE — Counts tokens (approximate)
# ════════════════════════════════════════════════════════════════════

class CostEstimationMiddleware(BaseMiddleware):
    """Approximate GPT-4o-mini cost: $0.00015/1K input tokens, $0.00060/1K output."""

    INPUT_COST_PER_1K  = 0.00015
    OUTPUT_COST_PER_1K = 0.00060

    def __init__(self):
        self._total_input_tokens  = 0
        self._total_output_tokens = 0

    def after_model(self, state: dict, **kwargs) -> Optional[dict]:
        # Use token usage from the last message if available (LangChain sets it)
        last = state["messages"][-1] if state.get("messages") else None
        if last and hasattr(last, "usage_metadata") and last.usage_metadata:
            usage = last.usage_metadata
            self._total_input_tokens  += usage.get("input_tokens", 0)
            self._total_output_tokens += usage.get("output_tokens", 0)
            cost = (
                self._total_input_tokens  / 1000 * self.INPUT_COST_PER_1K +
                self._total_output_tokens / 1000 * self.OUTPUT_COST_PER_1K
            )
            print(f"  [Cost] Running estimate: ${cost:.6f} "
                  f"({self._total_input_tokens}in / {self._total_output_tokens}out tokens)")
        return None


# ════════════════════════════════════════════════════════════════════
# 4. TOOL-SPECIFIC MIDDLEWARE — Only applies to named tools
# ════════════════════════════════════════════════════════════════════

class WeatherAuditMiddleware(BaseMiddleware):
    """Intercept only weather tool calls to validate the city name."""

    tools = ["get_weather"]  # ← Apply only to this tool

    def wrap_tool_call(self, tool_call: dict, call_tool, **kwargs) -> Any:
        city = tool_call.get("args", {}).get("city", "")
        if not city or any(char.isdigit() for char in city):
            print(f"  [WeatherAudit] ⚠️  Invalid city name: '{city}', defaulting to 'London'")
            tool_call = {**tool_call, "args": {**tool_call["args"], "city": "London"}}
        result = call_tool(tool_call, **kwargs)
        print(f"  [WeatherAudit] ✅ Audited weather call for: {city or 'London'}")
        return result


# ════════════════════════════════════════════════════════════════════
# DEMO 1 — Logging + Timing stacked
# ════════════════════════════════════════════════════════════════════

print("\n── Demo 1: LoggingMiddleware + TimingMiddleware ──────────────")

agent_logged = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, get_population],
    middleware=[LoggingMiddleware(), TimingMiddleware()],
    system_prompt="You are a helpful facts assistant.",
)

result = agent_logged.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
})
print(f"\nFinal response: {result['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# DEMO 2 — Cost Estimation
# ════════════════════════════════════════════════════════════════════

print("\n── Demo 2: CostEstimationMiddleware ─────────────────────────")

agent_cost = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, get_population],
    middleware=[CostEstimationMiddleware()],
    system_prompt="You are a helpful facts assistant.",
)

result2 = agent_cost.invoke({
    "messages": [{"role": "user", "content": "What is the population of Germany?"}]
})
print(f"\nFinal response: {result2['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# DEMO 3 — Tool-specific middleware
# ════════════════════════════════════════════════════════════════════

print("\n── Demo 3: Tool-Specific WeatherAuditMiddleware ─────────────")

agent_audit = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, get_population],
    middleware=[WeatherAuditMiddleware()],
    system_prompt="You are a helpful facts assistant.",
)

result3 = agent_audit.invoke({
    "messages": [{"role": "user", "content": "Get weather for city 123."}]
})
print(f"\nFinal response: {result3['messages'][-1].content}")

print("\n✅ Custom middleware demo complete.")
