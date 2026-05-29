"""
05_agent_loop_middleware.py
===========================
Demonstrates how middleware interacts with the agent's internal LOOP —
the cycle of model call → tool execution → model call that continues
until the agent produces a final answer.

Concepts covered:
  - How middleware hooks map to agent loop stages
  - Inspecting loop iteration number inside hooks
  - Early exit strategy — short-circuit the loop from middleware
  - before_agent vs before_model (when each fires during a multi-step run)
  - Rate-limiting pattern between iterations
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
print("Agent Loop Middleware Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════

@tool
def step_one(data: str) -> str:
    """First processing step — clean and validate input data."""
    print(f"  [Tool] step_one('{data}')")
    return f"Step 1 complete: data '{data}' validated."


@tool
def step_two(data: str) -> str:
    """Second processing step — transform the validated data."""
    print(f"  [Tool] step_two('{data}')")
    return f"Step 2 complete: data '{data}' transformed."


@tool
def step_three(data: str) -> str:
    """Third processing step — persist the transformed data."""
    print(f"  [Tool] step_three('{data}')")
    return f"Step 3 complete: data '{data}' persisted successfully."


# ════════════════════════════════════════════════════════════════════
# 1. LOOP OBSERVER MIDDLEWARE
#    Tracks how many model calls and tool calls have been made during
#    the current agent run and logs the progression.
# ════════════════════════════════════════════════════════════════════

class LoopObserverMiddleware(BaseMiddleware):
    """Observes each iteration of the agent's think-act loop."""

    def __init__(self):
        self._model_calls = 0
        self._tool_calls  = 0

    def before_agent(self, state: dict) -> Optional[dict]:
        self._model_calls = 0
        self._tool_calls  = 0
        print("  [LoopObserver] ─── Agent run started ───")
        return None

    def after_agent(self, state: dict) -> Optional[dict]:
        print(f"  [LoopObserver] ─── Agent run ended ───")
        print(f"  [LoopObserver] Summary: {self._model_calls} model calls, "
              f"{self._tool_calls} tool calls")
        return None

    def before_model(self, state: dict, **kwargs) -> Optional[dict]:
        self._model_calls += 1
        n_messages = len(state.get("messages", []))
        print(f"  [LoopObserver] LLM call #{self._model_calls} "
              f"(context: {n_messages} messages)")
        return None

    def wrap_tool_call(self, tool_call: dict, call_tool, **kwargs) -> Any:
        self._tool_calls += 1
        name = tool_call.get("name", "unknown")
        print(f"  [LoopObserver] Tool call #{self._tool_calls}: {name}")
        return call_tool(tool_call, **kwargs)


print("\n── 1. LoopObserverMiddleware ─────────────────────────────────")

agent_observed = create_agent(
    model="openai:gpt-4o-mini",
    tools=[step_one, step_two, step_three],
    middleware=[LoopObserverMiddleware()],
    system_prompt=(
        "You are a data pipeline agent. "
        "Always process data through step_one, then step_two, then step_three in order."
    ),
)

result = agent_observed.invoke({
    "messages": [{"role": "user", "content": "Process the dataset 'sales_q4'."}]
})
print(f"\nFinal response: {result['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 2. RATE-LIMITER MIDDLEWARE
#    Introduces a small delay between LLM calls to respect API rate
#    limits without hitting 429 errors in high-throughput scenarios.
# ════════════════════════════════════════════════════════════════════

class RateLimiterMiddleware(BaseMiddleware):
    """Adds a minimum delay between consecutive LLM invocations."""

    def __init__(self, min_delay_seconds: float = 0.5):
        self._min_delay = min_delay_seconds
        self._last_call  = 0.0

    def before_model(self, state: dict, **kwargs) -> Optional[dict]:
        now     = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._min_delay and self._last_call > 0:
            wait = self._min_delay - elapsed
            print(f"  [RateLimiter] Throttling — sleeping {wait:.2f}s")
            time.sleep(wait)
        self._last_call = time.monotonic()
        return None

    def after_model(self, state: dict, **kwargs) -> Optional[dict]:
        self._last_call = time.monotonic()
        return None


print("\n── 2. RateLimiterMiddleware ──────────────────────────────────")

agent_rate_limited = create_agent(
    model="openai:gpt-4o-mini",
    tools=[step_one, step_two],
    middleware=[RateLimiterMiddleware(min_delay_seconds=0.3)],
    system_prompt="You are a pipeline agent. Run step_one then step_two.",
)

t0 = time.perf_counter()
result_rl = agent_rate_limited.invoke({
    "messages": [{"role": "user", "content": "Process 'inventory_sync' through both steps."}]
})
elapsed = time.perf_counter() - t0
print(f"Response: {result_rl['messages'][-1].content[:100]}")
print(f"Total wall time: {elapsed:.2f}s (includes rate-limit delays)")


# ════════════════════════════════════════════════════════════════════
# 3. EARLY EXIT MIDDLEWARE
#    Terminates the agent loop early if a specific condition is met
#    (e.g. budget exhausted, confidence threshold met).
# ════════════════════════════════════════════════════════════════════

class EarlyExitMiddleware(BaseMiddleware):
    """Exits the agent loop once the answer contains a stop phrase."""

    STOP_PHRASE = "complete"

    def after_model(self, state: dict, **kwargs) -> Optional[dict]:
        messages = state.get("messages", [])
        if messages:
            last_content = getattr(messages[-1], "content", "") or ""
            if self.STOP_PHRASE.lower() in last_content.lower():
                print(f"  [EarlyExit] Stop phrase '{self.STOP_PHRASE}' detected — "
                      "signalling early termination.")
                # Inject a terminal flag recognised by create_agent's loop
                return {**state, "__early_exit__": True}
        return None


print("\n── 3. EarlyExitMiddleware ────────────────────────────────────")

agent_early_exit = create_agent(
    model="openai:gpt-4o-mini",
    tools=[step_one, step_two, step_three],
    middleware=[EarlyExitMiddleware(), LoopObserverMiddleware()],
    system_prompt=(
        "You are a pipeline agent. Run steps as needed. "
        "After running step_one, report the task is 'complete'."
    ),
)

result_ee = agent_early_exit.invoke({
    "messages": [{"role": "user", "content": "Process 'cache_warmup'."}]
})
print(f"Response: {result_ee['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# HOOK FIRING ORDER SUMMARY
# ════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("Hook Firing Order (for reference):")
print("  1. before_agent      — once, at the very start of the run")
print("  2. before_model      — before EACH LLM call in the loop")
print("  3. after_model       — after EACH LLM call in the loop")
print("  4. wrap_tool_call    — around EACH tool invocation")
print("  5. after_agent       — once, when the agent loop exits")
print("═" * 60)
print("\n✅ Agent loop middleware demo complete.")
