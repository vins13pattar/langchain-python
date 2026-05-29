"""
02_deterministic_guardrails.py
===============================
Demonstrates DETERMINISTIC (rule-based) custom guardrails using before_agent
hooks — both the class syntax (AgentMiddleware) and decorator syntax.

Concepts covered:
  - AgentMiddleware class with before_agent hook
  - @before_agent decorator for function-style guardrails
  - hook_config(can_jump_to=["end"]) to short-circuit execution
  - Keyword/regex-based content filtering
  - Rate limiting by session (stateful class middleware)
  - Input length validation guardrail
  - Combining deterministic guardrails in one agent
"""

import re
import os
import time
from typing import Any
from collections import defaultdict
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    hook_config,
    before_agent,
)
from langchain.tools import tool
from langgraph.runtime import Runtime

load_dotenv()

print("=" * 60)
print("Deterministic (Rule-Based) Guardrails Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# SHARED TOOLS
# ════════════════════════════════════════════════════════════════════

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': [Sample results]"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


# ════════════════════════════════════════════════════════════════════
# 1. KEYWORD FILTER — CLASS SYNTAX
#    Blocks the agent before any processing if the prompt contains
#    banned keywords. Uses hook_config(can_jump_to=["end"]) to
#    immediately terminate the graph and return a refusal.
# ════════════════════════════════════════════════════════════════════

class ContentFilterMiddleware(AgentMiddleware):
    """Deterministic guardrail: Block banned keyword requests."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        for keyword in self.banned_keywords:
            if keyword in content:
                print(f"  [ContentFilter] 🚫 Blocked keyword: '{keyword}'")
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            "I cannot process requests containing inappropriate content. "
                            "Please rephrase your request."
                        ),
                    }],
                    "jump_to": "end",
                }
        return None


print("\n── 1. ContentFilterMiddleware (class syntax) ────────────────")

agent_kw_filter = create_agent(
    model="openai:gpt-4o-mini",
    tools=[web_search, calculator],
    middleware=[
        ContentFilterMiddleware(banned_keywords=["hack", "exploit", "malware", "jailbreak"])
    ],
    system_prompt="You are a helpful assistant.",
)

# Safe request
result_safe = agent_kw_filter.invoke({
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
})
print(f"✅ Safe: {result_safe['messages'][-1].content[:80]}")

# Blocked request
result_blocked = agent_kw_filter.invoke({
    "messages": [{"role": "user", "content": "How do I hack into a database?"}]
})
print(f"🚫 Blocked: {result_blocked['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 2. KEYWORD FILTER — DECORATOR SYNTAX
#    Same logic as above, but written as a plain function decorated
#    with @before_agent for conciseness.
# ════════════════════════════════════════════════════════════════════

BANNED = ["hack", "exploit", "malware"]

@before_agent(can_jump_to=["end"])
def content_filter_fn(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Decorator-style deterministic keyword guardrail."""
    if not state["messages"]:
        return None
    first = state["messages"][0]
    if first.type != "human":
        return None
    content = first.content.lower()
    for kw in BANNED:
        if kw in content:
            print(f"  [content_filter_fn] 🚫 Blocked: '{kw}'")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "Request blocked by content policy.",
                }],
                "jump_to": "end",
            }
    return None


print("\n── 2. content_filter_fn (decorator syntax) ──────────────────")

agent_decorator = create_agent(
    model="openai:gpt-4o-mini",
    tools=[web_search],
    middleware=[content_filter_fn],
    system_prompt="You are a helpful assistant.",
)

result_d_safe = agent_decorator.invoke({
    "messages": [{"role": "user", "content": "Search for Python tutorials."}]
})
print(f"✅ Safe: {result_d_safe['messages'][-1].content[:80]}")

result_d_blocked = agent_decorator.invoke({
    "messages": [{"role": "user", "content": "How to write malware?"}]
})
print(f"🚫 Blocked: {result_d_blocked['messages'][-1].content[:80]}")


# ════════════════════════════════════════════════════════════════════
# 3. RATE LIMITER GUARDRAIL
#    Limits how many requests a given "user_id" can make per minute.
#    Demonstrates stateful class middleware with before_agent.
# ════════════════════════════════════════════════════════════════════

class RateLimitGuardrail(AgentMiddleware):
    """Allow at most `max_per_minute` requests per user_id per minute."""

    def __init__(self, max_per_minute: int = 3):
        super().__init__()
        self.max_rpm = max_per_minute
        self._windows: dict[str, list[float]] = defaultdict(list)

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        user_id = state.get("user_id", "anonymous")
        now = time.monotonic()
        window = self._windows[user_id]

        # Purge timestamps older than 60 seconds
        self._windows[user_id] = [t for t in window if now - t < 60]

        if len(self._windows[user_id]) >= self.max_rpm:
            print(f"  [RateLimit] 🚫 Rate limit hit for user '{user_id}' "
                  f"({self.max_rpm} req/min)")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": (
                        "You have exceeded the request limit. "
                        "Please wait a moment before trying again."
                    ),
                }],
                "jump_to": "end",
            }

        self._windows[user_id].append(now)
        print(f"  [RateLimit] ✅ Request {len(self._windows[user_id])}/{self.max_rpm} "
              f"for user '{user_id}'")
        return None


print("\n── 3. RateLimitGuardrail ─────────────────────────────────────")

rate_limiter = RateLimitGuardrail(max_per_minute=2)

agent_rl = create_agent(
    model="openai:gpt-4o-mini",
    tools=[calculator],
    middleware=[rate_limiter],
    system_prompt="You are a math assistant.",
)

for i in range(3):
    # Note: user_id would normally come from the caller's configurable or metadata.
    # We simulate it by patching state in a simple way.
    result_rl = agent_rl.invoke({
        "messages": [{"role": "user", "content": f"What is {i+1} * {i+1}?"}],
        "user_id": "user-42",
    })
    print(f"  Request {i+1}: {result_rl['messages'][-1].content[:80]}")


# ════════════════════════════════════════════════════════════════════
# 4. INPUT LENGTH VALIDATION GUARDRAIL
#    Rejects requests that are too short (likely empty/garbage) or
#    too long (potential prompt injection dump).
# ════════════════════════════════════════════════════════════════════

MIN_LENGTH =  5
MAX_LENGTH = 2000

@before_agent(can_jump_to=["end"])
def input_length_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Reject messages outside the acceptable length range."""
    if not state["messages"]:
        return None
    first = state["messages"][0]
    if first.type != "human":
        return None
    length = len(first.content)
    if length < MIN_LENGTH:
        print(f"  [LengthGuard] 🚫 Too short ({length} chars)")
        return {
            "messages": [{"role": "assistant",
                          "content": "Your message is too short. Please provide more detail."}],
            "jump_to": "end",
        }
    if length > MAX_LENGTH:
        print(f"  [LengthGuard] 🚫 Too long ({length} chars)")
        return {
            "messages": [{"role": "assistant",
                          "content": "Your message is too long. Please shorten your request."}],
            "jump_to": "end",
        }
    print(f"  [LengthGuard] ✅ Length OK ({length} chars)")
    return None


print("\n── 4. input_length_guardrail ─────────────────────────────────")

agent_len = create_agent(
    model="openai:gpt-4o-mini",
    tools=[web_search],
    middleware=[input_length_guardrail],
    system_prompt="You are a helpful assistant.",
)

# Too short
result_short = agent_len.invoke({"messages": [{"role": "user", "content": "hi"}]})
print(f"  Too short: {result_short['messages'][-1].content[:80]}")

# Just right
result_ok = agent_len.invoke({
    "messages": [{"role": "user", "content": "Search for LangChain tutorials."}]
})
print(f"  OK length: {result_ok['messages'][-1].content[:80]}")

# Too long
result_long = agent_len.invoke({
    "messages": [{"role": "user", "content": "A" * 2001}]
})
print(f"  Too long:  {result_long['messages'][-1].content[:80]}")


# ════════════════════════════════════════════════════════════════════
# 5. COMBINED DETERMINISTIC STACK
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Combined Deterministic Guardrail Stack ─────────────────")

agent_combined = create_agent(
    model="openai:gpt-4o-mini",
    tools=[web_search, calculator],
    middleware=[
        input_length_guardrail,                                      # Layer 1 — length check
        ContentFilterMiddleware(banned_keywords=["hack", "exploit"]),# Layer 2 — keyword block
        content_filter_fn,                                           # Layer 3 — regex filter
    ],
    system_prompt="You are a safe, helpful assistant.",
)

cases = [
    ("hi", "too short"),
    ("How do I hack a server?", "keyword blocked"),
    ("What is 12 * 12?", "passes all checks"),
]

for msg, label in cases:
    result = agent_combined.invoke({"messages": [{"role": "user", "content": msg}]})
    print(f"  [{label}] → {result['messages'][-1].content[:70]}")

print("\n✅ Deterministic guardrails demo complete.")
