"""
06_lifecycle_context.py
========================
Demonstrates LIFE-CYCLE CONTEXT — controlling what happens BETWEEN the
core agent steps (model call ↔ tool execution) using middleware hooks
for summarization, persistent state updates, and cross-cutting concerns.

Concepts covered:
  - SummarizationMiddleware — auto-condense long conversations (persistent)
  - before_model / after_model persistent state updates via Command
  - Transient (wrap_model_call) vs Persistent (before_model) updates
  - Custom summarization using before_model + explicit store writes
  - Logging middleware that writes audit entries to store
  - Distinction between life-cycle context and model context
"""

import os
import time
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    SummarizationMiddleware,
    before_model,
    after_model,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from typing import Callable

load_dotenv()

print("=" * 60)
print("Life-Cycle Context Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# SHARED TOOLS
# ════════════════════════════════════════════════════════════════════

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 24°C, humidity 55%."

@tool
def get_news(topic: str) -> str:
    """Get latest news for a topic."""
    return f"Latest {topic} news: [Headline A, Headline B, Headline C]."


# ════════════════════════════════════════════════════════════════════
# 1. SUMMARIZATION MIDDLEWARE
#    Automatically condenses conversation history when it grows too
#    long. PERSISTENT — replaces old messages in State permanently.
#    This is the most common life-cycle pattern.
# ════════════════════════════════════════════════════════════════════

print("\n── 1. SummarizationMiddleware ───────────────────────────────")

agent_summarized = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, get_news],
    checkpointer=MemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",   # model to use for summarization
            trigger={"messages": 6},       # summarize when > 6 messages
            keep={"messages": 2},          # keep last 2 messages after summary
        )
    ],
    system_prompt="You are a helpful assistant.",
)

config = {"configurable": {"thread_id": "summarize-demo"}}

questions = [
    "What's the weather in Paris?",
    "And in Tokyo?",
    "What's the latest tech news?",
    "Any sports news?",
    "Weather in New York?",
    "What about London weather?",
    "Summarize what we've discussed.",
]

for q in questions:
    r = agent_summarized.invoke(
        {"messages": [{"role": "user", "content": q}]},
        config=config,
    )
    msg_count = len(r.get("messages", []))
    print(f"  Q: '{q[:50]}' | msgs in state: {msg_count}")
    print(f"  A: {r['messages'][-1].content[:80]}")


# ════════════════════════════════════════════════════════════════════
# 2. PERSISTENT STATE UPDATE via before_model
#    Track turn count in State — a counter that PERSISTS across
#    turns (unlike transient wrap_model_call changes).
# ════════════════════════════════════════════════════════════════════

@before_model
def track_turn_count(state: AgentState, runtime: Runtime) -> dict | None:
    """Increment a persistent turn counter in State."""
    turns = state.get("turn_count", 0) + 1
    print(f"  [before_model] turn_count → {turns}")
    return {"turn_count": turns}


print("\n── 2. Persistent State Updates (turn counter) ───────────────")

agent_turns = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    checkpointer=MemorySaver(),
    middleware=[track_turn_count],
    system_prompt="You are a helpful assistant.",
)

config2 = {"configurable": {"thread_id": "turn-count-demo"}}
for i in range(3):
    r = agent_turns.invoke(
        {"messages": [{"role": "user", "content": f"Question {i+1}"}]},
        config=config2,
    )
    print(f"  Turn {r.get('turn_count', '?')}: {r['messages'][-1].content[:60]}")


# ════════════════════════════════════════════════════════════════════
# 3. AUDIT LOGGING to STORE via before_model
#    Every model call is logged as an audit entry in long-term
#    memory — this PERSISTS across sessions.
# ════════════════════════════════════════════════════════════════════

@dataclass
class AuditCtx:
    user_id: str


@before_model
def audit_to_store(state: AgentState, runtime: Runtime[AuditCtx]) -> dict | None:
    """Write an audit entry to Store for every model call."""
    store   = runtime.store
    user_id = runtime.context.user_id
    info    = runtime.execution_info

    if not store:
        return None

    # Read existing audit log
    existing  = store.get(("audit",), user_id)
    log_list  = existing.value.get("entries", []) if existing else []

    # Append new entry
    log_list.append({
        "timestamp": time.time(),
        "thread_id": info.thread_id,
        "run_id":    info.run_id[:8],
        "messages":  len(state.get("messages", [])),
    })

    # Write back (trimmed to last 50 entries)
    store.put(("audit",), user_id, {"entries": log_list[-50:]})
    print(f"  [audit_to_store] user={user_id}, total entries={len(log_list)}")
    return None


print("\n── 3. Audit Logging to Store ─────────────────────────────────")

audit_store = InMemoryStore()

agent_audit = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    context_schema=AuditCtx,
    store=audit_store,
    checkpointer=MemorySaver(),
    middleware=[audit_to_store],
    system_prompt="You are a weather assistant.",
)

for city in ("Rome", "Sydney"):
    agent_audit.invoke(
        {"messages": [{"role": "user", "content": f"Weather in {city}?"}]},
        context=AuditCtx(user_id="AUDIT-USR-1"),
        config={"configurable": {"thread_id": f"audit-{city.lower()}"}},
    )

audit_log = audit_store.get(("audit",), "AUDIT-USR-1")
print(f"Audit entries: {len(audit_log.value['entries']) if audit_log else 0}")
for entry in (audit_log.value["entries"] if audit_log else []):
    print(f"  thread={entry['thread_id']}, run={entry['run_id']}, msgs={entry['messages']}")


# ════════════════════════════════════════════════════════════════════
# 4. TRANSIENT vs PERSISTENT — side-by-side comparison
# ════════════════════════════════════════════════════════════════════

# TRANSIENT: wrap_model_call — only modifies what the LLM sees, NOT state
@wrap_model_call
def transient_injection(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Inject a transient hint — NOT saved to state."""
    extra = "Note: Always end your answer with a smiley face 😊"
    messages = [*request.messages, {"role": "user", "content": extra}]
    return handler(request.override(messages=messages))


# PERSISTENT: before_model — return dict updates state permanently
@before_model
def persistent_flag(state: AgentState, runtime: Runtime) -> dict | None:
    """Set a persistent flag in state that all future turns can read."""
    if not state.get("context_engineering_enabled"):
        print("  [persistent_flag] Setting context_engineering_enabled=True")
        return {"context_engineering_enabled": True}
    print(f"  [persistent_flag] Already set: {state.get('context_engineering_enabled')}")
    return None


print("\n── 4. Transient vs Persistent (comparison) ──────────────────")

agent_comparison = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    checkpointer=MemorySaver(),
    middleware=[
        transient_injection,   # Changes LLM input only — does NOT persist
        persistent_flag,       # Updates state — PERSISTS across turns
    ],
    system_prompt="You are a helpful assistant.",
)

config3 = {"configurable": {"thread_id": "transient-vs-persistent"}}

result_t1 = agent_comparison.invoke(
    {"messages": [{"role": "user", "content": "Say hello."}]},
    config=config3,
)
print(f"Turn 1 — flag in state: {result_t1.get('context_engineering_enabled')}")
print(f"Turn 1 — response: {result_t1['messages'][-1].content[:100]}")

result_t2 = agent_comparison.invoke(
    {"messages": [{"role": "user", "content": "Say goodbye."}]},
    config=config3,
)
print(f"Turn 2 — flag in state: {result_t2.get('context_engineering_enabled')} (persisted ✓)")
print(f"Turn 2 — response: {result_t2['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 5. COMBINED LIFE-CYCLE STACK
#    Summarization + turn tracking + audit logging together.
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Combined Life-Cycle Middleware Stack ───────────────────")

combined_store = InMemoryStore()

agent_lifecycle = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, get_news],
    context_schema=AuditCtx,
    store=combined_store,
    checkpointer=MemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger={"messages": 8},
            keep={"messages": 2},
        ),
        track_turn_count,          # Persistent counter
        audit_to_store,            # Persistent audit log
    ],
    system_prompt="You are a general knowledge assistant.",
)

config4 = {"configurable": {"thread_id": "lifecycle-combined"}}
queries  = ["Weather in Berlin?", "Latest AI news?", "Weather in Madrid?"]

for q in queries:
    r = agent_lifecycle.invoke(
        {"messages": [{"role": "user", "content": q}]},
        context=AuditCtx(user_id="LIFECYCLE-USR"),
        config=config4,
    )
    print(f"  Q: '{q[:40]}' | turns={r.get('turn_count','?')} | "
          f"msgs={len(r.get('messages',[]))}")

final_audit = combined_store.get(("audit",), "LIFECYCLE-USR")
print(f"\nFinal audit entries: {len(final_audit.value['entries']) if final_audit else 0}")

print("\n" + "═" * 60)
print("Life-Cycle Context — When to Use Each Pattern:")
print("  SummarizationMiddleware → auto-manage long conversations")
print("  before_model (return dict) → persist data across ALL turns")
print("  after_model  (return dict) → persist model response metadata")
print("  wrap_model_call            → transient per-call injection")
print("  store.put() in any hook    → persist across sessions")
print("═" * 60)
print("\n✅ Life-cycle context demo complete.")
