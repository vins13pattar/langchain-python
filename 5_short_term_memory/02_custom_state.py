"""
02_custom_state.py
==================
Demonstrates CUSTOM AGENT STATE — extending AgentState with your own fields.

Concepts covered:
  - AgentState            — base class with built-in "messages" key
  - Custom state fields   — add user_id, preferences, counters, flags
  - state_schema=         — pass your custom state class to create_agent
  - Passing initial state — set custom fields in agent.invoke()
  - Reading state in tools via runtime.state
  - Writing state via Command return from tools
  - State persists across turns within the same thread

The built-in AgentState only has "messages".
When you need to track extra information (user identity, counters,
preferences, authentication status, etc.) you extend it.
"""

import os
import uuid
from typing import Optional
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

print("=" * 60)
print("Custom Agent State Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. DEFINE CUSTOM STATE
# ════════════════════════════════════════════════════════════════════

class UserAgentState(AgentState):
    """Extended state with user-specific fields.

    AgentState already provides the 'messages' key with an add_messages
    reducer. Any extra fields you define here are also persisted to the
    checkpointer with each turn.
    """
    # User identity
    user_id:   str  = ""
    user_name: str  = ""

    # Preferences (persisted across turns in the same thread)
    language:  str  = "English"
    theme:     str  = "light"

    # Session counters
    query_count:   int  = 0
    tool_call_count: int = 0

    # Flags
    is_authenticated: bool = False


# ════════════════════════════════════════════════════════════════════
# 2. TOOLS THAT READ & WRITE STATE
# ════════════════════════════════════════════════════════════════════

@tool
def get_session_stats(runtime: ToolRuntime) -> str:
    """Return statistics about the current conversation session.

    No input needed — reads from conversation state.
    """
    state = runtime.state
    return (
        f"Session stats:\n"
        f"  User:        {state.get('user_name', 'Unknown') or 'Unknown'}\n"
        f"  Language:    {state.get('language', 'English')}\n"
        f"  Theme:       {state.get('theme', 'light')}\n"
        f"  Queries:     {state.get('query_count', 0)}\n"
        f"  Tool calls:  {state.get('tool_call_count', 0)}\n"
        f"  Auth status: {'✅ Authenticated' if state.get('is_authenticated') else '❌ Not authenticated'}"
    )


@tool
def set_preference(key: str, value: str, runtime: ToolRuntime) -> Command:
    """Update a user preference setting.

    Args:
        key:   Preference to set — 'language' or 'theme'
        value: New value (language: 'English'/'Hindi'/'French'; theme: 'light'/'dark')
    """
    allowed = {
        "language": ["English", "Hindi", "French", "Spanish"],
        "theme":    ["light", "dark"],
    }
    if key not in allowed:
        return Command(update={"messages": [
            ToolMessage(
                content=f"Unknown preference '{key}'. Valid keys: {list(allowed.keys())}",
                tool_call_id=runtime.tool_call_id,
            )
        ]})
    if value not in allowed[key]:
        return Command(update={"messages": [
            ToolMessage(
                content=f"Invalid value '{value}' for '{key}'. Valid values: {allowed[key]}",
                tool_call_id=runtime.tool_call_id,
            )
        ]})

    return Command(update={
        key: value,                              # ← write to state field
        "messages": [
            ToolMessage(
                content=f"✅ {key} set to '{value}'",
                tool_call_id=runtime.tool_call_id,
            )
        ],
    })


@tool
def increment_query_count(runtime: ToolRuntime) -> Command:
    """Increment the query counter in agent state.

    No arguments needed — updates state automatically.
    """
    current = runtime.state.get("query_count", 0)
    return Command(update={
        "query_count": current + 1,
        "messages": [
            ToolMessage(
                content=f"Query #{current + 1} recorded.",
                tool_call_id=runtime.tool_call_id,
            )
        ],
    })


# ════════════════════════════════════════════════════════════════════
# 3. CREATE AGENT WITH CUSTOM STATE
# ════════════════════════════════════════════════════════════════════

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_session_stats, set_preference, increment_query_count],
    state_schema=UserAgentState,           # ← custom state schema
    checkpointer=MemorySaver(),
    system_prompt=(
        "You are a personalised assistant. "
        "Use tools to manage user preferences and track session stats. "
        "Be concise."
    ),
)


# ════════════════════════════════════════════════════════════════════
# 4. PASSING INITIAL STATE IN invoke()
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Passing initial state in invoke() ─────────────────")

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# Pass initial values for custom fields alongside messages
r = agent.invoke(
    {
        "messages":         [{"role": "user", "content": "Hello! Show me my session stats."}],
        "user_id":          "u-007",       # ← initial state values
        "user_name":        "Vinod",
        "language":         "English",
        "is_authenticated": True,
    },
    config,
)
print(f"\n  🤖 {r['messages'][-1].content}")

# Verify state was stored
current_state = r
print(f"\n  user_id from state:  {current_state.get('user_id', 'N/A')}")
print(f"  user_name from state: {current_state.get('user_name', 'N/A')}")


# ════════════════════════════════════════════════════════════════════
# 5. STATE PERSISTS ACROSS TURNS
# ════════════════════════════════════════════════════════════════════

print("\n── 2. State persists across turns ───────────────────────")

# Turn 2 — set a preference (no need to re-pass user_id — it's in state)
r2 = agent.invoke(
    {"messages": [{"role": "user", "content": "Set my theme to dark."}]},
    config,   # same thread
)
print(f"\n  Turn 2: {r2['messages'][-1].content}")
print(f"  Theme in state: {r2.get('theme', 'N/A')}")

# Turn 3 — set language preference
r3 = agent.invoke(
    {"messages": [{"role": "user", "content": "Set my language to Hindi."}]},
    config,
)
print(f"\n  Turn 3: {r3['messages'][-1].content}")
print(f"  Language in state: {r3.get('language', 'N/A')}")

# Turn 4 — verify all preferences stuck
r4 = agent.invoke(
    {"messages": [{"role": "user", "content": "Show me my current session stats."}]},
    config,
)
print(f"\n  Turn 4 (all prefs):\n  {r4['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 6. MULTIPLE USERS — each thread has its OWN state
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Multiple users with isolated state ─────────────────")

alice_config = {"configurable": {"thread_id": "alice-session"}}
bob_config   = {"configurable": {"thread_id": "bob-session"}}

# Alice starts with dark theme + Hindi
agent.invoke(
    {"messages": [{"role": "user", "content": "Set my theme to dark."}],
     "user_name": "Alice"},
    alice_config,
)

# Bob uses default (light theme + English)
agent.invoke(
    {"messages": [{"role": "user", "content": "Show my stats."}],
     "user_name": "Bob"},
    bob_config,
)

# Verify isolation
ra = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my theme?"}]},
    alice_config,
)
rb = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my theme?"}]},
    bob_config,
)

print(f"\n  Alice's theme: {ra['messages'][-1].content}")
print(f"  Bob's theme:   {rb['messages'][-1].content}")
print(f"\n  ✅ Each user thread has completely isolated state.")


# ════════════════════════════════════════════════════════════════════
# 7. KEY POINTS ABOUT CUSTOM STATE
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Custom state — key points ─────────────────────────")
print("""
  AgentState provides:
    messages: list    ← always present, uses add_messages reducer

  Your custom fields:
    • Simple values (str, int, bool, dict) work out of the box
    • Last-write wins (no special reducer needed)
    • Access via runtime.state["field_name"] in tools
    • Update via Command(update={"field_name": new_value})

  Passing initial state:
    agent.invoke({"messages": [...], "user_id": "u-001", ...}, config)
    ↑ Only for first turn — subsequent turns read from checkpointer

  State vs Context:
    state    = mutable, auto-saved to checkpointer, per-thread
    context  = immutable, passed on every invoke(), per-run
""")
