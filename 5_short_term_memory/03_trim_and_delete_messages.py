"""
03_trim_and_delete_messages.py
================================
Demonstrates TRIMMING and DELETING messages to manage context window size.

Concepts covered:
  - @before_model middleware    — runs BEFORE each model call
  - @after_model middleware     — runs AFTER each model call
  - RemoveMessage               — mark a specific message for deletion
  - REMOVE_ALL_MESSAGES         — wipe the entire message list
  - Trimming to N recent msgs   — keep last N, remove the rest
  - Conditional trimming        — only trim when threshold is crossed
  - After-model cleanup         — clean up after the model responds

PROBLEM:
  Long conversations grow the message list → eventually exceeds context window.
  Models also degrade with very long contexts (distracted, slow, expensive).

SOLUTION:
  Trim (before model call) or delete (after model call) old messages
  so only the most relevant context reaches the model.

WARNING:
  Always ensure the remaining message history is valid:
  • Some providers require starting with a human/user message
  • AIMessages with tool_calls must be followed by ToolMessages
"""

import os
import uuid
from typing import Any
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langchain_core.messages import RemoveMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

load_dotenv()

print("=" * 60)
print("Trim & Delete Messages Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. BASIC MESSAGE TRIMMING WITH @before_model
#    Keeps system message + last N messages
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Trim to last N messages (@before_model) ───────────")


@before_model
def trim_to_last_3(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the first message (system/context) + the last 3 messages.

    @before_model runs before every model call, including after tool calls.
    This keeps the context window small without losing the current context.
    """
    messages = state["messages"]

    # Nothing to trim if ≤ 3 messages
    if len(messages) <= 3:
        return None   # return None = no changes

    # Keep first message (often a system-like intro) + last 3
    first_msg     = messages[0]
    recent        = messages[-3:]
    new_messages  = [first_msg] + recent

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),   # wipe current list
            *new_messages,                           # replace with trimmed list
        ]
    }


trim_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[trim_to_last_3],
    checkpointer=MemorySaver(),
    system_prompt="You are a helpful assistant. Remember facts the user shares.",
)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}


def send(msg: str) -> str:
    r = trim_agent.invoke({"messages": [{"role": "user", "content": msg}]}, config)
    msgs_in_state = len(r["messages"])
    reply = r["messages"][-1].content
    print(f"    [state has {msgs_in_state} messages]")
    return reply


print()
print(f"  T1: {send('My name is Vinod.')}")
print(f"  T2: {send('I live in Bengaluru.')}")
print(f"  T3: {send('I work as a software engineer.')}")
print(f"  T4: {send('My hobby is photography.')}")   # trimming kicks in
print(f"  T5: {send('Do you remember my name?')}")    # may or may not recall


# ════════════════════════════════════════════════════════════════════
# 2. CONDITIONAL TRIMMING — only trim above a threshold
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Conditional trimming (only above threshold) ────────")


@before_model
def conditional_trim(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Trim messages only when the conversation exceeds 10 messages."""
    messages = state["messages"]
    THRESHOLD = 10      # don't trim below this

    if len(messages) <= THRESHOLD:
        return None     # not yet — keep all messages

    # Above threshold: keep first + last 5
    first   = messages[0]
    recent  = messages[-5:]
    trimmed = [first] + recent

    print(f"      [TRIM] {len(messages)} → {len(trimmed)} messages")
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *trimmed,
        ]
    }


print(f"\n  Conditional trimming doesn't activate until > 10 messages.")


# ════════════════════════════════════════════════════════════════════
# 3. DELETE SPECIFIC MESSAGES with RemoveMessage
#    Target individual messages by their ID
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Delete specific messages (@after_model) ───────────")


@after_model
def delete_oldest_pair(state: AgentState, runtime: Runtime) -> dict | None:
    """After each model call, delete the oldest 2 messages (keep history lean).

    @after_model runs after the model responds but before the next turn.
    Deleting a human+AI pair keeps conversations manageable.
    """
    messages = state["messages"]

    # Only delete when we have more than 4 messages
    if len(messages) <= 4:
        return None

    # Remove the 2 oldest messages (first human + first AI reply)
    to_delete = messages[:2]
    ids_to_delete = [RemoveMessage(id=m.id) for m in to_delete]

    print(f"      [DELETE] removed {len(ids_to_delete)} old messages")
    return {"messages": ids_to_delete}


delete_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[delete_oldest_pair],
    checkpointer=MemorySaver(),
    system_prompt="You are a helpful assistant.",
)

config2 = {"configurable": {"thread_id": str(uuid.uuid4())}}

print()
for msg in [
    "My name is Vinod.",
    "I love Python programming.",
    "I enjoy hiking on weekends.",
    "What's 15 multiplied by 8?",
    "Can you recall my name?",
]:
    r = delete_agent.invoke({"messages": [{"role": "user", "content": msg}]}, config2)
    n = len(r["messages"])
    print(f"  [{n:2d} msgs] Q: {msg[:40]}")
    print(f"           A: {r['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 4. REMOVE ALL MESSAGES — full wipe
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Remove ALL messages (full wipe) ────────────────────")


@before_model
def wipe_on_keyword(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Wipe the entire message history when user says 'clear history'."""
    messages = state["messages"]
    if not messages:
        return None

    last = messages[-1]
    if hasattr(last, "content") and "clear history" in last.content.lower():
        print("      [WIPE] clearing entire message history")
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
        }
    return None


wipe_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[wipe_on_keyword],
    checkpointer=MemorySaver(),
    system_prompt="You are a helpful assistant. Tell me how many messages you can see when asked.",
)

config3 = {"configurable": {"thread_id": str(uuid.uuid4())}}

wipe_agent.invoke({"messages": [{"role": "user", "content": "My name is Vinod."}]}, config3)
wipe_agent.invoke({"messages": [{"role": "user", "content": "I work at TechCorp."}]}, config3)

r_before = wipe_agent.invoke({"messages": [{"role": "user", "content": "How many messages can you see?"}]}, config3)
print(f"\n  Before wipe: {r_before['messages'][-1].content}")
print(f"  State size:  {len(r_before['messages'])} messages")

# Trigger the wipe
wipe_agent.invoke({"messages": [{"role": "user", "content": "Please clear history now."}]}, config3)

r_after = wipe_agent.invoke({"messages": [{"role": "user", "content": "How many messages can you see now?"}]}, config3)
print(f"\n  After wipe:  {r_after['messages'][-1].content}")
print(f"  State size:  {len(r_after['messages'])} messages")


# ════════════════════════════════════════════════════════════════════
# 5. SAFE TRIMMING — preserve HumanMessage→AIMessage structure
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Safe trimming (preserve message structure) ────────")


@before_model
def safe_trim(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Trim to last N messages while ensuring the list starts with a HumanMessage.

    Providers expect the message list to alternate properly and not start
    with an AIMessage or ToolMessage.
    """
    messages = state["messages"]
    MAX_KEEP  = 6          # keep at most 6 messages

    if len(messages) <= MAX_KEEP:
        return None

    # Trim to last MAX_KEEP, then ensure we start with a human message
    recent = messages[-MAX_KEEP:]
    # Drop any leading AI/Tool messages to maintain valid structure
    while recent and not isinstance(recent[0], HumanMessage):
        recent = recent[1:]

    if not recent:
        return None  # safety: don't wipe if we'd end up with nothing

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *recent,
        ]
    }


print(f"\n  Safe trimming always starts the trimmed list with a HumanMessage.")
print(f"  This satisfies provider requirements and avoids validation errors.")


# ════════════════════════════════════════════════════════════════════
# 6. QUICK REFERENCE
# ════════════════════════════════════════════════════════════════════

print("\n── 6. Quick reference ────────────────────────────────────")
print("""
  # Delete a SPECIFIC message by ID
  from langchain_core.messages import RemoveMessage
  return {"messages": [RemoveMessage(id=msg.id)]}

  # Delete MULTIPLE specific messages
  return {"messages": [RemoveMessage(id=m.id) for m in messages[:3]]}

  # Delete ALL messages (full wipe)
  from langgraph.graph.message import REMOVE_ALL_MESSAGES
  return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

  # Replace with a new trimmed list
  return {
      "messages": [
          RemoveMessage(id=REMOVE_ALL_MESSAGES),
          *new_messages,
      ]
  }

  Middleware hooks:
    @before_model  → runs BEFORE each model call (trim before sending)
    @after_model   → runs AFTER model response (clean up after reply)
""")
