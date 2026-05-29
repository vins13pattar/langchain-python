"""
03_tool_runtime_context.py
==========================
Demonstrates RUNTIME CONTEXT — accessing state, context, and the store
from inside a tool via the ToolRuntime parameter.

Concepts covered:
  - ToolRuntime parameter       — hidden from model, auto-injected
  - runtime.state               — short-term memory (conversation messages + custom fields)
  - runtime.context             — immutable per-run data (user ID, permissions)
  - runtime.store               — long-term persistent memory across sessions
  - runtime.tool_call_id        — unique ID for this tool invocation
  - ToolRuntime[ContextType]    — typed context access
  - get_runtime()               — alternative access method inside tools

ToolRuntime is the single interface for all runtime injection.
It is INVISIBLE to the model — the model never sees these parameters.
"""

import os
import uuid
from dataclasses import dataclass
from typing import Any, Optional
from dotenv import load_dotenv

from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

load_dotenv()

print("=" * 60)
print("Tool Runtime & Context Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. ACCESSING STATE (short-term / conversation memory)
# ════════════════════════════════════════════════════════════════════

print("\n── 1. runtime.state — conversation state ─────────────────")

@tool
def get_message_count(runtime: ToolRuntime) -> str:
    """Return the number of messages in the current conversation.

    No arguments needed — reads directly from conversation state.
    """
    messages = runtime.state["messages"]
    return f"This conversation has {len(messages)} messages so far."


@tool
def get_last_user_message(runtime: ToolRuntime) -> str:
    """Return the most recent message from the user in the conversation.

    Useful for summarising or referencing what was just said.
    """
    from langchain_core.messages import HumanMessage
    messages = runtime.state["messages"]
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return f"Last user message: '{msg.content}'"
    return "No user messages found."


@tool
def summarise_conversation(runtime: ToolRuntime) -> str:
    """Produce a brief summary of the conversation so far.

    No input needed — reads all messages from state.
    """
    messages = runtime.state["messages"]
    total    = len(messages)
    human_n  = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_n     = total - human_n
    topics   = [m.content[:40] for m in messages if m.__class__.__name__ == "HumanMessage"]
    return (
        f"Conversation summary:\n"
        f"  Total messages: {total} ({human_n} user, {ai_n} AI)\n"
        f"  Topics discussed: {topics}"
    )


# Agent with state tools
state_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_message_count, get_last_user_message, summarise_conversation],
    checkpointer=MemorySaver(),
    system_prompt="You are a helpful assistant. Use tools to inspect the conversation.",
)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# First turn
r = state_agent.invoke(
    {"messages": [HumanMessage("Hello! My name is Vinod.")]},
    config=config,
)
print(f"\n  Turn 1: {r['messages'][-1].content}")

# Second turn — agent can now count messages
r = state_agent.invoke(
    {"messages": [HumanMessage("How many messages have we exchanged?")]},
    config=config,
)
print(f"  Turn 2: {r['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 2. ACCESSING CONTEXT (per-run immutable data)
# ════════════════════════════════════════════════════════════════════

print("\n── 2. runtime.context — per-run user data ────────────────")

@dataclass
class UserContext:
    user_id:  str
    username: str
    role:     str = "viewer"       # admin | editor | viewer
    plan:     str = "free"         # free | pro | enterprise


USER_DB = {
    "u-001": {"name": "Alice",  "account_type": "Premium", "balance": 5000.0},
    "u-002": {"name": "Bob",    "account_type": "Standard", "balance": 1200.0},
    "u-003": {"name": "Vinod",  "account_type": "Enterprise", "balance": 99000.0},
}


@tool
def get_my_account(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account details.

    No arguments needed — reads user identity from runtime context.
    """
    user_id = runtime.context.user_id
    user    = USER_DB.get(user_id)
    if not user:
        return f"No account found for user ID '{user_id}'"
    return (
        f"Account details:\n"
        f"  Name:    {user['name']}\n"
        f"  Type:    {user['account_type']}\n"
        f"  Balance: ${user['balance']:,.2f}"
    )


@tool
def perform_admin_task(task: str, runtime: ToolRuntime[UserContext]) -> str:
    """Perform an administrative task (admin role required).

    Args:
        task: The administrative task to perform
    """
    ctx = runtime.context
    if ctx.role != "admin":
        return (
            f"❌ Access denied. Your role is '{ctx.role}'. "
            f"This action requires 'admin' role."
        )
    return f"✅ Admin task '{task}' completed by {ctx.username} (role={ctx.role})"


@tool
def get_plan_features(runtime: ToolRuntime[UserContext]) -> str:
    """List the features available in the current user's plan.

    No input needed — reads plan from context.
    """
    plan_features = {
        "free":       ["5 requests/day", "Basic support"],
        "pro":        ["Unlimited requests", "Priority support", "API access"],
        "enterprise": ["Unlimited requests", "24/7 support", "API access", "Custom SLA"],
    }
    plan  = runtime.context.plan
    feats = plan_features.get(plan, ["Unknown plan"])
    return f"Your {plan.upper()} plan includes:\n" + "\n".join(f"  • {f}" for f in feats)


context_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_my_account, perform_admin_task, get_plan_features],
    context_schema=UserContext,
    checkpointer=MemorySaver(),
    system_prompt="You are a personalised banking assistant. Use tools to answer questions.",
)


def ask_as(user: UserContext, question: str) -> str:
    result = context_agent.invoke(
        {"messages": [HumanMessage(question)]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
        context=user,
    )
    return result["messages"][-1].content


admin = UserContext(user_id="u-003", username="Vinod", role="admin", plan="enterprise")
viewer = UserContext(user_id="u-002", username="Bob",   role="viewer", plan="free")

print(f"\n  Admin user asks for account info:")
print(f"  🤖 {ask_as(admin, 'Show me my account details.')}")

print(f"\n  Admin runs an admin task:")
print(f"  🤖 {ask_as(admin, 'Perform the admin task: generate_monthly_report')}")

print(f"\n  Viewer tries an admin task (should be denied):")
print(f"  🤖 {ask_as(viewer, 'Perform the admin task: delete_all_logs')}")

print(f"\n  Viewer asks about their plan:")
print(f"  🤖 {ask_as(viewer, 'What features do I have in my plan?')}")


# ════════════════════════════════════════════════════════════════════
# 3. LONG-TERM MEMORY (Store)
# ════════════════════════════════════════════════════════════════════

print("\n── 3. runtime.store — long-term memory ───────────────────")

@tool
def save_user_preference(key: str, value: str, runtime: ToolRuntime[UserContext]) -> str:
    """Save a user preference to persistent storage.

    Args:
        key:   Preference name (e.g. 'language', 'theme')
        value: Preference value (e.g. 'English', 'dark')
    """
    user_id = runtime.context.user_id
    store   = runtime.store

    # Get existing prefs
    existing = store.get(("preferences",), user_id)
    prefs = existing.value if existing else {}

    # Update
    prefs[key] = value
    store.put(("preferences",), user_id, prefs)
    return f"✅ Saved preference: {key} = '{value}' for user {user_id}"


@tool
def get_user_preferences(runtime: ToolRuntime[UserContext]) -> str:
    """Get all saved preferences for the current user.

    No input needed — reads from persistent store.
    """
    user_id = runtime.context.user_id
    stored  = runtime.store.get(("preferences",), user_id)
    if not stored or not stored.value:
        return "No saved preferences found."
    prefs = stored.value
    lines = [f"  {k}: {v}" for k, v in prefs.items()]
    return "Your saved preferences:\n" + "\n".join(lines)


@tool
def remember_fact(topic: str, fact: str, runtime: ToolRuntime[UserContext]) -> str:
    """Store a fact about a topic in the user's long-term memory.

    Args:
        topic: Category/topic for the fact (e.g. 'project', 'meeting')
        fact:  The information to remember
    """
    user_id = runtime.context.user_id
    store   = runtime.store

    existing = store.get(("facts",), user_id)
    facts = existing.value if existing else {}
    facts.setdefault(topic, []).append(fact)
    store.put(("facts",), user_id, facts)
    return f"✅ Remembered [{topic}]: {fact}"


@tool
def recall_facts(topic: str, runtime: ToolRuntime[UserContext]) -> str:
    """Recall stored facts about a topic.

    Args:
        topic: The topic to recall facts about
    """
    user_id  = runtime.context.user_id
    existing = runtime.store.get(("facts",), user_id)
    if not existing:
        return f"No facts stored for topic '{topic}'"
    facts = existing.value.get(topic, [])
    if not facts:
        return f"No facts found for topic '{topic}'"
    return f"Facts about '{topic}':\n" + "\n".join(f"  • {f}" for f in facts)


persistent_store = InMemoryStore()   # In production: PostgresStore

store_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[save_user_preference, get_user_preferences, remember_fact, recall_facts],
    context_schema=UserContext,
    checkpointer=MemorySaver(),
    store=persistent_store,           # ← inject the store
    system_prompt="You are a personal assistant with long-term memory. Save and recall information as requested.",
)

user = UserContext(user_id="u-001", username="Alice", role="editor", plan="pro")

# Session 1 — save preferences
print("\n  Session 1 — saving preferences and facts…")
r = store_agent.invoke(
    {"messages": [HumanMessage("Set my language preference to Hindi and remember that my project deadline is June 30th.")]},
    config={"configurable": {"thread_id": str(uuid.uuid4())}},
    context=user,
)
print(f"  🤖 {r['messages'][-1].content}")

# Session 2 — recall (new thread = fresh conversation, same store)
print("\n  Session 2 — recalling from persistent store…")
r = store_agent.invoke(
    {"messages": [HumanMessage("What are my preferences and what do you know about my project?")]},
    config={"configurable": {"thread_id": str(uuid.uuid4())}},  # new thread!
    context=user,
)
print(f"  🤖 {r['messages'][-1].content}")
