"""
05_tool_context_reads_writes.py
=================================
Demonstrates TOOL CONTEXT — how tools both READ from and WRITE to
State, Store, and Runtime Context to fetch inputs and persist outputs.

Concepts covered:
  - Reading runtime.state (session state) inside a tool
  - Reading runtime.store (long-term memory) inside a tool
  - Reading runtime.context (static config: user_id, api_key) inside a tool
  - Writing to State via Command(update={...}) return from a tool
  - Writing to Store via runtime.store.put() inside a tool
  - Merging existing store data with new values
  - Full read+write lifecycle in a single tool
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

load_dotenv()

print("=" * 60)
print("Tool Context — Reads and Writes")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# CONTEXT
# ════════════════════════════════════════════════════════════════════

@dataclass
class AppCtx:
    user_id:       str
    api_key:       str = "sk-demo-key-123"
    db_connection: str = "postgresql://localhost/demo"


# ════════════════════════════════════════════════════════════════════
# PART 1: READS
# ════════════════════════════════════════════════════════════════════

# ── 1a. Read from STATE ──────────────────────────────────────────────

@tool
def check_auth_status(runtime: ToolRuntime) -> str:
    """Check whether the current user is authenticated (from session state)."""
    # runtime.state mirrors the agent's current state dict
    current_state    = runtime.state
    is_authenticated = current_state.get("authenticated", False)
    upload_count     = len(current_state.get("uploaded_files", []))
    print(f"  [Tool] check_auth_status → auth={is_authenticated}, files={upload_count}")
    if is_authenticated:
        return f"User is authenticated. {upload_count} file(s) uploaded this session."
    return "User is NOT authenticated. Please log in."


# ── 1b. Read from STORE ──────────────────────────────────────────────

@tool
def get_user_preference(
    preference_key: str,
    runtime: ToolRuntime[AppCtx],
) -> str:
    """
    Retrieve a specific user preference from long-term memory.

    Args:
        preference_key: The preference key to look up (e.g. 'theme', 'language').
    """
    user_id = runtime.context.user_id
    store   = runtime.store

    if not store:
        return "No memory store configured."

    prefs_mem = store.get(("preferences",), user_id)
    if prefs_mem:
        value = prefs_mem.value.get(preference_key)
        print(f"  [Tool] get_user_preference → user={user_id}, key={preference_key}, value={value}")
        return f"{preference_key}: {value}" if value else f"No preference set for '{preference_key}'."
    return f"No preferences found for user {user_id}."


# ── 1c. Read from RUNTIME CONTEXT ───────────────────────────────────

@tool
def fetch_user_data(
    query: str,
    runtime: ToolRuntime[AppCtx],
) -> str:
    """
    Fetch data from the database using credentials from runtime context.

    Args:
        query: The data query or resource name to fetch.
    """
    user_id = runtime.context.user_id
    api_key = runtime.context.api_key
    db      = runtime.context.db_connection

    print(f"  [Tool] fetch_user_data → user={user_id}, query='{query}', db={db[:20]}...")
    # In production: use api_key + db to perform real query
    return (
        f"Data for '{query}' (user={user_id}): "
        "[Record-A, Record-B, Record-C] (via {db})"
    )


print("\n── PART 1: Tool READS ────────────────────────────────────────")

store = InMemoryStore()
store.put(("preferences",), "USR-R1",
          {"theme": "dark", "language": "English", "notifications": "email"})

agent_reads = create_agent(
    model="openai:gpt-4o-mini",
    tools=[check_auth_status, get_user_preference, fetch_user_data],
    context_schema=AppCtx,
    store=store,
    system_prompt="You are a data access assistant.",
)

# Read from state
result_state_read = agent_reads.invoke({
    "messages":       [{"role": "user", "content": "Am I authenticated?"}],
    "authenticated":  True,
    "uploaded_files": [{"name": "doc.pdf"}],
}, context=AppCtx(user_id="USR-R1"))
print(f"State read:   {result_state_read['messages'][-1].content[:100]}")

# Read from store
result_store_read = agent_reads.invoke(
    {"messages": [{"role": "user", "content": "What's my preferred theme?"}]},
    context=AppCtx(user_id="USR-R1"),
)
print(f"Store read:   {result_store_read['messages'][-1].content[:100]}")

# Read from context
result_ctx_read = agent_reads.invoke(
    {"messages": [{"role": "user", "content": "Fetch my orders data."}]},
    context=AppCtx(user_id="USR-R1", api_key="sk-real-key", db_connection="postgresql://prod/db"),
)
print(f"Context read: {result_ctx_read['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# PART 2: WRITES
# ════════════════════════════════════════════════════════════════════

# ── 2a. Write to STATE via Command ───────────────────────────────────

@tool
def authenticate_user(
    password: str,
    runtime: ToolRuntime,
) -> Command:
    """
    Authenticate the user and update session state.

    Args:
        password: The user's password to verify.
    """
    is_correct = password == "secret123"
    print(f"  [Tool] authenticate_user → correct={is_correct}")
    if is_correct:
        return Command(update={"authenticated": True, "auth_method": "password"})
    return Command(update={"authenticated": False, "auth_error": "Invalid password"})


@tool
def upload_file(
    filename: str,
    file_type: str,
    runtime: ToolRuntime,
) -> Command:
    """
    Register an uploaded file in the current session state.

    Args:
        filename:  The name of the file being uploaded.
        file_type: MIME type or file extension.
    """
    current_files = runtime.state.get("uploaded_files", [])
    new_file = {"name": filename, "type": file_type, "summary": f"Uploaded {file_type} file"}
    updated  = current_files + [new_file]
    print(f"  [Tool] upload_file → {filename} (total: {len(updated)} files)")
    return Command(update={"uploaded_files": updated})


# ── 2b. Write to STORE via runtime.store.put() ───────────────────────

@tool
def save_user_preference(
    preference_key: str,
    preference_value: str,
    runtime: ToolRuntime[AppCtx],
) -> str:
    """
    Save a user preference to long-term memory (persists across sessions).

    Args:
        preference_key:   The preference key (e.g. 'theme', 'language').
        preference_value: The value to store.
    """
    user_id = runtime.context.user_id
    store   = runtime.store

    if not store:
        return "No memory store configured."

    # Merge with existing preferences
    existing = store.get(("preferences",), user_id)
    prefs    = existing.value.copy() if existing else {}
    prefs[preference_key] = preference_value

    store.put(("preferences",), user_id, prefs)
    print(f"  [Tool] save_user_preference → user={user_id}, {preference_key}={preference_value}")
    return f"Preference saved: {preference_key} = {preference_value}."


@tool
def save_conversation_insight(
    insight: str,
    runtime: ToolRuntime[AppCtx],
) -> str:
    """
    Extract and save a key insight from this conversation to long-term memory.

    Args:
        insight: The insight or memory to persist across future sessions.
    """
    import time
    user_id = runtime.context.user_id
    store   = runtime.store

    if not store:
        return "No memory store configured."

    existing  = store.get(("insights",), user_id)
    insights  = existing.value.get("list", []) if existing else []
    insights.append({"text": insight, "timestamp": time.time()})

    store.put(("insights",), user_id, {"list": insights})
    print(f"  [Tool] save_conversation_insight → user={user_id}, insight='{insight[:50]}'")
    return f"Insight saved: '{insight}'. Total insights: {len(insights)}."


print("\n── PART 2: Tool WRITES ───────────────────────────────────────")

store2 = InMemoryStore()

agent_writes = create_agent(
    model="openai:gpt-4o-mini",
    tools=[authenticate_user, upload_file, save_user_preference,
           save_conversation_insight, check_auth_status],
    context_schema=AppCtx,
    store=store2,
    checkpointer=MemorySaver(),
    system_prompt="You are a session and preference management assistant.",
)

config = {"configurable": {"thread_id": "writes-demo-1"}}

# Write to state via Command
result_auth = agent_writes.invoke(
    {"messages": [{"role": "user",
                   "content": "Log me in with password 'secret123'."}]},
    context=AppCtx(user_id="USR-W1"),
    config=config,
)
print(f"Auth write:    {result_auth['messages'][-1].content[:100]}")
print(f"State after:   authenticated={result_auth.get('authenticated', False)}")

# Write to store (preferences)
result_pref = agent_writes.invoke(
    {"messages": [{"role": "user",
                   "content": "Save my preference: theme = dark mode."}]},
    context=AppCtx(user_id="USR-W1"),
    config={"configurable": {"thread_id": "writes-demo-2"}},
)
print(f"Pref write:    {result_pref['messages'][-1].content[:100]}")

# Verify it was saved
saved_pref = store2.get(("preferences",), "USR-W1")
print(f"Stored prefs:  {saved_pref.value if saved_pref else 'none'}")

# Write insight to store
result_insight = agent_writes.invoke(
    {"messages": [{"role": "user",
                   "content": "Remember that I always work in the finance domain."}]},
    context=AppCtx(user_id="USR-W1"),
    config={"configurable": {"thread_id": "writes-demo-3"}},
)
print(f"Insight write: {result_insight['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# PART 3: FULL READ + WRITE CYCLE
# ════════════════════════════════════════════════════════════════════

@tool
def update_notification_preferences(
    channel:  str,
    enabled:  bool,
    runtime:  ToolRuntime[AppCtx],
) -> str:
    """
    Toggle notification channel preference and return the full settings.

    Args:
        channel: Notification channel: 'email', 'sms', or 'push'.
        enabled: True to enable, False to disable.
    """
    user_id = runtime.context.user_id
    store   = runtime.store

    if not store:
        return "No store configured."

    # READ existing settings
    existing = store.get(("notifications",), user_id)
    settings = existing.value.copy() if existing else {"email": True, "sms": False, "push": False}

    # MODIFY
    settings[channel] = enabled

    # WRITE back
    store.put(("notifications",), user_id, settings)
    print(f"  [Tool] update_notifications → user={user_id}, {channel}={enabled}")
    return f"Notifications updated: {settings}."


print("\n── PART 3: Full Read + Write Cycle ──────────────────────────")

store3 = InMemoryStore()
store3.put(("notifications",), "USR-RW", {"email": True, "sms": False, "push": True})

agent_rw = create_agent(
    model="openai:gpt-4o-mini",
    tools=[update_notification_preferences],
    context_schema=AppCtx,
    store=store3,
    system_prompt="You are a notification settings assistant.",
)

result_rw = agent_rw.invoke(
    {"messages": [{"role": "user",
                   "content": "Please disable SMS notifications for me."}]},
    context=AppCtx(user_id="USR-RW"),
)
print(f"Response: {result_rw['messages'][-1].content[:100]}")
updated = store3.get(("notifications",), "USR-RW")
print(f"Store:    {updated.value if updated else 'none'}")

print("\n" + "═" * 60)
print("Tool Context Summary:")
print("  Reads:  runtime.state    — current session dict")
print("          runtime.store    — cross-session long-term memory")
print("          runtime.context  — injected config (user_id, api_key)")
print("  Writes: return Command(update={...}) — persists to state")
print("          runtime.store.put()          — persists to store")
print("═" * 60)
print("\n✅ Tool context reads & writes demo complete.")
