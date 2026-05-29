"""
02_tool_runtime.py
==================
Demonstrates ToolRuntime — accessing the full Runtime object inside tool
functions, including context, store (long-term memory), and stream writer.

Concepts covered:
  - ToolRuntime[Context] as a tool parameter (auto-injected, not passed by model)
  - Accessing runtime.context inside a tool
  - Accessing runtime.store for long-term memory reads/writes
  - Accessing runtime.writer to push custom streaming updates
  - Tools with mixed regular args + ToolRuntime parameter
  - Why ToolRuntime arguments are invisible to the LLM (not in schema)
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

load_dotenv()

print("=" * 60)
print("ToolRuntime — Accessing Runtime Inside Tools")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# CONTEXT DEFINITION
# ════════════════════════════════════════════════════════════════════

@dataclass
class UserContext:
    user_id:   str
    user_name: str
    tenant:    str


# ════════════════════════════════════════════════════════════════════
# 1. ACCESSING CONTEXT IN A TOOL
#    ToolRuntime[Context] is always the first parameter of a tool.
#    The LLM does NOT see it in the tool schema — it's injected by
#    the runtime automatically.
# ════════════════════════════════════════════════════════════════════

@tool
def get_personalized_greeting(runtime: ToolRuntime[UserContext]) -> str:
    """Return a personalized greeting for the current user."""
    ctx = runtime.context
    print(f"  [Tool] get_personalized_greeting → user={ctx.user_name}")
    return f"Welcome back, {ctx.user_name}! Your account ({ctx.user_id}) is active."


@tool
def get_account_summary(runtime: ToolRuntime[UserContext]) -> str:
    """Retrieve a summary of the current user's account."""
    ctx = runtime.context
    print(f"  [Tool] get_account_summary → user={ctx.user_id}, tenant={ctx.tenant}")
    return (
        f"Account summary for {ctx.user_name} (tenant: {ctx.tenant}): "
        "15 active projects, 3 pending invoices, 2 open support tickets."
    )


print("\n── 1. Accessing Context Inside Tools ────────────────────────")

agent_ctx = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_personalized_greeting, get_account_summary],
    context_schema=UserContext,
    system_prompt="You are a personalized account assistant.",
)

result = agent_ctx.invoke(
    {"messages": [{"role": "user", "content": "Show me my account summary."}]},
    context=UserContext(user_id="USR-881", user_name="Elena", tenant="globex-corp"),
)
print(f"Response: {result['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 2. MIXED TOOL — normal args + ToolRuntime
#    The LLM provides `query` and `top_k`. The runtime provides
#    `runtime` (ToolRuntime) automatically.
# ════════════════════════════════════════════════════════════════════

@tool
def search_user_documents(
    query: str,
    top_k: int,
    runtime: ToolRuntime[UserContext],
) -> str:
    """
    Search through the user's private document library.

    Args:
        query: The search query string.
        top_k: Number of results to return (1-10).
    """
    ctx = runtime.context
    print(f"  [Tool] search_user_documents → user={ctx.user_id}, query='{query}', k={top_k}")
    # In production, this would query a real per-user vector store
    return (
        f"Top {top_k} results for '{query}' in {ctx.user_name}'s library: "
        f"[Doc-A] Budget Q4, [Doc-B] Project Plan, [Doc-C] Vendor Contracts."
    )


print("\n── 2. Mixed Tool (args + ToolRuntime) ────────────────────────")

agent_mixed = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_user_documents],
    context_schema=UserContext,
    system_prompt="You are a document search assistant.",
)

result_mixed = agent_mixed.invoke(
    {"messages": [{"role": "user", "content":
        "Search my documents for 'budget' and show me the top 3 results."}]},
    context=UserContext(user_id="USR-002", user_name="Marco", tenant="finance-dept"),
)
print(f"Response: {result_mixed['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 3. ACCESSING STORE (LONG-TERM MEMORY) IN A TOOL
#    runtime.store is a BaseStore for reading/writing persistent
#    memories across sessions. It's None if no store is configured.
# ════════════════════════════════════════════════════════════════════

@tool
def fetch_user_preferences(runtime: ToolRuntime[UserContext]) -> str:
    """Fetch the user's saved preferences from long-term memory store."""
    ctx = runtime.context
    if runtime.store:
        # Try to read from store: namespace=(tenant, "prefs"), key=user_id
        memory = runtime.store.get((ctx.tenant, "prefs"), ctx.user_id)
        if memory:
            prefs = memory.value.get("preferences", "No preferences saved.")
            print(f"  [Tool] fetch_user_preferences → found in store: {prefs[:60]}")
            return f"Stored preferences for {ctx.user_name}: {prefs}"
        else:
            print(f"  [Tool] fetch_user_preferences → no store entry found")
    else:
        print("  [Tool] fetch_user_preferences → no store configured")
    return f"No saved preferences found for {ctx.user_name}. Using defaults."


@tool
def save_user_preferences(
    preferences: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """
    Save the user's preferences to long-term memory.

    Args:
        preferences: A description of the user's preferences to remember.
    """
    ctx = runtime.context
    if runtime.store:
        runtime.store.put(
            (ctx.tenant, "prefs"),
            ctx.user_id,
            {"preferences": preferences},
        )
        print(f"  [Tool] save_user_preferences → saved: '{preferences[:60]}'")
        return f"Preferences saved for {ctx.user_name}: {preferences}"
    print("  [Tool] save_user_preferences → no store configured")
    return "No store configured — preferences not saved."


print("\n── 3. Accessing Store in Tools ───────────────────────────────")

# Without a store (runtime.store is None)
agent_no_store = create_agent(
    model="openai:gpt-4o-mini",
    tools=[fetch_user_preferences, save_user_preferences],
    context_schema=UserContext,
    system_prompt="You are a preferences assistant.",
)

result_no_store = agent_no_store.invoke(
    {"messages": [{"role": "user", "content":
        "Fetch my preferences and then save: 'I prefer dark mode and compact layout'."}]},
    context=UserContext(user_id="USR-100", user_name="Nora", tenant="demo"),
)
print(f"Response (no store): {result_no_store['messages'][-1].content[:120]}")


# With an in-memory store
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

agent_with_store = create_agent(
    model="openai:gpt-4o-mini",
    tools=[fetch_user_preferences, save_user_preferences],
    context_schema=UserContext,
    store=store,
    system_prompt="You are a preferences assistant.",
)

# First call: save preferences
agent_with_store.invoke(
    {"messages": [{"role": "user", "content":
        "Save my preference: 'I prefer dark mode and English language'."}]},
    context=UserContext(user_id="USR-200", user_name="Oscar", tenant="prod"),
)

# Second call: fetch the saved preferences
result_with_store = agent_with_store.invoke(
    {"messages": [{"role": "user", "content": "What are my saved preferences?"}]},
    context=UserContext(user_id="USR-200", user_name="Oscar", tenant="prod"),
)
print(f"Response (with store): {result_with_store['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 4. STREAM WRITER — Push custom progress updates from a tool
# ════════════════════════════════════════════════════════════════════

@tool
def long_running_analysis(
    dataset: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """
    Run a long analysis on a dataset, emitting progress updates.

    Args:
        dataset: Name of the dataset to analyze.
    """
    import time
    ctx = runtime.context
    steps = ["Loading data", "Cleaning", "Analyzing", "Generating report"]

    for i, step in enumerate(steps, 1):
        msg = f"[{i}/{len(steps)}] {step} for {dataset}..."
        print(f"  [Tool] stream update: {msg}")
        if runtime.writer:
            runtime.writer({"progress": msg, "user": ctx.user_name})
        time.sleep(0.05)  # Simulate work

    return f"Analysis of '{dataset}' complete. 3 anomalies found."


print("\n── 4. Stream Writer (Custom Progress Updates) ───────────────")

agent_stream = create_agent(
    model="openai:gpt-4o-mini",
    tools=[long_running_analysis],
    context_schema=UserContext,
    system_prompt="You are a data analysis assistant.",
)

result_stream = agent_stream.invoke(
    {"messages": [{"role": "user", "content":
        "Run the analysis on dataset 'sales_2025_q1'."}]},
    context=UserContext(user_id="USR-300", user_name="Priya", tenant="analytics"),
)
print(f"Response: {result_stream['messages'][-1].content[:120]}")

print("\n" + "═" * 60)
print("ToolRuntime Capabilities:")
print("  runtime.context       — injected context dataclass instance")
print("  runtime.store         — BaseStore for long-term memory (or None)")
print("  runtime.writer        — stream writer for custom updates (or None)")
print("  runtime.execution_info — thread_id, run_id, attempt_number")
print("  runtime.server_info   — assistant_id, user (LangGraph Server only)")
print("  ToolRuntime args are NOT sent to the LLM — fully invisible")
print("═" * 60)
print("\n✅ ToolRuntime demo complete.")
