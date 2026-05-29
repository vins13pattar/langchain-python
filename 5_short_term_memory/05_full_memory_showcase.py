"""
05_full_memory_showcase.py
===========================
A COMPLETE showcase combining ALL short-term memory concepts.

Simulates a multi-turn PERSONAL PRODUCTIVITY ASSISTANT that:
  ✅ Persists conversation across turns (MemorySaver + thread_id)
  ✅ Maintains custom state (user name, preferences, task count)
  ✅ Trims messages on every before_model call to avoid context overflow
  ✅ Uses a dynamic system prompt personalised from state
  ✅ Has tools that READ and WRITE state (via ToolRuntime + Command)
  ✅ Demonstrates multiple isolated threads (different users)
  ✅ Shows the @after_model hook for PII scrubbing / content filtering

Run this file to see the full memory lifecycle in action.
"""

import os
import uuid
from typing import Any
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model, dynamic_prompt, ModelRequest
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import HumanMessage, RemoveMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command

load_dotenv()

print("=" * 60)
print("Full Short-Term Memory Showcase")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# CUSTOM STATE
# ════════════════════════════════════════════════════════════════════

class ProductivityState(AgentState):
    """Agent state for the productivity assistant."""
    user_name:    str  = ""
    timezone:     str  = "UTC"
    focus_mode:   bool = False       # suppress interruptions
    tasks_added:  int  = 0
    tasks_done:   int  = 0


# ════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════

# Simulated task DB (in-memory for demo)
TASK_DB: dict[str, list[dict]] = {}


@tool
def add_task(title: str, priority: str = "medium", runtime: ToolRuntime = None) -> Command:
    """Add a new task to the user's to-do list.

    Args:
        title:    Task title or description
        priority: 'low', 'medium', or 'high' (default: 'medium')
    """
    state      = runtime.state
    user       = state.get("user_name", "anonymous")
    tasks      = TASK_DB.setdefault(user, [])
    task_id    = len(tasks) + 1
    tasks.append({"id": task_id, "title": title, "priority": priority, "done": False})

    return Command(update={
        "tasks_added": state.get("tasks_added", 0) + 1,
        "messages": [ToolMessage(
            content=f"✅ Task #{task_id} added: '{title}' [{priority} priority]",
            tool_call_id=runtime.tool_call_id,
        )],
    })


@tool
def list_tasks(runtime: ToolRuntime) -> str:
    """List all current tasks.

    No input needed — reads tasks for the current user from state.
    """
    user  = runtime.state.get("user_name", "anonymous")
    tasks = TASK_DB.get(user, [])
    if not tasks:
        return "📋 No tasks yet. Use add_task to create one."
    lines = [
        f"  [{'✓' if t['done'] else ' '}] #{t['id']} {t['title']} ({t['priority']})"
        for t in tasks
    ]
    done  = sum(1 for t in tasks if t["done"])
    return f"📋 Tasks ({done}/{len(tasks)} done):\n" + "\n".join(lines)


@tool
def complete_task(task_id: int, runtime: ToolRuntime) -> Command:
    """Mark a task as completed.

    Args:
        task_id: The ID of the task to mark done
    """
    user  = runtime.state.get("user_name", "anonymous")
    tasks = TASK_DB.get(user, [])
    task  = next((t for t in tasks if t["id"] == task_id), None)

    if not task:
        return Command(update={"messages": [ToolMessage(
            content=f"Task #{task_id} not found.",
            tool_call_id=runtime.tool_call_id,
        )]})

    task["done"] = True
    return Command(update={
        "tasks_done": runtime.state.get("tasks_done", 0) + 1,
        "messages": [ToolMessage(
            content=f"🎉 Task #{task_id} completed: '{task['title']}'",
            tool_call_id=runtime.tool_call_id,
        )],
    })


@tool
def toggle_focus_mode(runtime: ToolRuntime) -> Command:
    """Toggle focus mode on or off.

    In focus mode, the assistant avoids off-topic responses.
    No arguments needed.
    """
    current = runtime.state.get("focus_mode", False)
    new_val = not current
    return Command(update={
        "focus_mode": new_val,
        "messages": [ToolMessage(
            content=f"🎯 Focus mode {'ON' if new_val else 'OFF'}.",
            tool_call_id=runtime.tool_call_id,
        )],
    })


@tool
def get_productivity_stats(runtime: ToolRuntime) -> str:
    """Get productivity statistics for this session.

    No input needed.
    """
    state = runtime.state
    added = state.get("tasks_added", 0)
    done  = state.get("tasks_done", 0)
    focus = state.get("focus_mode", False)
    name  = state.get("user_name", "Unknown")
    tz    = state.get("timezone", "UTC")

    pct = f"{done / added * 100:.0f}%" if added > 0 else "N/A"
    return (
        f"📊 Session stats for {name}:\n"
        f"  Tasks added:     {added}\n"
        f"  Tasks completed: {done}\n"
        f"  Completion rate: {pct}\n"
        f"  Focus mode:      {'ON 🎯' if focus else 'OFF'}\n"
        f"  Timezone:        {tz}"
    )


# ════════════════════════════════════════════════════════════════════
# MIDDLEWARE
# ════════════════════════════════════════════════════════════════════

@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    """Build system prompt from current agent state."""
    state = request.state
    name  = state.get("user_name", "there")
    focus = state.get("focus_mode", False)
    done  = state.get("tasks_done", 0)
    added = state.get("tasks_added", 0)

    base = (
        f"You are a personal productivity assistant.\n"
        f"The user's name is {name}. Always address them by name.\n"
        f"Session so far: {added} tasks added, {done} completed."
    )
    if focus:
        base += (
            "\n⚠️  FOCUS MODE is ON. "
            "Only respond to task-related queries. "
            "Politely redirect off-topic requests."
        )
    return base


@before_model
def trim_old_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep first message + last 6 messages to stay within context budget."""
    messages = state["messages"]
    MAX      = 8

    if len(messages) <= MAX:
        return None

    first   = messages[0]
    recent  = messages[-6:]
    trimmed = [first] + recent

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *trimmed,
        ]
    }


@after_model
def filter_sensitive_content(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove AI messages containing sensitive placeholder words."""
    BLOCKED = ["password", "secret_key", "api_key"]
    last    = state["messages"][-1]

    if hasattr(last, "content") and any(w in last.content.lower() for w in BLOCKED):
        print("    [FILTER] Blocked sensitive content from AI reply")
        return {"messages": [RemoveMessage(id=last.id)]}
    return None


# ════════════════════════════════════════════════════════════════════
# CREATE THE AGENT
# ════════════════════════════════════════════════════════════════════

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[add_task, list_tasks, complete_task, toggle_focus_mode, get_productivity_stats],
    middleware=[context_aware_prompt, trim_old_messages, filter_sensitive_content],
    state_schema=ProductivityState,
    checkpointer=MemorySaver(),
)


# ════════════════════════════════════════════════════════════════════
# RUN THE SHOWCASE
# ════════════════════════════════════════════════════════════════════

def chat(user_input: str, config: dict, **initial_state) -> str:
    payload = {"messages": [{"role": "user", "content": user_input}]}
    payload.update(initial_state)
    r = agent.invoke(payload, config)
    return r["messages"][-1].content


print("\n── Session: Vinod's Productivity Assistant ───────────────")

cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

print(f"\n🧑 Starting session with initial state…")
r = chat("Hi! Let's get to work.", cfg, user_name="Vinod", timezone="Asia/Kolkata")
print(f"🤖 {r}")

print(f"\n🧑 Add a task:")
print(f"🤖 {chat('Add a high priority task: Review PR for the payments module.', cfg)}")

print(f"\n🧑 Add more tasks:")
print(f"🤖 {chat('Add a medium task: Update the README docs. Also add a low priority task: Clean up old branches.', cfg)}")

print(f"\n🧑 List tasks:")
print(f"🤖 {chat('Show me all my tasks.', cfg)}")

print(f"\n🧑 Complete a task:")
print(f"🤖 {chat('Mark task #1 as done.', cfg)}")

print(f"\n🧑 Enable focus mode:")
print(f"🤖 {chat('Turn on focus mode.', cfg)}")

print(f"\n🧑 Off-topic (should be redirected in focus mode):")
print(f"🤖 {chat('Tell me a joke about penguins.', cfg)}")

print(f"\n🧑 Get stats:")
print(f"🤖 {chat('Show my productivity stats for this session.', cfg)}")


print("\n── Isolation: different users, different sessions ────────")

# Alice's session — completely isolated from Vinod's
alice_cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
chat("Add a task: Prepare slide deck for Monday meeting.", alice_cfg, user_name="Alice")
print(f"\n  Alice's tasks: {chat('List my tasks.', alice_cfg)}")
print(f"  Vinod's tasks: {chat('List my tasks.', cfg)}")
print(f"\n  ✅ Alice and Vinod have completely separate task lists and state.")
