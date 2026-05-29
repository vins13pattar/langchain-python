"""
01_checkpointer_basics.py
==========================
Demonstrates BASIC SHORT-TERM MEMORY with checkpointers.

Concepts covered:
  - InMemorySaver         — in-process checkpointer (dev/testing)
  - thread_id             — scopes memory to a conversation
  - Resuming a thread     — same thread_id = same conversation history
  - Multiple threads      — different thread_ids = isolated conversations
  - How state is stored   — agent state → checkpointer → resume on next call

Short-term memory = the agent remembers WITHIN a single thread.
Different threads = completely separate conversations with no cross-recall.

Without a checkpointer:
  Every agent.invoke() is stateless — no memory between calls.

With a checkpointer:
  Every agent.invoke() with the same thread_id picks up where it left off.
"""

import os
import uuid
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

print("=" * 60)
print("Checkpointer Basics Demo")
print("=" * 60)


# ── Tool (optional — agents work without tools too) ────────────────
@tool
def get_user_profile(user_id: str) -> str:
    """Fetch a user profile by ID.

    Args:
        user_id: The user's unique ID
    """
    profiles = {
        "u001": "Name: Alice, Role: Engineer, City: London",
        "u002": "Name: Bob,   Role: Designer, City: New York",
    }
    return profiles.get(user_id, "Profile not found")


# ════════════════════════════════════════════════════════════════════
# 1. AGENT WITHOUT CHECKPOINTER — stateless (no memory)
# ════════════════════════════════════════════════════════════════════

print("\n── 1. WITHOUT checkpointer (stateless) ───────────────────")

stateless_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_user_profile],
    system_prompt="You are a helpful assistant.",
)

r1 = stateless_agent.invoke({"messages": [{"role": "user", "content": "Hi! My name is Vinod."}]})
print(f"\n  Turn 1: {r1['messages'][-1].content}")

r2 = stateless_agent.invoke({"messages": [{"role": "user", "content": "What is my name?"}]})
print(f"  Turn 2: {r2['messages'][-1].content}")
print(f"\n  ⚠️  Without a checkpointer the agent has NO memory of turn 1.")


# ════════════════════════════════════════════════════════════════════
# 2. AGENT WITH InMemorySaver — short-term memory within a thread
# ════════════════════════════════════════════════════════════════════

print("\n── 2. WITH InMemorySaver (short-term memory) ─────────────")

memory_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_user_profile],
    checkpointer=MemorySaver(),           # ← add short-term memory
    system_prompt="You are a helpful assistant. Remember everything the user tells you.",
)

# A thread_id scopes the conversation — think of it like a chat session ID
thread_id = "session-vinod-001"
config    = {"configurable": {"thread_id": thread_id}}

r1 = memory_agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Vinod and I live in Bengaluru."}]},
    config,
)
print(f"\n  Turn 1: {r1['messages'][-1].content}")

r2 = memory_agent.invoke(
    {"messages": [{"role": "user", "content": "What is my name and where do I live?"}]},
    config,          # ← same thread_id = same conversation history
)
print(f"  Turn 2: {r2['messages'][-1].content}")

r3 = memory_agent.invoke(
    {"messages": [{"role": "user", "content": "What city does the person you know live in?"}]},
    config,
)
print(f"  Turn 3: {r3['messages'][-1].content}")

# Count how many messages accumulated in state
total_msgs = len(r3["messages"])
print(f"\n  Total messages in state after 3 turns: {total_msgs}")


# ════════════════════════════════════════════════════════════════════
# 3. MULTIPLE THREADS — isolated conversations
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Multiple isolated threads ──────────────────────────")

thread_alice = {"configurable": {"thread_id": "thread-alice"}}
thread_bob   = {"configurable": {"thread_id": "thread-bob"}}

# Alice's conversation
memory_agent.invoke(
    {"messages": [{"role": "user", "content": "My name is Alice and I'm a software engineer."}]},
    thread_alice,
)

# Bob's conversation (completely separate)
memory_agent.invoke(
    {"messages": [{"role": "user", "content": "My name is Bob and I'm a graphic designer."}]},
    thread_bob,
)

# Each thread remembers only its own history
ra = memory_agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name and job?"}]},
    thread_alice,
)
rb = memory_agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name and job?"}]},
    thread_bob,
)

print(f"\n  Thread (Alice): {ra['messages'][-1].content}")
print(f"  Thread (Bob):   {rb['messages'][-1].content}")
print(f"\n  ✅ Each thread has completely isolated memory.")


# ════════════════════════════════════════════════════════════════════
# 4. UNIQUE thread_ids IN PRODUCTION
#    Use uuid / user-session IDs — never hardcode in production
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Generating unique thread IDs ──────────────────────")

# In a real app, generate a unique ID per user session
def new_thread() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}

def resume_thread(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}

# Create a new session
session_config  = new_thread()
session_id      = session_config["configurable"]["thread_id"]

memory_agent.invoke(
    {"messages": [{"role": "user", "content": "Remember: the project deadline is June 30th."}]},
    session_config,
)

# Later — resume the same session by ID
resumed_config = resume_thread(session_id)
result = memory_agent.invoke(
    {"messages": [{"role": "user", "content": "What deadline did I mention?"}]},
    resumed_config,
)

print(f"\n  Session ID: {session_id}")
print(f"  Resumed:    {result['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 5. CHECKPOINTER SUMMARY
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Checkpointer options ──────────────────────────────")
print("""
  Development / Testing:
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()        ← in-process, lost on restart

  Production (persistent):
    # PostgreSQL
    from langgraph.checkpoint.postgres import PostgresSaver
    checkpointer = PostgresSaver.from_conn_string("postgresql://...")

    # SQLite (lightweight)
    from langgraph.checkpoint.sqlite import SqliteSaver
    checkpointer = SqliteSaver.from_conn_string("sqlite:///memory.db")

  ⚠️  MemorySaver is cleared when the Python process restarts.
     Use a database-backed checkpointer for real persistence.
""")
