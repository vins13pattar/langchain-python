"""
02_agent_with_memory.py
=======================
Demonstrates conversation MEMORY across multiple turns using a checkpointer.

Concepts covered:
  - MemorySaver / InMemorySaver — in-process checkpointer for local dev
  - thread_id — scopes a conversation so the agent can remember previous turns
  - Multi-turn dialogue — agent recalls earlier messages automatically

Without a checkpointer every invoke() is stateless — the agent forgets
everything between calls. Adding checkpointer=MemorySaver() + a thread_id
makes the agent remember the full conversation history.
"""

import os
import uuid
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver  # in-memory, for local dev

load_dotenv()


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def remember_fact(topic: str, fact: str) -> str:
    """Store a fact about a topic for later retrieval.

    Args:
        topic: The subject of the fact (e.g. 'user preference', 'project')
        fact:  The piece of information to remember
    """
    return f"✅ Noted: [{topic}] → {fact}"


@tool
def get_date() -> str:
    """Return today's date."""
    from datetime import date
    return str(date.today())


# ── Agent with persistent memory ─────────────────────────────────────────────
# MemorySaver stores state in-process (RAM).
# For production use SqliteSaver or RedisSaver instead.

checkpointer = MemorySaver()

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[remember_fact, get_date],
    checkpointer=checkpointer,         # ← enables persistence
    system_prompt=(
        "You are a personal assistant that remembers everything the user tells you. "
        "Reference earlier parts of our conversation when relevant."
    ),
)

# ── Conversation config ───────────────────────────────────────────────────────
# thread_id scopes the conversation — reuse the SAME id to continue a session,
# use a DIFFERENT id to start a fresh one.

THREAD_ID = str(uuid.uuid4())          # unique session for this run
config = {"configurable": {"thread_id": THREAD_ID}}


def chat(message: str) -> str:
    """Send one message and return the agent's reply."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,                 # same config keeps the thread alive
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    print("=" * 60)
    print("Agent with Conversation Memory Demo")
    print(f"Session thread_id: {THREAD_ID}")
    print("=" * 60)

    # Turn 1 — introduce yourself
    print("\n🧑 Turn 1: My name is Vinod and I am learning LangChain.")
    reply = chat("My name is Vinod and I am learning LangChain.")
    print(f"🤖 Agent: {reply}")

    # Turn 2 — test recall
    print("\n🧑 Turn 2: What's my name?")
    reply = chat("What's my name?")
    print(f"🤖 Agent: {reply}")

    # Turn 3 — add more context
    print("\n🧑 Turn 3: I prefer Python over JavaScript.")
    reply = chat("I prefer Python over JavaScript.")
    print(f"🤖 Agent: {reply}")

    # Turn 4 — recall all context
    print("\n🧑 Turn 4: Summarise what you know about me so far.")
    reply = chat("Summarise what you know about me so far.")
    print(f"🤖 Agent: {reply}")

    # ── Starting a FRESH conversation ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Starting fresh conversation with a NEW thread_id …")
    print("=" * 60)

    NEW_THREAD_ID = str(uuid.uuid4())
    new_config = {"configurable": {"thread_id": NEW_THREAD_ID}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's my name?"}]},
        config=new_config,
    )
    print(f"\n🧑 (new session) What's my name?")
    print(f"🤖 Agent: {result['messages'][-1].content}")
    print("  (Agent has no memory of Vinod — fresh thread!)")
