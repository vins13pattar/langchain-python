"""
04_summarization_and_dynamic_prompt.py
========================================
Demonstrates SUMMARIZATION MIDDLEWARE and DYNAMIC PROMPTS for memory management.

Concepts covered:
  - SummarizationMiddleware    — auto-summarise old messages when threshold is hit
  - trigger=("tokens", N)      — summarise when token count exceeds N
  - trigger=("messages", N)    — summarise when message count exceeds N
  - keep=("messages", N)       — keep N recent messages after summarisation
  - @dynamic_prompt middleware  — build system prompt dynamically from state/context
  - dynamic_prompt + context   — personalise prompts per user
  - dynamic_prompt + state     — include conversation-derived data in the prompt

SUMMARISATION vs TRIMMING:
  Trimming  → fast, cheap, loses info (old messages are gone)
  Summarising → slightly slower/costly, PRESERVES info as a condensed summary

When to use each:
  • Trimming: short-lived assistants where old context rarely matters
  • Summarising: long-running assistants where full history is important
    (therapy bots, research assistants, coding assistants with long sessions)
"""

import os
import uuid
from typing import TypedDict
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import SummarizationMiddleware, dynamic_prompt, ModelRequest
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

print("=" * 60)
print("Summarisation & Dynamic Prompt Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. SUMMARISATION MIDDLEWARE
#    Automatically compresses long conversations into a rolling summary
# ════════════════════════════════════════════════════════════════════

print("\n── 1. SummarizationMiddleware ────────────────────────────")


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


summarisation_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_current_time],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",      # model used to produce the summary
            trigger=("messages", 8),          # summarise when > 8 messages accumulate
            keep=("messages", 4),             # after summarising, keep 4 recent messages
        )
    ],
    checkpointer=MemorySaver(),
    system_prompt=(
        "You are a helpful assistant. "
        "Remember everything the user tells you throughout the conversation."
    ),
)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}


def chat_sum(msg: str) -> str:
    r = summarisation_agent.invoke(
        {"messages": [{"role": "user", "content": msg}]},
        config,
    )
    return r["messages"][-1].content


print(f"\n  Running long conversation (summarisation triggers at 8 messages)…\n")

turns = [
    "Hi! My name is Vinod.",
    "I am a software engineer.",
    "I live in Bengaluru, India.",
    "My favourite programming language is Python.",
    "I've been coding for 8 years.",
    "I love building AI applications.",
    "I also enjoy photography and hiking.",
    "My current project is a LangChain tutorial series.",
    # At this point summarisation should have triggered
    "What do you know about me so far?",   # recall test after summarisation
]

for i, msg in enumerate(turns, 1):
    reply = chat_sum(msg)
    print(f"  [{i:02d}] User:  {msg}")
    print(f"       Agent: {reply[:100]}{'…' if len(reply) > 100 else ''}")
    print()


# ════════════════════════════════════════════════════════════════════
# 2. SUMMARISATION TRIGGER OPTIONS
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Summarisation trigger options ──────────────────────")
print("""
  # Trigger by TOKEN count (most precise)
  SummarizationMiddleware(
      model="openai:gpt-4o-mini",
      trigger=("tokens", 4000),      # summarise when > 4000 tokens
      keep=("messages", 20),         # keep 20 recent messages
  )

  # Trigger by MESSAGE count (simpler)
  SummarizationMiddleware(
      model="openai:gpt-4o-mini",
      trigger=("messages", 10),      # summarise when > 10 messages
      keep=("messages", 5),          # keep 5 recent messages
  )

  # Recommended: token-based for production (more predictable)
  # Message-based: good for demos and testing

  The summary is inserted as a SystemMessage at the start of the
  trimmed history, so the model still has full context.
""")


# ════════════════════════════════════════════════════════════════════
# 3. @dynamic_prompt — build prompts dynamically from context
# ════════════════════════════════════════════════════════════════════

print("\n── 3. @dynamic_prompt with context ──────────────────────")

# Define context type — passed at invoke() time
class UserContext(TypedDict):
    user_name:  str
    user_role:  str
    language:   str


@tool
def get_weather_report(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name
    """
    data = {"london": "Cloudy 14°C", "tokyo": "Sunny 28°C", "mumbai": "Rainy 30°C"}
    return data.get(city.lower(), f"No weather data for '{city}'")


@dynamic_prompt
def personalised_system_prompt(request: ModelRequest) -> str:
    """Build a system prompt dynamically using context values."""
    ctx  = request.runtime.context
    name = ctx.get("user_name", "User")
    role = ctx.get("user_role", "general user")
    lang = ctx.get("language", "English")

    return (
        f"You are a highly personalised assistant.\n"
        f"The user's name is {name}. Always address them by name.\n"
        f"Their role is: {role}.\n"
        f"Respond in {lang}.\n"
        f"Adapt your tone to their role — technical for engineers, plain language for others."
    )


dynamic_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather_report],
    middleware=[personalised_system_prompt],
    context_schema=UserContext,
    checkpointer=MemorySaver(),
)


def ask_as_user(context: UserContext, question: str) -> str:
    r = dynamic_agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
        context=context,
    )
    return r["messages"][-1].content


print(f"\n  Engineering context:")
eng_ctx = UserContext(user_name="Vinod", user_role="Senior Python Engineer", language="English")
print(f"  🤖 {ask_as_user(eng_ctx, 'What is the weather in Tokyo and how does it affect API latency?')}")

print(f"\n  Executive context (Hindi language):")
exec_ctx = UserContext(user_name="Anita", user_role="Business Executive", language="Hindi")
print(f"  🤖 {ask_as_user(exec_ctx, 'What is the weather in London?')}")


# ════════════════════════════════════════════════════════════════════
# 4. @dynamic_prompt WITH STATE — use conversation-derived data
# ════════════════════════════════════════════════════════════════════

print("\n── 4. @dynamic_prompt with state ────────────────────────")

class SessionState(AgentState):
    user_name:    str = ""
    query_count:  int = 0


@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    """Build a prompt that adapts based on accumulated state."""
    state = request.state
    name  = state.get("user_name") or "Friend"
    count = state.get("query_count", 0)

    greeting = f"The user's name is {name}." if name != "Friend" else "The user has not introduced themselves yet."
    history  = (
        f"This is query #{count + 1} in this session."
        if count > 0 else
        "This is the user's first query in this session."
    )

    return (
        f"You are a helpful assistant.\n"
        f"{greeting}\n"
        f"{history}\n"
        "If you know the user's name, always use it when addressing them."
    )


state_prompt_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[state_aware_prompt],
    state_schema=SessionState,
    checkpointer=MemorySaver(),
)

cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

# First turn — no name in state yet
r1 = state_prompt_agent.invoke(
    {"messages": [{"role": "user", "content": "What is Python?"}],
     "user_name": "",
     "query_count": 0},
    cfg,
)
print(f"\n  Turn 1 (no name): {r1['messages'][-1].content[:120]}…")

# Second turn — name known in state
r2 = state_prompt_agent.invoke(
    {"messages": [{"role": "user", "content": "What is Python?"}],
     "user_name": "Vinod",
     "query_count": 1},
    cfg,
)
print(f"\n  Turn 2 (name=Vinod): {r2['messages'][-1].content[:120]}…")


# ════════════════════════════════════════════════════════════════════
# 5. MEMORY STRATEGY COMPARISON
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Memory strategy comparison ────────────────────────")
print("""
  ┌──────────────────────┬─────────────┬─────────────┬─────────────┐
  │ Strategy             │ Info loss?  │ Cost        │ Complexity  │
  ├──────────────────────┼─────────────┼─────────────┼─────────────┤
  │ Trim (@before_model) │ Yes         │ Low         │ Low         │
  │ Delete (@after_model)│ Yes         │ Low         │ Low         │
  │ Summarise (middleware)│ Minimal    │ Medium      │ Medium      │
  │ Long-term store      │ None        │ Variable    │ High        │
  └──────────────────────┴─────────────┴─────────────┴─────────────┘

  Recommended defaults:
    • Short sessions (< 20 msgs):     no trimming needed
    • Medium sessions (20-100 msgs):  SummarizationMiddleware
    • Long sessions / multi-day:      Summarise + long-term Store

  Dynamic prompts complement memory management by making the system
  prompt context-aware — no extra DB queries needed.
""")
