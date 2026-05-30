"""
01_model_context_system_prompt.py
==================================
Demonstrates dynamic SYSTEM PROMPT engineering — pulling instructions from
State, Store, and Runtime Context to give the LLM exactly the right base
instructions for the current invocation.

Concepts covered:
  - Static system_prompt (baseline)
  - @dynamic_prompt reading from request.messages (State shortcut)
  - @dynamic_prompt reading from runtime.store (long-term preferences)
  - @dynamic_prompt reading from runtime.context (deployment config)
  - Combining multiple prompt signals into one coherent prompt
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.tools import tool
from langgraph.store.memory import InMemoryStore

load_dotenv()

print("=" * 60)
print("Model Context — Dynamic System Prompt")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# SHARED TOOL
# ════════════════════════════════════════════════════════════════════

@tool
def get_help_topics() -> str:
    """Return a list of available help topics."""
    return "Available topics: billing, technical support, account management, product features."


# ════════════════════════════════════════════════════════════════════
# 1. STATE-BASED DYNAMIC PROMPT
#    Adapt instructions based on conversation length from state.
#    request.messages is a shortcut for request.state["messages"].
# ════════════════════════════════════════════════════════════════════

@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    """Adjust verbosity based on how long the conversation has been running."""
    message_count = len(request.messages)  # shortcut for request.state["messages"]
    print(f"  [DynamicPrompt/state] message_count={message_count}")

    base = "You are a helpful customer support assistant."

    if message_count > 10:
        base += "\nThis is a long conversation — be extra concise and summarize key points."
    elif message_count > 4:
        base += "\nThis is an ongoing conversation — reference previous context when useful."
    else:
        base += "\nThis is a new conversation — introduce yourself briefly."

    return base


print("\n── 1. State-Aware Prompt (conversation length) ──────────────")

agent_state = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_help_topics],
    middleware=[state_aware_prompt],
)

# Fresh conversation (0 messages → short greeting mode)
result_fresh = agent_state.invoke({
    "messages": [{"role": "user", "content": "Hi, I need some help."}]
})
print(f"Fresh (0 msgs): {result_fresh['messages'][-1].content[:120]}")

# Simulated long conversation
long_history = [{"role": "user" if i % 2 == 0 else "assistant",
                 "content": f"Message {i}"}
                for i in range(12)]
long_history.append({"role": "user", "content": "Can you help me now?"})

result_long = agent_state.invoke({"messages": long_history})
print(f"Long (12 msgs): {result_long['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 2. STORE-BASED DYNAMIC PROMPT
#    Read user preferences from long-term memory to personalize style.
# ════════════════════════════════════════════════════════════════════

@dataclass
class UserCtx:
    user_id: str


@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    """Tailor prompt style based on persisted user preferences."""
    user_id = request.runtime.context.user_id
    store   = request.runtime.store

    base = "You are a helpful assistant."
    if not store:
        return base

    user_prefs = store.get(("preferences",), user_id)
    if user_prefs:
        style = user_prefs.value.get("communication_style", "balanced")
        name  = user_prefs.value.get("name", "")
        print(f"  [DynamicPrompt/store] user={user_id}, style={style}, name={name}")
        if name:
            base += f"\nAddress the user as {name}."
        style_hints = {
            "concise":     "Keep all answers brief — maximum 2 sentences.",
            "detailed":    "Provide thorough, detailed explanations with examples.",
            "bullet_list": "Format all answers as bullet-point lists.",
            "balanced":    "Balance detail and brevity based on question complexity.",
        }
        base += f"\n{style_hints.get(style, '')}"
    else:
        print(f"  [DynamicPrompt/store] no prefs for user={user_id}, using defaults")

    return base


print("\n── 2. Store-Aware Prompt (user preferences) ─────────────────")

store = InMemoryStore()

# Seed preferences for two different users
store.put(("preferences",), "USR-A",
          {"name": "Alice", "communication_style": "bullet_list"})
store.put(("preferences",), "USR-B",
          {"name": "Bob",   "communication_style": "concise"})

agent_store = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    context_schema=UserCtx,
    store=store,
    middleware=[store_aware_prompt],
)

result_alice = agent_store.invoke(
    {"messages": [{"role": "user", "content": "Explain what a REST API is."}]},
    context=UserCtx(user_id="USR-A"),
)
print(f"Alice (bullet_list): {result_alice['messages'][-1].content[:150]}")

result_bob = agent_store.invoke(
    {"messages": [{"role": "user", "content": "Explain what a REST API is."}]},
    context=UserCtx(user_id="USR-B"),
)
print(f"Bob (concise):       {result_bob['messages'][-1].content[:150]}")


# ════════════════════════════════════════════════════════════════════
# 3. RUNTIME CONTEXT–BASED DYNAMIC PROMPT
#    Use per-invocation configuration (role, environment) to
#    determine the level of access and tone of the prompt.
# ════════════════════════════════════════════════════════════════════

@dataclass
class AppCtx:
    user_role:      str   # "admin", "editor", "viewer"
    deployment_env: str   # "production", "staging", "development"


@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    """Build role- and environment-aware system prompt."""
    ctx = request.runtime.context
    print(f"  [DynamicPrompt/context] role={ctx.user_role}, env={ctx.deployment_env}")

    base = "You are a platform management assistant."

    role_hints = {
        "admin":  "You have full admin access. You may discuss all operations.",
        "editor": "You have editor access. Guide the user to read and write operations only.",
        "viewer": "You have read-only access. Direct the user to read operations only.",
    }
    base += f"\n{role_hints.get(ctx.user_role, '')}"

    if ctx.deployment_env == "production":
        base += "\nIMPORTANT: Be extra careful with any data modification advice — this is production."
    elif ctx.deployment_env == "development":
        base += "\nThis is a development environment — feel free to suggest experimental approaches."

    return base


print("\n── 3. Context-Aware Prompt (role + environment) ─────────────")

agent_ctx = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_help_topics],
    context_schema=AppCtx,
    middleware=[context_aware_prompt],
)

result_admin = agent_ctx.invoke(
    {"messages": [{"role": "user", "content": "Can I delete the production database?"}]},
    context=AppCtx(user_role="admin", deployment_env="production"),
)
print(f"Admin (prod): {result_admin['messages'][-1].content[:120]}")

result_viewer = agent_ctx.invoke(
    {"messages": [{"role": "user", "content": "Can I delete the production database?"}]},
    context=AppCtx(user_role="viewer", deployment_env="production"),
)
print(f"Viewer (prod): {result_viewer['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 4. COMBINED PROMPT — all three data sources
# ════════════════════════════════════════════════════════════════════

@dataclass
class FullCtx:
    user_id:   str
    user_role: str
    language:  str


@dynamic_prompt
def combined_prompt(request: ModelRequest) -> str:
    """Compose prompt from state, store, and runtime context."""
    ctx        = request.runtime.context
    store      = request.runtime.store
    msg_count  = len(request.messages)

    parts = [f"You are a multilingual {ctx.user_role}-tier assistant."]

    # From runtime context
    parts.append(f"Always respond in {ctx.language}.")

    # From store
    if store:
        prefs = store.get(("preferences",), ctx.user_id)
        if prefs and prefs.value.get("verbose"):
            parts.append("Provide detailed, well-structured responses.")

    # From state
    if msg_count > 8:
        parts.append("This is a long session — keep answers brief.")

    result = "\n".join(parts)
    print(f"  [DynamicPrompt/combined] parts={len(parts)}")
    return result


print("\n── 4. Combined Prompt (state + store + context) ─────────────")

store2 = InMemoryStore()
store2.put(("preferences",), "USR-X", {"verbose": True})

agent_combined = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    context_schema=FullCtx,
    store=store2,
    middleware=[combined_prompt],
)

result_combined = agent_combined.invoke(
    {"messages": [{"role": "user", "content": "What is machine learning?"}]},
    context=FullCtx(user_id="USR-X", user_role="premium", language="English"),
)
print(f"Combined: {result_combined['messages'][-1].content[:150]}")

print("\n✅ Dynamic system prompt demo complete.")
