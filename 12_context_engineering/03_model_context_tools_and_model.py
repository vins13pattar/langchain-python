"""
03_model_context_tools_and_model.py
=====================================
Demonstrates dynamic TOOL SELECTION and dynamic MODEL SWITCHING using
wrap_model_call — adapting what tools and which model the LLM sees
at call time based on State, Store, and Runtime Context.

Concepts covered:
  - request.override(tools=filtered_tools) — dynamic tool filtering
  - State-based tool gating (auth status, conversation stage)
  - Store-based tool filtering (feature flags per user)
  - Runtime Context–based tool filtering (role-based permissions)
  - request.override(model=new_model) — dynamic model switching
  - State-based model selection (conversation length)
  - Store-based model preference (user preferred model)
  - Runtime Context–based model (cost tier / environment)
"""

import os
from dataclasses import dataclass
from typing import Callable
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.store.memory import InMemoryStore

load_dotenv()

print("=" * 60)
print("Model Context — Dynamic Tools & Model Selection")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# TOOLS (varying sensitivity levels)
# ════════════════════════════════════════════════════════════════════

@tool
def public_search(query: str) -> str:
    """Search public knowledge base (available to all users)."""
    return f"Public results for '{query}': [FAQ-1, FAQ-2, FAQ-3]."

@tool
def read_user_data(resource: str) -> str:
    """Read user account data (requires authentication)."""
    return f"User data for resource '{resource}': [sample data]."

@tool
def write_user_data(resource: str, value: str) -> str:
    """Write/update user data (requires editor role)."""
    return f"Updated '{resource}' to '{value}'."

@tool
def delete_data(resource: str) -> str:
    """Delete a data resource (admin only)."""
    return f"Deleted resource '{resource}'."

@tool
def advanced_analytics(metric: str) -> str:
    """Run advanced analytics queries (premium feature)."""
    return f"Analytics for '{metric}': trend up 12% this month."

@tool
def export_data(format: str) -> str:
    """Export data to a file format (premium feature)."""
    return f"Data exported to {format} format."


# ════════════════════════════════════════════════════════════════════
# 1. STATE-BASED TOOL GATING
#    Only expose sensitive tools after the user is authenticated.
# ════════════════════════════════════════════════════════════════════

@wrap_model_call
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Filter tools based on authentication status in State."""
    is_authenticated = request.state.get("authenticated", False)
    msg_count        = len(request.messages)
    print(f"  [ToolFilter/state] authenticated={is_authenticated}, msgs={msg_count}")

    if not is_authenticated:
        # Unauthenticated: public tools only
        tools = [t for t in request.tools if t.name == "public_search"]
    elif msg_count < 3:
        # Authenticated but early in conversation: basic tools
        tools = [t for t in request.tools if t.name in ("public_search", "read_user_data")]
    else:
        # Authenticated + established session: all tools
        tools = request.tools

    request = request.override(tools=tools)
    print(f"  [ToolFilter/state] {len(tools)} tools exposed: {[t.name for t in tools]}")
    return handler(request)


print("\n── 1. State-Based Tool Gating (auth status) ─────────────────")

agent_state_tools = create_agent(
    model="openai:gpt-4o-mini",
    tools=[public_search, read_user_data, write_user_data, delete_data],
    middleware=[state_based_tools],
    system_prompt="You are a data management assistant.",
)

result_unauth = agent_state_tools.invoke({
    "messages":       [{"role": "user", "content": "Search for account settings."}],
    "authenticated":  False,
})
print(f"Unauth:  {result_unauth['messages'][-1].content[:100]}")

result_auth = agent_state_tools.invoke({
    "messages":      [
        {"role": "user",      "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user",      "content": "Show me my account data."},
    ],
    "authenticated": True,
})
print(f"Auth:    {result_auth['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 2. STORE-BASED TOOL FILTERING (feature flags)
#    Each user has an individualized set of feature flags.
# ════════════════════════════════════════════════════════════════════

@dataclass
class UserCtx:
    user_id: str


@wrap_model_call
def store_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Filter tools based on user's feature flags in Store."""
    user_id = request.runtime.context.user_id
    store   = request.runtime.store

    if store:
        flags = store.get(("features",), user_id)
        if flags:
            enabled = set(flags.value.get("enabled_tools", []))
            tools   = [t for t in request.tools if t.name in enabled]
            print(f"  [ToolFilter/store] user={user_id}, enabled={enabled}")
            request = request.override(tools=tools)
        else:
            # No feature flags — default to public tools only
            tools = [t for t in request.tools if t.name == "public_search"]
            request = request.override(tools=tools)

    return handler(request)


print("\n── 2. Store-Based Tool Filtering (feature flags) ────────────")

feature_store = InMemoryStore()
feature_store.put(("features",), "FREE-1",
                  {"enabled_tools": ["public_search", "read_user_data"]})
feature_store.put(("features",), "PREM-1",
                  {"enabled_tools": ["public_search", "read_user_data",
                                     "write_user_data", "advanced_analytics", "export_data"]})

agent_store_tools = create_agent(
    model="openai:gpt-4o-mini",
    tools=[public_search, read_user_data, write_user_data, advanced_analytics, export_data],
    context_schema=UserCtx,
    store=feature_store,
    middleware=[store_based_tools],
    system_prompt="You are a data assistant.",
)

result_free = agent_store_tools.invoke(
    {"messages": [{"role": "user", "content": "Run analytics on my usage data."}]},
    context=UserCtx(user_id="FREE-1"),
)
print(f"Free tier:    {result_free['messages'][-1].content[:100]}")

result_prem = agent_store_tools.invoke(
    {"messages": [{"role": "user", "content": "Run analytics on my usage data."}]},
    context=UserCtx(user_id="PREM-1"),
)
print(f"Premium tier: {result_prem['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 3. RUNTIME CONTEXT–BASED TOOL FILTERING (RBAC)
# ════════════════════════════════════════════════════════════════════

@dataclass
class RoleCtx:
    user_role: str  # "admin", "editor", "viewer"


@wrap_model_call
def context_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Filter tools based on user role from Runtime Context."""
    role = request.runtime.context.user_role
    print(f"  [ToolFilter/context] role={role}")

    if role == "admin":
        tools = request.tools           # All tools
    elif role == "editor":
        tools = [t for t in request.tools if t.name != "delete_data"]
    else:  # viewer
        tools = [t for t in request.tools
                 if t.name in ("public_search", "read_user_data")]

    request = request.override(tools=tools)
    print(f"  [ToolFilter/context] {len(tools)} tools: {[t.name for t in tools]}")
    return handler(request)


print("\n── 3. Runtime Context–Based Tool Filtering (RBAC) ───────────")

agent_rbac_tools = create_agent(
    model="openai:gpt-4o-mini",
    tools=[public_search, read_user_data, write_user_data, delete_data],
    context_schema=RoleCtx,
    middleware=[context_based_tools],
    system_prompt="You are a data management assistant.",
)

for role in ("admin", "editor", "viewer"):
    r = agent_rbac_tools.invoke(
        {"messages": [{"role": "user", "content": "Delete the old data records."}]},
        context=RoleCtx(user_role=role),
    )
    print(f"  {role:8}: {r['messages'][-1].content[:80]}")


# ════════════════════════════════════════════════════════════════════
# 4. STATE-BASED MODEL SWITCHING
#    Use a cheaper model for short conversations, larger model for
#    long ones that need a bigger context window.
# ════════════════════════════════════════════════════════════════════

_MODELS = {
    "efficient": init_chat_model("openai:gpt-4o-mini"),
    "standard":  init_chat_model("openai:gpt-4o-mini"),  # same in example
}

@wrap_model_call
def state_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Select model based on conversation length in State."""
    msg_count = len(request.messages)
    print(f"  [ModelSelect/state] msg_count={msg_count}")

    if msg_count > 15:
        model = _MODELS["standard"]   # large context window
        print("  [ModelSelect/state] → standard model (long conversation)")
    else:
        model = _MODELS["efficient"]
        print("  [ModelSelect/state] → efficient model (short conversation)")

    return handler(request.override(model=model))


print("\n── 4. State-Based Model Selection (conversation length) ──────")

agent_model_state = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[state_based_model],
    system_prompt="You are a helpful assistant.",
)

r_short = agent_model_state.invoke({
    "messages": [{"role": "user", "content": "What is 2 + 2?"}]
})
print(f"Short: {r_short['messages'][-1].content[:80]}")


# ════════════════════════════════════════════════════════════════════
# 5. RUNTIME CONTEXT–BASED MODEL SELECTION (cost tier)
# ════════════════════════════════════════════════════════════════════

@dataclass
class CostCtx:
    cost_tier:   str   # "premium", "standard", "budget"
    environment: str   # "production", "staging"


@wrap_model_call
def context_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Select model based on cost tier and environment."""
    ctx = request.runtime.context
    print(f"  [ModelSelect/context] tier={ctx.cost_tier}, env={ctx.environment}")

    # In a real app you'd use distinct models; here we reuse gpt-4o-mini for demo
    model_name = {
        "premium":  "openai:gpt-4o",
        "standard": "openai:gpt-4o-mini",
        "budget":   "openai:gpt-4o-mini",
    }.get(ctx.cost_tier, "openai:gpt-4o-mini")
    print(f"  [ModelSelect/context] → {model_name}")

    model   = init_chat_model(model_name)
    request = request.override(model=model)
    return handler(request)


print("\n── 5. Context-Based Model Selection (cost tier) ─────────────")

agent_model_ctx = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    context_schema=CostCtx,
    middleware=[context_based_model],
    system_prompt="You are a helpful assistant.",
)

for tier in ("premium", "standard", "budget"):
    r = agent_model_ctx.invoke(
        {"messages": [{"role": "user", "content": "Summarize the benefits of cloud computing."}]},
        context=CostCtx(cost_tier=tier, environment="production"),
    )
    print(f"  {tier:10}: {r['messages'][-1].content[:80]}")

print("\n✅ Dynamic tools & model selection demo complete.")
