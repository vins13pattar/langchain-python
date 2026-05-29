"""
01_context_schema.py
====================
Demonstrates how to define and pass a RUNTIME CONTEXT to create_agent —
the primary dependency injection mechanism for LangChain agents.

Concepts covered:
  - Defining a context_schema with a Python dataclass
  - Passing context at invoke time via context=Context(...)
  - Accessing context inside tools using ToolRuntime[Context]
  - Accessing context inside middleware using Runtime[Context]
  - Why runtime context is better than global state or closures

Runtime context is analogous to dependency injection in web frameworks:
instead of hardcoding values or using globals, you inject per-request
configuration (user ID, tenant, feature flags, DB connections, etc.)
when calling agent.invoke().
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

load_dotenv()

print("=" * 60)
print("Runtime Context Schema Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. SIMPLE CONTEXT — single field
# ════════════════════════════════════════════════════════════════════

@dataclass
class UserContext:
    user_name: str


@tool
def greet_user(runtime: ToolRuntime[UserContext]) -> str:
    """Greet the user by their name from the runtime context."""
    name = runtime.context.user_name
    print(f"  [Tool] greet_user → user_name='{name}'")
    return f"Hello, {name}! How can I assist you today?"


print("\n── 1. Simple Context (user_name) ─────────────────────────────")

agent_simple = create_agent(
    model="openai:gpt-4o-mini",
    tools=[greet_user],
    context_schema=UserContext,
    system_prompt="You are a friendly personal assistant. Always greet the user.",
)

result_alice = agent_simple.invoke(
    {"messages": [{"role": "user", "content": "Hello!"}]},
    context=UserContext(user_name="Alice"),   # ← injected at call time
)
print(f"Alice: {result_alice['messages'][-1].content[:100]}")

result_bob = agent_simple.invoke(
    {"messages": [{"role": "user", "content": "Hello!"}]},
    context=UserContext(user_name="Bob"),     # ← different user, same agent
)
print(f"Bob:   {result_bob['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 2. RICH CONTEXT — multiple fields (user, tenant, feature flags)
# ════════════════════════════════════════════════════════════════════

@dataclass
class AppContext:
    user_id:    str
    user_name:  str
    tenant:     str
    is_premium: bool
    language:   str = "en"


@tool
def get_user_profile(runtime: ToolRuntime[AppContext]) -> str:
    """Return a summary of the current user's profile from context."""
    ctx = runtime.context
    plan = "Premium" if ctx.is_premium else "Free"
    print(f"  [Tool] get_user_profile → user={ctx.user_id}, tenant={ctx.tenant}")
    return (
        f"User profile: {ctx.user_name} (ID: {ctx.user_id}), "
        f"Tenant: {ctx.tenant}, Plan: {plan}, Language: {ctx.language}."
    )


@tool
def get_feature_flags(runtime: ToolRuntime[AppContext]) -> str:
    """Return enabled feature flags for the current user."""
    ctx = runtime.context
    flags = ["advanced_search", "export_csv"]
    if ctx.is_premium:
        flags += ["ai_insights", "priority_support"]
    print(f"  [Tool] get_feature_flags → premium={ctx.is_premium}")
    return f"Enabled flags for {ctx.user_name}: {', '.join(flags)}."


print("\n── 2. Rich Context (multi-field dataclass) ──────────────────")

agent_rich = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_user_profile, get_feature_flags],
    context_schema=AppContext,
    system_prompt="You are a user account assistant. Use the context to personalize responses.",
)

# Free-tier user
result_free = agent_rich.invoke(
    {"messages": [{"role": "user", "content": "What features do I have access to?"}]},
    context=AppContext(
        user_id="USR-001",
        user_name="Carlos",
        tenant="acme-corp",
        is_premium=False,
    ),
)
print(f"Free tier:    {result_free['messages'][-1].content[:120]}")

# Premium user
result_premium = agent_rich.invoke(
    {"messages": [{"role": "user", "content": "Show me my profile and feature flags."}]},
    context=AppContext(
        user_id="USR-002",
        user_name="Diana",
        tenant="enterprise-ltd",
        is_premium=True,
        language="fr",
    ),
)
print(f"Premium tier: {result_premium['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 3. CONTEXT WITHOUT TOOLS — access in system_prompt via middleware
# ════════════════════════════════════════════════════════════════════

from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dataclass
class LanguageContext:
    language: str
    formality: str = "casual"  # "casual" or "formal"


@dynamic_prompt
def localized_prompt(request: ModelRequest) -> str:
    """Generate a system prompt personalized to the user's language and formality."""
    ctx = request.runtime.context
    lang = ctx.language
    form = ctx.formality
    print(f"  [DynamicPrompt] language={lang}, formality={form}")
    return (
        f"You are a helpful assistant. "
        f"Always respond in {lang}. "
        f"Use a {'formal' if form == 'formal' else 'casual and friendly'} tone."
    )


print("\n── 3. Dynamic Prompt from Context ───────────────────────────")

agent_localized = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    context_schema=LanguageContext,
    middleware=[localized_prompt],
)

result_spanish = agent_localized.invoke(
    {"messages": [{"role": "user", "content": "Tell me a fun fact about space."}]},
    context=LanguageContext(language="Spanish", formality="casual"),
)
print(f"Spanish (casual):  {result_spanish['messages'][-1].content[:120]}")

result_formal = agent_localized.invoke(
    {"messages": [{"role": "user", "content": "Tell me a fun fact about space."}]},
    context=LanguageContext(language="English", formality="formal"),
)
print(f"English (formal):  {result_formal['messages'][-1].content[:120]}")

print("\n" + "═" * 60)
print("Context Schema Key Points:")
print("  - Define context as a Python @dataclass")
print("  - Pass context_schema=YourDataclass to create_agent")
print("  - Inject at call time: agent.invoke(..., context=YourDataclass(...))")
print("  - Access in tools: ToolRuntime[Context] parameter")
print("  - Access in middleware: Runtime[Context] parameter")
print("  - No global state — each invocation gets its own context")
print("═" * 60)
print("\n✅ Context schema demo complete.")
