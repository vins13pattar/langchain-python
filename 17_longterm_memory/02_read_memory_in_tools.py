"""
02_read_memory_in_tools.py
===========================
Demonstrates reading from long-term memory inside agent tools.
Tools access the store via ToolRuntime[Context], and the context
(user_id, org_id, etc.) is injected per-invocation.

Concepts covered:
  - context_schema — @dataclass defining per-request context
  - ToolRuntime[Context] — access store + context inside @tool
  - runtime.store → the same store passed to create_agent()
  - runtime.context → the Context object from agent.invoke(context=...)
  - get() in a tool — exact-key lookup by user_id
  - search() in a tool — semantic search within user's namespace
  - Multiple tools reading from different namespaces
  - Reading shared org-level knowledge from tools
  - Cross-namespace reads — org + user combined

Key pattern from the official LangChain docs:
  1. Create store
  2. Pre-populate with data
  3. Define @tool with ToolRuntime[Context] parameter
  4. Pass store= to create_agent()
  5. Pass context= to agent.invoke()
"""

from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

load_dotenv()

print("=" * 60)
print("Reading Long-Term Memory in Tools")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# PART 1: BASIC READ — get user info by ID from context
# Exactly as shown in the official docs.
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Basic User Info Read (official docs pattern) ──────────")

@dataclass
class UserContext:
    user_id: str


# Create and pre-populate store
store1 = InMemoryStore()
store1.put(("users",), "user_123", {
    "name": "John Smith",
    "language": "English",
    "plan": "pro",
})
store1.put(("users",), "user_456", {
    "name": "Alice Chen",
    "language": "Chinese",
    "plan": "enterprise",
})


@tool
def get_user_info(runtime: ToolRuntime[UserContext]) -> str:
    """Look up the current user's profile information."""
    assert runtime.store is not None
    user_id   = runtime.context.user_id
    user_info = runtime.store.get(("users",), user_id)
    if not user_info:
        return f"No profile found for user {user_id}"
    info = user_info.value
    return (
        f"User profile:\n"
        f"  Name:     {info.get('name', 'Unknown')}\n"
        f"  Language: {info.get('language', 'Unknown')}\n"
        f"  Plan:     {info.get('plan', 'Unknown')}"
    )


agent1 = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_user_info],
    store=store1,
    context_schema=UserContext,
    system_prompt="You are a helpful assistant. Use get_user_info to look up user details.",
)

# Invoke with different user contexts
for uid in ["user_123", "user_456"]:
    result = agent1.invoke(
        {"messages": [{"role": "user", "content": "What do you know about me?"}]},
        context=UserContext(user_id=uid),
    )
    print(f"\n  [{uid}]: {result['messages'][-1].content[:150]}")


# ════════════════════════════════════════════════════════════════════
# PART 2: READING USER PREFERENCES + MEMORIES
# Multiple namespaces within the same store — read both from tools.
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Multi-Namespace Reads (preferences + memories) ────────")

store2 = InMemoryStore()

# Populate preferences
store2.put(("user_123", "prefs"), "communication", {
    "style": "concise",
    "format": "bullet_points",
    "detail_level": "expert",
})
store2.put(("user_123", "prefs"), "coding", {
    "language": "Python",
    "style": "functional",
    "test_framework": "pytest",
})

# Populate memories
store2.put(("user_123", "memories"), "background", {
    "expertise": "Senior Python developer, 10 years exp",
    "current_project": "FastAPI microservice for payments",
    "team_size": 5,
})
store2.put(("user_123", "memories"), "last_session", {
    "topic": "async/await patterns",
    "questions_asked": 3,
    "satisfaction": "high",
})

# Populate a different user's data
store2.put(("user_456", "prefs"), "communication", {
    "style": "detailed",
    "format": "prose",
    "detail_level": "beginner",
})


@tool
def get_communication_prefs(runtime: ToolRuntime[UserContext]) -> str:
    """Retrieve how this user prefers to receive information."""
    assert runtime.store is not None
    uid   = runtime.context.user_id
    prefs = runtime.store.get((uid, "prefs"), "communication")
    if not prefs:
        return "No communication preferences set. Using defaults."
    p = prefs.value
    return (
        f"Communication style: {p.get('style')}, "
        f"format: {p.get('format')}, "
        f"detail level: {p.get('detail_level')}"
    )


@tool
def get_user_background(runtime: ToolRuntime[UserContext]) -> str:
    """Retrieve background information about the user."""
    assert runtime.store is not None
    uid    = runtime.context.user_id
    bg     = runtime.store.get((uid, "memories"), "background")
    if not bg:
        return "No background information stored for this user."
    b = bg.value
    return (
        f"Background:\n"
        f"  Expertise: {b.get('expertise')}\n"
        f"  Project:   {b.get('current_project')}\n"
        f"  Team:      {b.get('team_size')} engineers"
    )


@tool
def get_last_session_notes(runtime: ToolRuntime[UserContext]) -> str:
    """Get notes from the user's last interaction."""
    assert runtime.store is not None
    uid     = runtime.context.user_id
    session = runtime.store.get((uid, "memories"), "last_session")
    if not session:
        return "No previous session found."
    s = session.value
    return f"Last session: topic={s.get('topic')!r}, satisfaction={s.get('satisfaction')}"


agent2 = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_communication_prefs, get_user_background, get_last_session_notes],
    store=store2,
    context_schema=UserContext,
    system_prompt=(
        "You are a personalized assistant. Before answering, "
        "use get_communication_prefs to understand how to format your response, "
        "and get_user_background for relevant context. "
        "Reference what you know about the user."
    ),
)

result2 = agent2.invoke(
    {"messages": [{"role": "user", "content": "Help me understand Python async patterns."}]},
    context=UserContext(user_id="user_123"),
)
print(f"\n  Personalized response: {result2['messages'][-1].content[:250]}")


# ════════════════════════════════════════════════════════════════════
# PART 3: RICHER CONTEXT — org_id + user_id
# Multi-tenant apps need both user AND org context.
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Org + User Context (multi-tenant) ─────────────────────")

@dataclass
class TenantContext:
    user_id: str
    org_id:  str


store3 = InMemoryStore()

# Org-level shared data
store3.put(("org", "acme"), "config", {
    "product": "ACME CRM",
    "support_email": "support@acme.com",
    "tier": "enterprise",
})
store3.put(("org", "acme"), "style", {
    "tone": "professional",
    "language": "en",
    "brand_name": "ACME Corp",
})

# User-level data within org
store3.put(("org", "acme", "users"), "john", {
    "name": "John Doe",
    "role": "Sales Manager",
    "permissions": ["read", "write", "approve"],
})
store3.put(("org", "acme", "users"), "jane", {
    "name": "Jane Smith",
    "role": "Developer",
    "permissions": ["read", "write"],
})


@tool
def get_org_config(runtime: ToolRuntime[TenantContext]) -> str:
    """Get this organization's configuration and brand settings."""
    assert runtime.store is not None
    org_id = runtime.context.org_id
    config = runtime.store.get(("org", org_id), "config")
    style  = runtime.store.get(("org", org_id), "style")

    parts = []
    if config:
        c = config.value
        parts.append(f"Product: {c.get('product')}, Tier: {c.get('tier')}")
    if style:
        s = style.value
        parts.append(f"Brand: {s.get('brand_name')}, Tone: {s.get('tone')}")
    return "\n".join(parts) if parts else "No org config found."


@tool
def get_current_user_role(runtime: ToolRuntime[TenantContext]) -> str:
    """Get the current user's role and permissions within the organization."""
    assert runtime.store is not None
    org_id  = runtime.context.org_id
    user_id = runtime.context.user_id
    user    = runtime.store.get(("org", org_id, "users"), user_id)
    if not user:
        return f"User {user_id} not found in org {org_id}"
    u = user.value
    return f"User: {u['name']}, Role: {u['role']}, Permissions: {u['permissions']}"


agent3 = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_org_config, get_current_user_role],
    store=store3,
    context_schema=TenantContext,
    system_prompt=(
        "You are an enterprise assistant. Use get_org_config and get_current_user_role "
        "to personalize your responses to the user's organization and role."
    ),
)

result3 = agent3.invoke(
    {"messages": [{"role": "user", "content": "What product am I using and what can I do?"}]},
    context=TenantContext(user_id="john", org_id="acme"),
)
print(f"\n  Org-aware response: {result3['messages'][-1].content[:200]}")


# ════════════════════════════════════════════════════════════════════
# PART 4: SEARCH IN TOOLS — semantic + filtered recall
# Tools can use store.search() for fuzzy/semantic memory lookup.
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Semantic Memory Search in Tools ───────────────────────")

from collections.abc import Sequence
from langgraph.store.base import IndexConfig
from langchain_openai import OpenAIEmbeddings

openai_emb = OpenAIEmbeddings(model="text-embedding-3-small")

def embed_fn(texts: Sequence[str]) -> list[list[float]]:
    return openai_emb.embed_documents(list(texts))


store4 = InMemoryStore(index=IndexConfig(embed=embed_fn, dims=1536))

# Store memories for a user
for key, content, category in [
    ("m1", "User is an expert in machine learning and neural networks.", "expertise"),
    ("m2", "User struggles with Kubernetes networking configuration.", "weakness"),
    ("m3", "User prefers step-by-step explanations with code examples.", "preference"),
    ("m4", "User is building a recommendation system for e-commerce.", "project"),
    ("m5", "User's team uses GitHub Actions for CI/CD.", "tooling"),
    ("m6", "User asked about transformer attention last week.", "history"),
]:
    store4.put(("user_123", "memories"), key, {
        "content": content, "category": category
    })

print(f"  Stored 6 semantic memories for user_123")


@tool
def recall_relevant_memories(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Search the user's memory store for information relevant to the current question.
    Returns the 3 most semantically similar memories.
    """
    assert runtime.store is not None
    uid     = runtime.context.user_id
    results = runtime.store.search(
        (uid, "memories"),
        query=query,
        limit=3,
    )
    if not results:
        return "No relevant memories found."
    lines = [f"  [{r.value['category']}] {r.value['content']}" for r in results]
    return "Relevant memories:\n" + "\n".join(lines)


agent4 = create_agent(
    model="openai:gpt-4o-mini",
    tools=[recall_relevant_memories],
    store=store4,
    context_schema=UserContext,
    system_prompt=(
        "You are a personalized AI assistant with memory. "
        "When answering questions, use recall_relevant_memories to retrieve "
        "what you know about the user. Tailor your response accordingly."
    ),
)

for q in [
    "How do I improve my ML model's accuracy?",
    "What CI/CD tool should I use for my project?",
]:
    r = agent4.invoke(
        {"messages": [{"role": "user", "content": q}]},
        context=UserContext(user_id="user_123"),
    )
    print(f"\n  Q: {q}")
    print(f"  A: {r['messages'][-1].content[:180]}")

print("\n" + "═" * 60)
print("Read Memory in Tools Summary:")
print("  @dataclass Context           → per-request user_id, org_id")
print("  ToolRuntime[Context]         → runtime.store + runtime.context")
print("  runtime.store.get(ns, key)  → exact-key lookup")
print("  runtime.store.search(ns, ..) → filter + semantic search")
print("  context_schema=Context       → passed to create_agent()")
print("  context=Context(user_id=..)  → passed to agent.invoke()")
print("  Multi-namespace: user + org  → combine in same tool")
print("═" * 60)
print("\n✅ Reading long-term memory in tools demo complete.")
