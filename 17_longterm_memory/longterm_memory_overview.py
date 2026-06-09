"""
longterm_memory_overview.py — Long-Term Memory: all key concepts in one file
Covers: InMemoryStore CRUD, semantic search with IndexConfig, reading/writing in tools,
        memory types (episodic, semantic, procedural, profile), full personalised agent
"""

import time
from collections.abc import Sequence
from typing import List
from dotenv import load_dotenv

from langgraph.store.base import IndexConfig
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage

load_dotenv()

def section(title): print(f"\n{'─'*55}\n{title}\n{'─'*55}")

# Shared embedding function for IndexConfig
def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    return OpenAIEmbeddings(model="text-embedding-3-small").embed_documents(list(texts))


# ════════════════════════════════════════════════════════════════════
# 1. STORE BASICS — put / get / delete / search
# ════════════════════════════════════════════════════════════════════
section("1. STORE BASICS")

store = InMemoryStore()

# put(namespace, key, value_dict)
# Namespace = tuple (like a folder path); key = unique ID within namespace
store.put(("users",), "user_123", {"name": "John Smith", "language": "English", "plan": "pro"})
store.put(("users",), "user_456", {"name": "Alice Chen",  "language": "Chinese", "plan": "enterprise"})
store.put(("user_123", "prefs"), "ui",    {"theme": "dark", "font_size": "large"})
store.put(("user_123", "prefs"), "notifs",{"email": True, "sms": False})

# get(namespace, key) → StoreValue (or None)
item = store.get(("users",), "user_123")
print(f"get: {item.value}  key={item.key!r}  ns={item.namespace}")

prefs = store.get(("user_123", "prefs"), "ui")
print(f"prefs: theme={prefs.value['theme']!r}")

missing = store.get(("users",), "nonexistent")
print(f"missing key: {missing!r}")  # None

# delete
store.put(("temp",), "tmp", {"data": "ephemeral"})
store.delete(("temp",), "tmp")
print(f"after delete: {store.get(('temp',), 'tmp')!r}")

# search — filter by metadata
users_data = [
    ("alice",  {"name": "Alice",  "plan": "enterprise", "country": "US"}),
    ("bob",    {"name": "Bob",    "plan": "pro",        "country": "UK"}),
    ("carol",  {"name": "Carol",  "plan": "enterprise", "country": "US"}),
    ("dave",   {"name": "Dave",   "plan": "free",       "country": "DE"}),
]
for uid, data in users_data:
    store.put(("users",), uid, data)

enterprise = store.search(("users",), filter={"plan": "enterprise"})
print(f"\nEnterprise users: {[u.key for u in enterprise]}")

limited = store.search(("users",), limit=2)
print(f"Search limit=2: {[u.key for u in limited]}")


# ════════════════════════════════════════════════════════════════════
# 2. SEMANTIC SEARCH — IndexConfig + vector embeddings
# ════════════════════════════════════════════════════════════════════
section("2. SEMANTIC SEARCH")

semantic_store = InMemoryStore(
    index=IndexConfig(embed=embed_texts, dims=1536)  # text-embedding-3-small
)

memories = [
    ("mem_1", {"content": "User prefers concise, bullet-point answers.",       "type": "preference"}),
    ("mem_2", {"content": "User is an expert Python developer with 10 years.", "type": "background"}),
    ("mem_3", {"content": "User's current project is a FastAPI microservice.",  "type": "project"}),
    ("mem_4", {"content": "User finds regex confusing and avoids it.",          "type": "preference"}),
    ("mem_5", {"content": "User asked about async/await in Python last session.","type": "history"}),
]
for key, value in memories:
    semantic_store.put(("user_123", "memories"), key, value)

print(f"Indexed {len(memories)} memories")

# Semantic search by natural language query
for q in [
    "How does this user like to receive answers?",
    "What programming project is the user working on?",
]:
    results = semantic_store.search(("user_123", "memories"), query=q, limit=2)
    print(f"\nQuery: {q!r}")
    for r in results:
        print(f"  [{r.key}] {r.value['content'][:70]}")

# Combine semantic + filter
results = semantic_store.search(
    ("user_123", "memories"),
    query="how the user wants information formatted",
    filter={"type": "preference"},
    limit=2,
)
print(f"\nSemantic + filter (preference): {[r.key for r in results]}")


# ════════════════════════════════════════════════════════════════════
# 3. READING MEMORY IN TOOLS (runtime.store reads)
# ════════════════════════════════════════════════════════════════════
section("3. READING MEMORY IN TOOLS")

from dataclasses import dataclass

@dataclass
class UserCtx:
    user_id: str
    name:    str

read_store = InMemoryStore(index=IndexConfig(embed=embed_texts, dims=1536))
read_store.put(("u-001", "profile"), "basic", {"name": "Alice", "role": "engineer", "language": "Python"})
read_store.put(("u-001", "memories"), "mem_1", {"content": "Alice prefers step-by-step explanations."})
read_store.put(("u-001", "memories"), "mem_2", {"content": "Alice is working on a FastAPI project."})

@tool
def get_personalised_answer(question: str, runtime: ToolRuntime[UserCtx]) -> str:
    """Answer question using user profile and memories from long-term store. Args: question."""
    user_id = runtime.context.user_id

    # Read profile
    profile = runtime.store.get(("u-001", "profile"), "basic")
    profile_info = profile.value if profile else {}

    # Semantic search for relevant memories
    relevant = runtime.store.search(("u-001", "memories"), query=question, limit=2)
    memory_ctx = "\n".join(m.value["content"] for m in relevant)

    print(f"  [Tool] profile={profile_info}  memories={len(relevant)}")
    return (
        f"For {profile_info.get('name','user')} ({profile_info.get('role','?')}, "
        f"{profile_info.get('language','?')} dev):\n"
        f"Memories: {memory_ctx}\n\nAnswer: [contextualised response to: {question}]"
    )

agent_read = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_personalised_answer],
    context_schema=UserCtx,
    store=read_store,
    system_prompt="You are a personalised assistant. Use the tool to answer questions.",
)

r = agent_read.invoke(
    {"messages": [HumanMessage("How should I approach my current project?")]},
    context=UserCtx("u-001", "Alice"),
)
print("Personalised response:", r["messages"][-1].content[:150])


# ════════════════════════════════════════════════════════════════════
# 4. WRITING MEMORY FROM TOOLS (runtime.store writes)
# ════════════════════════════════════════════════════════════════════
section("4. WRITING MEMORY FROM TOOLS")

write_store = InMemoryStore(index=IndexConfig(embed=embed_texts, dims=1536))

@tool
def save_fact(fact: str, runtime: ToolRuntime[UserCtx]) -> str:
    """Save an important fact about the user to long-term memory. Args: fact."""
    user_id = runtime.context.user_id
    existing = write_store.get(("memories",), user_id)
    facts = existing.value.get("facts", []) if existing else []
    facts.append({"fact": fact, "ts": time.time()})
    write_store.put(("memories",), user_id, {"facts": facts})
    print(f"  [Tool] saved fact for {user_id}: {fact[:60]}")
    return f"Saved: '{fact}' (Total facts: {len(facts)})"

@tool
def recall_facts(runtime: ToolRuntime[UserCtx]) -> str:
    """Recall all facts about the user from long-term memory."""
    existing = write_store.get(("memories",), runtime.context.user_id)
    if not existing:
        return "No facts saved yet."
    return "\n".join(f"• {f['fact']}" for f in existing.value.get("facts", []))

agent_write = create_agent(
    model="openai:gpt-4o-mini",
    tools=[save_fact, recall_facts],
    context_schema=UserCtx,
    store=write_store,
    checkpointer=MemorySaver(),
    system_prompt="You are a memory-enabled assistant. Save important facts and recall them.",
)

user = UserCtx("u-001", "Alice")
cfg1 = {"configurable": {"thread_id": "t1"}}
r = agent_write.invoke({"messages": [HumanMessage("Remember: I use VSCode and prefer tabs over spaces.")]}, context=user, config=cfg1)
print("Save fact:", r["messages"][-1].content[:80])

cfg2 = {"configurable": {"thread_id": "t2"}}  # new thread
r = agent_write.invoke({"messages": [HumanMessage("What facts do you remember about me?")]}, context=user, config=cfg2)
print("Recall (new thread):", r["messages"][-1].content[:120])


# ════════════════════════════════════════════════════════════════════
# 5. MEMORY TYPES
# ════════════════════════════════════════════════════════════════════
section("5. MEMORY TYPES")

from pydantic import BaseModel, Field

# Episodic — time-stamped events / session history
class EpisodicMemory(BaseModel):
    event:   str = Field(description="What happened")
    summary: str = Field(description="Brief summary")
    ts:      float = Field(default_factory=time.time)

# Semantic — general knowledge facts
class SemanticMemory(BaseModel):
    fact:     str = Field(description="The fact or knowledge")
    category: str = Field(description="Category: language/framework/preference/etc")
    confidence: float = Field(default=1.0)

# Procedural — how-to steps
class ProceduralMemory(BaseModel):
    skill:   str       = Field(description="Skill name")
    steps:   List[str] = Field(description="Step-by-step instructions")
    context: str       = Field(description="When to use this skill")

# Profile — stable user attributes
class ProfileMemory(BaseModel):
    name:         str = Field(description="User's name")
    role:         str = Field(description="Job role")
    experience:   int = Field(description="Years of experience")
    primary_lang: str = Field(description="Primary programming language")

typed_store = InMemoryStore()

# Store each memory type in its own namespace
typed_store.put(("u-001", "episodic"),   "ep_1", EpisodicMemory(event="Deployed FastAPI service", summary="Successful deployment to prod").dict())
typed_store.put(("u-001", "semantic"),   "sm_1", SemanticMemory(fact="Python GIL limits true thread parallelism", category="language").dict())
typed_store.put(("u-001", "procedural"), "pr_1", ProceduralMemory(skill="Debug async code", steps=["Add logging", "Use asyncio.get_event_loop()", "Check for blocking calls"], context="When async code hangs").dict())
typed_store.put(("u-001", "profile"),    "base", ProfileMemory(name="Alice", role="Senior Engineer", experience=10, primary_lang="Python").dict())

for ns in ["episodic", "semantic", "procedural", "profile"]:
    results = typed_store.search(("u-001", ns), limit=5)
    print(f"\n{ns} memories ({len(results)} items):")
    for r in results:
        print(f"  {r.key}: {list(r.value.keys())}")


# ════════════════════════════════════════════════════════════════════
# 6. NAMESPACE PATTERNS
# ════════════════════════════════════════════════════════════════════
section("6. NAMESPACE PATTERNS")

ns_store = InMemoryStore()
# (user, user_id, category)  — user-scoped
ns_store.put(("user", "alice", "profile"),   "v1", {"name": "Alice", "role": "CTO"})
ns_store.put(("user", "alice", "memories"),  "m1", {"content": "Prefers async"})

# (org, org_id, category)  — shared org knowledge
ns_store.put(("org", "acme", "knowledge"),   "style", {"tone": "formal"})

# (session, session_id, context)  — semi-persistent
ns_store.put(("session", "sess99"), "topic", {"subject": "API design", "depth": "expert"})

# (app, defaults)  — global defaults
ns_store.put(("app", "defaults"), "cfg", {"max_tokens": 1000, "language": "en"})

p = ns_store.get(("user", "alice", "profile"), "v1")
print(f"User profile: {p.value}")
mems = ns_store.search(("user", "alice", "memories"), limit=10)
print(f"Alice memories: {[m.value['content'] for m in mems]}")

print("""
Long-Term Memory Quick Reference:
  InMemoryStore()   → in-memory (use DB-backed store in production)
  IndexConfig(embed=fn, dims=N) → enable semantic search
  
  store.put(namespace, key, dict)    → write
  store.get(namespace, key)          → read by key → StoreValue
  store.search(namespace, query=..., filter={...}, limit=N)  → query
  store.delete(namespace, key)       → delete
  
  StoreValue: .value .key .namespace
  
  Namespace convention: (user_id, category) or (org_id, type, ...)
  Memory types: episodic (events) | semantic (facts) | procedural (skills) | profile
  
  In tools: runtime.store.get/search/put  → access from ToolRuntime
  In create_agent: store=store  → makes store available in all tools
""")
