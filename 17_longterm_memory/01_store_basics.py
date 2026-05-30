"""
01_store_basics.py
===================
Demonstrates the InMemoryStore — the foundation of long-term memory
in LangChain agents. Covers all store operations: put, get, search,
delete, and semantic search with vector embeddings.

Key difference from short-term memory (MemorySaver):
  - Short-term: scoped to a single thread_id, lost between sessions
  - Long-term:  persists across ALL threads and sessions, queryable

Concepts covered:
  - InMemoryStore — in-memory dict (use DB-backed in production)
  - Namespace tuples — hierarchical organization (user_id, context)
  - Key — unique identifier within a namespace
  - put() / get() / delete() — CRUD operations
  - search() — content filtering + semantic search
  - IndexConfig — enable vector embeddings for semantic search
  - StoreValue — what store.get() and store.search() return
  - Namespace patterns — users, orgs, sessions, memories
"""

from collections.abc import Sequence
from dotenv import load_dotenv

from langgraph.store.base import IndexConfig
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

print("=" * 60)
print("Long-Term Memory — InMemoryStore Basics")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# PART 1: BASIC STORE — put / get / delete
# Namespace = tuple (like a folder path)
# Key       = string (like a filename within the folder)
# Value     = dict  (JSON-serializable data)
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Basic CRUD (put / get / delete) ───────────────────────")

store = InMemoryStore()

# put(namespace, key, value_dict)
# Namespace is a tuple — convention: (user_id, category)
store.put(("users",), "user_123", {
    "name": "John Smith",
    "language": "English",
    "plan": "pro",
    "joined": "2024-01",
})

store.put(("users",), "user_456", {
    "name": "Alice Chen",
    "language": "Chinese",
    "plan": "enterprise",
    "joined": "2024-03",
})

# User-scoped preferences namespace
store.put(("user_123", "preferences"), "ui", {
    "theme": "dark",
    "font_size": "large",
    "language": "English",
})

store.put(("user_123", "preferences"), "notifications", {
    "email":   True,
    "sms":     False,
    "push":    True,
    "frequency": "daily",
})

# get(namespace, key) → StoreValue | None
item = store.get(("users",), "user_123")
print(f"\n  get('users', 'user_123'):")
print(f"    item.value = {item.value}")
print(f"    item.key   = {item.key!r}")
print(f"    item.ns    = {item.namespace}")

prefs = store.get(("user_123", "preferences"), "ui")
print(f"\n  get('user_123/preferences', 'ui'):")
print(f"    theme = {prefs.value['theme']!r}")

# get a missing key → returns None
missing = store.get(("users",), "nonexistent")
print(f"\n  get missing key → {missing!r}")

# delete(namespace, key)
store.put(("temp",), "to_delete", {"data": "ephemeral"})
before = store.get(("temp",), "to_delete")
store.delete(("temp",), "to_delete")
after  = store.get(("temp",), "to_delete")
print(f"\n  delete: before={before.value['data']!r}, after={after!r}")


# ════════════════════════════════════════════════════════════════════
# PART 2: SEARCH — content filtering + listing
# search() returns a list[StoreValue], sorted by relevance when
# a semantic query is provided.
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Search with Content Filters ───────────────────────────")

# Populate a richer store for search demos
user_store = InMemoryStore()
users_data = [
    ("alice",  {"name": "Alice",  "plan": "enterprise", "country": "US", "role": "admin"}),
    ("bob",    {"name": "Bob",    "plan": "pro",        "country": "UK", "role": "user"}),
    ("carol",  {"name": "Carol",  "plan": "enterprise", "country": "US", "role": "user"}),
    ("dave",   {"name": "Dave",   "plan": "free",       "country": "DE", "role": "user"}),
    ("eve",    {"name": "Eve",    "plan": "pro",        "country": "US", "role": "admin"}),
]
for uid, data in users_data:
    user_store.put(("users",), uid, data)

# filter= is a dict of exact-match key-value conditions
enterprise_users = user_store.search(("users",), filter={"plan": "enterprise"})
print(f"\n  filter plan=enterprise: {len(enterprise_users)} users")
for u in enterprise_users:
    print(f"    {u.key}: {u.value['name']}, role={u.value['role']}")

us_admins = user_store.search(("users",), filter={"country": "US", "role": "admin"})
print(f"\n  filter country=US AND role=admin: {len(us_admins)} users")
for u in us_admins:
    print(f"    {u.key}: {u.value['name']}")

# limit= restricts results
limited = user_store.search(("users",), limit=2)
print(f"\n  search with limit=2: {len(limited)} users returned")


# ════════════════════════════════════════════════════════════════════
# PART 3: SEMANTIC SEARCH with IndexConfig
# Provide an embedding function → store indexes values for vector search
# query= in search() finds semantically similar items
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Semantic Search (IndexConfig + embeddings) ────────────")

# Using OpenAI embeddings for real semantic search
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embedding function compatible with IndexConfig."""
    return openai_embeddings.embed_documents(list(texts))


# InMemoryStore with semantic search capability
semantic_store = InMemoryStore(
    index=IndexConfig(
        embed=embed_texts,
        dims=1536,          # text-embedding-3-small dimension
    )
)

# Store memories with natural language content
memories = [
    ("mem_1", {"content": "User prefers concise, bullet-point answers over long paragraphs.", "type": "preference"}),
    ("mem_2", {"content": "User is an expert Python developer with 10 years experience.", "type": "background"}),
    ("mem_3", {"content": "User's current project is a FastAPI microservice for payment processing.", "type": "project"}),
    ("mem_4", {"content": "User finds regex confusing and prefers to avoid it when possible.", "type": "preference"}),
    ("mem_5", {"content": "User works in a team of 5 engineers and follows GitFlow branching.", "type": "background"}),
    ("mem_6", {"content": "User asked about async/await in Python last session.", "type": "history"}),
]

for key, value in memories:
    semantic_store.put(("user_123", "memories"), key, value)

print(f"  Indexed {len(memories)} memories with semantic embeddings")

# Semantic search — find memories relevant to a natural language query
queries = [
    "How does this user like to receive information?",
    "What is the user working on?",
    "What programming technologies does the user know?",
]

for query in queries:
    results = semantic_store.search(("user_123", "memories"), query=query, limit=2)
    print(f"\n  Query: {query!r}")
    for r in results:
        print(f"    [{r.key}] {r.value['content'][:70]}")

# Combine semantic search with content filter
print(f"\n  Semantic + filter (type=preference):")
pref_results = semantic_store.search(
    ("user_123", "memories"),
    query="how user wants answers formatted",
    filter={"type": "preference"},
    limit=2,
)
for r in pref_results:
    print(f"    [{r.key}] {r.value['content'][:80]}")


# ════════════════════════════════════════════════════════════════════
# PART 4: NAMESPACE PATTERNS
# Convention-driven namespace hierarchy for different use cases
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Namespace Patterns ────────────────────────────────────")

ns_store = InMemoryStore()

# Pattern 1: User-scoped memories
ns_store.put(("user", "alice", "profile"),   "basic",   {"name": "Alice", "role": "CTO"})
ns_store.put(("user", "alice", "memories"),  "mem_1",   {"content": "Prefers async tools"})
ns_store.put(("user", "alice", "memories"),  "mem_2",   {"content": "Uses Claude for coding"})

# Pattern 2: Org-scoped shared knowledge
ns_store.put(("org", "acme", "knowledge"),   "style",   {"tone": "formal", "language": "en"})
ns_store.put(("org", "acme", "knowledge"),   "product", {"name": "ACME CRM", "version": "2.4"})

# Pattern 3: Session-scoped context (semi-persistent)
ns_store.put(("session", "sess_99", "ctx"), "topic",   {"subject": "API design", "depth": "expert"})

# Pattern 4: Application-wide defaults
ns_store.put(("app", "defaults"),           "behavior", {"max_tokens": 1000, "language": "en"})

# Retrieve from different namespaces
alice_profile = ns_store.get(("user", "alice", "profile"), "basic")
org_style     = ns_store.get(("org", "acme", "knowledge"), "style")
alice_mems    = ns_store.search(("user", "alice", "memories"), limit=10)

print(f"  User profile: {alice_profile.value['name']}, role={alice_profile.value['role']}")
print(f"  Org style:    tone={org_style.value['tone']!r}")
print(f"  Alice memories: {len(alice_mems)} items")
for m in alice_mems:
    print(f"    {m.value['content']!r}")


# ════════════════════════════════════════════════════════════════════
# PART 5: StoreValue — understanding the return object
# ════════════════════════════════════════════════════════════════════

print("\n── 5. StoreValue Object ─────────────────────────────────────")

store.put(("demo",), "example", {"answer": 42, "label": "meaning of life"})
sv = store.get(("demo",), "example")

print(f"  sv.value:     {sv.value}")
print(f"  sv.key:       {sv.key!r}")
print(f"  sv.namespace: {sv.namespace}")

# StoreValue from search (may have score in semantic stores)
sem_results = semantic_store.search(("user_123", "memories"), query="Python expertise")
if sem_results:
    sv2 = sem_results[0]
    print(f"\n  Semantic search result:")
    print(f"    sv.value:  {sv2.value['content'][:60]}")
    print(f"    sv.key:    {sv2.key!r}")

print("\n" + "═" * 60)
print("Store Basics Summary:")
print("  InMemoryStore()               → in-memory (use DB in prod)")
print("  IndexConfig(embed=fn, dims=N) → enable semantic search")
print("  put(ns, key, dict)            → write a memory")
print("  get(ns, key) → StoreValue     → read by exact key")
print("  search(ns, filter={}, query)  → filter + semantic search")
print("  delete(ns, key)               → remove a memory")
print("  namespace = (user_id, cat)    → hierarchical organization")
print("  StoreValue: .value .key .namespace")
print("═" * 60)
print("\n✅ Store basics demo complete.")
