# 17_longterm_memory — Long-Term Memory

> **Long-term memory lets agents store and recall information across different conversations and sessions.**
> Unlike short-term memory (scoped to a thread), long-term memory persists across ALL threads
> and can be recalled at any time via semantic search.

---

## Files in this folder

| File | Concepts covered |
|------|-----------------|
| [`01_store_basics.py`](01_store_basics.py) | `InMemoryStore`, `IndexConfig`, `put/get/delete/search`, namespace patterns, `StoreValue` |
| [`02_read_memory_in_tools.py`](02_read_memory_in_tools.py) | `ToolRuntime[Context]`, `context_schema`, `runtime.store.get()`, multi-namespace reads |
| [`03_write_memory_from_tools.py`](03_write_memory_from_tools.py) | `TypedDict` schemas, `save_user_info`, episodic events, procedural rules, delete |
| [`04_memory_types.py`](04_memory_types.py) | Semantic, episodic, procedural memory, extraction LLM, cross-memory synthesis |
| [`05_full_longterm_memory_showcase.py`](05_full_longterm_memory_showcase.py) | Personal AI — learn from session 1, recall in session 2, multi-user isolation |

---

## Quick-start

```bash
python 17_longterm_memory/01_store_basics.py
python 17_longterm_memory/02_read_memory_in_tools.py
python 17_longterm_memory/05_full_longterm_memory_showcase.py
```

---

## Core Concepts

### InMemoryStore — the storage layer

Long-term memory is stored in a **LangGraph Store**. Each item is a JSON document organized by:
- **namespace** — a tuple acting like a folder path: `(user_id, "semantic")`
- **key** — a string identifier within the namespace: `"profile"`, `"evt_abc123"`
- **value** — any JSON-serializable dict

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import IndexConfig

# With semantic search (pass an embedding function)
store = InMemoryStore(
    index=IndexConfig(embed=my_embed_fn, dims=1536)
)

# CRUD
store.put(("user_123", "memories"), "profile", {"name": "Alice", "role": "developer"})
item    = store.get(("user_123", "memories"), "profile")   # → StoreValue
results = store.search(("user_123", "memories"), query="user skills", limit=5)
store.delete(("user_123", "memories"), "profile")

# StoreValue fields
print(item.value)      # {"name": "Alice", "role": "developer"}
print(item.key)        # "profile"
print(item.namespace)  # ("user_123", "memories")
```

---

### Reading from Store in Tools

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str           # injected per-request

store = InMemoryStore()
store.put(("users",), "user_123", {"name": "John", "language": "English"})

@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """Look up current user's profile."""
    assert runtime.store is not None
    user_id   = runtime.context.user_id
    user_info = runtime.store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_user_info],
    store=store,                   # ← pass the store
    context_schema=Context,        # ← define the context type
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Who am I?"}]},
    context=Context(user_id="user_123"),   # ← inject per request
)
```

---

### Writing to Store from Tools

```python
from typing_extensions import TypedDict

class UserInfo(TypedDict):
    name: str

@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """Save user info."""
    assert runtime.store is not None
    runtime.store.put(("users",), runtime.context.user_id, dict(user_info))
    return "Saved."

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[save_user_info],
    store=store,
    context_schema=Context,
)

agent.invoke(
    {"messages": [{"role": "user", "content": "My name is John Smith"}]},
    context=Context(user_id="user_123"),
)

# Verify
item = store.get(("users",), "user_123")
# → {"name": "John Smith"}
```

---

## The Three Memory Types

| Type | What it stores | Key strategy | Namespace |
|------|---------------|---|---|
| **Semantic** | Facts about the user (profile, skills, preferences) | Descriptive string: `"profile"`, `"skills"` | `(uid, "semantic")` |
| **Episodic** | Past events and interactions | Unique UUID per event: `"evt_abc123"` | `(uid, "episodes")` |
| **Procedural** | Rules about HOW the agent should behave | Unique UUID per rule: `"rule_xyz"` | `(uid, "rules")` |

### Memory Read Pattern (load_memory tool)

```python
@tool
def load_memory(query: str, runtime: ToolRuntime[Context]) -> str:
    """Load relevant memory at the start of every conversation."""
    uid = runtime.context.user_id

    sem   = runtime.store.search((uid, "semantic"),  query=query, limit=5)
    epi   = runtime.store.search((uid, "episodes"),  query=query, limit=3)
    rules = runtime.store.search((uid, "rules"),     limit=20)

    parts = []
    if sem:
        parts.append("Facts: " + " | ".join(r.value["content"] for r in sem))
    if epi:
        parts.append("History: " + " | ".join(r.value["summary"] for r in epi))
    if rules:
        parts.append("Rules: " + " | ".join(r.value["rule"] for r in rules))
    return "\n".join(parts) or "No prior memory."
```

---

## Namespace Conventions

```python
# User-scoped (most common)
("user_123", "semantic")     # factual knowledge about user
("user_123", "episodes")     # interaction history
("user_123", "rules")        # behavioral rules

# Multi-tenant: org + user
("org", "acme", "config")    # org-level shared config
("org", "acme", "users")     # org-scoped user roster

# Session-scoped (semi-persistent)
("session", "sess_99", "ctx")  # session-level context
```

---

## Semantic Search

```python
from langgraph.store.base import IndexConfig
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def embed_fn(texts):
    return embeddings.embed_documents(list(texts))

store = InMemoryStore(index=IndexConfig(embed=embed_fn, dims=1536))

# store.search() with query= performs vector similarity search
results = store.search(
    ("user_123", "memories"),
    query="Python async programming",   # semantic query
    filter={"category": "skills"},      # optional exact filter
    limit=3,
)
```

---

## Short-term vs Long-term Memory

| | Short-term (MemorySaver) | Long-term (Store) |
|---|---|---|
| **Scope** | Single thread (session) | All threads, all sessions |
| **Access** | Automatic (checkpoint) | Via `runtime.store` in tools |
| **Search** | Not searchable | Filter + semantic search |
| **Lifetime** | Until thread is cleared | Persistent (until deleted) |
| **Use case** | Conversation context | Cross-session personalization |

```python
# Use BOTH together
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=tools,
    checkpointer=MemorySaver(),   # short-term: current session
    store=store,                   # long-term: across sessions
    context_schema=Context,
)
```
