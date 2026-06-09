# 11_runtime — LangChain Runtime Examples

> **Runtime Provides Dependency Injection for Tools and Middleware**
>
> LangChain's `create_agent` runs on LangGraph's runtime. The `Runtime` object exposes
> per-invocation context, a long-term memory store, a stream writer, and identity
> metadata — all accessible inside tools and middleware hooks without global state.

---

## Files in this folder

| File | Concepts covered |
|------|-----------------|
| [`01_context_schema.py`](01_context_schema.py) | `context_schema=` dataclass, passing `context=` at invoke time, accessing via `ToolRuntime[Context]`, `@dynamic_prompt` from context |
| [`02_tool_runtime.py`](02_tool_runtime.py) | `ToolRuntime` parameter in tools, mixed args + ToolRuntime, `runtime.context`, `runtime.store` (read/write), `runtime.writer` (streaming) |
| [`03_runtime_in_middleware.py`](03_runtime_in_middleware.py) | `@dynamic_prompt`, `@before_model` / `@after_model` with `Runtime[Context]`, `execution_info`, `server_info`, role-based access middleware |
| [`04_execution_and_server_info.py`](04_execution_and_server_info.py) | `execution_info.thread_id/run_id/attempt`, retry detection, audit trail pattern, `server_info` local vs server detection, production auth gate |
| [`05_full_runtime_showcase.py`](05_full_runtime_showcase.py) | Multi-tenant CRM agent with context injection, RBAC, dynamic prompts, audit logging, store memory, and 4 real-world scenarios |
| [`runtime_overview.py`](runtime_overview.py) | Complete runtime overview in one file |

---

## Quick-start

```bash
pip install -r requirements.txt
python 11_runtime/01_context_schema.py
```

---

## What is the Runtime?

The `Runtime` object has 5 components:

| Component | Description | Available |
|-----------|-------------|-----------|
| `runtime.context` | Injected context dataclass instance | Always (if `context_schema` set) |
| `runtime.store` | `BaseStore` for long-term memory | When `store=` is passed to `create_agent` |
| `runtime.writer` | Stream writer for custom updates | When streaming is active |
| `runtime.execution_info` | `thread_id`, `run_id`, `attempt` | Always |
| `runtime.server_info` | `assistant_id`, `graph_id`, `user` | LangGraph Server only (None locally) |

---

## Context Schema Pattern

Define context as a Python `@dataclass`, then inject it at call time:

```python
from dataclasses import dataclass
from langchain.agents import create_agent

@dataclass
class AppContext:
    user_id:   str
    user_name: str
    tenant:    str
    is_premium: bool

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[...],
    context_schema=AppContext,          # ← register schema
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Hello!"}]},
    context=AppContext(                 # ← inject per-call
        user_id="USR-001",
        user_name="Alice",
        tenant="acme",
        is_premium=True,
    ),
)
```

> **Why not use global state?**
> Each `agent.invoke()` call gets its own isolated context — no thread-safety issues,
> no shared mutable state, and tools are fully reusable across users and tenants.

---

## ToolRuntime — Accessing Runtime Inside Tools

`ToolRuntime[Context]` is always the **first parameter** of a tool. It is auto-injected by the runtime and is **invisible to the LLM** (not included in the tool schema):

```python
from langchain.tools import tool, ToolRuntime

@dataclass
class Context:
    user_id: str

@tool
def get_my_data(query: str, runtime: ToolRuntime[Context]) -> str:
    """
    Search the user's private data.

    Args:
        query: The search query. (LLM sees this)
        # runtime is NOT shown to the LLM
    """
    user_id = runtime.context.user_id

    # Read from store
    if runtime.store:
        memory = runtime.store.get(("user_data",), user_id)

    # Write streaming progress
    if runtime.writer:
        runtime.writer({"progress": "Searching...", "user": user_id})

    return f"Results for user {user_id}: ..."
```

---

## Runtime in Middleware

Access `Runtime[Context]` in `@before_model` / `@after_model` decorator hooks:

```python
from langchain.agents import AgentState
from langchain.agents.middleware import before_model, after_model, dynamic_prompt, ModelRequest
from langgraph.runtime import Runtime

@dynamic_prompt
def my_prompt(request: ModelRequest) -> str:
    """Build system prompt from context."""
    ctx = request.runtime.context
    return f"You are an assistant. Address the user as {ctx.user_name}."

@before_model
def log_call(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    print(f"User: {runtime.context.user_name}, thread: {runtime.execution_info.thread_id}")
    return None  # Pass through unchanged
```

---

## Execution Info Reference

```python
@before_model
def audit(state: AgentState, runtime: Runtime) -> dict | None:
    info = runtime.execution_info
    print(f"thread={info.thread_id}")  # Conversation thread ID
    print(f"run={info.run_id}")        # Unique per invoke() call
    print(f"attempt={info.attempt}")   # 0 = first try, >0 = retry
    return None
```

---

## Server Info — Local vs LangGraph Server

```python
@before_model
def auth_gate(state: AgentState, runtime: Runtime) -> dict | None:
    server = runtime.server_info
    if server is None:
        return None  # Local dev — allow all

    # On LangGraph Server
    if server.user is None:
        raise ValueError("Authentication required")

    print(f"assistant={server.assistant_id}")
    print(f"graph={server.graph_id}")
    print(f"user={server.user.identity}")
    return None
```

> **Note:** `runtime.server_info` is `None` during local development and always populated when running on LangGraph Server.

---

## Key Rules

1. **`ToolRuntime` is always the first parameter** — place it before all other tool arguments.
2. **`ToolRuntime` is invisible to the LLM** — it never appears in the tool schema description.
3. **`runtime.store` may be `None`** — always guard with `if runtime.store:` before using it.
4. **`runtime.server_info` is `None` locally** — guard with `if server is not None:` for local/server portability.
5. **`@dynamic_prompt` replaces `system_prompt`** — if both are set, `@dynamic_prompt` takes precedence.
6. **`context` is per-invocation** — pass a fresh `context=YourDataclass(...)` to each `agent.invoke()` call.
