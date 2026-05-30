# 12_context_engineering — LangChain Context Engineering

> **Context Engineering: Providing the Right Information in the Right Format**
>
> When agents fail, it's almost always because the LLM didn't receive the right context —
> not because the model itself is incapable. **Context engineering** is the practice of
> carefully controlling what information and tools the LLM sees at every step of execution.
> It is the number one job of AI engineers building reliable agents.

---

## The Three Context Types

| Context Type | What You Control | Transient or Persistent |
|---|---|---|
| **Model Context** | What goes into model calls (prompt, messages, tools, model, format) | Transient |
| **Tool Context** | What tools can read/write (state, store, runtime context) | Persistent |
| **Life-cycle Context** | What happens between model and tool calls (summarization, logging) | Persistent |

## The Three Data Sources

| Data Source | Scope | Examples |
|---|---|---|
| **Runtime Context** | Invocation-scoped | user_id, API keys, role, jurisdiction, feature flags |
| **State** | Conversation-scoped | current messages, auth status, uploaded files |
| **Store** | Cross-conversation | user preferences, notes, memories, audit logs |

---

## Files in this folder

| File | Concepts covered |
|------|-----------------|
| [`01_model_context_system_prompt.py`](01_model_context_system_prompt.py) | `@dynamic_prompt` from State (conversation length), Store (verbosity preference), Runtime Context (role/env), and combined |
| [`02_model_context_messages.py`](02_model_context_messages.py) | `@wrap_model_call` message injection: file context from State, writing style from Store, compliance rules from Runtime Context |
| [`03_model_context_tools_and_model.py`](03_model_context_tools_and_model.py) | Dynamic tool filtering (state auth, store feature flags, context RBAC) and dynamic model switching by cost tier and conversation length |
| [`04_model_context_response_format.py`](04_model_context_response_format.py) | Dynamic Pydantic response schema from State (stage), Store (verbosity pref), Runtime Context (role/env), plus combined format + tools |
| [`05_tool_context_reads_writes.py`](05_tool_context_reads_writes.py) | Tool reads from `runtime.state`, `runtime.store`, `runtime.context`; writes via `Command(update=...)` and `store.put()` |
| [`06_lifecycle_context.py`](06_lifecycle_context.py) | `SummarizationMiddleware`, persistent state via `before_model`, audit logging to Store, transient vs persistent comparison |
| [`07_full_context_engineering_showcase.py`](07_full_context_engineering_showcase.py) | Smart Legal Research Agent — all 3 context types × all 3 data sources, 3 real-world scenarios |

---

## Quick-start

```bash
pip install -r requirements.txt
python 12_context_engineering/01_model_context_system_prompt.py
```

---

## Model Context

### System Prompt — `@dynamic_prompt`

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def my_prompt(request: ModelRequest) -> str:
    # From State
    msg_count = len(request.messages)   # shortcut for request.state["messages"]

    # From Store
    store = request.runtime.store
    prefs = store.get(("prefs",), "user_id") if store else None

    # From Runtime Context
    role = request.runtime.context.user_role

    return f"You are a {role} assistant. ..."
```

### Messages — `@wrap_model_call`

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def inject_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    # Build extra context message
    extra = {"role": "user", "content": "Additional context..."}
    messages = [*request.messages, extra]

    # TRANSIENT — does NOT modify saved state
    return handler(request.override(messages=messages))
```

### Tools — `request.override(tools=...)`

```python
@wrap_model_call
def filter_tools(request, handler):
    role = request.runtime.context.user_role
    if role == "viewer":
        tools = [t for t in request.tools if t.name.startswith("read_")]
        request = request.override(tools=tools)
    return handler(request)
```

### Model — `request.override(model=...)`

```python
from langchain.chat_models import init_chat_model

@wrap_model_call
def select_model(request, handler):
    tier  = request.runtime.context.cost_tier
    model = init_chat_model("openai:gpt-4o" if tier == "premium" else "openai:gpt-4o-mini")
    return handler(request.override(model=model))
```

### Response Format — `request.override(response_format=...)`

```python
from pydantic import BaseModel, Field

class DetailedResponse(BaseModel):
    answer:     str   = Field(description="Thorough answer.")
    confidence: float = Field(description="Confidence score 0.0–1.0.")

@wrap_model_call
def select_format(request, handler):
    msg_count = len(request.messages)
    schema    = DetailedResponse if msg_count > 5 else SimpleResponse
    return handler(request.override(response_format=schema))
```

---

## Tool Context

### Reads

```python
from langchain.tools import tool, ToolRuntime

@tool
def my_tool(query: str, runtime: ToolRuntime[MyCtx]) -> str:
    """..."""
    # From State
    auth = runtime.state.get("authenticated", False)

    # From Store
    prefs = runtime.store.get(("prefs",), runtime.context.user_id) if runtime.store else None

    # From Runtime Context
    user_id = runtime.context.user_id
    api_key = runtime.context.api_key
    ...
```

### Writes

```python
from langgraph.types import Command

@tool
def set_flag(value: str, runtime: ToolRuntime) -> Command:
    """Write to State."""
    return Command(update={"my_flag": value})

@tool
def save_memory(text: str, runtime: ToolRuntime[MyCtx]) -> str:
    """Write to Store."""
    if runtime.store:
        existing = runtime.store.get(("mem",), runtime.context.user_id)
        data     = existing.value if existing else {}
        data["saved"] = text
        runtime.store.put(("mem",), runtime.context.user_id, data)
    return "Saved."
```

---

## Life-cycle Context

### Auto-summarization

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[...],
    checkpointer=MemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger={"messages": 10},   # condense after 10 messages
            keep={"messages": 3},       # keep last 3 after summary
        )
    ],
)
```

### Persistent State Updates (`before_model`)

```python
from langchain.agents.middleware import before_model
from langchain.agents import AgentState

@before_model
def track_turns(state: AgentState, runtime: Runtime) -> dict | None:
    # Returning a dict PERSISTS the updates to state for all future turns
    return {"turn_count": state.get("turn_count", 0) + 1}
```

### Transient vs Persistent — Key Distinction

| Approach | Modifies What? | Persists? |
|----------|----------------|-----------|
| `wrap_model_call` with `request.override()` | What the LLM sees for ONE call | ❌ No |
| `before_model` / `after_model` returning a `dict` | Agent State | ✅ Yes |
| `runtime.store.put()` in any hook or tool | Long-term Store | ✅ Yes (cross-session) |

---

## Best Practices

1. **Start simple** — static prompt + tools; add dynamics only when reliability suffers.
2. **Deterministic first** — keyword/regex context filtering before LLM-based checks.
3. **Transient for presentation, persistent for truth** — inject hints transiently; write decisions persistently.
4. **Use built-in middleware** — `SummarizationMiddleware`, `PIIMiddleware`, `HumanInTheLoopMiddleware`.
5. **Document your context strategy** — make explicit what each middleware injects and why.
6. **Monitor token usage** — message injection increases prompt size; measure cost impact.
