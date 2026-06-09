# 4_tools — LangChain Tools Examples

> Tools extend what agents can do — letting them fetch data, run code,
> query databases, and take actions in the world.
>
> **Tool = Callable function + schema + description**
>
> The model reads the description to decide WHEN to call a tool,
> and uses the schema to know WHAT arguments to pass.

---

## Files in this folder

| File | Concepts covered |
|------|-----------------|
| [`01_basic_tools.py`](01_basic_tools.py) | `@tool`, docstrings, type hints, custom name/description, direct invocation |
| [`02_advanced_schemas.py`](02_advanced_schemas.py) | Pydantic `BaseModel`, `Literal`, `Optional`, nested models, JSON Schema |
| [`03_tool_runtime_context.py`](03_tool_runtime_context.py) | `ToolRuntime`, `runtime.state`, `runtime.context`, `runtime.store` |
| [`04_tool_return_values.py`](04_tool_return_values.py) | Return `str` / `dict` / `Command`, `wrap_tool_call` error handling |
| [`05_dynamic_tool_selection.py`](05_dynamic_tool_selection.py) | `wrap_model_call`, role-based filtering, state-based auth gates |
| [`06_full_tools_showcase.py`](06_full_tools_showcase.py) | All concepts combined — e-commerce assistant |
| [`tools_overview.py`](tools_overview.py) | Complete tools overview in one file |

---

## Quick-start

```bash
pip install langchain langchain-openai langgraph pydantic python-dotenv
echo "OPENAI_API_KEY=sk-..." > .env
python 01_basic_tools.py
```

---

## Core Concepts

### 1 — Minimal tool

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city.        ← description the model reads

    Args:
        city: City name (e.g. 'London')       ← arg description for the model
    """
    return f"Sunny, 22°C in {city}"
```

**Rules:**
- `@tool` decorator is all you need
- Docstring = tool description (be specific and informative)
- Type hints = input schema (required — no hints = broken tool)
- Use `snake_case` names only (`get_weather` not `"Get Weather"`)

---

### 2 — Rich schema with Pydantic

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class WeatherInput(BaseModel):
    location: str                          = Field(description="City name")
    units:    Literal["celsius", "fahrenheit"] = Field(default="celsius")
    forecast: bool                         = Field(default=False)

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", forecast: bool = False) -> str:
    """Get weather with optional forecast."""
    ...
```

---

### 3 — ToolRuntime (hidden from model)

```python
from langchain.tools import tool, ToolRuntime

@tool
def my_tool(user_input: str, runtime: ToolRuntime) -> str:
    """Does something. runtime is invisible to the model."""
    msgs  = runtime.state["messages"]       # short-term conversation state
    ctx   = runtime.context                 # per-run context (user ID, role)
    store = runtime.store                   # long-term persistent storage
    tid   = runtime.tool_call_id            # this call's unique ID
    return "result"
```

**Key: `runtime` is NEVER shown to the model.**

---

### 4 — Return types

```python
# String → model reads as text
@tool
def get_price(item: str) -> str:
    return f"{item} costs $9.99"

# Dict → model inspects fields
@tool
def get_product(id: str) -> dict:
    return {"id": id, "name": "Widget", "price": 9.99, "stock": 42}

# Command → update agent state
from langgraph.types import Command
from langchain_core.messages import ToolMessage

@tool
def set_language(lang: str, runtime: ToolRuntime) -> Command:
    return Command(update={
        "preferred_language": lang,
        "messages": [ToolMessage(
            content=f"Language set to {lang}",
            tool_call_id=runtime.tool_call_id,
        )],
    })
```

---

### 5 — Error handling

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest

@wrap_tool_call
def handle_errors(request: ToolCallRequest, handler) -> ToolMessage:
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool failed: {e}. Try again with different input.",
            tool_call_id=request.tool_call["id"],
        )

agent = create_agent(..., middleware=[handle_errors])
```

---

### 6 — Dynamic tool filtering

```python
from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def filter_by_role(request, handler):
    role    = request.runtime.context.role
    allowed = {"admin": all_tools, "viewer": read_only_tools}[role]
    filtered = [t for t in request.tools if t.name in allowed]
    return handler(request.override(tools=filtered))

agent = create_agent(..., middleware=[filter_by_role])
```

---

## ToolRuntime Components

| Component | Access via | Purpose |
|-----------|-----------|---------|
| State (short-term) | `runtime.state["messages"]` | Current conversation history + custom fields |
| Context (per-run) | `runtime.context.user_id` | Immutable data passed at invoke() time |
| Store (long-term) | `runtime.store.get(...)` | Persistent memory across sessions |
| Tool call ID | `runtime.tool_call_id` | Link ToolMessage to the correct tool call |
| Stream writer | `runtime.stream_writer(...)` | Emit real-time progress updates |
| Execution info | `runtime.execution_info.thread_id` | Thread/run ID, retry count |

---

## Key Rules

1. **Type hints are mandatory** — they define the tool's input schema.
2. **Docstrings are the model's guide** — be specific about when to use the tool.
3. **Use `snake_case` names** — spaces and special chars break some providers.
4. **`runtime` is invisible to the model** — always inject it last in the signature.
5. **`tool_call_id` is required for Command** — always use `runtime.tool_call_id` when returning a `Command` with `ToolMessage`.
6. **Reserved names**: `config` and `runtime` cannot be used as regular arguments.
