# 5_short_term_memory — LangChain Short-Term Memory Examples

> Short-term memory lets an agent **remember previous interactions within a single conversation thread**.
>
> Without memory: every `agent.invoke()` is stateless — like talking to a stranger each time.
> With memory: the agent picks up exactly where it left off.

---

## Files in this folder

| File | Concepts covered |
|------|-----------------|
| [`01_checkpointer_basics.py`](01_checkpointer_basics.py) | `MemorySaver`, `thread_id`, multiple isolated threads, unique session IDs |
| [`02_custom_state.py`](02_custom_state.py) | `AgentState` subclassing, custom fields, reading/writing via tools and `Command` |
| [`03_trim_and_delete_messages.py`](03_trim_and_delete_messages.py) | `@before_model`, `@after_model`, `RemoveMessage`, `REMOVE_ALL_MESSAGES`, safe trimming |
| [`04_summarization_and_dynamic_prompt.py`](04_summarization_and_dynamic_prompt.py) | `SummarizationMiddleware`, `@dynamic_prompt`, context/state-aware prompts |
| [`05_full_memory_showcase.py`](05_full_memory_showcase.py) | All concepts combined — personal productivity assistant |
| [`short_term_memory_overview.py`](short_term_memory_overview.py) | Complete short-term memory overview in one file |

---

## Quick-start

```bash
pip install langchain langchain-openai langgraph python-dotenv
echo "OPENAI_API_KEY=sk-..." > .env
python 01_checkpointer_basics.py
```

---

## Core Concepts

### 1 — Checkpointer + thread_id

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[...],
    checkpointer=MemorySaver(),          # ← adds short-term memory
)

config = {"configurable": {"thread_id": "user-session-123"}}

# Turn 1
agent.invoke({"messages": [{"role": "user", "content": "My name is Vinod."}]}, config)

# Turn 2 — agent remembers Turn 1
agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config)
```

**Rules:**
- Same `thread_id` = same conversation history
- Different `thread_id` = completely isolated conversation
- Without `checkpointer` = no memory at all

---

### 2 — Custom State

```python
from langchain.agents import create_agent, AgentState

class MyState(AgentState):       # extend AgentState
    user_name:  str  = ""        # persisted across turns
    query_count: int = 0         # updated by tools via Command

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[...],
    state_schema=MyState,        # pass your schema here
    checkpointer=MemorySaver(),
)

# Pass initial values in the first invoke()
agent.invoke({"messages": [...], "user_name": "Vinod"}, config)
```

---

### 3 — Trim Messages

```python
from langchain.agents.middleware import before_model
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

@before_model
def trim_messages(state, runtime):
    messages = state["messages"]
    if len(messages) <= 6:
        return None                      # nothing to do

    trimmed = [messages[0]] + messages[-4:]   # keep first + last 4
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *trimmed,
        ]
    }

agent = create_agent(..., middleware=[trim_messages], checkpointer=MemorySaver())
```

---

### 4 — Summarise Messages

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger=("messages", 10),    # summarise when > 10 messages
            keep=("messages", 5),        # keep 5 recent after summarising
        )
    ],
    checkpointer=MemorySaver(),
)
```

---

### 5 — Dynamic Prompt

```python
from langchain.agents.middleware import dynamic_prompt

@dynamic_prompt
def my_system_prompt(request) -> str:
    name = request.runtime.context.get("user_name", "User")
    return f"You are a helpful assistant. Address the user as {name}."

agent = create_agent(..., middleware=[my_system_prompt])
```

---

## Memory Strategy Guide

| Strategy | Info loss | Cost | Best for |
|----------|-----------|------|----------|
| **No trimming** | None | Low | Short sessions (< 20 msgs) |
| **Trim** (`@before_model`) | Yes | Very low | Assistants where old context doesn't matter |
| **Delete** (`@after_model`) | Yes | Very low | Cleaning specific messages |
| **Summarise** | Minimal | Medium | Long sessions where history matters |
| **Long-term store** | None | Variable | Cross-session persistence |

---

## Key Rules

1. **Always include a `checkpointer`** — without it, every call is stateless.
2. **Use unique `thread_id`s** — one per user session. Never reuse across users.
3. **`MemorySaver` is lost on restart** — use `PostgresSaver` / `SqliteSaver` in production.
4. **Trimmed lists must stay valid** — don't start with an `AIMessage`, ensure tool calls have matching `ToolMessage`s.
5. **`REMOVE_ALL_MESSAGES` wipes everything** — always add your desired messages after the wipe.
6. **`@before_model` runs before EVERY model call** — including after tool calls.
