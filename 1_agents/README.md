# 1_agents — LangChain Agents Examples

> **Agent = Model + Harness**
>
> A harness is everything around the agent loop: the model, its prompt, its
> tools, and any middleware that shapes its behaviour.

---

## What is an Agent?

An agent is a **model calling tools in a loop** until a given task is complete.

```
User message
    │
    ▼
┌─────────────────────────────────┐
│         create_agent()          │  ← harness
│  ┌──────────────────────────┐  │
│  │  Model (LLM)             │  │  ← decides what to do
│  │  ↓ calls tool(s)         │  │
│  │  Tool 1 / Tool 2 / …     │  │  ← takes action
│  │  ↓ result fed back       │  │
│  │  Model again …           │  │  ← reflects on result
│  └──────────────────────────┘  │
│         (loop until done)       │
└─────────────────────────────────┘
    │
    ▼
Final answer (str or structured object)
```

---

## Files in this folder

| File | Concepts covered |
|------|-----------------|
| [`01_basic_agent.py`](01_basic_agent.py) | `create_agent`, `@tool`, `system_prompt`, `invoke()` |
| [`02_agent_with_memory.py`](02_agent_with_memory.py) | `MemorySaver`, `thread_id`, multi-turn conversations |
| [`03_structured_output.py`](03_structured_output.py) | `response_format`, Pydantic schemas, `structured_response` |
| [`04_streaming.py`](04_streaming.py) | `agent.stream()`, `stream_mode="values"`, real-time output |
| [`05_middleware.py`](05_middleware.py) | `HumanInTheLoopMiddleware`, `ModelRetryMiddleware`, `PIIMiddleware` |
| [`06_context_and_runtime.py`](06_context_and_runtime.py) | `context_schema`, `context=`, `get_runtime()`, RBAC in tools |
| [`07_full_agent_showcase.py`](07_full_agent_showcase.py) | All concepts combined in a production-ready agent |
| [`agents_overview.py`](agents_overview.py) | Complete agents overview in one file |

---

## Quick-start

```bash
# 1. Install dependencies
pip install langchain langchain-openai langgraph pydantic python-dotenv

# 2. Set your API key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Run any example
python 01_basic_agent.py
```

---

## Core Concepts at a Glance

### 1 — Minimal agent

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

agent = create_agent("openai:gpt-4o-mini", tools=[search])
result = agent.invoke({"messages": [{"role": "user", "content": "Search for AI news"}]})
print(result["messages"][-1].content)
```

### 2 — With memory (multi-turn)

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_agent("openai:gpt-4o-mini", tools=[], checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "session-1"}}

agent.invoke({"messages": [{"role": "user", "content": "My name is Vinod"}]}, config=config)
result = agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config=config)
# → "Your name is Vinod"
```

### 3 — Structured output

```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    points: list[str]
    confidence: float

agent = create_agent("openai:gpt-4o-mini", tools=[], response_format=Summary)
result = agent.invoke({"messages": [{"role": "user", "content": "Summarise LangChain"}]})
report: Summary = result["structured_response"]   # fully typed object
```

### 4 — Middleware (fault tolerance, guardrails, HITL)

```python
from langchain.agents.middleware import ModelRetryMiddleware, PIIMiddleware

agent = create_agent(
    "openai:gpt-4o-mini",
    tools=[...],
    middleware=[
        ModelRetryMiddleware(max_retries=3),   # auto-retry on errors
        PIIMiddleware(),                        # scrub PII before LLM sees it
    ],
)
```

---

## Middleware Categories (from the docs)

| Category | Middleware | What it does |
|----------|-----------|--------------|
| Execution | `FilesystemMiddleware` | Gives agent read/write filesystem access |
| Context | `SummarizationMiddleware` | Compresses history before context overflow |
| Context | `MemoryMiddleware` | Loads persistent instructions at startup |
| Planning | `TodoListMiddleware` | Structured task planning |
| Planning | `SubAgentMiddleware` | Delegates to child agents |
| Fault tolerance | `ModelRetryMiddleware` | Retries model on rate-limits / timeouts |
| Fault tolerance | `ToolRetryMiddleware` | Retries tools on transient errors |
| Guardrails | `PIIMiddleware` | Scrubs PII before it reaches the model |
| Steering | `HumanInTheLoopMiddleware` | Pause & await human approval |

---

## Key Rules

1. **Always use `create_agent()`** — `AgentExecutor` is the old pattern.
2. **Access result correctly**: `result["messages"][-1].content` — not `result.content`.
3. **Conversation memory requires both** `checkpointer=` AND `thread_id` in config.
4. **`thread_id`** scopes history; **`context`** carries per-run data. Use both.
5. **Tool docstrings matter** — the model reads them to decide when to call each tool.
