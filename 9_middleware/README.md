# 9_middleware — LangChain Middleware Examples

> **Middleware Intercepts, Extends, and Safeguards Every Stage of Agent Execution**
>
> Middleware provides a way to tightly control what happens inside the agent loop.
> It is useful for logging, analytics, retries, fallbacks, rate limits, guardrails,
> PII detection, and human-in-the-loop approval workflows — all without modifying
> the core agent logic.

---

## Files in this folder

| File | Concepts covered |
|------|-----------------|
| [`01_built_in_middleware.py`](01_built_in_middleware.py) | `SummarizationMiddleware`, `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`, `ToolRetryMiddleware`, stacking multiple middleware |
| [`02_human_in_the_loop.py`](02_human_in_the_loop.py) | `HumanInTheLoopMiddleware`, approve / edit / reject workflows, per-tool HITL policies, `Command(resume=...)` |
| [`03_custom_middleware.py`](03_custom_middleware.py) | `BaseMiddleware`, `before_agent`, `after_agent`, `before_model`, `after_model`, `wrap_tool_call`, tool-specific middleware |
| [`04_pii_detection_and_guardrails.py`](04_pii_detection_and_guardrails.py) | `PIIDetectionMiddleware`, content guardrail, input validation, output sanitization |
| [`05_agent_loop_middleware.py`](05_agent_loop_middleware.py) | Loop observer, rate-limiter between iterations, early exit strategy, hook firing order |
| [`06_full_middleware_showcase.py`](06_full_middleware_showcase.py) | Production-ready customer support triage agent with 6 stacked middleware layers |
| [`middleware_overview.py`](middleware_overview.py) | Complete middleware overview in one file |

---

## Quick-start

```bash
pip install -r requirements.txt
python 9_middleware/01_built_in_middleware.py
```

---

## How Middleware Works

Middleware is passed as a list to `create_agent`. Each item intercepts the agent
loop at defined hook points. Hooks run in **declaration order** (top → bottom for
`before_*` hooks, bottom → top for `after_*` hooks).

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolRetryMiddleware

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[...],
    middleware=[
        ModelCallLimitMiddleware(max_calls=5),
        ToolRetryMiddleware(max_retries=3),
    ],
)
```

---

## Middleware Hook Lifecycle

```
User Input
    │
    ▼
before_agent()          ← fires ONCE at the start of the entire run
    │
    ▼
┌── AGENT LOOP ─────────────────────────────────────────────────┐
│       │                                                       │
│   before_model()      ← fires before EACH LLM call           │
│       │                                                       │
│   [LLM generates response or tool call]                       │
│       │                                                       │
│   after_model()       ← fires after EACH LLM call            │
│       │                                                       │
│   wrap_tool_call()    ← wraps EACH tool invocation            │
│       │                                                       │
│   [Tool executes, result returned to LLM]                     │
│       │                                                       │
│   (loop continues until agent finishes)                       │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
after_agent()           ← fires ONCE after the loop exits
    │
    ▼
Final Result
```

---

## Built-in Middleware Reference

| Middleware | Import | Key Parameters |
|-----------|--------|----------------|
| `SummarizationMiddleware` | `langchain.agents.middleware` | `model`, `max_tokens` |
| `ModelCallLimitMiddleware` | `langchain.agents.middleware` | `max_calls` |
| `ToolCallLimitMiddleware` | `langchain.agents.middleware` | `limits={"tool_name": N}` |
| `ToolRetryMiddleware` | `langchain.agents.middleware` | `max_retries` |
| `HumanInTheLoopMiddleware` | `langchain.agents.middleware` | `interrupt_on={...}` |
| `PIIDetectionMiddleware` | `langchain.agents.middleware` | `redact`, `raise_on_detect` |

---

## Human-in-the-Loop (HITL) Quick Reference

HITL requires a **checkpointer** and a **`thread_id`** in config:

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[send_email],
    checkpointer=MemorySaver(),          # Required
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
            }
        )
    ],
)

config = {"configurable": {"thread_id": "session-1"}}

# Run until interrupt
result = agent.invoke({"messages": [...]}, config=config)

if "__interrupt__" in result:
    # Human approves
    agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)

    # Human edits arguments
    agent.invoke(Command(resume={
        "decisions": [{"type": "edit", "edited_action": {"name": "send_email", "args": {...}}}]
    }), config=config)

    # Human rejects with feedback
    agent.invoke(Command(resume={
        "decisions": [{"type": "reject", "feedback": "Reason for rejection"}]
    }), config=config)
```

---

## Custom Middleware Quick Reference

```python
from langchain.agents.middleware import BaseMiddleware
from typing import Any, Optional

class MyMiddleware(BaseMiddleware):

    tools = ["specific_tool"]  # Optional: apply only to these tools

    def before_agent(self, state: dict) -> Optional[dict]:
        """Return modified state or None to pass through unchanged."""
        return None

    def after_agent(self, state: dict) -> Optional[dict]:
        return None

    def before_model(self, state: dict, **kwargs) -> Optional[dict]:
        return None

    def after_model(self, state: dict, **kwargs) -> Optional[dict]:
        return None

    def wrap_tool_call(self, tool_call: dict, call_tool, **kwargs) -> Any:
        # Intercept, modify args, or short-circuit
        return call_tool(tool_call, **kwargs)
```

---

## Key Rules

1. **Order matters** — Middleware is applied top-to-bottom. Put security middleware (guardrails, PII) first so they run before any processing occurs.
2. **HITL needs a checkpointer** — `HumanInTheLoopMiddleware` requires `checkpointer=MemorySaver()` and a `thread_id` in the config.
3. **Return `None` to pass through** — All `before_*` and `after_*` hooks return `Optional[dict]`. Returning `None` leaves state unchanged; returning a dict replaces the state.
4. **`wrap_tool_call` must return a value** — Always call `call_tool(tool_call, **kwargs)` or return an error string. Never return `None` from `wrap_tool_call`.
5. **Tool-specific middleware** — Set `tools = ["tool_name"]` as a class attribute on your `BaseMiddleware` subclass to restrict it to specific tools only.
