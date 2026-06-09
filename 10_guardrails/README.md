# 10_guardrails — LangChain Guardrails Examples

> **Guardrails Validate, Filter, and Protect Every Stage of Agent Execution**
>
> Guardrails help you build safe, compliant AI applications by intercepting content at key
> points in your agent's execution. They can detect sensitive information, enforce content
> policies, validate outputs, and prevent unsafe behaviors before they cause problems.

Common use cases:
- Preventing PII leakage
- Detecting and blocking prompt injection attacks
- Blocking inappropriate or harmful content
- Enforcing business rules and compliance requirements
- Validating output quality and accuracy

---

## Files in this folder

| File | Concepts covered |
|------|-----------------|
| [`01_pii_middleware.py`](01_pii_middleware.py) | `PIIMiddleware` with `redact` / `mask` / `hash` / `block` strategies, custom regex detector, `apply_to_input` / `apply_to_output` flags, stacking multiple PII instances |
| [`02_deterministic_guardrails.py`](02_deterministic_guardrails.py) | `AgentMiddleware.before_agent` class syntax, `@before_agent` decorator, keyword filter, rate limiter, input length validation, combined stack |
| [`03_model_based_guardrails.py`](03_model_based_guardrails.py) | `AgentMiddleware.after_agent` class syntax, `@after_agent` decorator, LLM-as-judge safety classifier, topic relevance check, output quality gate |
| [`04_hitl_as_guardrail.py`](04_hitl_as_guardrail.py) | `HumanInTheLoopMiddleware` for financial / database / email operations, full approve / edit / reject lifecycle, HITL + deterministic pre-filter combined |
| [`05_full_guardrails_showcase.py`](05_full_guardrails_showcase.py) | Financial advisory agent with 7-layer guardrail stack across 5 real-world scenarios |
| [`guardrails_overview.py`](guardrails_overview.py) | Complete guardrails overview in one file |

---

## Quick-start

```bash
pip install -r requirements.txt
python 10_guardrails/01_pii_middleware.py
```

---

## Two Approaches to Guardrails

| Approach | How | Speed | Cost | Best for |
|----------|-----|-------|------|---------|
| **Deterministic** | Regex, keywords, length checks | ⚡ Fast | 💚 Free | Known patterns, high-volume blocking |
| **Model-based** | Secondary LLM evaluates content | 🐢 Slower | 💛 Tokens | Subtle violations, nuanced safety |

> **Best practice:** Put deterministic guardrails **first** (cheap, fast) and model-based guardrails **last** (expensive but thorough).

---

## When Hooks Fire

```
User Input
    │
    ▼
before_agent()    ← fires ONCE — input validation, PII, keyword filters
    │
    ▼
[Agent loop — model calls, tool calls]
    │
    ▼
after_agent()     ← fires ONCE — output safety, quality checks, PII scrub
    │
    ▼
Final Result
```

Middleware placed in the list fires **top → bottom** for `before_*` hooks and **bottom → top** for `after_*` hooks.

---

## Built-in Guardrails

### PIIMiddleware

Detect and handle Personally Identifiable Information:

```python
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[...],
    middleware=[
        # Redact email addresses → [REDACTED_EMAIL]
        PIIMiddleware("email", strategy="redact", apply_to_input=True),

        # Mask credit cards → ****-****-****-1234
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),

        # Block custom API key pattern — raise exception
        PIIMiddleware("api_key",
                      detector=r"sk-[a-zA-Z0-9]{32}",
                      strategy="block",
                      apply_to_input=True),
    ],
)
```

**PII Strategies:**

| Strategy | Output | Use when |
|----------|--------|----------|
| `redact` | `[REDACTED_EMAIL]` | Compliance logging, audit trails |
| `mask`   | `****-****-****-1234` | Showing partial info to users |
| `hash`   | `a8f5f167...` | Consistent pseudonymization |
| `block`  | Raises exception | Zero-tolerance policies |

**Built-in PII types:** `email`, `credit_card`, `ip`, `mac_address`, `url`

**Configuration flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `apply_to_input` | Check user messages before model call | `True` |
| `apply_to_output` | Check AI messages after model call | `False` |
| `apply_to_tool_results` | Check tool result messages | `False` |

---

### HumanInTheLoopMiddleware

Require human approval before executing sensitive operations:

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[send_email, delete_db, search],
    checkpointer=InMemorySaver(),           # Required
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={
            "send_email": True,             # Always interrupt
            "delete_db":  True,             # Always interrupt
            "search":     False,            # Safe — never interrupt
        })
    ],
)

config = {"configurable": {"thread_id": "session-1"}}
result = agent.invoke({"messages": [...]}, config=config)

if "__interrupt__" in result:
    agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)
```

---

## Custom Guardrails

### Before Agent (input protection)

```python
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config, before_agent
from langgraph.runtime import Runtime
from typing import Any

# Class syntax
class MyInputGuardrail(AgentMiddleware):

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        first = state["messages"][0] if state["messages"] else None
        if first and "forbidden" in first.content.lower():
            return {
                "messages": [{"role": "assistant", "content": "Request blocked."}],
                "jump_to": "end",
            }
        return None  # Pass through

# Decorator syntax
@before_agent(can_jump_to=["end"])
def my_input_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    # ... same logic ...
    return None
```

### After Agent (output protection)

```python
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config, after_agent
from langchain.messages import AIMessage
from typing import Any

# Class syntax
class MyOutputGuardrail(AgentMiddleware):

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last = state["messages"][-1] if state["messages"] else None
        if isinstance(last, AIMessage) and "unsafe_content" in last.content:
            safe = AIMessage(content="I cannot provide that response.")
            return {**state, "messages": state["messages"][:-1] + [safe]}
        return None  # Pass through

# Decorator syntax
@after_agent(can_jump_to=["end"])
def my_output_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    # ... same logic ...
    return None
```

---

## Layered Guardrail Stack (Best Practice)

```python
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        # Layer 1: Deterministic input filter (before agent) — fast & cheap
        ContentFilterMiddleware(banned_keywords=["hack", "exploit"]),

        # Layer 2: PII protection on input (before model call)
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),

        # Layer 3: Human approval for sensitive tools (during loop)
        HumanInTheLoopMiddleware(interrupt_on={"send_email_tool": True}),

        # Layer 4: PII scrub on output (after model call)
        PIIMiddleware("email", strategy="redact",
                      apply_to_input=False, apply_to_output=True),

        # Layer 5: Model-based safety check (after agent) — thorough but costly
        SafetyGuardrailMiddleware(),
    ],
)
```

---

## Key Rules

1. **Order matters** — Deterministic guardrails (cheap, fast) go first; model-based (slow, expensive) go last.
2. **Return `None` to pass through** — All hooks must return `None` to let execution continue or a dict with `jump_to: "end"` to terminate.
3. **`hook_config(can_jump_to=["end"])` is required** to use `jump_to` in your returned dict — without it, the jump is silently ignored.
4. **HITL requires a checkpointer** — `HumanInTheLoopMiddleware` will not work without `checkpointer=InMemorySaver()` (or equivalent) and a `thread_id`.
5. **`apply_to_output` needs LangChain ≥ 1.3.2** — Streaming output redaction via `PIIMiddleware` requires this minimum version.
