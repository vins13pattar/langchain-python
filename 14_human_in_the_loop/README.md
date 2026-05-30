# 14_human_in_the_loop — Human-in-the-Loop (HITL)

> **Human-in-the-Loop lets you add human oversight to any agent tool call.**
>
> When a model proposes a dangerous or sensitive action, HITL middleware
> pauses execution, persists the graph state, and waits for a human decision.
> Execution resumes exactly where it stopped — no state is lost.

---

## Files in this folder

| File | Concepts covered |
|------|-----------------|
| [`01_hitl_basics.py`](01_hitl_basics.py) | `HumanInTheLoopMiddleware`, `interrupt_on`, `version="v2"`, approve, reject, auto-approve |
| [`02_decision_types.py`](02_decision_types.py) | All 4 types: approve, edit (args + tool swap), reject, respond |
| [`03_multiple_decisions.py`](03_multiple_decisions.py) | Multiple simultaneous interrupts, mixed decisions, sequential rounds |
| [`04_hitl_streaming.py`](04_hitl_streaming.py) | `stream()` with `stream_mode=["updates","messages"]`, interrupt detection in stream |
| [`05_full_hitl_showcase.py`](05_full_hitl_showcase.py) | Secure Financial Operations Agent — risk-tiered policies, all 4 decisions, streaming |

---

## Quick-start

```bash
python 14_human_in_the_loop/01_hitl_basics.py
```

---

## Core Pattern

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[write_file, execute_sql, read_data],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file":  True,                                    # all decisions
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},  # no editing
                "read_data":   False,                                   # auto-approve
            },
            description_prefix="Tool execution pending approval",
        )
    ],
    checkpointer=MemorySaver(),   # REQUIRED
)

config = {"configurable": {"thread_id": "my-thread"}}

# Step 1: Run until interrupt
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Delete old log records."}]},
    config=config,
    version="v2",         # returns GraphOutput with .interrupts
)

# Step 2: Inspect
print(result.interrupts[0].value["action_requests"])   # what the agent wants to do
print(result.interrupts[0].value["review_configs"])    # allowed decisions per action

# Step 3: Resume with a decision
agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,        # SAME thread_id to resume
    version="v2",
)
```

---

## The Four Decision Types

| Decision | When to use | Required fields |
|----------|------------|-----------------|
| `approve` | Execute exactly as proposed | `type` only |
| `edit` | Execute with modified args or different tool | `type`, `edited_action.name`, `edited_action.args` |
| `reject` | Block the action + send feedback to agent | `type`, `message` |
| `respond` | Skip tool; human reply IS the result | `type`, `message` |

```python
# approve — execute as-is
{"type": "approve"}

# edit — change args or even the tool name
{"type": "edit", "edited_action": {"name": "archive_data", "args": {"table": "logs", "days_old": 90}}}

# reject — block with feedback
{"type": "reject", "message": "Never delete directly. Archive first."}

# respond — human answer replaces tool execution (for ask_user tools)
{"type": "respond", "message": "Use the 'transactions' table, records older than 60 days."}
```

---

## interrupt_on Configuration

```python
HumanInTheLoopMiddleware(
    interrupt_on={
        "tool_name": True,    # all decisions allowed
        "tool_name": False,   # auto-approve, never interrupt
        "tool_name": {
            "allowed_decisions": ["approve", "reject"],   # restrict decisions
            "description": "Custom message shown in interrupt",
        },
    },
    description_prefix="Default prefix for all interrupts",
)
```

---

## Streaming with HITL

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "..."}]},
    config=config,
    stream_mode=["updates", "messages"],   # both tokens + updates
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        print(token.content, end="", flush=True)   # LLM token

    elif chunk["type"] == "updates":
        if "__interrupt__" in chunk["data"]:
            interrupts = chunk["data"]["__interrupt__"]
            # HITL pause — collect human decision then resume

# Resume (same API as invoke):
for chunk in agent.stream(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
    stream_mode=["updates", "messages"],
    version="v2",
):
    ...
```

---

## Multiple Simultaneous Decisions

```python
# When multiple tools are proposed at once, provide one decision per action
# in the SAME ORDER as they appear in action_requests
agent.invoke(
    Command(resume={
        "decisions": [
            {"type": "approve"},                              # [0] first action
            {"type": "edit", "edited_action": {"name": "send_notification",
                                               "args": {"channel": "#ops", "message": "Updated"}}},
            {"type": "reject", "message": "Not allowed."},   # [2] third action
        ]
    }),
    config=config,
    version="v2",
)
```

---

## Execution Lifecycle

```
1. agent.invoke(user_message, config=config, version="v2")
       ↓
2. LLM generates tool call proposals
       ↓
3. HumanInTheLoopMiddleware (after_model hook) inspects calls
       ↓
4. Matching tools → build HITLRequest → call interrupt()
       ↓
5. Graph state saved to checkpointer → execution PAUSED
       ↓
6. result.interrupts → show to human reviewer
       ↓
7. Human provides decisions
       ↓
8. agent.invoke(Command(resume={decisions: [...]}), config=config)
       ↓
9. Approved/edited → execute tool → continue agent loop
   Rejected         → ToolMessage with rejection feedback
   Responded        → ToolMessage with human reply (tool skipped)
```

---

## Key Rules

1. **Checkpointer is required** — HITL needs `MemorySaver` (dev) or `AsyncPostgresSaver` (prod).
2. **Same `thread_id` to resume** — the `config` must be identical between initial invoke and resume.
3. **`version="v2"` for `GraphOutput`** — gives you `.interrupts` and `.value` attributes.
4. **Decisions list order = action_requests order** — mismatch causes wrong actions.
5. **Edit conservatively** — large arg changes can confuse the agent's reasoning.
6. **`respond` skips tool execution** — the human's message becomes the `ToolMessage` content directly.
