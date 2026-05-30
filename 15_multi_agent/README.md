# 15_multi_agent — Multi-Agent Systems

> **Multi-agent systems coordinate specialized components to tackle complex workflows.**
>
> Not every complex task needs multiple agents — a single agent with the right tools
> can often achieve similar results. Use multi-agent patterns when you need:
> **context isolation**, **distributed development**, or **parallelization**.

---

## Files in this folder

| File | Pattern | Concepts covered |
|------|---------|-----------------|
| [`01_subagents.py`](01_subagents.py) | Subagents | Tool-per-agent, `ToolRuntime`, `Command`+`InjectedToolCallId`, parallel calls |
| [`02_subagents_dispatch.py`](02_subagents_dispatch.py) | Subagents | Single dispatch `task` tool, enum constraint, tool-based discovery, async jobs |
| [`03_handoffs.py`](03_handoffs.py) | Handoffs | `current_step` state, `Command` transitions, `@dynamic_prompt`, `@wrap_model_call` |
| [`04_skills.py`](04_skills.py) | Skills | On-demand skill loading, stateful reuse, Store cache, dynamic `load_skill` |
| [`05_router.py`](05_router.py) | Router | LLM structured output, keyword routing, fan-out+merge, async parallel, nested |
| [`06_full_multi_agent_showcase.py`](06_full_multi_agent_showcase.py) | All | Enterprise Assistant: Router → Skills / Subagents / Handoffs |

---

## Quick-start

```bash
python 15_multi_agent/01_subagents.py
python 15_multi_agent/03_handoffs.py
python 15_multi_agent/06_full_multi_agent_showcase.py
```

---

## The Four Patterns

### 🤖 Subagents — Centralized Supervisor

A main agent coordinates specialists as tools. All routing passes through the supervisor.

```python
from langchain.agents import create_agent
from langchain.tools import tool

# Create specialist
researcher = create_agent(model="openai:gpt-4o-mini", tools=[...],
                          system_prompt="You are a research specialist.")

# Wrap as a tool
@tool("research", description="Research a topic and return findings")
def call_researcher(topic: str) -> str:
    result = researcher.invoke({"messages": [{"role": "user", "content": topic}]})
    return result["messages"][-1].content

# Supervisor coordinates
supervisor = create_agent(model="openai:gpt-4o-mini", tools=[call_researcher])
```

**Best for:** Multiple distinct domains, centralized control, parallel execution.

---

### 🔄 Single Dispatch Tool

One `task` tool routes to any subagent by name — better for distributed teams.

```python
from enum import Enum

class AgentName(str, Enum):
    RESEARCH = "research"
    WRITER   = "writer"

@tool
def task(agent_name: AgentName, description: str) -> str:
    """Launch an ephemeral subagent for a task."""
    agent = REGISTRY[agent_name.value]
    result = agent.invoke({"messages": [{"role": "user", "content": description}]})
    return result["messages"][-1].content
```

---

### ↔️ Handoffs — State-Driven Behavior

Behavior changes dynamically based on a state variable. Tools return `Command` to transition.

```python
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.tools import tool, InjectedToolCallId
from langchain.messages import ToolMessage
from langgraph.types import Command
from typing import Annotated

class MyState(AgentState):
    current_step: str   # drives behavior

@tool
def advance_to_next(
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Transition to the next step."""
    return Command(update={
        "current_step": "next_step",
        "messages": [ToolMessage(content="Advanced.", tool_call_id=tool_call_id)]
    })

@dynamic_prompt
def step_prompt(request: ModelRequest) -> str:
    step = request.runtime.state.get("current_step", "start")
    return PROMPTS[step]

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[advance_to_next],
    middleware=[step_prompt],
    checkpointer=MemorySaver(),
)
```

**Best for:** Sequential constraints, multi-stage conversational flows, customer support.

---

### 🎯 Skills — On-Demand Context Loading

A single agent loads specialized knowledge via tools when needed.

```python
@tool
def load_skill(skill_name: str) -> str:
    """Load domain expertise into agent context."""
    return SKILLS[skill_name]["context"]   # becomes ToolMessage in history

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[load_skill],
    checkpointer=MemorySaver(),   # persists loaded skills across turns
)
```

**Best for:** Conversational queries, repeat requests (skill reused from history), narrow domains.

---

### 🔀 Router — Dedicated Classification Step

A routing step classifies input and dispatches to the right specialist.

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class RouteDecision:
    domain: Literal["coding", "legal", "finance", "general"]

router_llm = llm.with_structured_output(RouteDecision)

def route_and_invoke(query: str) -> str:
    decision = router_llm.invoke(f"Classify: {query}")
    return specialists[decision.domain].invoke({"messages": [{"role": "user", "content": query}]})["messages"][-1].content
```

**Best for:** Many independent specialist agents, parallel execution, clear domain boundaries.

---

## Choosing a Pattern

| Optimize for | Subagents | Handoffs | Skills | Router |
|---|:---:|:---:|:---:|:---:|
| Single requests | | ✅ | ✅ | ✅ |
| Repeat requests | | ✅ | ✅ | |
| Parallel execution | ✅ | | | ✅ |
| Large-context domains | ✅ | | | ✅ |
| Simple focused tasks | | | ✅ | |
| Sequential constraints | | ✅ | | |
| Direct user interaction | | ✅ | ✅ | ✅ |
| Distributed teams | ✅ | | ✅ | |

---

## Performance Comparison

| Pattern | One-shot calls | Repeat calls | Multi-domain (3) |
|---------|:-:|:-:|:-:|
| Subagents | 4 | 8 (4+4) | 5, ~9K tokens |
| Handoffs | 3 | 5 (3+2) | 7+, ~14K tokens |
| Skills | 3 | 5 (3+2) | 3, ~15K tokens |
| Router | 3 | 6 (3+3) | 5, ~9K tokens |

**Key insight:** Subagents and Router win on multi-domain token efficiency (context isolation).
Handoffs and Skills win on repeat requests (skip routing/reloading). You can mix patterns!

---

## Custom Subagent Inputs/Outputs

```python
from langchain.agents import AgentState
from langchain.tools import ToolRuntime, InjectedToolCallId
from langchain.messages import ToolMessage
from langgraph.types import Command
from typing import Annotated

class MyState(AgentState):
    extra_key: str

# Custom INPUT — inject state into subagent
@tool("my_subagent", description="...")
def call_subagent_with_input(query: str, runtime: ToolRuntime[None, MyState]) -> str:
    state  = runtime.state
    result = subagent.invoke({"messages": [{"role": "user", "content": query}],
                               "extra_key": state.get("extra_key", "")})
    return result["messages"][-1].content

# Custom OUTPUT — update supervisor state via Command
@tool("my_subagent_cmd", description="...")
def call_subagent_with_output(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return Command(update={
        "extra_key": result["extra_key"],
        "messages":  [ToolMessage(content=result["messages"][-1].content,
                                   tool_call_id=tool_call_id)]
    })
```

---

## Key Rules

1. **Subagents are stateless by design** — fresh context per invocation.
2. **Handoffs require `MemorySaver`** — state must persist across turns.
3. **Skills use the same agent** — context accumulates in conversation history.
4. **Router is stateless** — classification step, then fresh agent per request.
5. **You can mix patterns** — a subagents supervisor can invoke router agents; skills can trigger subagents.
6. **Context is everything** — the quality of your multi-agent system depends on ensuring each agent gets the right data.
