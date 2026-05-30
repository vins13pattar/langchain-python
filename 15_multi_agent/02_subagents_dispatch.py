"""
02_subagents_dispatch.py
=========================
Demonstrates the SINGLE DISPATCH TOOL pattern — a single parameterized
`task` tool routes to any registered subagent by name, enabling
distributed development and dynamic registries.

Concepts covered:
  - Single dispatch `task` tool with agent_name parameter
  - Agent registry as a plain dict
  - System prompt enumeration — static list of agents in prompt
  - Enum constraint — type-safe agent name selection
  - Tool-based discovery — `list_agents` for large/dynamic registries
  - Async background jobs — kick off, check status, get result
  - Context isolation as primary reason for subagents (not just specialization)
"""

import asyncio
import time
import uuid
from enum import Enum
from typing import Annotated
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, InjectedToolCallId
from langchain.messages import ToolMessage
from langgraph.types import Command

load_dotenv()

print("=" * 60)
print("Single Dispatch Subagent Pattern")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# SUBAGENT REGISTRY — developed independently by different teams
# ════════════════════════════════════════════════════════════════════

def make_subagent(name: str, specialty: str, system_prompt: str):
    return create_agent(
        model="openai:gpt-4o-mini",
        tools=[],
        system_prompt=(
            f"You are the {name} specialist — {specialty}. "
            f"{system_prompt} "
            "Include all your results in your final message. "
            "Be concise (2-4 sentences max)."
        ),
    )


SUBAGENTS = {
    "research":    make_subagent("research",  "information gathering",
                                 "Find 3 key facts about the topic."),
    "writer":      make_subagent("writer",    "content creation",
                                 "Write a clear, engaging summary."),
    "code":        make_subagent("code",      "software engineering",
                                 "Provide a concise code example or technical explanation."),
    "data":        make_subagent("data",      "data analysis and statistics",
                                 "Provide data insights or statistical context."),
    "legal":       make_subagent("legal",     "legal and compliance",
                                 "Highlight key legal considerations."),
    "marketing":   make_subagent("marketing", "marketing strategy",
                                 "Provide marketing angles and positioning ideas."),
}

AGENT_DESCRIPTIONS = {
    "research":  "Research and fact-finding for any topic",
    "writer":    "Content creation and copywriting",
    "code":      "Software engineering and technical implementation",
    "data":      "Data analysis, statistics, and metrics",
    "legal":     "Legal considerations and compliance requirements",
    "marketing": "Marketing strategy and positioning",
}


# ════════════════════════════════════════════════════════════════════
# PART 1: SYSTEM PROMPT ENUMERATION
# List available agents in the main agent's system prompt.
# Best for: small, static registries (< 10 agents).
# ════════════════════════════════════════════════════════════════════

print("\n── 1. System Prompt Enumeration ─────────────────────────────")

@tool
def task(agent_name: str, description: str) -> str:
    """Launch an ephemeral subagent for a specific task.

    Available agents:
    - research: Research and fact-finding
    - writer: Content creation and copywriting
    - code: Software engineering and technical implementation
    - data: Data analysis, statistics, and metrics
    - legal: Legal considerations and compliance
    - marketing: Marketing strategy and positioning
    """
    agent = SUBAGENTS.get(agent_name)
    if not agent:
        return f"Agent '{agent_name}' not found. Available: {list(SUBAGENTS.keys())}"

    print(f"  [Dispatch] → {agent_name}: {description[:60]}")
    result = agent.invoke({
        "messages": [{"role": "user", "content": description}]
    })
    content = result["messages"][-1].content
    print(f"  [Dispatch] ← {agent_name}: {content[:60]}")
    return content


main_agent_prompt = create_agent(
    model="openai:gpt-4o-mini",
    tools=[task],
    system_prompt=(
        "You coordinate specialized sub-agents. "
        "Delegate work using the task tool. "
        "Available agents: research (fact-finding), writer (content), "
        "code (technical), data (analytics), legal (compliance), marketing (strategy). "
        "Use multiple agents when appropriate."
    ),
)

result1 = main_agent_prompt.invoke({
    "messages": [{"role": "user", "content":
        "I want to launch a Python-based SaaS product. What should I know?"}]
})
print(f"\nResult: {result1['messages'][-1].content[:300]}")


# ════════════════════════════════════════════════════════════════════
# PART 2: ENUM CONSTRAINT
# Type-safe agent name using Python Enum.
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Enum Constraint (type-safe agent names) ────────────────")

class AgentName(str, Enum):
    RESEARCH  = "research"
    WRITER    = "writer"
    CODE      = "code"
    DATA      = "data"
    LEGAL     = "legal"
    MARKETING = "marketing"


@tool
def task_typed(agent_name: AgentName, description: str) -> str:
    """Launch an ephemeral subagent. The agent_name must be one of the enum values."""
    agent = SUBAGENTS[agent_name.value]
    print(f"  [Dispatch] → {agent_name.value}: {description[:60]}")
    result = agent.invoke({
        "messages": [{"role": "user", "content": description}]
    })
    content = result["messages"][-1].content
    print(f"  [Dispatch] ← {agent_name.value}: {content[:60]}")
    return content


main_agent_enum = create_agent(
    model="openai:gpt-4o-mini",
    tools=[task_typed],
    system_prompt=(
        "You coordinate specialized sub-agents via the task_typed tool. "
        "The agent_name must be one of the AgentName enum values. "
        "Always delegate domain work to the appropriate specialist."
    ),
)

result2 = main_agent_enum.invoke({
    "messages": [{"role": "user", "content":
        "Give me a quick Python code snippet for reading environment variables."}]
})
print(f"\nResult: {result2['messages'][-1].content[:200]}")


# ════════════════════════════════════════════════════════════════════
# PART 3: TOOL-BASED DISCOVERY
# Dynamic agent registry — main agent calls list_agents to discover
# available agents, then uses task to invoke them.
# Best for: large or frequently changing registries.
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Tool-Based Discovery (dynamic registry) ───────────────")

@tool
def list_agents(query: str = "") -> str:
    """List available subagents, optionally filtered by a keyword query."""
    matches = {
        name: desc for name, desc in AGENT_DESCRIPTIONS.items()
        if not query or query.lower() in name.lower() or query.lower() in desc.lower()
    }
    if not matches:
        return f"No agents found matching '{query}'. All: {list(AGENT_DESCRIPTIONS.keys())}"
    lines = [f"- {name}: {desc}" for name, desc in matches.items()]
    return "Available agents:\n" + "\n".join(lines)


main_agent_discovery = create_agent(
    model="openai:gpt-4o-mini",
    tools=[list_agents, task],
    system_prompt=(
        "You coordinate specialized sub-agents. "
        "Use list_agents to discover which agents are available before delegating work. "
        "Then use task to invoke the appropriate agent."
    ),
)

result3 = main_agent_discovery.invoke({
    "messages": [{"role": "user", "content":
        "I need help with data and marketing for a new analytics feature."}]
})
print(f"\nResult: {result3['messages'][-1].content[:300]}")


# ════════════════════════════════════════════════════════════════════
# PART 4: MULTI-AGENT PARALLEL DISPATCH
# Dispatch multiple agents in parallel for independent tasks.
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Parallel Multi-Agent Dispatch ─────────────────────────")
print("  (Supervisor invokes research + code + marketing in one turn)")

main_agent_parallel = create_agent(
    model="openai:gpt-4o-mini",
    tools=[task],
    system_prompt=(
        "You coordinate specialists. For multi-domain questions, "
        "invoke multiple agents simultaneously using separate task calls. "
        "Available agents: research, writer, code, data, legal, marketing."
    ),
)

result4 = main_agent_parallel.invoke({
    "messages": [{"role": "user", "content":
        "I'm building a developer tool. Give me research on the market, "
        "a code example, and marketing positioning ideas."}]
})
print(f"\nParallel result: {result4['messages'][-1].content[:400]}")


# ════════════════════════════════════════════════════════════════════
# PART 5: ASYNC BACKGROUND JOBS
# For long-running tasks — start job, check status, get result.
# Main agent stays responsive while work happens in background.
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Async Background Jobs ─────────────────────────────────")

# Simplified in-process job store (real app: Redis, DB)
_job_store: dict = {}


@tool
def start_background_job(agent_name: str, task_description: str) -> str:
    """Start a long-running subagent task in the background. Returns a job_id."""
    job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"
    _job_store[job_id] = {"status": "running", "result": None, "agent": agent_name}
    print(f"  [BackgroundJob] Started {job_id} → {agent_name}")

    # Simulate async work (in real app: use threading, celery, asyncio, etc.)
    agent = SUBAGENTS.get(agent_name, SUBAGENTS["research"])
    result = agent.invoke({"messages": [{"role": "user", "content": task_description}]})
    _job_store[job_id]["result"] = result["messages"][-1].content
    _job_store[job_id]["status"] = "completed"
    print(f"  [BackgroundJob] {job_id} completed")

    return f"Job started: {job_id}. Check status with check_job_status."


@tool
def check_job_status(job_id: str) -> str:
    """Check the status of a background job."""
    job = _job_store.get(job_id)
    if not job:
        return f"Job '{job_id}' not found."
    return f"Job {job_id}: status={job['status']}, agent={job['agent']}"


@tool
def get_job_result(job_id: str) -> str:
    """Retrieve the result of a completed background job."""
    job = _job_store.get(job_id)
    if not job:
        return f"Job '{job_id}' not found."
    if job["status"] != "completed":
        return f"Job {job_id} not yet complete. Status: {job['status']}"
    return f"Job {job_id} result:\n{job['result']}"


async_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[start_background_job, check_job_status, get_job_result],
    system_prompt=(
        "You coordinate background research tasks. "
        "Start tasks with start_background_job, check with check_job_status, "
        "and retrieve results with get_job_result. "
        "Available agents: research, writer, code, data, legal, marketing."
    ),
)

r_start = async_agent.invoke({
    "messages": [{"role": "user", "content":
        "Start a background research job on 'blockchain in supply chain'."}]
})
print(f"\nStart result: {r_start['messages'][-1].content[:150]}")

r_result = async_agent.invoke({
    "messages": [{"role": "user", "content":
        f"Get the result of the job from: {r_start['messages'][-1].content}"}]
})
print(f"Get result: {r_result['messages'][-1].content[:200]}")

print("\n" + "═" * 60)
print("Single Dispatch Subagent Summary:")
print("  task(agent_name, description) → invoke any agent by name")
print("  System prompt enumeration     → static registry in prompt")
print("  Enum constraint               → type-safe AgentName enum")
print("  Tool-based discovery          → list_agents → task flow")
print("  Background jobs               → start → check → get result")
print("  Parallel dispatch             → multiple task calls at once")
print("═" * 60)
print("\n✅ Single dispatch subagent demo complete.")
