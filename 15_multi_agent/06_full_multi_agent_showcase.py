"""
06_full_multi_agent_showcase.py
================================
Production-ready showcase: an INTELLIGENT ENTERPRISE ASSISTANT that
combines ALL four multi-agent patterns in a unified system.

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │             Intelligent Enterprise Assistant                     │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │  LAYER 1 — ROUTER (entry point)                                 │
  │    LLM classifies intent: task_type × domains                   │
  │    Simple queries → SKILLS (2-3 calls)                          │
  │    Multi-domain complex → SUBAGENTS (parallel)                  │
  │    Multi-turn guided flows → HANDOFFS (sequential)              │
  │                                                                 │
  │  LAYER 2A — SKILLS (single agent, loaded context)               │
  │    load_skill(domain) → inject expertise → answer               │
  │    Best for: conversational, focused queries                    │
  │                                                                 │
  │  LAYER 2B — SUBAGENTS (parallel specialists)                    │
  │    research + write + review in parallel                        │
  │    Best for: multi-domain, independent tasks                    │
  │                                                                 │
  │  LAYER 2C — HANDOFFS (stateful sequential workflow)             │
  │    intake → classify → resolve → close                          │
  │    Best for: multi-step flows requiring sequential constraints  │
  │                                                                 │
  │  Scenarios:                                                      │
  │    A. Simple query → Skills pattern                             │
  │    B. Complex multi-domain → Subagents (parallel)               │
  │    C. Customer support flow → Handoffs (sequential)             │
  │    D. Cross-pattern: router dispatches to correct layer         │
  └─────────────────────────────────────────────────────────────────┘
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Annotated, Literal
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime, InjectedToolCallId
from langchain.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

load_dotenv()

print("=" * 60)
print("Intelligent Enterprise Assistant — Full Multi-Agent Showcase")
print("=" * 60)

llm = init_chat_model("openai:gpt-4o-mini")


# ════════════════════════════════════════════════════════════════════
# SHARED KNOWLEDGE BASE (Skills content)
# ════════════════════════════════════════════════════════════════════

SKILL_CONTENT = {
    "engineering": "You have deep expertise in software architecture, APIs, microservices, databases, and DevOps best practices.",
    "product":     "You have deep expertise in product strategy, user research, roadmaps, OKRs, and go-to-market planning.",
    "legal":       "You have deep expertise in SaaS contracts, IP law, GDPR compliance, SLAs, and enterprise agreement structures.",
    "finance":     "You have deep expertise in SaaS metrics (MRR, ARR, churn), unit economics, fundraising, and financial modeling.",
    "marketing":   "You have deep expertise in B2B marketing, demand generation, positioning, messaging, and content strategy.",
}


# ════════════════════════════════════════════════════════════════════
# SPECIALIST SUBAGENTS
# ════════════════════════════════════════════════════════════════════

def make_expert(domain: str) -> object:
    return create_agent(
        model="openai:gpt-4o-mini",
        tools=[],
        system_prompt=(
            f"You are an enterprise {domain} expert. "
            f"{SKILL_CONTENT[domain]} "
            "Provide concrete, actionable insights in 2-4 sentences. "
            "Always include your complete answer in your final message."
        ),
    )


EXPERTS = {domain: make_expert(domain) for domain in SKILL_CONTENT}


# ════════════════════════════════════════════════════════════════════
# LAYER 2A — SKILLS AGENT
# ════════════════════════════════════════════════════════════════════

@tool
def load_enterprise_skill(domain: str) -> str:
    """Load enterprise domain expertise into context.
    Available domains: engineering, product, legal, finance, marketing
    """
    if domain not in SKILL_CONTENT:
        return f"Unknown domain. Available: {list(SKILL_CONTENT.keys())}"
    print(f"  [Skill] Loading: {domain}")
    return f"[{domain.upper()} EXPERTISE LOADED]\n{SKILL_CONTENT[domain]}"


skills_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[load_enterprise_skill],
    checkpointer=MemorySaver(),
    system_prompt=(
        "You are an enterprise assistant. Load relevant expertise skill "
        "before answering domain-specific questions. Then provide expert guidance."
    ),
)


# ════════════════════════════════════════════════════════════════════
# LAYER 2B — SUBAGENTS (parallel specialists)
# ════════════════════════════════════════════════════════════════════

@tool("engineering_expert", description="Get software engineering and architecture guidance")
def call_engineering(query: str) -> str:
    result = EXPERTS["engineering"].invoke({"messages": [{"role": "user", "content": query}]})
    print(f"  [Subagent] engineering: {result['messages'][-1].content[:60]}")
    return result["messages"][-1].content


@tool("product_expert", description="Get product strategy and roadmap guidance")
def call_product(query: str) -> str:
    result = EXPERTS["product"].invoke({"messages": [{"role": "user", "content": query}]})
    print(f"  [Subagent] product: {result['messages'][-1].content[:60]}")
    return result["messages"][-1].content


@tool("legal_expert", description="Get legal and compliance guidance")
def call_legal(query: str) -> str:
    result = EXPERTS["legal"].invoke({"messages": [{"role": "user", "content": query}]})
    print(f"  [Subagent] legal: {result['messages'][-1].content[:60]}")
    return result["messages"][-1].content


@tool("finance_expert", description="Get financial analysis and SaaS metrics guidance")
def call_finance(query: str) -> str:
    result = EXPERTS["finance"].invoke({"messages": [{"role": "user", "content": query}]})
    print(f"  [Subagent] finance: {result['messages'][-1].content[:60]}")
    return result["messages"][-1].content


@tool("marketing_expert", description="Get marketing strategy and positioning guidance")
def call_marketing(query: str) -> str:
    result = EXPERTS["marketing"].invoke({"messages": [{"role": "user", "content": query}]})
    print(f"  [Subagent] marketing: {result['messages'][-1].content[:60]}")
    return result["messages"][-1].content


subagents_supervisor = create_agent(
    model="openai:gpt-4o-mini",
    tools=[call_engineering, call_product, call_legal, call_finance, call_marketing],
    system_prompt=(
        "You are an enterprise coordinator with access to domain specialists. "
        "For complex multi-domain questions, invoke all relevant specialists simultaneously. "
        "Synthesize their perspectives into a unified, actionable response."
    ),
)


# ════════════════════════════════════════════════════════════════════
# LAYER 2C — HANDOFFS (sequential support workflow)
# ════════════════════════════════════════════════════════════════════

class SupportState(AgentState):
    current_step:    str
    ticket_id:       str
    issue_category:  str


SUPPORT_PROMPTS = {
    "intake":   "You are collecting the customer's issue. Ask for their account ID and a description of the problem.",
    "classify": "You are classifying the issue. Based on the description, categorize it and route appropriately.",
    "resolve":  "You are the resolution specialist. Provide a concrete solution. Escalate if unsolvable.",
    "close":    "You are closing the ticket. Confirm resolution with the customer and provide ticket reference.",
}


@tool
def record_intake(
    account_id: str,
    issue_description: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Record customer intake details and advance to classification."""
    import uuid
    ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
    return Command(update={
        "ticket_id":    ticket_id,
        "current_step": "classify",
        "messages": [ToolMessage(
            content=f"Intake recorded. Ticket: {ticket_id}. Account: {account_id}.",
            tool_call_id=tool_call_id,
        )]
    })


@tool
def classify_issue(
    category: Literal["technical", "billing", "general"],
    priority: Literal["high", "medium", "low"],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Classify the issue and advance to resolution."""
    return Command(update={
        "issue_category": f"{category}/{priority}",
        "current_step":   "resolve",
        "messages": [ToolMessage(
            content=f"Classified: {category} ({priority} priority). Moving to resolution.",
            tool_call_id=tool_call_id,
        )]
    })


@tool
def close_support_ticket(
    resolution_summary: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Close the ticket with a resolution summary."""
    return Command(update={
        "current_step": "closed",
        "messages": [ToolMessage(
            content=f"Ticket closed. Summary: {resolution_summary}",
            tool_call_id=tool_call_id,
        )]
    })


@dynamic_prompt
def support_step_prompt(request: ModelRequest) -> str:
    state = request.runtime.state
    step  = state.get("current_step", "intake")
    return SUPPORT_PROMPTS.get(step, SUPPORT_PROMPTS["intake"])


handoffs_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[record_intake, classify_issue, close_support_ticket],
    middleware=[support_step_prompt],
    checkpointer=MemorySaver(),
    system_prompt="Enterprise support agent.",
)


# ════════════════════════════════════════════════════════════════════
# LAYER 1 — ROUTER
# Classifies intent and dispatches to the correct pattern layer.
# ════════════════════════════════════════════════════════════════════

@dataclass
class RouterDecision:
    pattern:    Literal["skills", "subagents", "handoffs"]
    domains:    list[str]
    complexity: Literal["simple", "complex"]
    reason:     str


router_llm = llm.with_structured_output(RouterDecision)


def route_request(query: str) -> RouterDecision:
    prompt = (
        "Classify this enterprise assistant request.\n\n"
        "Patterns:\n"
        "- skills:    single domain, conversational, focused (2-3 model calls)\n"
        "- subagents: multi-domain, complex, benefits from parallel specialists\n"
        "- handoffs:  multi-step flow, sequential constraints (support, onboarding)\n\n"
        "Available domains: engineering, product, legal, finance, marketing\n\n"
        f"Request: {query}"
    )
    decision = router_llm.invoke(prompt)
    print(f"\n  [Router] pattern={decision.pattern!r}, domains={decision.domains}, "
          f"complexity={decision.complexity!r}")
    return decision


def enterprise_assistant(query: str, thread_id: str = "default") -> str:
    """Main entry point — routes to the appropriate pattern."""
    decision = route_request(query)

    if decision.pattern == "skills":
        cfg    = {"configurable": {"thread_id": f"skills-{thread_id}"}}
        result = skills_agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config=cfg,
        )
        return result["messages"][-1].content

    elif decision.pattern == "subagents":
        result = subagents_supervisor.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        return result["messages"][-1].content

    elif decision.pattern == "handoffs":
        cfg    = {"configurable": {"thread_id": f"handoffs-{thread_id}"}}
        result = handoffs_agent.invoke(
            {"messages": [{"role": "user", "content": query}],
             "current_step": "intake", "ticket_id": "", "issue_category": ""},
            config=cfg,
        )
        return result["messages"][-1].content

    return "Unable to route request."


# ════════════════════════════════════════════════════════════════════
# SCENARIOS
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO A — Simple Query → Skills Pattern")
print("─" * 60)
r_a = enterprise_assistant(
    "What are the key SaaS metrics I should track in year one?",
    thread_id="scenario-a"
)
print(f"\nResponse: {r_a[:300]}")


print("\n" + "─" * 60)
print("SCENARIO B — Complex Multi-Domain → Subagents Pattern")
print("─" * 60)
r_b = enterprise_assistant(
    "We're launching a B2B SaaS product next quarter. What do we need "
    "to consider across engineering, legal, finance, and marketing?",
    thread_id="scenario-b"
)
print(f"\nResponse: {r_b[:400]}")


print("\n" + "─" * 60)
print("SCENARIO C — Support Flow → Handoffs Pattern")
print("─" * 60)

cfg_c  = {"configurable": {"thread_id": "showcase-support"}}
init_c = {"current_step": "intake", "ticket_id": "", "issue_category": ""}

for turn_msg in [
    "I'm having trouble with the API rate limits on my account.",
    "My account ID is ENT-9821. The 429 errors started after yesterday's release.",
]:
    r_c = handoffs_agent.invoke(
        {**init_c, "messages": [{"role": "user", "content": turn_msg}]},
        config=cfg_c,
    )
    step = r_c.get("current_step", "?")
    print(f"\n  [step={step!r}] User: {turn_msg[:60]}")
    print(f"  Agent: {r_c['messages'][-1].content[:120]}")


print("\n" + "─" * 60)
print("SCENARIO D — Cross-Pattern: Router Selects Best Fit")
print("─" * 60)

cross_queries = [
    ("What is churn rate?", "skills"),
    ("Build a GTM strategy for our new enterprise tier.", "subagents"),
]

for q, expected_pattern in cross_queries:
    decision = route_request(q)
    match    = "✅" if decision.pattern == expected_pattern else "⚠️ "
    print(f"\n  {match} Q: {q}")
    print(f"     Routed to: {decision.pattern!r} (expected: {expected_pattern!r})")
    print(f"     Reason: {decision.reason[:80]}")


print("\n" + "═" * 60)
print("Full Multi-Agent Showcase — Patterns Used:")
print("  ROUTER    — LLM structured output classifies pattern + domains")
print("  SKILLS    — load_enterprise_skill injects domain context")
print("  SUBAGENTS — parallel specialist tool calls, supervisor synthesizes")
print("  HANDOFFS  — current_step drives sequential support workflow")
print("  All patterns share MemorySaver for multi-turn conversations")
print("═" * 60)
print("\n✅ Full multi-agent showcase complete.")
