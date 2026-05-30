"""
05_router.py
=============
Demonstrates the ROUTER multi-agent pattern — a dedicated routing
step classifies input and directs it to one or more specialized agents.
Results are synthesized into a combined response.

Concepts covered:
  - LLM-based router using structured output (Literal type)
  - Rule-based router using keyword matching (deterministic, fast)
  - Single-domain routing — dispatch to exactly one specialist
  - Multi-domain routing — dispatch to several agents in parallel
  - Fan-out + merge — parallel execution, result synthesis
  - Router chain — nested routing (route → sub-route)
  - Async parallel routing for maximum throughput
  - Router as a standalone step vs. as part of a graph

Key difference from Subagents:
  - Subagents:  main AGENT decides routing (LLM call per turn)
  - Router:     dedicated ROUTING STEP classifies first, then routes
  
Key difference from Handoffs:
  - Handoffs:   stateful, sequential, direct user interaction per step
  - Router:     stateless per request, parallel dispatch possible
"""

import asyncio
import re
from typing import Literal
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()

print("=" * 60)
print("Router Pattern")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# SPECIALIST AGENTS — each has a narrow domain
# ════════════════════════════════════════════════════════════════════

def make_specialist(name: str, prompt: str):
    return create_agent(
        model="openai:gpt-4o-mini",
        tools=[],
        system_prompt=f"{prompt} Be concise (2-3 sentences).",
    )


specialists = {
    "coding":    make_specialist("coding",    "You are a software engineer. Answer coding questions."),
    "data":      make_specialist("data",      "You are a data scientist. Answer data/ML questions."),
    "legal":     make_specialist("legal",     "You are a legal advisor. Summarize legal considerations."),
    "finance":   make_specialist("finance",   "You are a financial analyst. Provide financial insights."),
    "hr":        make_specialist("hr",        "You are an HR specialist. Handle people/org questions."),
    "general":   make_specialist("general",   "You are a helpful assistant. Answer general questions."),
}


def invoke_specialist(name: str, query: str) -> str:
    agent = specialists.get(name, specialists["general"])
    print(f"  [Specialist] → {name}: {query[:50]}")
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    content = result["messages"][-1].content
    print(f"  [Specialist] ← {name}: {content[:60]}")
    return content


# ════════════════════════════════════════════════════════════════════
# PART 1: LLM-BASED ROUTER — structured output classification
# Uses with_structured_output to classify into a fixed domain set.
# ════════════════════════════════════════════════════════════════════

print("\n── 1. LLM-Based Router (structured output) ──────────────────")

DomainType = Literal["coding", "data", "legal", "finance", "hr", "general"]

@dataclass
class RouteDecision:
    domain: DomainType
    confidence: float
    reason: str


llm = init_chat_model("openai:gpt-4o-mini")
router_llm = llm.with_structured_output(RouteDecision)

def llm_route(query: str) -> RouteDecision:
    """Classify a query into a domain using the LLM."""
    prompt = (
        f"Classify this query into exactly one domain.\n"
        f"Domains: coding, data, legal, finance, hr, general\n\n"
        f"Query: {query}"
    )
    decision = router_llm.invoke(prompt)
    print(f"  [LLMRouter] domain={decision.domain!r}, conf={decision.confidence:.2f}")
    return decision


def llm_router_pipeline(query: str) -> str:
    """Route → classify → invoke specialist → return result."""
    decision = llm_route(query)
    return invoke_specialist(decision.domain, query)


# Test LLM router with several domains
test_queries = [
    "How do I implement a binary search tree in Python?",
    "What are the GDPR implications of storing user location data?",
    "Explain the difference between a P/E ratio and EV/EBITDA.",
]

for q in test_queries:
    print(f"\n  Q: {q}")
    result = llm_router_pipeline(q)
    print(f"  A: {result[:120]}")


# ════════════════════════════════════════════════════════════════════
# PART 2: RULE-BASED ROUTER — keyword matching (deterministic, fast)
# No LLM call for routing — O(1), predictable, zero latency cost.
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Rule-Based Router (keyword matching) ──────────────────")

DOMAIN_KEYWORDS = {
    "coding":  {"python", "code", "function", "bug", "api", "sql", "algorithm", "class"},
    "data":    {"machine learning", "ml", "model", "dataset", "neural", "training", "accuracy"},
    "legal":   {"gdpr", "compliance", "contract", "liability", "regulation", "law"},
    "finance": {"revenue", "profit", "investment", "roi", "budget", "valuation", "cash flow"},
    "hr":      {"hiring", "interview", "employee", "performance review", "onboarding", "team"},
}


def keyword_route(query: str) -> str:
    """Deterministic routing via keyword matching."""
    lower = query.lower()
    scores = {domain: sum(1 for kw in kws if kw in lower)
              for domain, kws in DOMAIN_KEYWORDS.items()}
    best_domain  = max(scores, key=scores.get)
    best_score   = scores[best_domain]
    domain       = best_domain if best_score > 0 else "general"
    print(f"  [KeywordRouter] scores={scores}, → {domain!r}")
    return domain


def keyword_router_pipeline(query: str) -> str:
    domain = keyword_route(query)
    return invoke_specialist(domain, query)


for q in [
    "I have a bug in my Python function",
    "Do I need GDPR consent for analytics cookies?",
    "What's a good employee onboarding checklist?",
]:
    print(f"\n  Q: {q}")
    r = keyword_router_pipeline(q)
    print(f"  A: {r[:120]}")


# ════════════════════════════════════════════════════════════════════
# PART 3: MULTI-DOMAIN ROUTING (fan-out + merge)
# Some queries span multiple domains — route to several specialists
# in parallel, then synthesize results.
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Multi-Domain Fan-Out + Merge ──────────────────────────")

@dataclass
class MultiRouteDecision:
    domains: list[DomainType]
    reason: str


multi_router_llm = llm.with_structured_output(MultiRouteDecision)


def multi_llm_route(query: str) -> MultiRouteDecision:
    prompt = (
        f"Identify ALL relevant domains for this query (can be multiple).\n"
        f"Domains: coding, data, legal, finance, hr, general\n\n"
        f"Query: {query}"
    )
    decision = multi_router_llm.invoke(prompt)
    print(f"  [MultiRouter] domains={decision.domains}")
    return decision


def multi_router_pipeline(query: str) -> str:
    """Route to multiple specialists in parallel, merge results."""
    decision = multi_llm_route(query)

    # Invoke all matching specialists
    responses = {domain: invoke_specialist(domain, query) for domain in decision.domains}

    # Merge via synthesis LLM call
    if len(responses) == 1:
        return next(iter(responses.values()))

    context = "\n\n".join(f"[{domain.upper()}]\n{resp}" for domain, resp in responses.items())
    synthesis_prompt = (
        f"Synthesize these specialist responses into one coherent answer:\n\n"
        f"{context}\n\n"
        f"Original question: {query}\n"
        f"Provide a unified, concise response (3-5 sentences)."
    )
    synthesis = llm.invoke(synthesis_prompt)
    return synthesis.content


multi_query = "I'm building an AI startup — what code, legal, and finance considerations matter?"
print(f"\n  Q: {multi_query}")
merged = multi_router_pipeline(multi_query)
print(f"\n  Merged answer: {merged[:300]}")


# ════════════════════════════════════════════════════════════════════
# PART 4: ASYNC PARALLEL ROUTING
# Execute multiple specialists concurrently for minimum latency.
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Async Parallel Routing ────────────────────────────────")

async def invoke_specialist_async(name: str, query: str) -> tuple[str, str]:
    """Async wrapper for specialist invocation."""
    loop  = asyncio.get_event_loop()
    # Run blocking invoke in thread pool
    result = await loop.run_in_executor(None, invoke_specialist, name, query)
    return name, result


async def async_parallel_router(query: str, domains: list[str]) -> str:
    """Invoke all specified specialists in parallel."""
    tasks   = [invoke_specialist_async(d, query) for d in domains]
    results = await asyncio.gather(*tasks)

    # Synthesis
    context = "\n\n".join(f"[{name.upper()}]\n{content}" for name, content in results)
    prompt  = (
        f"Synthesize these expert perspectives:\n\n{context}\n\n"
        f"Question: {query}\nUnified answer (3 sentences):"
    )
    synthesis = llm.invoke(prompt)
    return synthesis.content


async def demo_async_routing():
    q = "How should I structure a Python ML pipeline for a fintech startup?"
    domains = llm_route(q).domain  # single domain for demo
    print(f"\n  Q: {q}")

    # Parallel routing across coding + data + finance
    parallel_result = await async_parallel_router(q, ["coding", "data", "finance"])
    print(f"\n  Parallel result: {parallel_result[:300]}")


asyncio.run(demo_async_routing())


# ════════════════════════════════════════════════════════════════════
# PART 5: ROUTER CHAIN — nested routing
# Route → sub-route for more granular classification.
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Router Chain (nested routing) ─────────────────────────")

# Sub-domains under "coding"
SUB_DOMAINS = {
    "frontend":  make_specialist("frontend",  "You are a frontend engineer (React/CSS/JS)."),
    "backend":   make_specialist("backend",   "You are a backend engineer (APIs/databases)."),
    "devops":    make_specialist("devops",    "You are a DevOps engineer (CI/CD/containers)."),
    "mobile":    make_specialist("mobile",    "You are a mobile engineer (iOS/Android)."),
}

SubDomainType = Literal["frontend", "backend", "devops", "mobile"]

@dataclass
class SubRouteDecision:
    sub_domain: SubDomainType


sub_router_llm = llm.with_structured_output(SubRouteDecision)

def nested_router_pipeline(query: str) -> str:
    """First route to domain, then sub-route within coding."""
    top_decision = llm_route(query)
    if top_decision.domain != "coding":
        return invoke_specialist(top_decision.domain, query)

    # Coding detected — sub-route
    sub_prompt   = f"Classify into frontend, backend, devops, or mobile:\n{query}"
    sub_decision = sub_router_llm.invoke(sub_prompt)
    print(f"  [SubRouter] sub_domain={sub_decision.sub_domain!r}")

    agent  = SUB_DOMAINS.get(sub_decision.sub_domain, specialists["coding"])
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


for q in [
    "How do I set up a Kubernetes CI/CD pipeline?",
    "What's the best React state management library?",
    "How do I design a REST API for user authentication?",
]:
    print(f"\n  Q: {q}")
    r = nested_router_pipeline(q)
    print(f"  A: {r[:120]}")

print("\n" + "═" * 60)
print("Router Pattern Summary:")
print("  LLM router:     with_structured_output → Literal domain type")
print("  Rule router:    keyword matching → deterministic, zero latency")
print("  Fan-out+merge:  multi-domain routing → parallel → synthesis")
print("  Async parallel: asyncio.gather → invoke all specialists at once")
print("  Router chain:   top-level routing → sub-domain classification")
print("  Cost: 3 model calls (route + specialist + synthesize) vs. subagent 5")
print("═" * 60)
print("\n✅ Router pattern demo complete.")
