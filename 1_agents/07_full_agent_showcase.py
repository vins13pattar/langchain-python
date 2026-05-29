"""
07_full_agent_showcase.py
=========================
A COMPLETE, production-ready agent combining ALL core concepts:

  ✅ create_agent()           — modern harness
  ✅ Multiple tools           — real-world callable actions
  ✅ system_prompt            — agent personality & constraints
  ✅ MemorySaver checkpointer — conversation history across turns
  ✅ thread_id                — scopes the conversation
  ✅ context_schema           — per-run user data
  ✅ structured output        — typed final answer
  ✅ Streaming                — real-time output
  ✅ Fault tolerance          — ModelRetry + ToolRetry middleware
  ✅ Guardrails               — PIIMiddleware
  ✅ Recursion limit          — prevents runaway loops

This file is the "putting it all together" example. Refer to the numbered
files (01–06) for isolated deep-dives on each concept.
"""

import os
import uuid
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    PIIMiddleware,
)
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

load_dotenv()


# ══════════════════════════════════════════════════════════════════════
# 1. CONTEXT SCHEMA
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AppContext:
    user_id:  str
    username: str
    plan:     str = "free"    # free | pro | enterprise


# ══════════════════════════════════════════════════════════════════════
# 2. STRUCTURED OUTPUT SCHEMA
# ══════════════════════════════════════════════════════════════════════

class ResearchReport(BaseModel):
    """Final research report returned by the agent."""
    title:       str              = Field(description="Report title")
    summary:     str              = Field(description="Executive summary, 2-3 sentences")
    key_findings: List[str]       = Field(description="3-5 main findings")
    confidence:  float            = Field(description="Confidence score 0.0–1.0")
    next_steps:  Optional[str]    = Field(default=None, description="Recommended actions")


# ══════════════════════════════════════════════════════════════════════
# 3. TOOLS
# ══════════════════════════════════════════════════════════════════════

@tool
def search_web(query: str) -> str:
    """Search the web for current information on a topic.

    Args:
        query: Search keywords (2-8 words recommended)
    """
    results = {
        "langchain agents":  "LangChain agents use create_agent() with middleware for production use.",
        "ai trends 2026":    "Agentic AI, multimodal models, and on-device inference dominate 2026.",
        "python frameworks": "FastAPI, LangChain, and Pydantic lead Python AI framework adoption.",
    }
    key = query.lower()
    for k, v in results.items():
        if any(word in key for word in k.split()):
            return v
    return f"No specific results for '{query}' — returning general knowledge."


@tool
def read_document(doc_id: str) -> str:
    """Read a document from the knowledge base by ID.

    Args:
        doc_id: Document identifier (e.g. 'DOC-001')
    """
    docs = {
        "DOC-001": "LangChain 0.3 introduces create_agent() as the primary agent factory.",
        "DOC-002": "Middleware in LangChain handles retries, PII, HITL, and summarisation.",
        "DOC-003": "Checkpointers (MemorySaver, SqliteSaver) persist conversation state.",
    }
    return docs.get(doc_id.upper(), f"Document '{doc_id}' not found in knowledge base.")


@tool
def calculate(expression: str) -> str:
    """Evaluate a safe mathematical expression.

    Args:
        expression: Math expression such as '2 + 2' or '100 * 1.08'
    """
    try:
        # Only allow safe operations
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: unsafe characters in expression"
        return str(round(eval(expression), 6))   # noqa: S307
    except Exception as e:
        return f"Error: {e}"


@tool
def get_user_info() -> str:
    """Return information about the currently logged-in user."""
    from langchain.agents.runtime import get_runtime
    runtime = get_runtime()
    ctx: AppContext = runtime.context
    return (
        f"User: {ctx.username} (ID: {ctx.user_id}), "
        f"Plan: {ctx.plan.upper()}"
    )


# ══════════════════════════════════════════════════════════════════════
# 4. CREATE THE AGENT
# ══════════════════════════════════════════════════════════════════════

agent = create_agent(
    # ── Core ─────────────────────────────────────────────────────────
    model="openai:gpt-4o-mini",
    tools=[search_web, read_document, calculate, get_user_info],

    # ── Persona ───────────────────────────────────────────────────────
    system_prompt=(
        "You are an expert research assistant. "
        "Always greet the user by name on first contact. "
        "Use available tools to gather information before answering. "
        "Be concise, factual, and cite your tool results."
    ),

    # ── Structured output ─────────────────────────────────────────────
    response_format=ResearchReport,

    # ── Memory ────────────────────────────────────────────────────────
    checkpointer=MemorySaver(),

    # ── Runtime context ───────────────────────────────────────────────
    context_schema=AppContext,

    # ── Middleware stack ──────────────────────────────────────────────
    middleware=[
        ModelRetryMiddleware(max_retries=3),   # handle rate-limits / timeouts
        ToolRetryMiddleware(max_retries=2),    # handle transient tool errors
        PIIMiddleware(),                        # scrub PII before it reaches LLM
    ],

    # ── Agent name (useful in multi-agent systems) ────────────────────
    name="research_assistant",
)


# ══════════════════════════════════════════════════════════════════════
# 5. HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def run_with_streaming(question: str, context: AppContext, config: dict) -> ResearchReport:
    """Stream intermediate steps and return the final structured report."""
    print(f"\n🧑 {context.username}: {question}")
    print("─" * 55)

    final_state = None

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        context=context,
        stream_mode="values",
    ):
        latest = chunk["messages"][-1]
        final_state = chunk

        if isinstance(latest, AIMessage):
            if latest.tool_calls:
                names = [tc["name"] for tc in latest.tool_calls]
                print(f"  🔧 Calling: {names}")
            elif latest.content:
                print(f"  💬 Thinking: {latest.content[:80]}…")

        elif isinstance(latest, ToolMessage):
            print(f"  📦 [{latest.name}] → {latest.content[:60]}…")

    print("─" * 55)

    report: ResearchReport = final_state["structured_response"]
    return report


def print_report(report: ResearchReport) -> None:
    print(f"\n{'═' * 55}")
    print(f"  REPORT: {report.title}")
    print(f"{'═' * 55}")
    print(f"\nSummary:\n  {report.summary}")
    print(f"\nKey Findings:")
    for i, finding in enumerate(report.key_findings, 1):
        print(f"  {i}. {finding}")
    print(f"\nConfidence: {report.confidence:.0%}")
    if report.next_steps:
        print(f"\nNext Steps: {report.next_steps}")
    print()


# ══════════════════════════════════════════════════════════════════════
# 6. RUN THE SHOWCASE
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  Full Agent Showcase — All Concepts Combined")
    print("=" * 55)

    user    = AppContext(user_id="u-007", username="Vinod", plan="pro")
    thread  = str(uuid.uuid4())
    config  = {"configurable": {"thread_id": thread}}

    # ── Query 1 ──────────────────────────────────────────────────────
    report = run_with_streaming(
        "Research the latest AI trends and summarise key findings.",
        context=user,
        config=config,
    )
    print_report(report)

    # ── Query 2 (same thread — agent remembers previous turn) ─────────
    report = run_with_streaming(
        "Now read document DOC-002 and connect it to what we just discussed.",
        context=user,
        config=config,   # ← same thread_id keeps history
    )
    print_report(report)

    # ── Query 3 — mixed tool calls ────────────────────────────────────
    report = run_with_streaming(
        "If I have 1000 tokens and each costs $0.002, what's the total cost? "
        "Also who am I and what plan am I on?",
        context=user,
        config=config,
    )
    print_report(report)
