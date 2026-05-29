"""
04_hitl_as_guardrail.py
=======================
Demonstrates Human-in-the-Loop (HITL) as a safety guardrail — the most
effective protection for high-stakes, irreversible operations.

Concepts covered:
  - HumanInTheLoopMiddleware as a safety gate for financial operations
  - HumanInTheLoopMiddleware for production data modifications
  - HumanInTheLoopMiddleware for external communications
  - Combining HITL with deterministic pre-filters in one agent
  - Full approve / edit / reject / feedback lifecycle
  - InMemorySaver (alias for MemorySaver) as checkpointer
"""

import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    AgentMiddleware,
    AgentState,
    hook_config,
)
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver   # alias for MemorySaver
from langgraph.types import Command
from langgraph.runtime import Runtime
from typing import Any

load_dotenv()

print("=" * 60)
print("Human-in-the-Loop as a Safety Guardrail Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# TOOLS — operations of varying risk level
# ════════════════════════════════════════════════════════════════════

@tool
def search_tool(query: str) -> str:
    """Search for information (safe — no approval needed)."""
    print(f"  [Tool] search_tool: '{query}'")
    return f"Search results for '{query}': [Sample results]"


@tool
def send_email_tool(to: str, subject: str, body: str) -> str:
    """Send an email to an external recipient. Requires human approval."""
    print(f"  [Tool] send_email_tool: to={to}, subject='{subject}'")
    return f"Email sent to {to}."


@tool
def delete_database_tool(table: str, where_clause: str) -> str:
    """Delete records from the database. IRREVERSIBLE — requires human approval."""
    print(f"  [Tool] delete_database_tool: table={table}, where={where_clause}")
    return f"Deleted records from {table} where {where_clause}."


@tool
def transfer_funds(from_account: str, to_account: str, amount: float) -> str:
    """Transfer funds between accounts. High-risk — requires human approval."""
    print(f"  [Tool] transfer_funds: {from_account} → {to_account}, ${amount:.2f}")
    return f"Transferred ${amount:.2f} from {from_account} to {to_account}."


# ════════════════════════════════════════════════════════════════════
# 1. BASIC HITL GUARDRAIL
#    Requires approval before sending emails and deleting data.
#    Search is auto-approved (safe operation).
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Basic HITL Guardrail ──────────────────────────────────")

agent_hitl = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_tool, send_email_tool, delete_database_tool],
    checkpointer=InMemorySaver(),   # Required for HITL state persistence
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool":      True,   # Always interrupt
                "delete_database_tool": True,   # Always interrupt
                "search_tool":          False,  # Safe — never interrupt
            }
        )
    ],
    system_prompt="You are a business operations assistant.",
)

config_1 = {"configurable": {"thread_id": "hitl-guardrail-1"}}

# Agent hits send_email_tool — pauses for approval
result_1a = agent_hitl.invoke(
    {"messages": [{"role": "user", "content":
        "Send an email to team@company.com with subject 'Quarterly Update' "
        "and body 'Please review the attached Q4 report.'"}]},
    config=config_1,
)

if "__interrupt__" in result_1a:
    action = result_1a["__interrupt__"][0].value.get("action", {})
    print(f"⏸  Paused before: {action.get('name')}({action.get('args')})")

    # Human approves
    result_1b = agent_hitl.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config_1,
    )
    print(f"✅ Approved. Response: {result_1b['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 2. HITL WITH FINANCIAL GUARDRAIL
#    All fund transfer operations require explicit approval.
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Financial HITL Guardrail ──────────────────────────────")

agent_finance = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_tool, transfer_funds],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "transfer_funds": {"allowed_decisions": ["approve", "edit", "reject"]},
                "search_tool":    False,
            }
        )
    ],
    system_prompt="You are a financial operations assistant.",
)

config_2 = {"configurable": {"thread_id": "hitl-finance-1"}}

result_2a = agent_finance.invoke(
    {"messages": [{"role": "user", "content":
        "Transfer $5,000 from account ACC-001 to account ACC-999."}]},
    config=config_2,
)

if "__interrupt__" in result_2a:
    action = result_2a["__interrupt__"][0].value.get("action", {})
    print(f"⏸  Paused before transfer: {action.get('args')}")

    # Human edits the amount before approving
    print("👤 Human decision: EDIT (reduce amount to $500)")
    result_2b = agent_finance.invoke(
        Command(resume={
            "decisions": [{
                "type": "edit",
                "edited_action": {
                    "name": "transfer_funds",
                    "args": {
                        "from_account": "ACC-001",
                        "to_account":   "ACC-999",
                        "amount":        500.0,   # ← Reduced by human reviewer
                    },
                },
            }]
        }),
        config=config_2,
    )
    print(f"✏️  Edited & approved. Response: {result_2b['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 3. HITL REJECTION WORKFLOW
#    Human rejects a dangerous database delete operation with feedback.
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Rejection Workflow ────────────────────────────────────")

config_3 = {"configurable": {"thread_id": "hitl-rejection-1"}}

result_3a = agent_hitl.invoke(
    {"messages": [{"role": "user", "content":
        "Delete all records from the users table where status='inactive'."}]},
    config=config_3,
)

if "__interrupt__" in result_3a:
    action = result_3a["__interrupt__"][0].value.get("action", {})
    print(f"⏸  Paused before: {action.get('name')}")
    print("👤 Human decision: REJECT")

    result_3b = agent_hitl.invoke(
        Command(resume={
            "decisions": [{
                "type":     "reject",
                "feedback": "Mass delete requires data-team sign-off. Contact your manager.",
            }]
        }),
        config=config_3,
    )
    print(f"🚫 Rejected. Response: {result_3b['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 4. HITL + DETERMINISTIC PRE-FILTER COMBINED
#    Apply a keyword guardrail BEFORE HITL to block obvious abuse,
#    then use HITL for all remaining sensitive operations.
# ════════════════════════════════════════════════════════════════════

class ContentFilterMiddleware(AgentMiddleware):
    BANNED = ["hack", "malware", "ransomware"]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
        first = state["messages"][0]
        if first.type != "human":
            return None
        for kw in self.BANNED:
            if kw in first.content.lower():
                print(f"  [Filter] 🚫 Keyword '{kw}' blocked before HITL check.")
                return {
                    "messages": [{"role": "assistant",
                                  "content": "Request blocked by content policy."}],
                    "jump_to": "end",
                }
        return None


print("\n── 4. HITL + Deterministic Pre-Filter ───────────────────────")

agent_combined = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        ContentFilterMiddleware(),           # Layer 1 — block abuse early
        HumanInTheLoopMiddleware(            # Layer 2 — HITL for sensitive ops
            interrupt_on={"send_email_tool": True, "search_tool": False}
        ),
    ],
    system_prompt="You are a business assistant.",
)

# Blocked by content filter — HITL never reached
config_4a = {"configurable": {"thread_id": "combined-blocked"}}
result_4a = agent_combined.invoke(
    {"messages": [{"role": "user", "content": "Send a malware report to external.com."}]},
    config=config_4a,
)
print(f"🚫 Content-filtered: {result_4a['messages'][-1].content[:80]}")

# Passes filter — pauses at HITL
config_4b = {"configurable": {"thread_id": "combined-hitl"}}
result_4b = agent_combined.invoke(
    {"messages": [{"role": "user", "content":
        "Send an update email to board@company.com with subject 'Meeting Notes'."}]},
    config=config_4b,
)
if "__interrupt__" in result_4b:
    print("⏸  Passed content filter — paused at HITL gate.")
    result_4c = agent_combined.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config_4b,
    )
    print(f"✅ HITL approved: {result_4c['messages'][-1].content[:80]}")

print("\n✅ HITL guardrail demo complete.")
