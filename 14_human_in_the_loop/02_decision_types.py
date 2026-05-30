"""
02_decision_types.py
=====================
Deep dive into all four HITL decision types:
  ✅ approve  — execute tool as-is
  ✏️  edit    — modify tool args before execution
  ❌ reject   — block tool + send feedback to agent
  💬 respond  — skip tool; human reply IS the tool result

Concepts covered:
  - approve: simple pass-through
  - edit: edited_action with new name/args
  - reject: message as feedback to the agent
  - respond: "ask_user" pattern — human answers the tool
  - allowed_decisions restriction per tool
  - InterruptOnConfig with custom descriptions
  - Using pprint to inspect interrupt structure
"""

import os
from pprint import pprint
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

print("=" * 60)
print("HITL — All Four Decision Types")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    print(f"  [Tool] send_email(to={to!r}, subject={subject!r})")
    return f"Email sent to {to} — subject: '{subject}'."


@tool
def delete_records(table: str, condition: str) -> str:
    """Delete records from a database table matching a condition."""
    print(f"  [Tool] delete_records(table={table!r}, condition={condition!r})")
    return f"Deleted records from '{table}' WHERE {condition}."


@tool
def ask_user(question: str) -> str:
    """Ask the human user a clarifying question and return their answer."""
    # Tool body is never called when decision is "respond" —
    # the human's message IS the result.
    print(f"  [Tool] ask_user({question!r}) — this should not execute when 'respond' is used")
    return "Tool fallback response (should not appear with respond decision)."


@tool
def archive_data(table: str, days_old: int = 30) -> str:
    """Archive data older than N days from a table."""
    print(f"  [Tool] archive_data(table={table!r}, days_old={days_old})")
    return f"Archived records from '{table}' older than {days_old} days."


# ════════════════════════════════════════════════════════════════════
# AGENT
# ════════════════════════════════════════════════════════════════════

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[send_email, delete_records, ask_user, archive_data],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # All 4 decision types allowed for email
                "send_email": True,

                # Only approve/reject for deletes (no editing dangerous SQL)
                "delete_records": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "DANGER: Permanent data deletion requested",
                },

                # "respond" for the ask_user tool — human IS the answer
                "ask_user": {
                    "allowed_decisions": ["respond"],
                    "description": "The agent is asking you a question",
                },

                # Archive allows approve/edit/reject
                "archive_data": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "Data archival operation needs review",
                },
            }
        )
    ],
    checkpointer=MemorySaver(),
    system_prompt=(
        "You are a database and communications assistant. "
        "When you need clarification, use the ask_user tool."
    ),
)


# ════════════════════════════════════════════════════════════════════
# ✅ DECISION 1: APPROVE — execute exactly as proposed
# ════════════════════════════════════════════════════════════════════

print("\n── ✅ APPROVE — Execute tool as-is ──────────────────────────")

cfg_approve = {"configurable": {"thread_id": "decision-approve"}}

r1 = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Send an email to alice@example.com with subject 'Monthly Report' "
        "and body 'Please find attached the Q4 report.'"}]},
    config=cfg_approve,
    version="v2",
)

print(f"Interrupt: {r1.interrupts[0].value['action_requests'][0]['name']}")
print(f"Args: {r1.interrupts[0].value['action_requests'][0]['arguments']}")

final_approve = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=cfg_approve,
    version="v2",
)
print(f"Result: {final_approve.value['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# ✏️ DECISION 2: EDIT — modify before execution
# ════════════════════════════════════════════════════════════════════

print("\n── ✏️  EDIT — Modify tool args before execution ──────────────")

cfg_edit = {"configurable": {"thread_id": "decision-edit"}}

r2 = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Send an email to all-hands@example.com about the company offsite."}]},
    config=cfg_edit,
    version="v2",
)

if r2.interrupts:
    original_args = r2.interrupts[0].value["action_requests"][0]["arguments"]
    print(f"Original args: {original_args}")

    # Edit: change recipient and add more professional subject
    final_edit = agent.invoke(
        Command(resume={
            "decisions": [{
                "type": "edit",
                "edited_action": {
                    "name": "send_email",       # same tool name
                    "args": {
                        "to":      "leadership@example.com",  # narrower audience
                        "subject": original_args.get("subject", "Company Offsite"),
                        "body":    original_args.get("body", "") + "\n\n[Reviewed and approved by comms team]",
                    }
                }
            }]
        }),
        config=cfg_edit,
        version="v2",
    )
    print(f"After edit: {final_edit.value['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# ❌ DECISION 3: REJECT — block + provide feedback
# ════════════════════════════════════════════════════════════════════

print("\n── ❌ REJECT — Block tool and send feedback to agent ─────────")

cfg_reject = {"configurable": {"thread_id": "decision-reject"}}

r3 = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Delete all records from the users table where status = 'inactive'."}]},
    config=cfg_reject,
    version="v2",
)

if r3.interrupts:
    action = r3.interrupts[0].value["action_requests"][0]
    print(f"Proposed: {action['name']}({action['arguments']})")
    print(f"Allowed decisions: {r3.interrupts[0].value['review_configs'][0]['allowed_decisions']}")

    # Reject with guidance on what to do instead
    final_reject = agent.invoke(
        Command(resume={
            "decisions": [{
                "type":    "reject",
                "message": (
                    "Do NOT delete records. Instead, set status = 'archived' "
                    "using an UPDATE statement. Deletion is irreversible."
                ),
            }]
        }),
        config=cfg_reject,
        version="v2",
    )
    print(f"After rejection: {final_reject.value['messages'][-1].content[:200]}")


# ════════════════════════════════════════════════════════════════════
# 💬 DECISION 4: RESPOND — human reply IS the tool result
# ════════════════════════════════════════════════════════════════════

print("\n── 💬 RESPOND — Human answer replaces tool execution ─────────")

cfg_respond = {"configurable": {"thread_id": "decision-respond"}}

r4 = agent.invoke(
    {"messages": [{"role": "user", "content":
        "I need to archive some data. Ask me which table and how old."}]},
    config=cfg_respond,
    version="v2",
)

if r4.interrupts:
    action = r4.interrupts[0].value["action_requests"][0]
    print(f"Agent is asking: {action['arguments'].get('question', action['arguments'])}")

    # Respond: the human's reply becomes the ToolMessage content directly
    final_respond = agent.invoke(
        Command(resume={
            "decisions": [{
                "type":    "respond",
                "message": "Archive the 'audit_logs' table for records older than 90 days.",
            }]
        }),
        config=cfg_respond,
        version="v2",
    )
    print(f"After respond: {final_respond.value['messages'][-1].content[:180]}")


# ════════════════════════════════════════════════════════════════════
# ✏️ EDIT — change to a different tool entirely
# ════════════════════════════════════════════════════════════════════

print("\n── ✏️  EDIT — Change tool entirely (delete → archive) ─────────")

cfg_edit2 = {"configurable": {"thread_id": "decision-edit-tool"}}

r5 = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Delete records from 'transactions' older than 60 days."}]},
    config=cfg_edit2,
    version="v2",
)

if r5.interrupts:
    print(f"Proposed: {r5.interrupts[0].value['action_requests'][0]['name']}")

    # Edit: switch to archive_data instead of delete_records
    final_edit2 = agent.invoke(
        Command(resume={
            "decisions": [{
                "type": "edit",
                "edited_action": {
                    "name": "archive_data",    # different tool!
                    "args": {"table": "transactions", "days_old": 60},
                }
            }]
        }),
        config=cfg_edit2,
        version="v2",
    )
    print(f"After tool swap: {final_edit2.value['messages'][-1].content[:120]}")

print("\n" + "═" * 60)
print("Decision Type Reference:")
print("  approve  → {\"type\": \"approve\"}")
print("  edit     → {\"type\": \"edit\", \"edited_action\": {\"name\": ..., \"args\": {...}}}")
print("  reject   → {\"type\": \"reject\", \"message\": \"reason...\"}")
print("  respond  → {\"type\": \"respond\", \"message\": \"human answer\"}")
print("  Decisions list order must match action_requests order in interrupt")
print("═" * 60)
print("\n✅ All four decision types demo complete.")
