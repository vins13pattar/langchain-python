"""
02_human_in_the_loop.py
=======================
Demonstrates Human-in-the-Loop (HITL) middleware — pausing agent execution
before dangerous tool calls and resuming based on human decisions.

Concepts covered:
  - HumanInTheLoopMiddleware setup (requires checkpointer + thread_id)
  - Detecting __interrupt__ in the agent result
  - Resume with: approve / edit / reject decisions
  - Per-tool HITL policies (different rules for different tools)
  - Simulating the full approve → edit → reject lifecycle
"""

import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

print("=" * 60)
print("Human-in-the-Loop (HITL) Middleware Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# TOOLS — email and file operations of varying risk levels
# ════════════════════════════════════════════════════════════════════

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient with a subject and body."""
    print(f"  [Tool] send_email → to={to}, subject='{subject}'")
    return f"Email sent to {to} with subject '{subject}'."


@tool
def read_email(folder: str = "inbox") -> str:
    """Read emails from a specified folder (no approval needed)."""
    print(f"  [Tool] read_email → folder={folder}")
    return f"3 unread emails found in '{folder}'."


@tool
def delete_email(email_id: str) -> str:
    """Permanently delete an email by its ID."""
    print(f"  [Tool] delete_email → id={email_id}")
    return f"Email {email_id} permanently deleted."


# ════════════════════════════════════════════════════════════════════
# 1. BASIC HITL — APPROVE WORKFLOW
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Approve Workflow ──────────────────────────────────────")

agent_hitl = create_agent(
    model="openai:gpt-4o-mini",
    tools=[send_email, read_email, delete_email],
    checkpointer=MemorySaver(),  # ← Required for HITL state persistence
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email":   {"allowed_decisions": ["approve", "edit", "reject"]},
                "delete_email": {"allowed_decisions": ["approve", "reject"]},
                # read_email is not listed → no interruption needed
            }
        )
    ],
    system_prompt="You are an email management assistant.",
)

config_approve = {"configurable": {"thread_id": "hitl-session-approve"}}

# Step 1 — agent reaches the tool call and pauses
result1 = agent_hitl.invoke(
    {"messages": [{"role": "user", "content":
        "Send an email to alice@example.com with subject 'Meeting Tomorrow' "
        "and body 'Hi Alice, let us catch up tomorrow at 10am!'"}]},
    config=config_approve,
)

if "__interrupt__" in result1:
    interrupt_info = result1["__interrupt__"]
    print(f"⏸  Agent paused — waiting for human approval.")
    print(f"   Tool: {interrupt_info[0].value.get('action', {}).get('name', 'unknown')}")
    print(f"   Args: {interrupt_info[0].value.get('action', {}).get('args', {})}")

    # Step 2 — human approves
    result2 = agent_hitl.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config_approve,
    )
    print(f"✅ Approved. Response: {result2['messages'][-1].content[:100]}")
else:
    print(f"Response: {result1['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 2. EDIT WORKFLOW — Human corrects the tool arguments before approval
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Edit Workflow ─────────────────────────────────────────")

config_edit = {"configurable": {"thread_id": "hitl-session-edit"}}

result_e1 = agent_hitl.invoke(
    {"messages": [{"role": "user", "content":
        "Send a meeting invite email to bob@example.com with subject "
        "'Q4 Planning' and body 'Please join our Q4 planning session.'"}]},
    config=config_edit,
)

if "__interrupt__" in result_e1:
    print(f"⏸  Agent paused — human will edit the email address.")

    # Human fixes the recipient before approving
    result_e2 = agent_hitl.invoke(
        Command(resume={
            "decisions": [{
                "type": "edit",
                "edited_action": {
                    "name": "send_email",
                    "args": {
                        "to": "bob.smith@company.com",   # corrected address
                        "subject": "Q4 Planning Session",
                        "body": "Please join our Q4 planning session.",
                    },
                },
            }]
        }),
        config=config_edit,
    )
    print(f"✏️  Edited & approved. Response: {result_e2['messages'][-1].content[:100]}")
else:
    print(f"Response: {result_e1['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 3. REJECT WORKFLOW — Human blocks a dangerous action with feedback
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Reject Workflow ───────────────────────────────────────")

config_reject = {"configurable": {"thread_id": "hitl-session-reject"}}

result_r1 = agent_hitl.invoke(
    {"messages": [{"role": "user", "content":
        "Delete email with ID email-99 immediately."}]},
    config=config_reject,
)

if "__interrupt__" in result_r1:
    print("⏸  Agent paused — human will reject the delete operation.")

    result_r2 = agent_hitl.invoke(
        Command(resume={
            "decisions": [{
                "type": "reject",
                "feedback": "Deletion requires manager approval first. Do not delete.",
            }]
        }),
        config=config_reject,
    )
    print(f"🚫 Rejected. Response: {result_r2['messages'][-1].content[:120]}")
else:
    print(f"Response: {result_r1['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 4. PER-TOOL POLICIES — Different rules for different risk levels
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Per-Tool Policies ─────────────────────────────────────")

agent_per_tool = create_agent(
    model="openai:gpt-4o-mini",
    tools=[send_email, read_email, delete_email],
    checkpointer=MemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email":   {"allowed_decisions": ["approve", "edit", "reject"]},
                "delete_email": {"allowed_decisions": ["approve", "reject"]},
                "read_email":   False,   # Safe — no interruption needed
            }
        )
    ],
)

config_safe = {"configurable": {"thread_id": "hitl-session-safe-read"}}

# read_email should complete without interruption
result_safe = agent_per_tool.invoke(
    {"messages": [{"role": "user", "content": "Check my inbox for new emails."}]},
    config=config_safe,
)

if "__interrupt__" not in result_safe:
    print(f"✅ read_email ran without interruption (low-risk op).")
    print(f"   Response: {result_safe['messages'][-1].content[:100]}")
else:
    print("Unexpected interrupt for read_email!")

print("\n✅ Human-in-the-Loop demo complete.")
