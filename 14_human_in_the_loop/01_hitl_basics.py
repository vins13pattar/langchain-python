"""
01_hitl_basics.py
==================
Introduction to Human-in-the-Loop (HITL) middleware in LangChain.

Concepts covered:
  - HumanInTheLoopMiddleware setup and interrupt_on configuration
  - Checkpointer requirement for interrupt state persistence
  - version="v2" for GraphOutput with .interrupts attribute
  - thread_id config for conversation association
  - Inspecting interrupt details (action_requests, review_configs)
  - Simple approve and reject decisions via Command(resume=...)
  - description_prefix for custom interrupt messages

The HITL middleware pauses agent execution when a tool call matches
the interrupt policy, saves state to the checkpointer, and waits
for a human decision before resuming.
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
print("Human-in-the-Loop — Basics")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# TOOLS — varying risk levels
# ════════════════════════════════════════════════════════════════════

@tool
def read_data(table: str) -> str:
    """Read data from a database table (safe — no approval needed)."""
    print(f"  [Tool] read_data({table!r}) — executing")
    return f"Data from '{table}': [row1, row2, row3]"


@tool
def execute_sql(query: str) -> str:
    """Execute a SQL statement against the database."""
    print(f"  [Tool] execute_sql({query!r}) — executing")
    return f"SQL executed: {query} → 42 rows affected."


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file on disk."""
    print(f"  [Tool] write_file({path!r}) — executing")
    return f"File written to {path} ({len(content)} bytes)."


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    print(f"  [Tool] send_email(to={to!r}) — executing")
    return f"Email sent to {to} with subject '{subject}'."


# ════════════════════════════════════════════════════════════════════
# 1. BASIC HITL SETUP
#    Configure which tools require approval and what decisions
#    are allowed for each.
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Basic HITL Configuration ──────────────────────────────")

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[read_data, execute_sql, write_file, send_email],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # True = all decisions (approve, edit, reject, respond) allowed
                "write_file":  True,

                # Restrict to only approve/reject — no editing SQL queries
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},

                # Send email allows all decisions + custom description
                "send_email":  {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "Email pending review before sending",
                },

                # False = auto-approve, no interrupt (safe operation)
                "read_data":   False,
            },
            description_prefix="Tool execution pending approval",
        )
    ],
    # Checkpointer is REQUIRED — persists state across the interrupt
    checkpointer=MemorySaver(),
    system_prompt="You are a database and file management assistant.",
)

# ── 1a. Interrupt on write_file ──────────────────────────────────

print("\n── 1a. Interrupt — write_file ───────────────────────────────")

config = {"configurable": {"thread_id": "hitl-basics-write"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Write a config file to /etc/app/config.yaml with content 'debug: true'"}]},
    config=config,
    version="v2",   # ← required for GraphOutput with .interrupts
)

print(f"\nInterrupts raised: {len(result.interrupts)}")
if result.interrupts:
    interrupt = result.interrupts[0]
    value     = interrupt.value
    actions   = value.get("action_requests", [])
    configs   = value.get("review_configs", [])
    print(f"Action to review: {actions[0]['name']}({actions[0]['arguments']})")
    print(f"Allowed decisions: {configs[0]['allowed_decisions']}")
    print(f"Description: {actions[0]['description'][:80]}")

    # APPROVE — execute the tool as-is
    print("\n  → Approving write_file...")
    final = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
        version="v2",
    )
    print(f"After approval: {final.value['messages'][-1].content[:100]}")


# ── 1b. Interrupt on execute_sql — reject ──────────────────────

print("\n── 1b. Interrupt — execute_sql (reject) ─────────────────────")

config2 = {"configurable": {"thread_id": "hitl-basics-sql"}}

result2 = agent.invoke(
    {"messages": [{"role": "user",
                   "content": "Delete all records from the logs table older than 90 days."}]},
    config=config2,
    version="v2",
)

print(f"\nInterrupts: {len(result2.interrupts)}")
if result2.interrupts:
    action = result2.interrupts[0].value["action_requests"][0]
    print(f"SQL proposed: {action['arguments'].get('query', '')[:80]}")

    # REJECT — don't execute, send feedback to the agent
    print("  → Rejecting SQL — asking for safer approach...")
    final2 = agent.invoke(
        Command(resume={
            "decisions": [{
                "type":    "reject",
                "message": "Deleting data is too risky. Archive it to a backup table instead.",
            }]
        }),
        config=config2,
        version="v2",
    )
    print(f"After rejection: {final2.value['messages'][-1].content[:150]}")


# ── 1c. Safe tool — read_data auto-approved (no interrupt) ───────

print("\n── 1c. Safe Tool — read_data (no interrupt) ─────────────────")

config3 = {"configurable": {"thread_id": "hitl-basics-read"}}

result3 = agent.invoke(
    {"messages": [{"role": "user", "content": "Read data from the users table."}]},
    config=config3,
    version="v2",
)

print(f"Interrupts: {len(result3.interrupts)} (expected 0 — read_data is auto-approved)")
if not result3.interrupts:
    final_msg = result3.value["messages"][-1].content
    print(f"Direct result: {final_msg[:120]}")

print("\n" + "═" * 60)
print("HITL Basics Key Points:")
print("  HumanInTheLoopMiddleware(interrupt_on={...})")
print("    True        — all decisions allowed (approve/edit/reject/respond)")
print("    False       — auto-approve, never interrupt")
print("    {...}       — InterruptOnConfig with allowed_decisions + description")
print("  checkpointer  — REQUIRED to persist state across interrupts")
print("  version='v2'  — returns GraphOutput with .interrupts")
print("  config thread_id — ties invocations to a conversation thread")
print("  Command(resume={'decisions': [...]}) — resume after interrupt")
print("═" * 60)
print("\n✅ HITL basics demo complete.")
