"""
05_middleware.py
================
Demonstrates the SIX middleware categories from the LangChain docs.

Concepts covered:
  - middleware= parameter on create_agent()
  - HumanInTheLoopMiddleware   — pause & await approval before sensitive actions
  - ModelRetryMiddleware        — auto-retry on model errors / rate limits
  - ToolRetryMiddleware         — auto-retry on tool failures
  - PIIMiddleware               — scrub personal data before it reaches the model
  - (FilesystemMiddleware shown conceptually — requires deepagents package)

Middleware is the primary extensibility primitive in create_agent().
Each piece handles ONE concern, composes freely, and never requires
restructuring your tools or business logic.

Middleware execution order (simplified):
  User message
    → [Middleware 1 before_model]
    → [Middleware 2 before_model]
    → Model call
    → [Middleware 2 after_model]
    → [Middleware 1 after_model]
    → Tool execution
    → … loop continues …
"""

import os
import uuid
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    PIIMiddleware,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()


# ── Shared tools ──────────────────────────────────────────────────────────────

@tool
def read_file(path: str) -> str:
    """Read and return the contents of a file.

    Args:
        path: File path to read
    """
    return f"[Contents of {path}]: This is sample file content for demonstration."


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file (DESTRUCTIVE — triggers HITL approval).

    Args:
        path:    Destination file path
        content: Content to write
    """
    return f"✅ Written {len(content)} bytes to {path}"


@tool
def delete_file(path: str) -> str:
    """Delete a file permanently (DESTRUCTIVE — triggers HITL approval).

    Args:
        path: File path to delete
    """
    return f"🗑️  Deleted {path}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (requires HITL approval).

    Args:
        to:      Recipient email address
        subject: Email subject line
        body:    Email body text
    """
    return f"📧 Email sent to {to}: '{subject}'"


# ════════════════════════════════════════════════════════════════════
# EXAMPLE 1 — Fault Tolerance (ModelRetry + ToolRetry)
# ════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Example 1 — Fault Tolerance Middleware")
print("=" * 60)

fault_tolerant_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[read_file],
    middleware=[
        ModelRetryMiddleware(max_retries=3),   # retry model on timeout / rate-limit
        ToolRetryMiddleware(max_retries=2),    # retry tool on transient failure
    ],
    system_prompt=(
        "You are a file assistant. Read files when asked. "
        "On errors, try again."
    ),
)

result = fault_tolerant_agent.invoke({
    "messages": [{"role": "user", "content": "Read the file at /docs/readme.txt"}]
})
print(f"\n🤖 {result['messages'][-1].content}\n")


# ════════════════════════════════════════════════════════════════════
# EXAMPLE 2 — Guardrails (PIIMiddleware)
# ════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Example 2 — PII Guardrail Middleware")
print("=" * 60)
print("(PII like phone numbers / SSNs are scrubbed before reaching the model)")

pii_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[
        PIIMiddleware(),    # intercepts & masks SSNs, credit cards, phone numbers, etc.
    ],
    system_prompt=(
        "You are a data processor. Summarise any text sent to you. "
        "Never repeat raw personal data back."
    ),
)

result = pii_agent.invoke({
    "messages": [{
        "role": "user",
        "content": (
            "Process this record: Name=John Doe, SSN=123-45-6789, "
            "Phone=555-867-5309, Email=john@example.com"
        ),
    }]
})
print(f"\n🤖 {result['messages'][-1].content}\n")


# ════════════════════════════════════════════════════════════════════
# EXAMPLE 3 — Human-in-the-Loop (HITL) Steering
# ════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Example 3 — Human-in-the-Loop (HITL) Middleware")
print("=" * 60)
print(
    "The agent pauses BEFORE write_file / delete_file / send_email "
    "and waits for human approval.\n"
)

checkpointer = MemorySaver()            # required for HITL — stores the paused state
thread_id    = str(uuid.uuid4())
config       = {"configurable": {"thread_id": thread_id}}

hitl_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[read_file, write_file, delete_file, send_email],
    checkpointer=checkpointer,          # HITL requires a checkpointer
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file":  True,    # pause before any write
                "delete_file": True,    # pause before any delete
                "send_email":  True,    # pause before sending email
            }
        )
    ],
    system_prompt=(
        "You are a file and email assistant. "
        "Always read before writing. Ask before sending emails."
    ),
)

# ── Step A: Ask agent to write a file ────────────────────────────────────────
print("🧑 User: Write a summary to /reports/summary.txt")
try:
    result = hitl_agent.invoke(
        {"messages": [{"role": "user", "content": "Write a short summary to /reports/summary.txt"}]},
        config=config,
    )
    # If we reach here the agent finished without hitting HITL
    print(f"🤖 Agent: {result['messages'][-1].content}")

except Exception as interrupt:
    # HITL raises an interrupt — the agent is paused
    print(f"\n⏸️  AGENT PAUSED — waiting for human approval")
    print(f"   Interrupt details: {interrupt}")

    # ── Step B: Human reviews and approves ───────────────────────────────────
    approval = input("\n✅ Approve this action? (yes/no): ").strip().lower()

    if approval == "yes":
        print("\nResuming agent with approval …")
        result = hitl_agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config,             # SAME config/thread_id — resumes from checkpoint
        )
        print(f"🤖 Agent: {result['messages'][-1].content}")
    else:
        print("\nResuming agent with rejection …")
        result = hitl_agent.invoke(
            Command(resume={"decisions": [{"type": "reject", "reason": "Not authorised"}]}),
            config=config,
        )
        print(f"🤖 Agent: {result['messages'][-1].content}")
