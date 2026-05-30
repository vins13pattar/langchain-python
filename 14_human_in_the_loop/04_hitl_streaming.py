"""
04_hitl_streaming.py
=====================
Demonstrates streaming with HITL — using stream() instead of invoke()
to get real-time LLM token output and detect interrupts as they occur.

Concepts covered:
  - agent.stream() with stream_mode=["updates", "messages"]
  - version="v2" for unified streaming format
  - chunk["type"] == "messages" for LLM tokens
  - chunk["type"] == "updates" with "__interrupt__" for HITL pauses
  - Streaming resume after human decision
  - Accumulating streamed tokens into full responses
  - Combining token streaming with interrupt detection
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
print("HITL — Streaming with stream_mode v2")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════

@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query against the production database."""
    print(f"\n  [Tool] execute_sql({query!r})")
    return f"Query executed: {query} → 15 rows affected."


@tool
def deploy_code(branch: str, environment: str) -> str:
    """Deploy code from a branch to an environment."""
    print(f"\n  [Tool] deploy_code(branch={branch!r}, env={environment!r})")
    return f"Deployed '{branch}' to {environment} successfully."


@tool
def send_alert(severity: str, message: str) -> str:
    """Send an operational alert to the on-call team."""
    print(f"\n  [Tool] send_alert(severity={severity!r})")
    return f"Alert sent: [{severity.upper()}] {message}"


# ════════════════════════════════════════════════════════════════════
# AGENT
# ════════════════════════════════════════════════════════════════════

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[execute_sql, deploy_code, send_alert],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "execute_sql":  True,
                "deploy_code":  {"allowed_decisions": ["approve", "reject"]},
                "send_alert":   {"allowed_decisions": ["approve", "edit", "reject"]},
            },
            description_prefix="Production operation pending human approval",
        )
    ],
    checkpointer=MemorySaver(),
    system_prompt=(
        "You are a production operations assistant. "
        "All database and deployment operations require human approval."
    ),
)


# ════════════════════════════════════════════════════════════════════
# HELPER — stream until interrupt or completion
# ════════════════════════════════════════════════════════════════════

def stream_until_interrupt(config: dict, input_data) -> list:
    """
    Stream the agent and collect tokens + detect interrupts.
    Returns list of interrupts found (may be empty if agent completed).
    """
    interrupts_found = []
    current_response = []

    print("\n  [Streaming] ", end="", flush=True)

    for chunk in agent.stream(
        input_data,
        config=config,
        stream_mode=["updates", "messages"],
        version="v2",
    ):
        chunk_type = chunk.get("type")

        if chunk_type == "messages":
            # LLM token chunk
            token, metadata = chunk["data"]
            if token.content:
                text = token.content if isinstance(token.content, str) else str(token.content)
                print(text, end="", flush=True)
                current_response.append(text)

        elif chunk_type == "updates":
            update_data = chunk.get("data", {})
            if "__interrupt__" in update_data:
                # HITL interrupt detected
                interrupts_found = update_data["__interrupt__"]
                print("\n  [INTERRUPTED]")

    if not interrupts_found:
        print()  # newline after streaming completes

    return interrupts_found


# ════════════════════════════════════════════════════════════════════
# 1. STREAM UNTIL INTERRUPT, THEN RESUME
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Stream → Interrupt → Approve ─────────────────────────")

cfg1 = {"configurable": {"thread_id": "stream-approve"}}

# Initial stream
interrupts = stream_until_interrupt(
    config=cfg1,
    input_data={
        "messages": [{"role": "user", "content":
            "Clean up the database: delete all sessions older than 7 days."}]
    }
)

if interrupts:
    interrupt = interrupts[0]
    actions   = interrupt.value.get("action_requests", [])
    print(f"\n  Interrupt — action: {actions[0]['name']}")
    print(f"  SQL:       {actions[0]['arguments'].get('query', '')[:80]}")

    print("\n  → Human approves. Resuming with streaming...")
    post_interrupts = stream_until_interrupt(
        config=cfg1,
        input_data=Command(resume={"decisions": [{"type": "approve"}]}),
    )
    if not post_interrupts:
        print("\n  ✅ Agent completed after approval.")


# ════════════════════════════════════════════════════════════════════
# 2. STREAM → INTERRUPT → REJECT → STREAM CONTINUATION
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Stream → Interrupt → Reject → Continue ────────────────")

cfg2 = {"configurable": {"thread_id": "stream-reject"}}

interrupts2 = stream_until_interrupt(
    config=cfg2,
    input_data={
        "messages": [{"role": "user", "content":
            "Deploy the main branch to production right now."}]
    }
)

if interrupts2:
    actions2 = interrupts2[0].value.get("action_requests", [])
    print(f"\n  Interrupt — {actions2[0]['name']}({actions2[0]['arguments']})")

    print("\n  → Human rejects deployment. Resuming...")
    post2 = stream_until_interrupt(
        config=cfg2,
        input_data=Command(resume={
            "decisions": [{
                "type":    "reject",
                "message": "Cannot deploy directly to production. Create a staging PR first.",
            }]
        }),
    )
    print("\n  ✅ Agent acknowledged rejection.")


# ════════════════════════════════════════════════════════════════════
# 3. STREAM → INTERRUPT → EDIT → FINAL STREAM
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Stream → Interrupt → Edit → Final Stream ──────────────")

cfg3 = {"configurable": {"thread_id": "stream-edit"}}

interrupts3 = stream_until_interrupt(
    config=cfg3,
    input_data={
        "messages": [{"role": "user", "content":
            "Send a critical alert saying the payment service is down."}]
    }
)

if interrupts3:
    actions3 = interrupts3[0].value.get("action_requests", [])
    print(f"\n  Interrupt — alert: {actions3[0]['arguments']}")

    print("\n  → Human edits severity from 'critical' to 'warning'...")
    post3 = stream_until_interrupt(
        config=cfg3,
        input_data=Command(resume={
            "decisions": [{
                "type": "edit",
                "edited_action": {
                    "name": "send_alert",
                    "args": {
                        "severity": "warning",    # downgrade severity
                        "message":  actions3[0]["arguments"].get("message", ""),
                    }
                }
            }]
        }),
    )
    print("\n  ✅ Alert sent with edited severity.")


# ════════════════════════════════════════════════════════════════════
# 4. INVOKE vs STREAM — comparison
# ════════════════════════════════════════════════════════════════════

def show_comparison():
    print("\n── 4. invoke() vs stream() Comparison ───────────────────────")
    print("""
  invoke() — batch mode:
    result = agent.invoke(input, config=config, version="v2")
    # result.interrupts — list of interrupts
    # result.value — final state dict
    # Use for: simple scripts, batch processing

  stream() — streaming mode:
    for chunk in agent.stream(
        input,
        config=config,
        stream_mode=["updates", "messages"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]   # LLM token
            print(token.content, end="")
        elif chunk["type"] == "updates":
            if "__interrupt__" in chunk["data"]:
                interrupts = chunk["data"]["__interrupt__"]
    # Use for: real-time UIs, token-by-token display

  Resuming is the same in both modes:
    agent.invoke(Command(resume={...}), config=config, version="v2")
    agent.stream(Command(resume={...}), config=config, ...)
    """)


show_comparison()

print("═" * 60)
print("Streaming HITL Key Points:")
print("  stream_mode=['updates','messages'] + version='v2'")
print("  chunk['type'] == 'messages'  → LLM token: (token, metadata)")
print("  chunk['type'] == 'updates'   → state update dict")
print("  '__interrupt__' in updates   → HITL pause detected")
print("  Resume identically to invoke() using Command(resume={...})")
print("═" * 60)
print("\n✅ HITL streaming demo complete.")
