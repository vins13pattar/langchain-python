"""
03_multiple_decisions.py
=========================
Demonstrates handling multiple simultaneous HITL interrupts —
when the agent proposes several tool calls at once, each requiring
a separate decision in the correct order.

Concepts covered:
  - Multiple action_requests in a single interrupt
  - Decisions list order must match action_requests order
  - Mixed decisions: approve first, reject second, edit third
  - review_configs per tool (different allowed decisions)
  - Inspecting the full interrupt structure
  - Multi-step conversations with repeated interrupts
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
print("HITL — Multiple Simultaneous Decisions")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════

@tool
def backup_database(db_name: str) -> str:
    """Create a backup of a database."""
    print(f"  [Tool] backup_database({db_name!r})")
    return f"Backup of '{db_name}' created successfully."


@tool
def send_notification(channel: str, message: str) -> str:
    """Send a notification to a Slack/Teams channel."""
    print(f"  [Tool] send_notification(channel={channel!r})")
    return f"Notification sent to '{channel}': {message[:50]}"


@tool
def update_config(key: str, value: str) -> str:
    """Update an application configuration setting."""
    print(f"  [Tool] update_config({key!r}={value!r})")
    return f"Config updated: {key} = {value}"


@tool
def restart_service(service_name: str) -> str:
    """Restart a named service on the server."""
    print(f"  [Tool] restart_service({service_name!r})")
    return f"Service '{service_name}' restarted successfully."


@tool
def read_logs(service_name: str, lines: int = 50) -> str:
    """Read the last N lines of logs for a service (safe)."""
    print(f"  [Tool] read_logs({service_name!r}, {lines})")
    return f"Last {lines} log lines for '{service_name}': [log1, log2, ...]"


# ════════════════════════════════════════════════════════════════════
# AGENT
# ════════════════════════════════════════════════════════════════════

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[backup_database, send_notification, update_config, restart_service, read_logs],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "backup_database":   True,   # all decisions
                "send_notification": {"allowed_decisions": ["approve", "edit", "reject"]},
                "update_config":     True,   # all decisions
                "restart_service":   {"allowed_decisions": ["approve", "reject"]},
                "read_logs":         False,  # safe — no interrupt
            },
            description_prefix="Operations Engineering approval required",
        )
    ],
    checkpointer=MemorySaver(),
    system_prompt=(
        "You are a DevOps assistant. When asked to perform multiple operations, "
        "you may execute several tools in sequence to complete the task."
    ),
)


# ════════════════════════════════════════════════════════════════════
# 1. MULTIPLE SIMULTANEOUS INTERRUPTS — mixed decisions
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Multiple Simultaneous Interrupts ──────────────────────")
print("   (Model proposes backup + notification in one step)")

cfg1 = {"configurable": {"thread_id": "multi-decision-1"}}

result1 = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Perform a maintenance window: backup the 'production' database, "
        "then notify the #ops channel that maintenance is starting."}]},
    config=cfg1,
    version="v2",
)

print(f"\nInterrupts: {len(result1.interrupts)}")
if result1.interrupts:
    value   = result1.interrupts[0].value
    actions = value.get("action_requests", [])
    configs = value.get("review_configs", [])

    print(f"Actions under review ({len(actions)}):")
    for i, (action, cfg) in enumerate(zip(actions, configs)):
        print(f"  [{i}] {action['name']}({action['arguments']}) "
              f"→ allowed: {cfg['allowed_decisions']}")

    # Mixed decisions: approve backup, edit the notification message
    print("\n  → Decision: approve backup, edit notification")
    final1 = agent.invoke(
        Command(resume={
            "decisions": [
                # [0] backup_database → approve as-is
                {"type": "approve"},
                # [1] send_notification → edit the message
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "send_notification",
                        "args": {
                            "channel": actions[1]["arguments"].get("channel", "#ops"),
                            "message": "🔧 Maintenance started at 02:00 UTC. ETA: 30 mins. DBA team on call.",
                        }
                    }
                }
            ]
        }),
        config=cfg1,
        version="v2",
    )
    print(f"\nAfter mixed decisions: {final1.value['messages'][-1].content[:150]}")


# ════════════════════════════════════════════════════════════════════
# 2. THREE SIMULTANEOUS DECISIONS — approve/reject/edit
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Three Simultaneous Decisions ──────────────────────────")

cfg2 = {"configurable": {"thread_id": "multi-decision-3"}}

result2 = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Deploy the new release: "
        "update config 'app_version' to '2.1.0', "
        "restart the 'web-api' service, "
        "and backup the 'staging' database."}]},
    config=cfg2,
    version="v2",
)

print(f"\nInterrupts: {len(result2.interrupts)}")
if result2.interrupts:
    actions = result2.interrupts[0].value.get("action_requests", [])
    print(f"Actions ({len(actions)}):")
    for i, a in enumerate(actions):
        print(f"  [{i}] {a['name']}({a['arguments']})")

    # Three decisions in order matching actions list
    print("\n  → Decisions: edit config, approve restart, reject backup (do it manually)")
    final2 = agent.invoke(
        Command(resume={
            "decisions": [
                # Decisions must be in the SAME ORDER as action_requests
                # [0] update_config → edit: different version
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "update_config",
                        "args": {"key": "app_version", "value": "2.0.9-hotfix"}
                    }
                },
                # [1] restart_service → approve
                {"type": "approve"},
                # [2] backup_database → reject (use backup service instead)
                {
                    "type": "reject",
                    "message": "Do not use this backup tool. Use the automated backup service at 03:00 UTC instead."
                },
            ]
        }),
        config=cfg2,
        version="v2",
    )
    print(f"\nFinal response: {final2.value['messages'][-1].content[:200]}")


# ════════════════════════════════════════════════════════════════════
# 3. SEQUENTIAL INTERRUPTS — multiple rounds of approval
#    First round: backup approved → agent continues, proposes restart
#    Second round: restart is then reviewed
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Sequential Interrupts (multi-round review) ────────────")

cfg3 = {"configurable": {"thread_id": "multi-sequential"}}

# Round 1
r_round1 = agent.invoke(
    {"messages": [{"role": "user", "content":
        "First backup 'prod-db', then restart the 'cache' service."}]},
    config=cfg3,
    version="v2",
)
print(f"\nRound 1 interrupts: {len(r_round1.interrupts)}")
if r_round1.interrupts:
    a1 = r_round1.interrupts[0].value["action_requests"]
    print(f"  Actions: {[a['name'] for a in a1]}")

    # Approve what's in round 1
    r_round2 = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}] * len(a1)}),
        config=cfg3,
        version="v2",
    )
    print(f"Round 2 interrupts: {len(r_round2.interrupts)}")
    if r_round2.interrupts:
        a2 = r_round2.interrupts[0].value["action_requests"]
        print(f"  Actions: {[a['name'] for a in a2]}")

        # Approve the second round too
        final3 = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}] * len(a2)}),
            config=cfg3,
            version="v2",
        )
        print(f"Final: {final3.value['messages'][-1].content[:120]}")
    else:
        print(f"Completed: {r_round2.value['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 4. ALL-REJECT — agent adapts based on feedback
# ════════════════════════════════════════════════════════════════════

print("\n── 4. All-Reject — Agent Adapts to Feedback ─────────────────")

cfg4 = {"configurable": {"thread_id": "multi-all-reject"}}

r4 = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Update config 'debug_mode' to 'true' and restart the 'api' service."}]},
    config=cfg4,
    version="v2",
)

if r4.interrupts:
    actions = r4.interrupts[0].value["action_requests"]
    print(f"Proposed: {[a['name'] for a in actions]}")

    # Reject both with clear guidance
    final4 = agent.invoke(
        Command(resume={
            "decisions": [
                {"type": "reject",
                 "message": "Never enable debug_mode in production environments."},
                {"type": "reject",
                 "message": "Service restarts require a maintenance window ticket first."},
            ]
        }),
        config=cfg4,
        version="v2",
    )
    print(f"Agent adapted: {final4.value['messages'][-1].content[:200]}")

print("\n" + "═" * 60)
print("Multiple Decisions Key Points:")
print("  - Decisions list MUST match action_requests order")
print("  - Each action gets its own decision from the list")
print("  - Mixed decisions allowed (approve some, reject others)")
print("  - Agent adapts when actions are rejected with feedback")
print("  - Sequential interrupts: each round waits for decisions")
print("═" * 60)
print("\n✅ Multiple decisions demo complete.")
