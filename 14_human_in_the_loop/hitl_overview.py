"""
hitl_overview.py — Human-in-the-Loop (HITL): all key concepts in one file
Covers: HumanInTheLoopMiddleware setup, all 4 decision types (approve/edit/reject/respond),
        multiple decisions, HITL streaming, version="v2" GraphOutput
"""

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

def section(title): print(f"\n{'─'*55}\n{title}\n{'─'*55}")


# ── Tools at various risk levels ──────────────────────────────────
@tool
def read_data(table: str) -> str:
    """Read data from a database table (safe). Args: table: Table name."""
    return f"Data from '{table}': [row1, row2, row3]"

@tool
def execute_sql(query: str) -> str:
    """Execute a SQL statement. Args: query: SQL query string."""
    print(f"  [Tool] execute_sql: {query[:60]}")
    return f"SQL executed: {query} → 42 rows affected."

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. Args: to, subject, body."""
    print(f"  [Tool] send_email to={to!r}")
    return f"Email sent to {to} — subject: '{subject}'."

@tool
def delete_records(table: str, condition: str) -> str:
    """Delete records from a table. Args: table, condition."""
    print(f"  [Tool] delete_records: {table!r} WHERE {condition}")
    return f"Deleted from '{table}' WHERE {condition}."

@tool
def archive_data(table: str, days_old: int = 30) -> str:
    """Archive old data. Args: table, days_old."""
    print(f"  [Tool] archive_data: {table!r}, {days_old}d")
    return f"Archived records from '{table}' older than {days_old} days."

@tool
def ask_user(question: str) -> str:
    """Ask the human user a clarifying question. Args: question."""
    return "Tool fallback — should not appear when 'respond' is used."

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file. Args: path, content."""
    print(f"  [Tool] write_file to {path!r}")
    return f"Written {len(content)} bytes to {path}."


# ════════════════════════════════════════════════════════════════════
# 1. BASIC SETUP — interrupt_on configuration
# ════════════════════════════════════════════════════════════════════
section("1. BASIC SETUP")

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[read_data, execute_sql, write_file, send_email, delete_records, archive_data, ask_user],
    checkpointer=MemorySaver(),   # REQUIRED — persists state across interrupt
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file":   True,          # all decisions: approve/edit/reject/respond
                "execute_sql":  {"allowed_decisions": ["approve", "reject"]},  # no edit for SQL
                "send_email":   {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "Email pending review before sending",
                },
                "delete_records": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "DANGER: Permanent data deletion",
                },
                "archive_data":   {"allowed_decisions": ["approve", "edit", "reject"]},
                "ask_user":       {"allowed_decisions": ["respond"]},  # human IS the answer
                "read_data":  False,          # auto-approve, never interrupt
            },
            description_prefix="Tool execution pending approval",
        )
    ],
    system_prompt="You are a database and file management assistant. Use ask_user for clarifications.",
)

# Safe read — no interrupt
cfg = {"configurable": {"thread_id": "hitl-read"}}
r = agent.invoke({"messages": [{"role": "user", "content": "Read data from the users table."}]}, config=cfg, version="v2")
print(f"Safe (no interrupt): {r.interrupts}  result: {r.value['messages'][-1].content[:80]}")


# ════════════════════════════════════════════════════════════════════
# 2. DECISION TYPE: APPROVE — execute tool as-is
# ════════════════════════════════════════════════════════════════════
section("2. APPROVE")

cfg_ap = {"configurable": {"thread_id": "hitl-approve"}}
r = agent.invoke(
    {"messages": [{"role": "user", "content": "Write a config file to /etc/app/config.yaml with content 'debug: true'"}]},
    config=cfg_ap, version="v2"
)
if r.interrupts:
    action = r.interrupts[0].value["action_requests"][0]
    print(f"Paused: {action['name']}({action['arguments']})")
    final = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=cfg_ap, version="v2")
    print(f"Approved: {final.value['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 3. DECISION TYPE: EDIT — modify args before execution
# ════════════════════════════════════════════════════════════════════
section("3. EDIT (modify args or swap tool)")

cfg_ed = {"configurable": {"thread_id": "hitl-edit"}}
r = agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to all-hands@co.com about the company offsite."}]},
    config=cfg_ed, version="v2"
)
if r.interrupts:
    orig = r.interrupts[0].value["action_requests"][0]["arguments"]
    print(f"Original: {orig}")
    final = agent.invoke(
        Command(resume={"decisions": [{
            "type": "edit",
            "edited_action": {
                "name": "send_email",
                "args": {
                    "to":      "leadership@co.com",       # narrower audience
                    "subject": orig.get("subject", "Offsite"),
                    "body":    orig.get("body", "") + "\n[Reviewed by comms team]",
                }
            }
        }]}),
        config=cfg_ed, version="v2"
    )
    print(f"After edit: {final.value['messages'][-1].content[:100]}")

# Edit — change to a different tool entirely (delete → archive)
cfg_ed2 = {"configurable": {"thread_id": "hitl-edit-tool"}}
r = agent.invoke(
    {"messages": [{"role": "user", "content": "Delete records from transactions older than 60 days."}]},
    config=cfg_ed2, version="v2"
)
if r.interrupts:
    print(f"Proposed tool: {r.interrupts[0].value['action_requests'][0]['name']}")
    final = agent.invoke(
        Command(resume={"decisions": [{"type": "edit", "edited_action": {"name": "archive_data", "args": {"table": "transactions", "days_old": 60}}}]}),
        config=cfg_ed2, version="v2"
    )
    print(f"After tool swap: {final.value['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 4. DECISION TYPE: REJECT — block + send feedback to agent
# ════════════════════════════════════════════════════════════════════
section("4. REJECT (with feedback message)")

cfg_re = {"configurable": {"thread_id": "hitl-reject"}}
r = agent.invoke(
    {"messages": [{"role": "user", "content": "Delete all records from users table where status = 'inactive'."}]},
    config=cfg_re, version="v2"
)
if r.interrupts:
    action = r.interrupts[0].value["action_requests"][0]
    print(f"Proposed: {action['name']}({action['arguments']})")
    final = agent.invoke(
        Command(resume={"decisions": [{
            "type": "reject",
            "message": "Do NOT delete. Set status='archived' with an UPDATE instead."
        }]}),
        config=cfg_re, version="v2"
    )
    print(f"After rejection: {final.value['messages'][-1].content[:180]}")


# ════════════════════════════════════════════════════════════════════
# 5. DECISION TYPE: RESPOND — human reply IS the tool result
# ════════════════════════════════════════════════════════════════════
section("5. RESPOND (human answer replaces tool execution)")

cfg_rs = {"configurable": {"thread_id": "hitl-respond"}}
r = agent.invoke(
    {"messages": [{"role": "user", "content": "I need to archive some data. Ask me which table."}]},
    config=cfg_rs, version="v2"
)
if r.interrupts:
    question = r.interrupts[0].value["action_requests"][0]["arguments"]
    print(f"Agent asks: {question}")
    final = agent.invoke(
        Command(resume={"decisions": [{"type": "respond", "message": "Archive the 'audit_logs' table for records older than 90 days."}]}),
        config=cfg_rs, version="v2"
    )
    print(f"After respond: {final.value['messages'][-1].content[:180]}")


# ════════════════════════════════════════════════════════════════════
# 6. MULTIPLE TOOL CALLS IN ONE TURN — parallel decisions
# ════════════════════════════════════════════════════════════════════
section("6. MULTIPLE DECISIONS (parallel tool calls)")

cfg_multi = {"configurable": {"thread_id": "hitl-multi"}}
r = agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to alice@co.com about the meeting, and execute SQL: SELECT * FROM orders."}]},
    config=cfg_multi, version="v2"
)
if r.interrupts:
    actions = r.interrupts[0].value["action_requests"]
    print(f"Paused with {len(actions)} tool(s): {[a['name'] for a in actions]}")
    # One decision per tool call, in order
    decisions = [{"type": "approve"} for _ in actions]
    final = agent.invoke(Command(resume={"decisions": decisions}), config=cfg_multi, version="v2")
    print(f"After approving all: {final.value['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 7. HITL WITH STREAMING — inspect interrupt during stream
# ════════════════════════════════════════════════════════════════════
section("7. HITL WITH STREAMING")

cfg_st = {"configurable": {"thread_id": "hitl-stream"}}
interrupted = False
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Write 'hello world' to /tmp/test.txt"}]},
    config=cfg_st, stream_mode="updates", version="v2"
):
    if chunk.get("type") == "interrupt":
        interrupted = True
        action = chunk["data"]["action_requests"][0]
        print(f"Interrupt during stream: {action['name']}({action['arguments']})")

if interrupted:
    final = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=cfg_st, version="v2")
    print(f"Resumed: {final.value['messages'][-1].content[:80]}")


print("""
HITL Quick Reference:
  HumanInTheLoopMiddleware(interrupt_on={
    "tool_name": True,              # all decisions allowed
    "tool_name": False,             # never interrupt (auto-approve)
    "tool_name": {                  # per-tool config
        "allowed_decisions": ["approve", "edit", "reject"],
        "description": "Custom message shown to human reviewer",
    },
  })
  
  Requires: checkpointer=MemorySaver()
  Returns:  version="v2" → GraphOutput with .interrupts + .value
  Resume:   Command(resume={"decisions": [{...}]})

  Decision types:
    approve  → {"type": "approve"}
    edit     → {"type": "edit", "edited_action": {"name": "...", "args": {...}}}
    reject   → {"type": "reject", "message": "reason"}
    respond  → {"type": "respond", "message": "human answer"}
""")
