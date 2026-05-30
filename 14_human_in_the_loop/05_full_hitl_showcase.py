"""
05_full_hitl_showcase.py
=========================
Production-ready showcase: a SECURE FINANCIAL OPERATIONS AGENT with
multi-layer HITL controls, streaming, custom policies, and audit logging.

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │           Secure Financial Operations Agent                      │
  ├─────────────────────────────────────────────────────────────────┤
  │  Tools (risk-tiered):                                            │
  │    read_balance      — safe, no approval                        │
  │    transfer_funds    — HIGH RISK: approve/reject only            │
  │    execute_trade     — HIGH RISK: all decisions                  │
  │    generate_report   — MEDIUM: approve/edit/reject               │
  │    ask_user          — CLARIFICATION: respond only               │
  │    send_alert        — MEDIUM: approve/edit/reject               │
  │                                                                 │
  │  HITL Policies:                                                  │
  │    transfer_funds    — only approve/reject, custom description   │
  │    execute_trade     — all decisions, requires review            │
  │    generate_report   — approve/edit/reject (can modify params)   │
  │    ask_user          — respond only (human IS the answer)        │
  │    send_alert        — approve/edit/reject                       │
  │                                                                 │
  │  Life-cycle Middleware:                                          │
  │    @before_model     — audit every LLM call to store            │
  │    @dynamic_prompt   — trader role + risk-level system prompt   │
  │                                                                 │
  │  Scenarios:                                                      │
  │    A. Fund transfer — approve flow                               │
  │    B. Trade execution — edit to reduce quantity                  │
  │    C. Report generation — edit parameters                        │
  │    D. Clarification — respond with human answer                 │
  │    E. Multi-operation — mixed decisions in one round             │
  │    F. Streaming — real-time token output with interrupt detect   │
  └─────────────────────────────────────────────────────────────────┘
"""

import os
import time
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    dynamic_prompt,
    ModelRequest,
    before_model,
)
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

load_dotenv()

print("=" * 60)
print("Secure Financial Operations Agent — Full HITL Showcase")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# CONTEXT SCHEMA
# ════════════════════════════════════════════════════════════════════

@dataclass
class TraderCtx:
    trader_id:   str
    role:        str    # "junior", "senior", "risk_officer"
    desk:        str    # "equity", "fixed_income", "fx"
    daily_limit: float  # max single-trade value


# ════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════

@tool
def read_balance(account_id: str) -> str:
    """Read the current balance of an account (safe — no approval needed)."""
    balances = {"ACC-001": "$1,250,000", "ACC-002": "$450,000", "MAIN": "$8,500,000"}
    balance  = balances.get(account_id, "$0")
    print(f"  [Tool] read_balance({account_id!r}) = {balance}")
    return f"Account {account_id} balance: {balance}"


@tool
def transfer_funds(from_account: str, to_account: str, amount: float, currency: str = "USD") -> str:
    """Transfer funds between accounts (requires approval)."""
    print(f"  [Tool] transfer_funds({from_account}→{to_account}, {amount} {currency})")
    return f"Transferred {currency} {amount:,.2f} from {from_account} to {to_account}. Ref: TXN-{int(time.time())}"


@tool
def execute_trade(symbol: str, action: str, quantity: int, price: float) -> str:
    """Execute a securities trade (requires approval)."""
    value = quantity * price
    print(f"  [Tool] execute_trade({action} {quantity}x {symbol} @ {price})")
    return f"Trade executed: {action} {quantity} {symbol} @ ${price:.2f} = ${value:,.2f}. OrderID: ORD-{int(time.time())}"


@tool
def generate_report(report_type: str, period: str, include_pnl: bool = True) -> str:
    """Generate a financial report (requires approval before distribution)."""
    print(f"  [Tool] generate_report({report_type}, {period})")
    return (
        f"Report generated: {report_type} for {period}. "
        f"{'P&L included.' if include_pnl else 'P&L excluded.'} "
        f"Ready for distribution."
    )


@tool
def ask_user(question: str) -> str:
    """Ask the human operator a clarifying question."""
    print(f"  [Tool] ask_user({question!r}) — awaiting human response")
    return "Fallback (not used when decision is 'respond')"


@tool
def send_alert(recipient: str, severity: str, message: str) -> str:
    """Send a risk alert to a recipient (requires review)."""
    print(f"  [Tool] send_alert(to={recipient!r}, severity={severity!r})")
    return f"Alert sent to {recipient}: [{severity.upper()}] {message}"


# ════════════════════════════════════════════════════════════════════
# LANGCHAIN MIDDLEWARE
# ════════════════════════════════════════════════════════════════════

@dynamic_prompt
def trading_prompt(request: ModelRequest) -> str:
    """Risk-aware, role-specific system prompt."""
    ctx = request.runtime.context
    role_note = {
        "junior":       "You require approval for all trades and transfers.",
        "senior":       "You may suggest larger trades but they still require approval.",
        "risk_officer": "You have full authority but all actions are logged.",
    }.get(ctx.role, "")
    return (
        f"You are a financial operations assistant for the {ctx.desk} desk. "
        f"Trader: {ctx.trader_id}, Role: {ctx.role}. {role_note} "
        f"Daily limit: ${ctx.daily_limit:,.0f}. "
        f"Always be precise with amounts and symbols."
    )


_audit_store = InMemoryStore()
_model_calls = []

@before_model
def model_audit(state: AgentState, runtime: Runtime[TraderCtx]) -> dict | None:
    """Record every LLM call in the audit trail."""
    entry = {
        "trader": runtime.context.trader_id,
        "role":   runtime.context.role,
        "desk":   runtime.context.desk,
        "run_id": runtime.execution_info.run_id[:8],
        "msgs":   len(state.get("messages", [])),
        "ts":     time.time(),
    }
    _model_calls.append(entry)

    store   = runtime.store
    user_id = runtime.context.trader_id
    if store:
        existing = store.get(("audit",), user_id)
        log      = existing.value.get("calls", []) if existing else []
        log.append(entry)
        store.put(("audit",), user_id, {"calls": log})
    return None


# ════════════════════════════════════════════════════════════════════
# BUILD AGENT
# ════════════════════════════════════════════════════════════════════

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[read_balance, transfer_funds, execute_trade,
           generate_report, ask_user, send_alert],
    context_schema=TraderCtx,
    store=_audit_store,
    checkpointer=MemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "read_balance":   False,   # Safe — no interrupt
                "transfer_funds": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "⚠️ FUNDS TRANSFER — Requires compliance approval",
                },
                "execute_trade": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "📈 TRADE ORDER — Risk desk review required",
                },
                "generate_report": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "📊 REPORT — Review before distribution",
                },
                "ask_user": {
                    "allowed_decisions": ["respond"],
                    "description": "❓ CLARIFICATION — Operator input required",
                },
                "send_alert": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "🚨 ALERT — Review before sending",
                },
            },
            description_prefix="Financial Operations Control",
        ),
        trading_prompt,
        model_audit,
    ],
    system_prompt="You are a financial operations assistant.",
)


# ════════════════════════════════════════════════════════════════════
# SCENARIO A — Transfer: approve
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO A — Fund Transfer (approve)")
print("─" * 60)

ctx_a = TraderCtx(trader_id="TDR-001", role="senior", desk="fx", daily_limit=500_000)
cfg_a = {"configurable": {"thread_id": "hitl-transfer"}}

r_a = agent.invoke(
    {"messages": [{"role": "user",
                   "content": "Transfer $50,000 USD from ACC-001 to ACC-002."}]},
    context=ctx_a, config=cfg_a, version="v2",
)

if r_a.interrupts:
    action = r_a.interrupts[0].value["action_requests"][0]
    print(f"\nProposed: {action['name']}({action['arguments']})")
    print("Human review: APPROVED ✅")
    final_a = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        context=ctx_a, config=cfg_a, version="v2",
    )
    print(f"Result: {final_a.value['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# SCENARIO B — Trade: edit (reduce quantity)
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO B — Trade Execution (edit quantity)")
print("─" * 60)

ctx_b = TraderCtx(trader_id="TDR-002", role="junior", desk="equity", daily_limit=100_000)
cfg_b = {"configurable": {"thread_id": "hitl-trade"}}

r_b = agent.invoke(
    {"messages": [{"role": "user",
                   "content": "Buy 1000 shares of AAPL at $190.00."}]},
    context=ctx_b, config=cfg_b, version="v2",
)

if r_b.interrupts:
    action_b = r_b.interrupts[0].value["action_requests"][0]
    orig_qty = action_b["arguments"].get("quantity", 1000)
    print(f"\nProposed: {action_b['name']}({action_b['arguments']})")
    print(f"Risk desk: Exceeds daily limit — reducing to 200 shares ✏️")
    final_b = agent.invoke(
        Command(resume={
            "decisions": [{
                "type": "edit",
                "edited_action": {
                    "name": "execute_trade",
                    "args": {**action_b["arguments"], "quantity": 200}
                }
            }]
        }),
        context=ctx_b, config=cfg_b, version="v2",
    )
    print(f"Result: {final_b.value['messages'][-1].content[:150]}")


# ════════════════════════════════════════════════════════════════════
# SCENARIO C — Report: reject (wrong period)
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO C — Report Generation (reject + re-request)")
print("─" * 60)

ctx_c = TraderCtx(trader_id="TDR-001", role="senior", desk="fx", daily_limit=500_000)
cfg_c = {"configurable": {"thread_id": "hitl-report"}}

r_c = agent.invoke(
    {"messages": [{"role": "user",
                   "content": "Generate a trading report for Q3 2024, include P&L."}]},
    context=ctx_c, config=cfg_c, version="v2",
)

if r_c.interrupts:
    print(f"\nProposed: {r_c.interrupts[0].value['action_requests'][0]['arguments']}")
    print("Risk officer: Wrong quarter — Q4 is needed ❌")
    final_c = agent.invoke(
        Command(resume={
            "decisions": [{
                "type":    "reject",
                "message": "Wrong period. Generate for Q4 2024, not Q3. Also exclude P&L for external distribution.",
            }]
        }),
        context=ctx_c, config=cfg_c, version="v2",
    )
    print(f"After rejection: {final_c.value['messages'][-1].content[:200]}")


# ════════════════════════════════════════════════════════════════════
# SCENARIO D — Clarification via respond
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO D — Clarification (respond decision)")
print("─" * 60)

ctx_d = TraderCtx(trader_id="TDR-003", role="senior", desk="fixed_income", daily_limit=1_000_000)
cfg_d = {"configurable": {"thread_id": "hitl-clarify"}}

r_d = agent.invoke(
    {"messages": [{"role": "user",
                   "content": "I need to execute a bond trade but I'm not sure of the details yet."}]},
    context=ctx_d, config=cfg_d, version="v2",
)

if r_d.interrupts:
    action_d = r_d.interrupts[0].value["action_requests"][0]
    print(f"\nAgent asking: {action_d['arguments'].get('question', action_d['arguments'])}")
    final_d = agent.invoke(
        Command(resume={
            "decisions": [{
                "type":    "respond",
                "message": "Buy 500 units of US Treasury 10Y at 4.25% yield.",
            }]
        }),
        context=ctx_d, config=cfg_d, version="v2",
    )
    print(f"After respond: {final_d.value['messages'][-1].content[:180]}")


# ════════════════════════════════════════════════════════════════════
# SCENARIO E — Streaming with real-time token output
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO E — Streaming with Interrupt Detection")
print("─" * 60)

ctx_e = TraderCtx(trader_id="TDR-001", role="senior", desk="equity", daily_limit=500_000)
cfg_e = {"configurable": {"thread_id": "hitl-streaming"}}

print("\nStreaming: ", end="", flush=True)
found_interrupts = []

for chunk in agent.stream(
    {"messages": [{"role": "user",
                   "content": "Read my ACC-001 balance, then send a low-risk alert to risk@bank.com."}]},
    context=ctx_e,
    config=cfg_e,
    stream_mode=["updates", "messages"],
    version="v2",
):
    if chunk.get("type") == "messages":
        token, _ = chunk["data"]
        if token.content:
            text = token.content if isinstance(token.content, str) else ""
            print(text, end="", flush=True)
    elif chunk.get("type") == "updates":
        if "__interrupt__" in chunk.get("data", {}):
            found_interrupts = chunk["data"]["__interrupt__"]
            print("\n  [INTERRUPT DETECTED]")

if found_interrupts:
    action_e = found_interrupts[0].value["action_requests"][0]
    print(f"\nAlert to review: {action_e['arguments']}")
    print("Operator: Upgrading severity to 'medium' ✏️")

    for chunk in agent.stream(
        Command(resume={
            "decisions": [{
                "type": "edit",
                "edited_action": {
                    "name": "send_alert",
                    "args": {
                        **action_e["arguments"],
                        "severity": "medium",
                    }
                }
            }]
        }),
        context=ctx_e,
        config=cfg_e,
        stream_mode=["updates", "messages"],
        version="v2",
    ):
        if chunk.get("type") == "messages":
            token, _ = chunk["data"]
            if token.content:
                print(token.content if isinstance(token.content, str) else "", end="", flush=True)

    print("\n  ✅ Streaming resume complete.")


# ════════════════════════════════════════════════════════════════════
# AUDIT SUMMARY
# ════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("Audit Summary:")
for trader_id in ("TDR-001", "TDR-002", "TDR-003"):
    audit = _audit_store.get(("audit",), trader_id)
    if audit:
        calls = audit.value.get("calls", [])
        print(f"  {trader_id}: {len(calls)} LLM calls logged")

print(f"\nTotal model calls across all traders: {len(_model_calls)}")

print("\n" + "═" * 60)
print("Full HITL Showcase — Components Used:")
print("  HumanInTheLoopMiddleware — risk-tiered interrupt_on policy")
print("  @dynamic_prompt          — role + desk system prompt")
print("  @before_model audit      — persistent model call log")
print("  Scenarios A–E            — approve, edit, reject, respond, stream")
print("  MemorySaver              — interrupt state persistence")
print("  InMemoryStore            — cross-session audit trail")
print("═" * 60)
print("\n✅ Full HITL showcase complete.")
