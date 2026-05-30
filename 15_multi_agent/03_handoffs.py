"""
03_handoffs.py
===============
Demonstrates the HANDOFFS multi-agent pattern — behavior changes
dynamically based on state. Tools update a state variable (current_step)
that determines the agent's system prompt and tools each turn.

Concepts covered:
  - State-driven behavior via current_step state variable
  - Tools returning Command(update={...}) to trigger transitions
  - @dynamic_prompt reading state to adjust system prompt per turn
  - @wrap_model_call for state-dependent tool configuration
  - Multi-stage conversational workflows (e.g., customer support)
  - Sequential constraints — unlock capabilities after preconditions
  - Agent-to-agent handoff pattern (transfer_to_X tools)
  - Persistent state across conversation turns

Key difference from Subagents:
  - Subagents:  routing through main agent, subagents are stateless
  - Handoffs:   state variable changes BEHAVIOR/ROUTING, direct user interaction
"""

from typing import Annotated
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest, wrap_model_call, ModelCall
from langchain.tools import tool, ToolRuntime, InjectedToolCallId
from langchain.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

print("=" * 60)
print("Handoffs Pattern")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# PART 1: SEQUENTIAL CUSTOMER SUPPORT WORKFLOW
# Steps: greet → verify_identity → diagnose → resolve
# Each step unlocks the next via Command(update={"current_step": ...})
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Sequential Support Workflow (multi-step) ─────────────")

class SupportState(AgentState):
    current_step: str    # Controls active behavior
    customer_id:  str    # Collected in step 1
    issue_type:   str    # Classified in step 2
    warranty_status: str # Checked in step 2


# Step-specific system prompts
STEP_PROMPTS = {
    "greet": (
        "You are a friendly customer support agent. "
        "Start by warmly greeting the customer and collecting their customer ID. "
        "Use the record_customer_id tool once you have it."
    ),
    "diagnose": (
        "You are a technical support diagnostician. "
        "Ask the customer to describe their issue in detail. "
        "Classify it using record_issue_classification once you understand it."
    ),
    "resolve": (
        "You are a resolution specialist. "
        "Based on the issue type, provide a specific, actionable solution. "
        "Offer escalation if the issue can't be resolved in this session."
    ),
    "escalate": (
        "You are an escalation agent. "
        "Summarize the issue and inform the customer they're being connected to a senior specialist. "
        "Use the close_ticket tool when done."
    ),
}

STEP_TOOL_SETS = {
    "greet":    ["record_customer_id"],
    "diagnose": ["record_issue_classification", "check_warranty"],
    "resolve":  ["apply_solution", "transfer_to_escalation"],
    "escalate": ["close_ticket"],
}


# ─── TOOLS ─────────────────────────────────────────────────────────

@tool
def record_customer_id(
    customer_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Record the customer's ID and advance to the diagnose step."""
    print(f"  [Step] record_customer_id → '{customer_id}', advancing to 'diagnose'")
    return Command(update={
        "customer_id":  customer_id,
        "current_step": "diagnose",
        "messages": [ToolMessage(
            content=f"Customer ID {customer_id} recorded. Moving to diagnosis.",
            tool_call_id=tool_call_id,
        )]
    })


@tool
def check_warranty(customer_id: str) -> str:
    """Check warranty status for a customer."""
    # Simulated lookup
    status = "in_warranty" if customer_id.startswith("PREM") else "out_of_warranty"
    print(f"  [Tool] check_warranty({customer_id!r}) = {status}")
    return f"Warranty status for {customer_id}: {status}"


@tool
def record_issue_classification(
    issue_type: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Classify the issue and advance to the resolve step."""
    print(f"  [Step] record_issue_classification → '{issue_type}', advancing to 'resolve'")
    return Command(update={
        "issue_type":   issue_type,
        "current_step": "resolve",
        "messages": [ToolMessage(
            content=f"Issue classified as '{issue_type}'. Moving to resolution.",
            tool_call_id=tool_call_id,
        )]
    })


@tool
def apply_solution(solution: str) -> str:
    """Apply a resolution solution for the customer's issue."""
    print(f"  [Tool] apply_solution: {solution[:60]}")
    return f"Solution applied: {solution}. Confirmation: TKT-{hash(solution) % 10000:04d}"


@tool
def transfer_to_escalation(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfer the case to the escalation team."""
    print(f"  [Step] transfer_to_escalation → reason: {reason[:60]}")
    return Command(update={
        "current_step": "escalate",
        "messages": [ToolMessage(
            content=f"Escalating: {reason}",
            tool_call_id=tool_call_id,
        )]
    })


@tool
def close_ticket(resolution_notes: str) -> str:
    """Close the support ticket with resolution notes."""
    print(f"  [Tool] close_ticket: {resolution_notes[:60]}")
    return f"Ticket closed. Notes: {resolution_notes}"


# ─── DYNAMIC PROMPT — reads current_step from state ────────────────

@dynamic_prompt
def support_prompt(request: ModelRequest) -> str:
    state = request.runtime.state
    step  = state.get("current_step", "greet")
    return STEP_PROMPTS.get(step, STEP_PROMPTS["greet"])


# ─── WRAP_MODEL_CALL — filter tools based on current_step ──────────

ALL_SUPPORT_TOOLS = [
    record_customer_id, check_warranty, record_issue_classification,
    apply_solution, transfer_to_escalation, close_ticket,
]

@wrap_model_call
def filter_tools_by_step(call: ModelCall, runtime) -> ModelCall:
    """Only pass tools relevant to the current step to the model."""
    step         = runtime.state.get("current_step", "greet")
    allowed      = STEP_TOOL_SETS.get(step, [])
    filtered     = [t for t in call.tools if t.name in allowed]
    print(f"  [ToolFilter] step={step!r} → {[t.name for t in filtered]}")
    return call.replace(tools=filtered)


# ─── AGENT ─────────────────────────────────────────────────────────

support_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=ALL_SUPPORT_TOOLS,
    middleware=[support_prompt, filter_tools_by_step],
    checkpointer=MemorySaver(),
    system_prompt="You are a customer support agent.",   # overridden by dynamic_prompt
)

# ─── SIMULATE MULTI-TURN CONVERSATION ──────────────────────────────

config = {"configurable": {"thread_id": "handoff-support-1"}}

turns = [
    ("greet",    "Hi, I need help with my device."),
    ("id",       "My customer ID is PREM-4521."),
    ("diagnose", "The screen keeps flickering when I open heavy apps."),
    ("resolve",  "Yes please, fix it."),
]

for label, user_msg in turns:
    result = support_agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}],
         "current_step": "greet", "customer_id": "", "issue_type": "", "warranty_status": ""},
        config=config,
    )
    step = result.get("current_step", "?")
    print(f"\n  Turn [{label}] step={step!r}")
    print(f"  Agent: {result['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# PART 2: AGENT-TO-AGENT HANDOFF
# "Transfer to another agent" pattern — one agent calls
# transfer_to_specialist to hand off the conversation to a different
# specialized agent configuration.
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Agent-to-Agent Handoffs ───────────────────────────────")

class HandoffState(AgentState):
    active_agent: str    # Which agent/configuration is active


@tool
def transfer_to_billing(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfer the conversation to the billing specialist."""
    print(f"  [Handoff] → billing: {reason}")
    return Command(update={
        "active_agent": "billing",
        "messages": [ToolMessage(
            content="Transferred to Billing. How can I help with your account?",
            tool_call_id=tool_call_id,
        )]
    })


@tool
def transfer_to_technical(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Transfer the conversation to the technical specialist."""
    print(f"  [Handoff] → technical: {reason}")
    return Command(update={
        "active_agent": "technical",
        "messages": [ToolMessage(
            content="Transferred to Technical Support. Let's solve your issue.",
            tool_call_id=tool_call_id,
        )]
    })


@tool
def process_refund(amount: float, reason: str) -> str:
    """Process a billing refund (only available in billing mode)."""
    print(f"  [Tool] process_refund(${amount}, {reason[:40]})")
    return f"Refund of ${amount:.2f} processed. Reference: REF-{hash(reason) % 9999:04d}"


@tool
def run_diagnostics(device_id: str) -> str:
    """Run technical diagnostics (only available in technical mode)."""
    print(f"  [Tool] run_diagnostics({device_id!r})")
    return f"Diagnostics for {device_id}: No hardware faults. Firmware v3.2.1."


AGENT_PROMPTS = {
    "general":   "You are a general customer service rep. Route to specialists as needed.",
    "billing":   "You are a billing specialist. Handle refunds, charges, and account queries.",
    "technical": "You are a technical specialist. Diagnose and fix device issues.",
}

AGENT_TOOLS = {
    "general":   [transfer_to_billing, transfer_to_technical],
    "billing":   [process_refund, transfer_to_technical],
    "technical": [run_diagnostics, transfer_to_billing],
}

ALL_HANDOFF_TOOLS = [
    transfer_to_billing, transfer_to_technical, process_refund, run_diagnostics
]


@dynamic_prompt
def agent_routing_prompt(request: ModelRequest) -> str:
    state        = request.runtime.state
    active_agent = state.get("active_agent", "general")
    return AGENT_PROMPTS.get(active_agent, AGENT_PROMPTS["general"])


@wrap_model_call
def filter_agent_tools(call: ModelCall, runtime) -> ModelCall:
    active_agent = runtime.state.get("active_agent", "general")
    allowed      = [t.name for t in AGENT_TOOLS.get(active_agent, [])]
    filtered     = [t for t in call.tools if t.name in allowed]
    print(f"  [ToolFilter] active_agent={active_agent!r} → {[t.name for t in filtered]}")
    return call.replace(tools=filtered)


handoff_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=ALL_HANDOFF_TOOLS,
    middleware=[agent_routing_prompt, filter_agent_tools],
    checkpointer=MemorySaver(),
    system_prompt="Customer service agent.",
)

cfg2  = {"configurable": {"thread_id": "handoff-agent-1"}}
init2 = {"active_agent": "general"}

for msg in [
    "I was charged twice for my subscription.",
    "Yes I want a refund please.",
]:
    r = handoff_agent.invoke(
        {**init2, "messages": [{"role": "user", "content": msg}]},
        config=cfg2,
    )
    print(f"\n  agent={r.get('active_agent','?')!r}")
    print(f"  → {r['messages'][-1].content[:120]}")

print("\n" + "═" * 60)
print("Handoffs Pattern Summary:")
print("  current_step / active_agent — state variable drives behavior")
print("  Command(update={'current_step': ...}) — transition trigger")
print("  @dynamic_prompt reads state → different system prompt per step")
print("  @wrap_model_call filters tools → different tool set per step")
print("  MemorySaver — required for multi-turn persistent state")
print("  Direct user interaction — agent talks to user across all steps")
print("═" * 60)
print("\n✅ Handoffs pattern demo complete.")
