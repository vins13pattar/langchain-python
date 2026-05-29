"""
05_full_runtime_showcase.py
============================
Production-ready showcase: a MULTI-TENANT CRM AGENT that uses the full
Runtime system for dependency injection, personalization, and audit control.

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │                Multi-Tenant CRM Agent                            │
  ├─────────────────────────────────────────────────────────────────┤
  │  Context Schema (CRMContext):                                    │
  │    - user_id, user_name, role, tenant_id, language              │
  │                                                                 │
  │  Runtime Usage:                                                  │
  │    - @dynamic_prompt     → personalized system prompt per user  │
  │    - before_agent (RBAC) → role-based access control            │
  │    - before_model audit  → log every LLM call with identity     │
  │    - ToolRuntime context → tenant-scoped data access            │
  │    - ToolRuntime store   → per-tenant long-term memory          │
  │    - ToolRuntime writer  → stream progress for long ops         │
  │    - execution_info      → run_id for deduplication/tracing     │
  └─────────────────────────────────────────────────────────────────┘

Scenarios:
  A. Admin user — full access, personalized prompt, audit trail
  B. Sales rep  — tenant-scoped data, read-only, personalized
  C. Guest user — blocked by RBAC middleware
  D. Store usage — save/load user preferences across calls
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    AgentMiddleware,
    dynamic_prompt,
    ModelRequest,
    before_model,
    before_agent,
    hook_config,
)
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

load_dotenv()

print("=" * 60)
print("Multi-Tenant CRM Agent — Full Runtime Showcase")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# CONTEXT SCHEMA
# ════════════════════════════════════════════════════════════════════

@dataclass
class CRMContext:
    user_id:   str
    user_name: str
    role:      str          # "admin", "sales", "support", "guest"
    tenant_id: str
    language:  str = "English"


# ════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════

# Simulated CRM data store
_CRM_DATA = {
    "tenant-001": {
        "contacts": ["Alice Brown", "Bob Chen", "Carol Davis"],
        "deals":    [("Deal-A", "$50K"), ("Deal-B", "$120K")],
        "revenue":  "$1.2M",
    },
    "tenant-002": {
        "contacts": ["Dave Evans", "Eva Fischer"],
        "deals":    [("Deal-X", "$30K")],
        "revenue":  "$380K",
    },
}


@tool
def list_contacts(runtime: ToolRuntime[CRMContext]) -> str:
    """List all contacts for the current tenant."""
    ctx = runtime.context
    data = _CRM_DATA.get(ctx.tenant_id, {})
    contacts = data.get("contacts", [])
    print(f"  [Tool] list_contacts → tenant={ctx.tenant_id}, user={ctx.user_name}")
    return f"Contacts for tenant {ctx.tenant_id}: {', '.join(contacts)}."


@tool
def get_revenue_summary(runtime: ToolRuntime[CRMContext]) -> str:
    """Get revenue summary for the current tenant. Admin only."""
    ctx = runtime.context
    if ctx.role not in ("admin",):
        return "Access denied: revenue data is restricted to admin users."
    data = _CRM_DATA.get(ctx.tenant_id, {})
    revenue = data.get("revenue", "N/A")
    deals   = data.get("deals", [])
    print(f"  [Tool] get_revenue_summary → tenant={ctx.tenant_id}, revenue={revenue}")
    deal_str = ", ".join(f"{d[0]} ({d[1]})" for d in deals)
    return f"Revenue for {ctx.tenant_id}: {revenue}. Active deals: {deal_str}."


@tool
def get_user_notes(runtime: ToolRuntime[CRMContext]) -> str:
    """Retrieve saved notes for the current user from long-term memory."""
    ctx = runtime.context
    if runtime.store:
        memory = runtime.store.get(("notes", ctx.tenant_id), ctx.user_id)
        if memory:
            notes = memory.value.get("notes", "")
            print(f"  [Tool] get_user_notes → found: '{notes[:60]}'")
            return f"Your saved notes: {notes}"
    print("  [Tool] get_user_notes → no notes found")
    return "No saved notes found for you."


@tool
def save_user_notes(
    notes: str,
    runtime: ToolRuntime[CRMContext],
) -> str:
    """
    Save notes for the current user to long-term memory.

    Args:
        notes: The notes content to save.
    """
    ctx = runtime.context
    if runtime.store:
        runtime.store.put(
            ("notes", ctx.tenant_id),
            ctx.user_id,
            {"notes": notes},
        )
        print(f"  [Tool] save_user_notes → saved: '{notes[:60]}'")
        return f"Notes saved: {notes}"
    return "No store configured — notes not saved."


@tool
def run_report(
    report_type: str,
    runtime: ToolRuntime[CRMContext],
) -> str:
    """
    Run a CRM report (streams progress updates).

    Args:
        report_type: Type of report ('contacts', 'deals', 'revenue').
    """
    ctx  = runtime.context
    info = runtime.execution_info
    print(f"  [Tool] run_report → type={report_type}, run_id={info.run_id[:8]}...")

    stages = ["Fetching data", "Aggregating", "Formatting", "Finalizing"]
    for i, stage in enumerate(stages, 1):
        progress = f"[{i}/{len(stages)}] {stage}..."
        print(f"  [Tool] progress: {progress}")
        if runtime.writer:
            runtime.writer({"stage": progress, "user": ctx.user_name})
        time.sleep(0.02)

    data = _CRM_DATA.get(ctx.tenant_id, {})
    if report_type == "revenue":
        return f"Revenue Report: {data.get('revenue', 'N/A')}"
    elif report_type == "contacts":
        return f"Contacts Report: {len(data.get('contacts', []))} total"
    elif report_type == "deals":
        return f"Deals Report: {len(data.get('deals', []))} active deals"
    return f"Report '{report_type}' generated."


# ════════════════════════════════════════════════════════════════════
# MIDDLEWARE
# ════════════════════════════════════════════════════════════════════

@dynamic_prompt
def crm_system_prompt(request: ModelRequest) -> str:
    """Build a personalized, role-aware system prompt."""
    ctx = request.runtime.context
    role_hint = {
        "admin":   "You have full access to all CRM data including revenue.",
        "sales":   "You can view contacts and deals for your tenant.",
        "support": "You can view contacts and run contact reports.",
        "guest":   "You have read-only access to basic information.",
    }.get(ctx.role, "")
    print(f"  [DynamicPrompt] user={ctx.user_name}, role={ctx.role}")
    return (
        f"You are a CRM assistant for tenant '{ctx.tenant_id}'. "
        f"Address the user as {ctx.user_name}. "
        f"Always respond in {ctx.language}. "
        f"{role_hint}"
    )


class RBACMiddleware(AgentMiddleware):
    """Block guest users from the CRM agent."""

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime[CRMContext]) -> dict[str, Any] | None:
        role = runtime.context.role
        name = runtime.context.user_name
        if role == "guest":
            print(f"  [RBAC] 🚫 Guest '{name}' blocked")
            return {
                "messages": [{"role": "assistant",
                              "content": "Access denied. Guest users cannot access CRM data."}],
                "jump_to": "end",
            }
        print(f"  [RBAC] ✅ role='{role}', user='{name}'")
        return None


_audit_log: list[dict] = []

@before_model
def crm_audit(state: AgentState, runtime: Runtime[CRMContext]) -> dict | None:
    """Record each LLM call with user and execution identity."""
    entry = {
        "user":      runtime.context.user_name,
        "tenant":    runtime.context.tenant_id,
        "role":      runtime.context.role,
        "thread_id": runtime.execution_info.thread_id,
        "run_id":    runtime.execution_info.run_id[:8],
        "messages":  len(state.get("messages", [])),
    }
    _audit_log.append(entry)
    print(f"  [Audit] {entry}")
    return None


# ════════════════════════════════════════════════════════════════════
# BUILD THE AGENT
# ════════════════════════════════════════════════════════════════════

store = InMemoryStore()

crm_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[list_contacts, get_revenue_summary, get_user_notes, save_user_notes, run_report],
    context_schema=CRMContext,
    checkpointer=MemorySaver(),
    store=store,
    middleware=[
        RBACMiddleware(),        # Block unauthorized roles
        crm_system_prompt,       # Dynamic, personalized prompt
        crm_audit,               # Audit every LLM call
    ],
    system_prompt="You are a CRM assistant.",  # Overridden by dynamic_prompt
)


# ════════════════════════════════════════════════════════════════════
# SCENARIO A — Admin user: full access
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO A — Admin (full access, revenue visible)")
print("─" * 60)

cfg_a = {"configurable": {"thread_id": "crm-admin-thread"}}
result_a = crm_agent.invoke(
    {"messages": [{"role": "user", "content":
        "Show me the revenue summary and list all contacts."}]},
    context=CRMContext(user_id="ADM-1", user_name="Alex", role="admin",
                       tenant_id="tenant-001", language="English"),
    config=cfg_a,
)
print(f"\nResponse: {result_a['messages'][-1].content[:200]}")


# ════════════════════════════════════════════════════════════════════
# SCENARIO B — Sales rep: tenant-scoped, no revenue
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO B — Sales Rep (contacts only, different tenant)")
print("─" * 60)

cfg_b = {"configurable": {"thread_id": "crm-sales-thread"}}
result_b = crm_agent.invoke(
    {"messages": [{"role": "user", "content":
        "List all my contacts and run a contacts report."}]},
    context=CRMContext(user_id="SLS-1", user_name="Betty", role="sales",
                       tenant_id="tenant-002", language="English"),
    config=cfg_b,
)
print(f"\nResponse: {result_b['messages'][-1].content[:200]}")


# ════════════════════════════════════════════════════════════════════
# SCENARIO C — Guest user: blocked by RBAC
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO C — Guest (blocked by RBAC)")
print("─" * 60)

cfg_c = {"configurable": {"thread_id": "crm-guest-thread"}}
result_c = crm_agent.invoke(
    {"messages": [{"role": "user", "content": "Show me the contacts."}]},
    context=CRMContext(user_id="GST-1", user_name="Charlie", role="guest",
                       tenant_id="tenant-001"),
    config=cfg_c,
)
print(f"\nBlocked: {result_c['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# SCENARIO D — Long-term memory: save then retrieve notes
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO D — Long-Term Memory (save + retrieve notes)")
print("─" * 60)

cfg_d = {"configurable": {"thread_id": "crm-notes-save"}}
crm_agent.invoke(
    {"messages": [{"role": "user", "content":
        "Save this note: 'Follow up with Alice Brown re: renewal by Friday.'"}]},
    context=CRMContext(user_id="SLS-2", user_name="Diana", role="sales",
                       tenant_id="tenant-001"),
    config=cfg_d,
)

cfg_d2 = {"configurable": {"thread_id": "crm-notes-fetch"}}
result_d = crm_agent.invoke(
    {"messages": [{"role": "user", "content": "What notes do I have saved?"}]},
    context=CRMContext(user_id="SLS-2", user_name="Diana", role="sales",
                       tenant_id="tenant-001"),
    config=cfg_d2,
)
print(f"\nNotes retrieved: {result_d['messages'][-1].content[:150]}")


# ════════════════════════════════════════════════════════════════════
# AUDIT LOG SUMMARY
# ════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print(f"Audit log — {len(_audit_log)} LLM calls recorded:")
for entry in _audit_log:
    print(f"  user={entry['user']:<10} tenant={entry['tenant']:<12} "
          f"role={entry['role']:<8} run={entry['run_id']} msgs={entry['messages']}")

print("\n" + "═" * 60)
print("Full Runtime Showcase — Components Used:")
print("  context_schema   — CRMContext injected at invoke() time")
print("  @dynamic_prompt  — personalized prompt per user/role/language")
print("  RBACMiddleware   — before_agent + Runtime[Context] for access control")
print("  crm_audit        — before_model audit log using execution_info")
print("  ToolRuntime      — tenant-scoped tool data + store memory")
print("  InMemoryStore    — per-tenant long-term notes storage")
print("  MemorySaver      — session continuity across turns")
print("═" * 60)
print("\n✅ Full runtime showcase complete.")
