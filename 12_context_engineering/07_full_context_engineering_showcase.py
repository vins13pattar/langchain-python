"""
07_full_context_engineering_showcase.py
=========================================
Production-ready showcase: a SMART LEGAL RESEARCH AGENT that applies
all three context types (Model, Tool, Life-cycle) across all three data
sources (State, Store, Runtime Context).

Architecture overview:
  ┌─────────────────────────────────────────────────────────────────┐
  │               Smart Legal Research Agent                         │
  ├─────────────────────────────────────────────────────────────────┤
  │  Context Schema (LegalCtx):                                      │
  │    user_id, role, jurisdiction, compliance_frameworks            │
  │                                                                 │
  │  MODEL CONTEXT:                                                  │
  │    @dynamic_prompt       → role + jurisdiction system prompt     │
  │    inject_jurisdiction   → compliance rules in messages          │
  │    context_based_tools   → role-gated tool access                │
  │    context_based_format  → admin vs user response schema         │
  │                                                                 │
  │  TOOL CONTEXT:                                                   │
  │    search_legal_db       → reads runtime.context (jurisdiction)  │
  │    save_research_note    → writes to store (persistent notes)    │
  │    get_saved_notes       → reads from store                      │
  │    authenticate_session  → writes to state via Command           │
  │                                                                 │
  │  LIFE-CYCLE CONTEXT:                                             │
  │    SummarizationMiddleware → auto-condense long sessions         │
  │    audit_logger            → log every call to store             │
  │    track_research_session  → persist session metadata to state   │
  └─────────────────────────────────────────────────────────────────┘

Scenarios:
  A. Paralegal — jurisdiction-filtered search, notes persistence
  B. Admin user — full tool access, admin response format
  C. Long session — summarization triggered after 8 messages
  D. Cross-session — saved notes retrieved in a new session
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from dotenv import load_dotenv
from pydantic import BaseModel, Field as PField

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    AgentMiddleware,
    SummarizationMiddleware,
    dynamic_prompt,
    ModelRequest,
    ModelResponse,
    before_model,
    wrap_model_call,
    hook_config,
)
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

load_dotenv()

print("=" * 60)
print("Smart Legal Research Agent — Full Context Engineering")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# CONTEXT SCHEMA
# ════════════════════════════════════════════════════════════════════

@dataclass
class LegalCtx:
    user_id:               str
    role:                  str             # "paralegal", "attorney", "admin"
    jurisdiction:          str             # "EU", "US", "UK"
    compliance_frameworks: list[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════
# RESPONSE SCHEMAS
# ════════════════════════════════════════════════════════════════════

class LegalResearchResponse(BaseModel):
    """Standard legal research response."""
    summary:    str        = PField(description="Summary of legal findings.")
    key_points: list[str]  = PField(description="Key legal points identified.")
    disclaimer: str        = PField(description="Legal disclaimer.")


class AdminLegalResponse(BaseModel):
    """Admin response with full metadata."""
    summary:      str       = PField(description="Summary of legal findings.")
    key_points:   list[str] = PField(description="Key legal points.")
    sources:      list[str] = PField(description="Cases and statutes consulted.", default_factory=list)
    confidence:   float     = PField(description="Confidence score 0.0–1.0.")
    disclaimer:   str       = PField(description="Legal disclaimer.")


# ════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════

_LEGAL_DB = {
    "EU":  {"gdpr": "GDPR Article 6 requires lawful basis for processing.",
             "contract": "EU contract law requires mutual consent and consideration."},
    "US":  {"privacy": "CCPA grants CA residents data rights since 2020.",
             "contract": "US contract law requires offer, acceptance, consideration."},
    "UK":  {"gdpr": "UK GDPR (post-Brexit) mirrors EU GDPR with some differences.",
             "contract": "UK contract law closely follows common law principles."},
}


@tool
def search_legal_db(
    query: str,
    runtime: ToolRuntime[LegalCtx],
) -> str:
    """
    Search the legal database filtered to the user's jurisdiction.

    Args:
        query: The legal topic or question to search for.
    """
    ctx = runtime.context
    db  = _LEGAL_DB.get(ctx.jurisdiction, {})
    print(f"  [Tool] search_legal_db → jurisdiction={ctx.jurisdiction}, query='{query}'")
    results = [v for k, v in db.items() if any(w in k for w in query.lower().split())]
    if results:
        return f"[{ctx.jurisdiction}] Legal results for '{query}':\n" + "\n".join(f"  • {r}" for r in results)
    return f"No specific results for '{query}' in {ctx.jurisdiction} database."


@tool
def save_research_note(
    note: str,
    case_ref: str,
    runtime: ToolRuntime[LegalCtx],
) -> str:
    """
    Save a research note to long-term memory for this user.

    Args:
        note:     The research finding or note to save.
        case_ref: Reference code for the case or topic.
    """
    ctx   = runtime.context
    store = runtime.store
    if not store:
        return "No store configured."

    existing = store.get(("legal_notes", ctx.jurisdiction), ctx.user_id)
    notes    = existing.value.get("notes", []) if existing else []
    notes.append({"ref": case_ref, "note": note, "saved_at": time.time()})
    store.put(("legal_notes", ctx.jurisdiction), ctx.user_id, {"notes": notes})
    print(f"  [Tool] save_research_note → user={ctx.user_id}, ref={case_ref}")
    return f"Note saved under ref '{case_ref}'. Total notes: {len(notes)}."


@tool
def get_saved_notes(runtime: ToolRuntime[LegalCtx]) -> str:
    """Retrieve all saved research notes for the current user."""
    ctx   = runtime.context
    store = runtime.store
    if not store:
        return "No store configured."

    existing = store.get(("legal_notes", ctx.jurisdiction), ctx.user_id)
    if not existing:
        return f"No saved notes for user {ctx.user_id} in {ctx.jurisdiction}."
    notes = existing.value.get("notes", [])
    print(f"  [Tool] get_saved_notes → {len(notes)} notes")
    formatted = "\n".join(f"  [{n['ref']}] {n['note']}" for n in notes)
    return f"Saved notes ({ctx.jurisdiction}):\n{formatted}"


@tool
def authenticate_session(password: str, runtime: ToolRuntime) -> Command:
    """
    Authenticate the user for this session.

    Args:
        password: The session password to verify.
    """
    if password == "legal123":
        print("  [Tool] authenticate_session → success")
        return Command(update={"authenticated": True, "session_start": time.time()})
    print("  [Tool] authenticate_session → failed")
    return Command(update={"authenticated": False})


@tool
def get_compliance_checklist(runtime: ToolRuntime[LegalCtx]) -> str:
    """Generate a compliance checklist based on the user's frameworks."""
    ctx = runtime.context
    items = []
    if "GDPR" in ctx.compliance_frameworks:
        items += ["☐ Data processing agreement in place",
                  "☐ Lawful basis documented", "☐ DPO appointed (if required)"]
    if "HIPAA" in ctx.compliance_frameworks:
        items += ["☐ BAA signed with business associates",
                  "☐ PHI encryption implemented"]
    if not items:
        items = ["☐ No specific compliance frameworks configured"]
    return "Compliance checklist:\n" + "\n".join(items)


# ════════════════════════════════════════════════════════════════════
# MIDDLEWARE — MODEL CONTEXT
# ════════════════════════════════════════════════════════════════════

@dynamic_prompt
def legal_system_prompt(request: ModelRequest) -> str:
    """Role + jurisdiction system prompt."""
    ctx = request.runtime.context
    role_hints = {
        "paralegal": "You assist with legal research and document preparation.",
        "attorney":  "You are a legal expert. Provide authoritative but careful analysis.",
        "admin":     "You have full access to all research data and system metadata.",
    }
    return (
        f"You are a legal research assistant for {ctx.jurisdiction} law. "
        f"{role_hints.get(ctx.role, '')} "
        f"Always include a disclaimer that responses are not legal advice."
    )


@wrap_model_call
def inject_jurisdiction_rules(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Inject compliance rules from runtime context into messages."""
    ctx   = request.runtime.context
    rules = []
    if "GDPR" in ctx.compliance_frameworks:
        rules.append("GDPR: Obtain explicit consent; right to erasure applies.")
    if "HIPAA" in ctx.compliance_frameworks:
        rules.append("HIPAA: PHI cannot be shared without authorization.")
    if rules:
        msg = f"Active compliance constraints for {ctx.jurisdiction}:\n" + "\n".join(f"• {r}" for r in rules)
        messages = [*request.messages, {"role": "user", "content": msg}]
        request  = request.override(messages=messages)
        print(f"  [inject_jurisdiction_rules] {len(rules)} rules injected")
    return handler(request)


@wrap_model_call
def role_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Filter tools based on role from runtime context."""
    role = request.runtime.context.role
    if role == "admin":
        tools = request.tools
    elif role == "attorney":
        tools = [t for t in request.tools if t.name != "authenticate_session"]
    else:  # paralegal
        tools = [t for t in request.tools
                 if t.name in ("search_legal_db", "get_saved_notes", "save_research_note")]
    print(f"  [role_based_tools] role={role}, tools={[t.name for t in tools]}")
    return handler(request.override(tools=tools))


@wrap_model_call
def role_based_format(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Select response schema based on role."""
    role = request.runtime.context.role
    schema = AdminLegalResponse if role == "admin" else LegalResearchResponse
    print(f"  [role_based_format] schema={schema.__name__}")
    return handler(request.override(response_format=schema))


# ════════════════════════════════════════════════════════════════════
# MIDDLEWARE — LIFE-CYCLE CONTEXT
# ════════════════════════════════════════════════════════════════════

@before_model
def track_research_session(state: AgentState, runtime: Runtime[LegalCtx]) -> dict | None:
    """Persist session turn count and jurisdiction to state."""
    turns = state.get("research_turns", 0) + 1
    print(f"  [track_session] turn={turns}")
    return {
        "research_turns": turns,
        "last_jurisdiction": runtime.context.jurisdiction,
    }


@before_model
def store_audit(state: AgentState, runtime: Runtime[LegalCtx]) -> dict | None:
    """Audit every model call to store."""
    store   = runtime.store
    user_id = runtime.context.user_id
    if not store:
        return None
    existing = store.get(("audit",), user_id)
    entries  = existing.value.get("entries", []) if existing else []
    entries.append({
        "ts":    time.time(),
        "run":   runtime.execution_info.run_id[:8],
        "turns": state.get("research_turns", 0),
        "jur":   runtime.context.jurisdiction,
    })
    store.put(("audit",), user_id, {"entries": entries[-100:]})
    return None


# ════════════════════════════════════════════════════════════════════
# BUILD THE AGENT
# ════════════════════════════════════════════════════════════════════

shared_store = InMemoryStore()

legal_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_legal_db, save_research_note, get_saved_notes,
           authenticate_session, get_compliance_checklist],
    context_schema=LegalCtx,
    store=shared_store,
    checkpointer=MemorySaver(),
    middleware=[
        # MODEL CONTEXT
        legal_system_prompt,         # dynamic system prompt
        inject_jurisdiction_rules,   # compliance messages (transient)
        role_based_tools,            # RBAC tool filtering
        role_based_format,           # dynamic response schema
        # LIFE-CYCLE CONTEXT
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            trigger={"messages": 8},
            keep={"messages": 2},
        ),
        track_research_session,      # persistent turn counter
        store_audit,                 # persistent audit log
    ],
)


# ════════════════════════════════════════════════════════════════════
# SCENARIO A — Paralegal: jurisdiction-filtered search + notes
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO A — Paralegal (EU, GDPR, search + save notes)")
print("─" * 60)

cfg_a = {"configurable": {"thread_id": "legal-paralegal-1"}}
ctx_a = LegalCtx(user_id="PAR-001", role="paralegal",
                  jurisdiction="EU", compliance_frameworks=["GDPR"])

r_a1 = legal_agent.invoke(
    {"messages": [{"role": "user", "content": "Search for GDPR data processing rules."}]},
    context=ctx_a, config=cfg_a,
)
print(f"\nSearch result: {r_a1['messages'][-1].content[:200]}")

r_a2 = legal_agent.invoke(
    {"messages": [{"role": "user", "content":
        "Save this note under ref GDPR-001: 'Lawful basis must be documented before processing.'"}]},
    context=ctx_a, config=cfg_a,
)
print(f"Note saved:    {r_a2['messages'][-1].content[:120]}")

r_a3 = legal_agent.invoke(
    {"messages": [{"role": "user", "content": "Show me all my saved notes."}]},
    context=ctx_a, config=cfg_a,
)
print(f"Notes fetch:   {r_a3['messages'][-1].content[:200]}")


# ════════════════════════════════════════════════════════════════════
# SCENARIO B — Admin: full access, AdminLegalResponse schema
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO B — Admin (US, full access, admin response format)")
print("─" * 60)

cfg_b = {"configurable": {"thread_id": "legal-admin-1"}}
ctx_b = LegalCtx(user_id="ADM-001", role="admin",
                  jurisdiction="US", compliance_frameworks=["CCPA"])

r_b = legal_agent.invoke(
    {"messages": [{"role": "user", "content":
        "Search for US contract law and give me a full analysis."}]},
    context=ctx_b, config=cfg_b,
)
print(f"Admin response: {r_b['messages'][-1].content[:250]}")
print(f"Research turns: {r_b.get('research_turns', '?')}")


# ════════════════════════════════════════════════════════════════════
# SCENARIO C — Cross-session note retrieval
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO C — Cross-Session Note Retrieval (new thread)")
print("─" * 60)

cfg_c = {"configurable": {"thread_id": "legal-paralegal-NEW-SESSION"}}  # New thread!
r_c = legal_agent.invoke(
    {"messages": [{"role": "user", "content":
        "Retrieve all my saved research notes from previous sessions."}]},
    context=ctx_a,   # Same user, same store — notes persist!
    config=cfg_c,
)
print(f"Notes in new session: {r_c['messages'][-1].content[:200]}")


# ════════════════════════════════════════════════════════════════════
# AUDIT SUMMARY
# ════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("Audit Summary:")
for user_id in ("PAR-001", "ADM-001"):
    audit = shared_store.get(("audit",), user_id)
    if audit:
        entries = audit.value.get("entries", [])
        print(f"  {user_id}: {len(entries)} model calls logged")

print("\n" + "═" * 60)
print("Full Context Engineering Stack:")
print("  MODEL:     @dynamic_prompt      — role + jurisdiction prompt")
print("             inject_jurisdiction   — compliance rules (transient)")
print("             role_based_tools      — RBAC tool gating")
print("             role_based_format     — admin vs user response schema")
print("  TOOL:      search_legal_db       — reads runtime.context")
print("             save_research_note    — writes to store")
print("             get_saved_notes       — reads from store")
print("             authenticate_session  — writes to state (Command)")
print("  LIFECYCLE: SummarizationMiddleware — auto-condense conversations")
print("             track_research_session  — persistent turn counter")
print("             store_audit             — persistent audit log")
print("═" * 60)
print("\n✅ Full context engineering showcase complete.")
