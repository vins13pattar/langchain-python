"""
context_engineering_overview.py — Context Engineering: all key concepts in one file
Covers: model context (system prompt, messages, tools, format), tool context
        (state/store/Command reads & writes), lifecycle context (summarization, audit)
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    SummarizationMiddleware,
    dynamic_prompt, ModelRequest, ModelResponse,
    before_model,
    wrap_model_call,
)
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

def section(title): print(f"\n{'─'*55}\n{title}\n{'─'*55}")


# ════════════════════════════════════════════════════════════════════
# SHARED CONTEXT SCHEMA
# ════════════════════════════════════════════════════════════════════

@dataclass
class LegalCtx:
    user_id:               str
    role:                  str         # paralegal | attorney | admin
    jurisdiction:          str         # EU | US | UK
    compliance_frameworks: List[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════
# 1. MODEL CONTEXT — what the LLM sees: system prompt, messages, tools, format
# ════════════════════════════════════════════════════════════════════
section("1. MODEL CONTEXT")

# 1a. Dynamic system prompt
@dynamic_prompt
def legal_system_prompt(request: ModelRequest) -> str:
    ctx = request.runtime.context
    hints = {
        "paralegal": "You assist with research and document preparation.",
        "attorney":  "You are a legal expert. Be authoritative but careful.",
        "admin":     "You have full access to all research data.",
    }
    return (
        f"You are a legal assistant for {ctx.jurisdiction} law. "
        f"{hints.get(ctx.role, '')} "
        f"Always include a disclaimer that responses are not legal advice."
    )

# 1b. Inject compliance rules into messages (transient model context)
@wrap_model_call
def inject_compliance_rules(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    ctx = request.runtime.context
    rules = []
    if "GDPR" in ctx.compliance_frameworks:
        rules.append("GDPR: Explicit consent required; right to erasure applies.")
    if "HIPAA" in ctx.compliance_frameworks:
        rules.append("HIPAA: PHI cannot be shared without authorization.")
    if rules:
        msg = f"Active constraints ({ctx.jurisdiction}):\n" + "\n".join(f"• {r}" for r in rules)
        request = request.override(messages=[*request.messages, {"role": "user", "content": msg}])
        print(f"  [inject_compliance] {len(rules)} rules injected")
    return handler(request)

# 1c. Role-based tool filtering (model context — dynamic tool list)
@wrap_model_call
def role_based_tools(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    role = request.runtime.context.role
    ALLOWED = {
        "paralegal": {"search_legal_db", "save_research_note", "get_saved_notes"},
        "attorney":  {"search_legal_db", "save_research_note", "get_saved_notes", "compliance_checklist"},
        "admin":     {t.name for t in request.tools},
    }
    allowed = ALLOWED.get(role, ALLOWED["paralegal"])
    tools = [t for t in request.tools if t.name in allowed]
    print(f"  [role_tools] role={role}  {len(tools)}/{len(request.tools)} tools")
    return handler(request.override(tools=tools))

# 1d. Dynamic response format based on role
class StandardResponse(BaseModel):
    summary:    str      = Field(description="Summary of findings")
    key_points: List[str]= Field(description="Key points")
    disclaimer: str      = Field(description="Legal disclaimer")

class AdminResponse(BaseModel):
    summary:    str      = Field(description="Summary of findings")
    key_points: List[str]= Field(description="Key points")
    sources:    List[str]= Field(description="Cases and statutes", default_factory=list)
    confidence: float    = Field(description="Confidence score 0-1")
    disclaimer: str      = Field(description="Legal disclaimer")

@wrap_model_call
def role_based_format(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    schema = AdminResponse if request.runtime.context.role == "admin" else StandardResponse
    print(f"  [role_format] schema={schema.__name__}")
    return handler(request.override(response_format=schema))


# ════════════════════════════════════════════════════════════════════
# 2. TOOL CONTEXT — reading & writing state/store in tools
# ════════════════════════════════════════════════════════════════════
section("2. TOOL CONTEXT")

_LEGAL_DB = {
    "EU": {"gdpr": "GDPR Art.6: lawful basis required.", "contract": "EU contract law: consent + consideration."},
    "US": {"privacy": "CCPA grants CA data rights.", "contract": "US contract: offer + acceptance + consideration."},
}

@tool
def search_legal_db(query: str, runtime: ToolRuntime[LegalCtx]) -> str:
    """Search legal database for the user's jurisdiction. Args: query: Topic to search."""
    ctx = runtime.context
    db = _LEGAL_DB.get(ctx.jurisdiction, {})
    results = [v for k, v in db.items() if any(w in k for w in query.lower().split())]
    print(f"  [Tool] search → jur={ctx.jurisdiction}  q='{query}'  hits={len(results)}")
    return f"[{ctx.jurisdiction}] " + ("\n".join(results) if results else "No results.")

@tool
def save_research_note(note: str, case_ref: str, runtime: ToolRuntime[LegalCtx]) -> str:
    """Save a research note to persistent store. Args: note, case_ref."""
    existing = runtime.store.get(("notes", runtime.context.jurisdiction), runtime.context.user_id)
    notes = existing.value.get("notes", []) if existing else []
    notes.append({"ref": case_ref, "note": note, "ts": time.time()})
    runtime.store.put(("notes", runtime.context.jurisdiction), runtime.context.user_id, {"notes": notes})
    print(f"  [Tool] save_note → user={runtime.context.user_id}  ref={case_ref}")
    return f"Saved under '{case_ref}'. Total: {len(notes)} notes."

@tool
def get_saved_notes(runtime: ToolRuntime[LegalCtx]) -> str:
    """Retrieve all saved research notes."""
    stored = runtime.store.get(("notes", runtime.context.jurisdiction), runtime.context.user_id)
    if not stored:
        return "No saved notes."
    notes = stored.value.get("notes", [])
    return "\n".join(f"[{n['ref']}] {n['note']}" for n in notes)

@tool
def compliance_checklist(runtime: ToolRuntime[LegalCtx]) -> str:
    """Generate compliance checklist from frameworks in context."""
    items = []
    if "GDPR" in runtime.context.compliance_frameworks:
        items += ["☐ Data processing agreement", "☐ Lawful basis documented", "☐ DPO appointed"]
    if "HIPAA" in runtime.context.compliance_frameworks:
        items += ["☐ BAA signed", "☐ PHI encryption implemented"]
    return "Checklist:\n" + "\n".join(items or ["☐ No frameworks configured"])

@tool
def authenticate(password: str, runtime: ToolRuntime) -> Command:
    """Authenticate for this session. Args: password: Use 'legal123'."""
    ok = password == "legal123"
    print(f"  [Tool] authenticate → {'success' if ok else 'fail'}")
    return Command(update={
        "authenticated": ok,
        "messages": [ToolMessage(
            content="✅ Authenticated!" if ok else "❌ Wrong password.",
            tool_call_id=runtime.tool_call_id,
        )],
    })


# ════════════════════════════════════════════════════════════════════
# 3. LIFECYCLE CONTEXT — summarization, audit, session tracking
# ════════════════════════════════════════════════════════════════════
section("3. LIFECYCLE CONTEXT")

_audit: list[dict] = []

@before_model
def audit_and_track(state: AgentState, runtime: Runtime[LegalCtx]) -> dict | None:
    """Increment turn counter and audit every LLM call."""
    turns = state.get("research_turns", 0) + 1
    entry = {
        "user":   runtime.context.user_id,
        "jur":    runtime.context.jurisdiction,
        "run":    runtime.execution_info.run_id[:8],
        "turns":  turns,
    }
    _audit.append(entry)
    print(f"  [Audit+Track] {entry}")
    return {"research_turns": turns}


# ════════════════════════════════════════════════════════════════════
# 4. FULL AGENT COMBINING ALL CONTEXT TYPES
# ════════════════════════════════════════════════════════════════════
section("4. FULL AGENT: ALL CONTEXT TYPES")

shared_store = InMemoryStore()

legal_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_legal_db, save_research_note, get_saved_notes, compliance_checklist, authenticate],
    context_schema=LegalCtx,
    store=shared_store,
    checkpointer=MemorySaver(),
    middleware=[
        # MODEL CONTEXT
        legal_system_prompt,       # dynamic system prompt
        inject_compliance_rules,   # compliance messages (transient)
        role_based_tools,          # RBAC tool filtering
        role_based_format,         # dynamic response schema
        # LIFECYCLE CONTEXT
        SummarizationMiddleware(model="openai:gpt-4o-mini", trigger=("messages", 8), keep=("messages", 2)),
        audit_and_track,           # persistent turn counter + audit
    ],
)

# Scenario A: Paralegal — EU GDPR search + save notes
ctx_a = LegalCtx("PAR-001", "paralegal", "EU", ["GDPR"])
cfg_a = {"configurable": {"thread_id": str(uuid.uuid4())}}

r = legal_agent.invoke({"messages": [HumanMessage("Search for GDPR data processing rules.")]}, context=ctx_a, config=cfg_a)
print("Search:", r["messages"][-1].content[:120])

r = legal_agent.invoke({"messages": [HumanMessage("Save note under GDPR-001: 'Lawful basis must be documented.'")]}, context=ctx_a, config=cfg_a)
print("Save note:", r["messages"][-1].content[:80])

r = legal_agent.invoke({"messages": [HumanMessage("Show my saved notes.")]}, context=ctx_a, config=cfg_a)
print("Get notes:", r["messages"][-1].content[:120])

# Scenario B: Admin — US, full tool access, AdminResponse schema
ctx_b = LegalCtx("ADM-001", "admin", "US", ["CCPA"])
cfg_b = {"configurable": {"thread_id": str(uuid.uuid4())}}
r = legal_agent.invoke({"messages": [HumanMessage("Analyse US contract law.")]}, context=ctx_b, config=cfg_b)
print("Admin response:", r["messages"][-1].content[:150])
print("Research turns:", r.get("research_turns", "?"))

# Cross-session: same store, new thread — notes persist
cfg_c = {"configurable": {"thread_id": str(uuid.uuid4())}}
r = legal_agent.invoke({"messages": [HumanMessage("Retrieve my saved notes from previous sessions.")]}, context=ctx_a, config=cfg_c)
print("Cross-session notes:", r["messages"][-1].content[:150])

print(f"\nAudit log: {len(_audit)} LLM calls")
for e in _audit:
    print(f"  {e}")

print("""
Context types:
  MODEL CONTEXT    → what the LLM sees per call (prompt, messages, tools, format)
    @dynamic_prompt        — role/user-aware system prompt
    inject_* (wrap_model)  — transient extra messages
    role_based_tools       — RBAC tool gating
    role_based_format      — dynamic response schema per role

  TOOL CONTEXT     → data tools read/write during execution
    runtime.context  → immutable per-run data (role, jurisdiction)
    runtime.store    → persistent cross-thread memory (store.get/put)
    runtime.state    → current graph state (read-only in tool)
    Command(update)  → write to state from a tool

  LIFECYCLE CONTEXT → accumulated, session-wide data
    SummarizationMiddleware → auto-compress long conversations
    @before_model           → track turns, audit, inject session state
    runtime.execution_info  → run_id, thread_id for tracing
""")
