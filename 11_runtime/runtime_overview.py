"""
runtime_overview.py — LangChain Runtime: all key concepts in one file
Covers: context_schema, ToolRuntime (state/context/store/writer/execution_info),
        runtime in middleware, dynamic_prompt, RBAC, audit logging
"""

import time
import uuid
from dataclasses import dataclass
from typing import Any
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    AgentMiddleware, hook_config,
    dynamic_prompt, ModelRequest,
    before_model, before_agent,
)
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

def section(title): print(f"\n{'─'*55}\n{title}\n{'─'*55}")


# ════════════════════════════════════════════════════════════════════
# 1. CONTEXT SCHEMA — inject per-run data at invoke() time
# ════════════════════════════════════════════════════════════════════
section("1. CONTEXT SCHEMA")

@dataclass
class UserContext:
    user_id:   str
    user_name: str
    role:      str = "viewer"   # admin | editor | viewer

@tool
def get_my_profile(runtime: ToolRuntime[UserContext]) -> str:
    """Return current user's profile."""
    ctx = runtime.context
    return f"User: {ctx.user_name}  ID: {ctx.user_id}  Role: {ctx.role}"

@tool
def admin_action(task: str, runtime: ToolRuntime[UserContext]) -> str:
    """Perform an admin action (admin only). Args: task: Action to perform."""
    if runtime.context.role != "admin":
        return f"❌ Denied for role '{runtime.context.role}'"
    return f"✅ Admin task '{task}' done by {runtime.context.user_name}"

ctx_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_my_profile, admin_action],
    context_schema=UserContext,
    checkpointer=MemorySaver(),
    system_prompt="You are a personalised assistant. Greet users by name.",
)

def ask_as(user: UserContext, q: str) -> str:
    return ctx_agent.invoke(
        {"messages": [{"role": "user", "content": q}]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
        context=user,
    )["messages"][-1].content

admin  = UserContext("u-001", "Alice", "admin")
viewer = UserContext("u-002", "Bob",   "viewer")
print("Admin profile:", ask_as(admin,  "Show my profile.")[:80])
print("Admin task:   ", ask_as(admin,  "Perform admin task: flush_cache")[:80])
print("Viewer task:  ", ask_as(viewer, "Perform admin task: delete_logs")[:80])


# ════════════════════════════════════════════════════════════════════
# 2. TOOLRUNTIME — state, context, store, execution_info, writer
# ════════════════════════════════════════════════════════════════════
section("2. TOOLRUNTIME")

# a) runtime.state — read current conversation state
@tool
def get_message_count(runtime: ToolRuntime) -> str:
    """Count messages in the current conversation."""
    return f"This conversation has {len(runtime.state['messages'])} messages."

# b) runtime.context — access per-run context (type-safe)
@tool
def greet_user(runtime: ToolRuntime[UserContext]) -> str:
    """Greet the current user by name."""
    return f"Hello, {runtime.context.user_name}! Your role is '{runtime.context.role}'."

# c) runtime.store — persistent memory across threads
@tool
def save_pref(key: str, value: str, runtime: ToolRuntime[UserContext]) -> str:
    """Save a user preference to persistent storage. Args: key, value."""
    stored = (runtime.store.get(("prefs",), runtime.context.user_id) or type("", (), {"value": {}})()).value
    stored[key] = value
    runtime.store.put(("prefs",), runtime.context.user_id, stored)
    return f"Saved {key}={value}"

@tool
def get_prefs(runtime: ToolRuntime[UserContext]) -> str:
    """Retrieve all saved preferences."""
    stored = runtime.store.get(("prefs",), runtime.context.user_id)
    return str(stored.value if stored else "No preferences")

# d) runtime.execution_info — run metadata
@tool
def show_run_info(runtime: ToolRuntime) -> str:
    """Show metadata about the current agent run."""
    info = runtime.execution_info
    return f"Run ID: {info.run_id[:8]}...  Thread: {info.thread_id}"

# e) runtime.writer — stream progress updates
@tool
def run_long_report(report_type: str, runtime: ToolRuntime) -> str:
    """Run a long report and stream progress. Args: report_type: e.g. contacts, revenue."""
    stages = ["Fetching", "Aggregating", "Formatting", "Done"]
    for i, stage in enumerate(stages, 1):
        msg = f"[{i}/{len(stages)}] {stage}..."
        if runtime.writer:
            runtime.writer({"stage": msg})
        time.sleep(0.01)
    return f"{report_type} report complete."

store = InMemoryStore()
runtime_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_message_count, greet_user, save_pref, get_prefs, show_run_info, run_long_report],
    context_schema=UserContext,
    checkpointer=MemorySaver(),
    store=store,
    system_prompt="You are a personalised assistant.",
)

user = UserContext("u-001", "Alice", "admin")
cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

r = runtime_agent.invoke({"messages": [HumanMessage("Greet me and show message count.")]}, config=cfg, context=user)
print("Greet+count:", r["messages"][-1].content[:100])

r = runtime_agent.invoke({"messages": [HumanMessage("Save my theme pref to dark.")]}, config=cfg, context=user)
print("Save pref:", r["messages"][-1].content[:80])

# New thread, same store — pref persists
cfg2 = {"configurable": {"thread_id": str(uuid.uuid4())}}
r = runtime_agent.invoke({"messages": [HumanMessage("What are my preferences?")]}, config=cfg2, context=user)
print("Recall pref (new thread):", r["messages"][-1].content[:80])


# ════════════════════════════════════════════════════════════════════
# 3. RUNTIME IN MIDDLEWARE — context + execution_info in hooks
# ════════════════════════════════════════════════════════════════════
section("3. RUNTIME IN MIDDLEWARE")

_audit_log: list[dict] = []

@before_model
def audit_logger(state: AgentState, runtime: Runtime[UserContext]) -> dict | None:
    """Log every LLM call with user identity and execution metadata."""
    entry = {
        "user":    runtime.context.user_name,
        "role":    runtime.context.role,
        "run_id":  runtime.execution_info.run_id[:8],
        "thread":  runtime.execution_info.thread_id[:8],
        "msgs":    len(state.get("messages", [])),
    }
    _audit_log.append(entry)
    print(f"  [Audit] {entry}")
    return None

class RBACMiddleware(AgentMiddleware):
    """Block guest users with early exit."""
    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime[UserContext]) -> dict | Any | None:
        if runtime.context.role == "guest":
            print(f"  [RBAC] 🚫 Blocked guest {runtime.context.user_name}")
            return {
                "messages": [{"role": "assistant", "content": "Access denied for guest users."}],
                "jump_to": "end",
            }
        print(f"  [RBAC] ✅ Allowed role={runtime.context.role}")
        return None

rbac_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_my_profile],
    context_schema=UserContext,
    checkpointer=MemorySaver(),
    middleware=[RBACMiddleware(), audit_logger],
    system_prompt="You are a secure assistant.",
)

for user_ctx, q in [
    (UserContext("u-001", "Alice", "admin"),  "Show my profile."),
    (UserContext("u-002", "Bob",   "guest"),  "Show my profile."),
]:
    r = rbac_agent.invoke(
        {"messages": [{"role": "user", "content": q}]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
        context=user_ctx,
    )
    print(f"  [{user_ctx.role}] {r['messages'][-1].content[:80]}")

print(f"\nAudit log: {len(_audit_log)} LLM calls recorded")
for entry in _audit_log:
    print(f"  {entry}")


# ════════════════════════════════════════════════════════════════════
# 4. dynamic_prompt — build system prompt dynamically from context
# ════════════════════════════════════════════════════════════════════
section("4. DYNAMIC PROMPT")

@dataclass
class CRMContext:
    user_name: str
    role:      str
    tenant_id: str
    language:  str = "English"

@tool
def list_contacts(runtime: ToolRuntime[CRMContext]) -> str:
    """List contacts for the current tenant."""
    return f"Contacts for {runtime.context.tenant_id}: Alice, Bob, Carol."

@dynamic_prompt
def crm_prompt(request: ModelRequest) -> str:
    ctx = request.runtime.context
    role_hint = {"admin": "Full access.", "sales": "View contacts only.", "support": "View contacts and cases."}.get(ctx.role, "")
    return (
        f"You are a CRM assistant for tenant '{ctx.tenant_id}'. "
        f"Address the user as {ctx.user_name}. Respond in {ctx.language}. {role_hint}"
    )

crm_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[list_contacts],
    context_schema=CRMContext,
    middleware=[crm_prompt],
    checkpointer=MemorySaver(),
)

for ctx, q in [
    (CRMContext("Alex",  "admin",  "tenant-001"),           "List all contacts."),
    (CRMContext("Betty", "sales",  "tenant-002", "Hindi"),  "List all contacts."),
]:
    r = crm_agent.invoke(
        {"messages": [{"role": "user", "content": q}]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
        context=ctx,
    )
    print(f"  [{ctx.role}/{ctx.language}] {r['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 5. execution_info — run_id, thread_id for deduplication/tracing
# ════════════════════════════════════════════════════════════════════
section("5. EXECUTION INFO")

@tool
def get_execution_info(runtime: ToolRuntime) -> str:
    """Return execution metadata for the current run."""
    info = runtime.execution_info
    return f"run_id={info.run_id}  thread_id={info.thread_id}"

info_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_execution_info],
    checkpointer=MemorySaver(),
    system_prompt="Use get_execution_info when asked about run metadata.",
)
cfg3 = {"configurable": {"thread_id": "stable-thread-001"}}
r = info_agent.invoke({"messages": [HumanMessage("Show me the run and thread IDs.")]}, config=cfg3)
print("Execution info:", r["messages"][-1].content[:150])
