"""
03_runtime_in_middleware.py
===========================
Demonstrates how to access the Runtime object inside middleware hooks —
for dynamic prompts, before/after model hooks, and authentication gates.

Concepts covered:
  - @dynamic_prompt — generate a system prompt from context at call time
  - @before_model with Runtime[Context] — log/modify state before LLM call
  - @after_model with Runtime[Context] — inspect/modify state after LLM call
  - runtime.execution_info — thread_id, run_id, attempt_number
  - runtime.server_info   — assistant_id, user (LangGraph Server only)
  - Auth gate middleware using server_info.user
  - Combining runtime-aware middleware with context_schema
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    dynamic_prompt,
    ModelRequest,
    before_model,
    after_model,
    before_agent,
    hook_config,
    AgentMiddleware,
)
from langchain.tools import tool
from langgraph.runtime import Runtime

load_dotenv()

print("=" * 60)
print("Runtime in Middleware Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# CONTEXT DEFINITION
# ════════════════════════════════════════════════════════════════════

@dataclass
class UserContext:
    user_name: str
    user_id:   str
    role:      str = "user"  # "user", "admin", "guest"
    language:  str = "English"


# ════════════════════════════════════════════════════════════════════
# SHARED TOOL
# ════════════════════════════════════════════════════════════════════

@tool
def get_system_status() -> str:
    """Return the current system status."""
    return "All systems operational. CPU: 23%, Memory: 61%, Disk: 45%."


# ════════════════════════════════════════════════════════════════════
# 1. DYNAMIC PROMPT — generate system prompt from runtime context
#    @dynamic_prompt reads ModelRequest which exposes runtime.context
# ════════════════════════════════════════════════════════════════════

@dynamic_prompt
def personalized_system_prompt(request: ModelRequest) -> str:
    """Build a personalized system prompt from the injected context."""
    ctx = request.runtime.context
    print(f"  [DynamicPrompt] Building prompt for user='{ctx.user_name}', lang={ctx.language}")
    role_note = ""
    if ctx.role == "admin":
        role_note = " You may reveal system internals to this user."
    elif ctx.role == "guest":
        role_note = " Provide only basic information to guest users."
    return (
        f"You are a helpful assistant. "
        f"Address the user as {ctx.user_name}. "
        f"Always respond in {ctx.language}.{role_note}"
    )


print("\n── 1. Dynamic Prompt from Runtime Context ───────────────────")

agent_dynamic = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_system_status],
    context_schema=UserContext,
    middleware=[personalized_system_prompt],
)

result_en = agent_dynamic.invoke(
    {"messages": [{"role": "user", "content": "Check the system status."}]},
    context=UserContext(user_id="USR-1", user_name="Alice", role="admin", language="English"),
)
print(f"English (admin):  {result_en['messages'][-1].content[:100]}")

result_es = agent_dynamic.invoke(
    {"messages": [{"role": "user", "content": "Check the system status."}]},
    context=UserContext(user_id="USR-2", user_name="Carlos", role="user", language="Spanish"),
)
print(f"Spanish (user):   {result_es['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 2. BEFORE MODEL + AFTER MODEL with Runtime[Context]
#    Use these hooks to log, count, or modify per-user behavior
#    around every LLM call in the agent loop.
# ════════════════════════════════════════════════════════════════════

@before_model
def log_before_model(state: AgentState, runtime: Runtime[UserContext]) -> dict | None:
    """Log each LLM call with user context."""
    user = runtime.context.user_name
    n_msgs = len(state.get("messages", []))
    print(f"  [before_model] User='{user}', messages in context: {n_msgs}")
    return None  # Pass through unchanged


@after_model
def log_after_model(state: AgentState, runtime: Runtime[UserContext]) -> dict | None:
    """Log the model response with user context."""
    user = runtime.context.user_name
    last = state["messages"][-1] if state.get("messages") else None
    preview = (last.content[:60] + "...") if last and last.content else "(no content)"
    print(f"  [after_model]  User='{user}', response: {preview}")
    return None


print("\n── 2. before_model + after_model with Runtime[Context] ──────")

agent_logged = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_system_status],
    context_schema=UserContext,
    middleware=[personalized_system_prompt, log_before_model, log_after_model],
)

result_logged = agent_logged.invoke(
    {"messages": [{"role": "user", "content": "What's the current status?"}]},
    context=UserContext(user_id="USR-3", user_name="Diana", language="English"),
)
print(f"\nFinal response: {result_logged['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 3. EXECUTION INFO — thread_id, run_id, attempt_number
#    Available from runtime.execution_info in any hook.
#    Useful for audit logging, deduplication, retry detection.
# ════════════════════════════════════════════════════════════════════

@before_model
def audit_logger(state: AgentState, runtime: Runtime[UserContext]) -> dict | None:
    """Log execution identity for audit trail."""
    info = runtime.execution_info
    user = runtime.context.user_name
    print(
        f"  [AuditLog] user={user}, "
        f"thread={info.thread_id}, "
        f"run={info.run_id}, "
        f"attempt={info.attempt}"
    )
    return None


print("\n── 3. Execution Info (thread_id, run_id, attempt) ───────────")

from langgraph.checkpoint.memory import MemorySaver

agent_exec = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    context_schema=UserContext,
    checkpointer=MemorySaver(),
    middleware=[audit_logger],
    system_prompt="You are a helpful assistant.",
)

config = {"configurable": {"thread_id": "thread-audit-001"}}
agent_exec.invoke(
    {"messages": [{"role": "user", "content": "Hello!"}]},
    context=UserContext(user_id="USR-9", user_name="Frank"),
    config=config,
)
print("  (execution info logged above ☝)")


# ════════════════════════════════════════════════════════════════════
# 4. SERVER INFO — assistant_id, user (LangGraph Server only)
#    runtime.server_info is None during local development.
#    Use it for authentication gates in production deployments.
# ════════════════════════════════════════════════════════════════════

@before_model
def auth_gate(state: AgentState, runtime: Runtime) -> dict | None:
    """Block unauthenticated users when running on LangGraph Server."""
    server = runtime.server_info
    if server is not None:
        # Running on LangGraph Server — check authentication
        if server.user is None:
            raise ValueError("Authentication required to use this agent.")
        print(f"  [AuthGate] Authenticated user: {server.user.identity}")
        print(f"  [AuthGate] Assistant: {server.assistant_id}")
    else:
        # Local development — skip auth check
        print("  [AuthGate] Local mode — skipping server auth check")
    return None


print("\n── 4. Server Info (LangGraph Server auth gate) ──────────────")

agent_auth = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_system_status],
    context_schema=UserContext,
    middleware=[auth_gate, personalized_system_prompt],
)

result_auth = agent_auth.invoke(
    {"messages": [{"role": "user", "content": "What is the system status?"}]},
    context=UserContext(user_id="USR-5", user_name="Grace"),
)
print(f"Response: {result_auth['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 5. ROLE-BASED ACCESS CONTROL via before_agent
#    Demonstrate using context.role to gate access to the agent.
# ════════════════════════════════════════════════════════════════════

from typing import Any

class RoleBasedAccessMiddleware(AgentMiddleware):
    """Block guest users from accessing the agent entirely."""

    def __init__(self, allowed_roles: list[str]):
        super().__init__()
        self.allowed_roles = allowed_roles

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
        role = runtime.context.role
        name = runtime.context.user_name
        if role not in self.allowed_roles:
            print(f"  [RBAC] 🚫 Blocked role='{role}' for user='{name}'")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": f"Access denied. Your role '{role}' does not have permission.",
                }],
                "jump_to": "end",
            }
        print(f"  [RBAC] ✅ Access granted: role='{role}', user='{name}'")
        return None


print("\n── 5. Role-Based Access Control via Runtime Context ─────────")

agent_rbac = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_system_status],
    context_schema=UserContext,
    middleware=[RoleBasedAccessMiddleware(allowed_roles=["user", "admin"])],
    system_prompt="You are a system monitoring assistant.",
)

# Admin — allowed
result_admin = agent_rbac.invoke(
    {"messages": [{"role": "user", "content": "System status?"}]},
    context=UserContext(user_id="ADM-1", user_name="Henry", role="admin"),
)
print(f"Admin: {result_admin['messages'][-1].content[:80]}")

# Guest — blocked
result_guest = agent_rbac.invoke(
    {"messages": [{"role": "user", "content": "System status?"}]},
    context=UserContext(user_id="GST-1", user_name="Irene", role="guest"),
)
print(f"Guest: {result_guest['messages'][-1].content[:80]}")

print("\n✅ Runtime in middleware demo complete.")
