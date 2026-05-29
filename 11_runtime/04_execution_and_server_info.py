"""
04_execution_and_server_info.py
================================
Deep-dive into runtime.execution_info and runtime.server_info —
the two sources of identity and deployment metadata on the Runtime object.

Concepts covered:
  - runtime.execution_info: thread_id, run_id, attempt (always available)
  - runtime.server_info: assistant_id, graph_id, user (LangGraph Server only)
  - Detecting retry attempts inside tools
  - Audit trail pattern using execution_info
  - Conditional logic based on server vs local environment
  - User identity from server_info.user.identity
"""

import os
import uuid
from dataclasses import dataclass, field
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime

load_dotenv()

print("=" * 60)
print("Execution Info & Server Info Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. EXECUTION INFO — thread_id, run_id, attempt
#    Always present regardless of deployment mode.
# ════════════════════════════════════════════════════════════════════

@tool
def echo_execution_info(runtime: ToolRuntime) -> str:
    """Return the current execution identity from the runtime."""
    info = runtime.execution_info
    print(f"  [Tool] execution_info → "
          f"thread={info.thread_id}, run={info.run_id}, attempt={info.attempt}")
    return (
        f"Execution identity: "
        f"thread_id={info.thread_id}, "
        f"run_id={info.run_id}, "
        f"attempt={info.attempt}."
    )


print("\n── 1. Accessing execution_info in a Tool ────────────────────")

agent_exec = create_agent(
    model="openai:gpt-4o-mini",
    tools=[echo_execution_info],
    checkpointer=MemorySaver(),
    system_prompt="You are an identity assistant.",
)

config = {"configurable": {"thread_id": "exec-demo-thread-001"}}
result = agent_exec.invoke(
    {"messages": [{"role": "user", "content": "Show me the current execution info."}]},
    config=config,
)
print(f"Response: {result['messages'][-1].content[:150]}")


# ════════════════════════════════════════════════════════════════════
# 2. RETRY DETECTION — using attempt number inside a tool
#    attempt=0 is the first try. attempt>0 means the tool is being
#    retried (e.g. after a transient failure).
# ════════════════════════════════════════════════════════════════════

_call_tracker: dict[str, int] = {}

@tool
def flaky_external_api(endpoint: str, runtime: ToolRuntime) -> str:
    """
    Call an external API endpoint that may fail transiently.

    Args:
        endpoint: The API endpoint path to call.
    """
    info  = runtime.execution_info
    key   = f"{info.run_id}:{endpoint}"
    count = _call_tracker.get(key, 0) + 1
    _call_tracker[key] = count

    print(f"  [Tool] flaky_external_api → endpoint={endpoint}, "
          f"attempt={info.attempt}, call_count={count}")

    if info.attempt == 0 and count == 1:
        # First attempt of the run — simulate a transient failure
        raise ConnectionError(f"Timeout calling {endpoint} — please retry.")

    return f"API {endpoint} responded: 200 OK (succeeded on attempt {info.attempt})."


print("\n── 2. Retry Detection via attempt Number ─────────────────────")

from langchain.agents.middleware import ToolRetryMiddleware

agent_retry = create_agent(
    model="openai:gpt-4o-mini",
    tools=[flaky_external_api],
    middleware=[ToolRetryMiddleware(max_retries=3)],
    system_prompt="You are an API integration assistant.",
)

result_retry = agent_retry.invoke({
    "messages": [{"role": "user", "content": "Call the /health endpoint."}]
})
print(f"Response: {result_retry['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 3. AUDIT TRAIL — using run_id for deduplication / logging
#    Each agent.invoke() produces a unique run_id. Use it to
#    deduplicate webhook callbacks or audit logs.
# ════════════════════════════════════════════════════════════════════

audit_log: list[dict] = []

@before_model
def audit_middleware(state: AgentState, runtime: Runtime) -> dict | None:
    """Record each model call in an audit log with run/thread identity."""
    info = runtime.execution_info
    entry = {
        "thread_id": info.thread_id,
        "run_id":    info.run_id,
        "attempt":   info.attempt,
        "messages":  len(state.get("messages", [])),
    }
    audit_log.append(entry)
    print(f"  [AuditLog] {entry}")
    return None


print("\n── 3. Audit Trail Pattern ────────────────────────────────────")

agent_audit = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    checkpointer=MemorySaver(),
    middleware=[audit_middleware],
    system_prompt="You are a helpful assistant.",
)

config_a = {"configurable": {"thread_id": "audit-thread-A"}}
agent_audit.invoke(
    {"messages": [{"role": "user", "content": "Tell me a joke."}]},
    config=config_a,
)
agent_audit.invoke(
    {"messages": [{"role": "user", "content": "And another one."}]},
    config=config_a,
)

print(f"\nAudit log entries recorded: {len(audit_log)}")
for entry in audit_log:
    print(f"  thread={entry['thread_id']}, run={entry['run_id']}, "
          f"attempt={entry['attempt']}, msgs={entry['messages']}")


# ════════════════════════════════════════════════════════════════════
# 4. SERVER INFO — local vs LangGraph Server detection
#    server_info is None during local dev, populated in production.
# ════════════════════════════════════════════════════════════════════

@tool
def deployment_info(runtime: ToolRuntime) -> str:
    """Return deployment context — local dev or LangGraph Server."""
    server = runtime.server_info
    if server is None:
        print("  [Tool] deployment_info → local development mode")
        return "Running in local development mode. No server metadata available."

    # On LangGraph Server
    user_info = "anonymous"
    if server.user is not None:
        user_info = server.user.identity
    print(f"  [Tool] deployment_info → server mode, assistant={server.assistant_id}")
    return (
        f"Running on LangGraph Server. "
        f"Assistant ID: {server.assistant_id}, "
        f"Graph ID: {server.graph_id}, "
        f"Authenticated user: {user_info}."
    )


print("\n── 4. Server Info (local vs LangGraph Server) ───────────────")

agent_deploy = create_agent(
    model="openai:gpt-4o-mini",
    tools=[deployment_info],
    system_prompt="You are a deployment info assistant.",
)

result_deploy = agent_deploy.invoke({
    "messages": [{"role": "user", "content": "Where is this agent running?"}]
})
print(f"Response: {result_deploy['messages'][-1].content[:150]}")


# ════════════════════════════════════════════════════════════════════
# 5. AUTH GATE using server_info — production deployment pattern
#    In production (LangGraph Server), block requests from
#    unauthenticated users using server_info.user.
# ════════════════════════════════════════════════════════════════════

from langchain.agents.middleware import AgentMiddleware, hook_config
from typing import Any

class ProductionAuthMiddleware(AgentMiddleware):
    """Gate agent access: require authenticated users on LangGraph Server."""

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        server = runtime.server_info
        if server is None:
            # Local development — allow all
            print("  [AuthMiddleware] Local mode — allowing all requests")
            return None

        # LangGraph Server — require authentication
        if server.user is None:
            print("  [AuthMiddleware] 🚫 Unauthenticated request blocked")
            return {
                "messages": [{"role": "assistant",
                              "content": "Authentication required. Please log in."}],
                "jump_to": "end",
            }

        print(f"  [AuthMiddleware] ✅ Authenticated: {server.user.identity}")
        return None


print("\n── 5. Production Auth Gate Pattern ──────────────────────────")

agent_prod = create_agent(
    model="openai:gpt-4o-mini",
    tools=[deployment_info],
    middleware=[ProductionAuthMiddleware()],
    system_prompt="You are a secure production assistant.",
)

result_prod = agent_prod.invoke({
    "messages": [{"role": "user", "content": "What deployment am I connected to?"}]
})
print(f"Response (local): {result_prod['messages'][-1].content[:120]}")

print("\n" + "═" * 60)
print("Execution & Server Info Reference:")
print("  runtime.execution_info.thread_id  — conversation thread ID")
print("  runtime.execution_info.run_id     — unique ID per invoke() call")
print("  runtime.execution_info.attempt    — retry count (0 = first try)")
print("  runtime.server_info               — None in local dev")
print("  runtime.server_info.assistant_id  — LangGraph Server assistant")
print("  runtime.server_info.graph_id      — deployed graph identifier")
print("  runtime.server_info.user          — authenticated user or None")
print("  runtime.server_info.user.identity — user identity string")
print("═" * 60)
print("\n✅ Execution & server info demo complete.")
