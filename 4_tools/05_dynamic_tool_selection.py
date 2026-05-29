"""
05_dynamic_tool_selection.py
============================
Demonstrates DYNAMIC TOOL SELECTION — adapting available tools at runtime
based on user role, conversation state, feature flags, or context.

Concepts covered:
  - wrap_model_call middleware    — intercept model calls to filter tools
  - request.override(tools=...)  — replace the tool list for this call
  - Role-based tool filtering    — admin / editor / viewer tool sets
  - State-based filtering        — authentication gates advanced tools
  - Context-based filtering      — per-user feature flag filtering
  - Pre-registered tools         — all tools known at startup, filtered dynamically
  - Runtime tool registration    — tools loaded/created at runtime (advanced)

Dynamic selection solves two problems:
  1. Too many tools overwhelm the model → reduce to what's relevant
  2. Tool availability should follow permissions → enforce at runtime
"""

import os
import uuid
from dataclasses import dataclass
from typing import Callable
from dotenv import load_dotenv

from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

print("=" * 60)
print("Dynamic Tool Selection Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. DEFINE ALL POSSIBLE TOOLS
#    (registered upfront; filtered at runtime)
# ════════════════════════════════════════════════════════════════════

# ── Public tools (everyone) ──────────────────────────────────────
@tool
def read_public_docs(topic: str) -> str:
    """Read public documentation about a topic.

    Args:
        topic: The topic to look up
    """
    return f"📄 Public docs for '{topic}': [general information available to all users]"


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base.

    Args:
        query: Search query
    """
    return f"🔍 Knowledge base results for '{query}': [3 articles found]"


# ── Editor tools (editor + admin) ──────────────────────────────
@tool
def create_document(title: str, content: str) -> str:
    """Create a new document in the system.

    Args:
        title:   Document title
        content: Document body text
    """
    return f"📝 Document created: '{title}' (ID: DOC-{abs(hash(title)) % 10000:04d})"


@tool
def update_document(doc_id: str, new_content: str) -> str:
    """Update an existing document.

    Args:
        doc_id:      Document ID (e.g. DOC-1234)
        new_content: New content for the document
    """
    return f"✏️  Document {doc_id} updated successfully."


# ── Admin tools (admin only) ────────────────────────────────────
@tool
def delete_document(doc_id: str) -> str:
    """Permanently delete a document from the system.

    Args:
        doc_id: Document ID to delete
    """
    return f"🗑️  Document {doc_id} deleted permanently."


@tool
def manage_users(action: str, user_email: str) -> str:
    """Manage user accounts (create, suspend, delete).

    Args:
        action:     'create', 'suspend', or 'delete'
        user_email: User's email address
    """
    return f"👤 User action '{action}' applied to {user_email}"


@tool
def view_audit_logs(days: int = 7) -> str:
    """View system audit logs for the past N days.

    Args:
        days: Number of days of logs to retrieve (default: 7)
    """
    return f"📋 Audit logs (last {days} days): [45 events found]"


ALL_TOOLS = [
    read_public_docs,
    search_knowledge_base,
    create_document,
    update_document,
    delete_document,
    manage_users,
    view_audit_logs,
]

ROLE_TOOLS = {
    "viewer": {"read_public_docs", "search_knowledge_base"},
    "editor": {"read_public_docs", "search_knowledge_base", "create_document", "update_document"},
    "admin":  {t.name for t in ALL_TOOLS},   # admins get everything
}


# ════════════════════════════════════════════════════════════════════
# 2. ROLE-BASED FILTERING (from Runtime Context)
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Role-based tool filtering ──────────────────────────")


@dataclass
class UserCtx:
    user_id: str
    role:    str = "viewer"


@wrap_model_call
def role_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Filter tools based on the user's role from context."""
    ctx = request.runtime.context if request.runtime else None

    if ctx and hasattr(ctx, "role"):
        allowed = ROLE_TOOLS.get(ctx.role, ROLE_TOOLS["viewer"])
        filtered = [t for t in request.tools if t.name in allowed]
        print(f"    [middleware] role={ctx.role!r} → {len(filtered)}/{len(request.tools)} tools available")
        request = request.override(tools=filtered)

    return handler(request)


role_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=ALL_TOOLS,
    middleware=[role_based_tools],
    context_schema=UserCtx,
    checkpointer=MemorySaver(),
    system_prompt="You are a document management assistant. Use only the tools available to you.",
)


def ask_as_role(role: str, question: str) -> str:
    user = UserCtx(user_id=f"u-{role}", role=role)
    result = role_agent.invoke(
        {"messages": [HumanMessage(question)]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
        context=user,
    )
    return result["messages"][-1].content


print(f"\n  👤 ADMIN asks to delete DOC-001:")
print(f"  🤖 {ask_as_role('admin', 'Delete the document with ID DOC-001.')}")

print(f"\n  📝 EDITOR asks to create a document:")
print(f"  🤖 {ask_as_role('editor', 'Create a new document titled Meeting Notes with content: Discussed Q3 goals.')}")

print(f"\n  👁️  VIEWER tries to delete (should fail gracefully):")
print(f"  🤖 {ask_as_role('viewer', 'Delete the document with ID DOC-001.')}")


# ════════════════════════════════════════════════════════════════════
# 3. STATE-BASED FILTERING (authentication gate)
# ════════════════════════════════════════════════════════════════════

print("\n── 3. State-based tool filtering ────────────────────────")

# Only show advanced tools AFTER user has authenticated in the conversation.
# "authenticated" is a custom field in agent state.

from langchain.agents import AgentState
from langgraph.types import Command
from langchain_core.messages import ToolMessage


class AuthState(AgentState):
    authenticated: bool = False


@tool
def authenticate(password: str, runtime: ToolRuntime) -> Command:
    """Authenticate the user to unlock advanced tools.

    Args:
        password: User's password (use 'secret123' for this demo)
    """
    is_valid = password == "secret123"
    return Command(
        update={
            "authenticated": is_valid,
            "messages": [
                ToolMessage(
                    content="✅ Authentication successful! Advanced tools are now unlocked." if is_valid
                    else "❌ Invalid password. Please try again.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def access_sensitive_data(data_type: str) -> str:
    """Access sensitive business data (requires authentication).

    Args:
        data_type: Type of data to access (e.g. 'financials', 'hr_records')
    """
    return f"🔐 Sensitive data ({data_type}): [confidential records]"


@wrap_model_call
def auth_gate_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Only expose sensitive tools after authentication."""
    state = request.state or {}
    is_auth = state.get("authenticated", False)

    if not is_auth:
        # Only show authenticate tool until user is logged in
        safe_tools = [t for t in request.tools if t.name == "authenticate"]
        print(f"    [middleware] not authenticated → {len(safe_tools)} tool(s) available")
    else:
        safe_tools = request.tools
        print(f"    [middleware] authenticated → all {len(safe_tools)} tools available")

    return handler(request.override(tools=safe_tools))


auth_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[authenticate, access_sensitive_data, search_knowledge_base],
    middleware=[auth_gate_tools],
    state_schema=AuthState,
    checkpointer=MemorySaver(),
    system_prompt=(
        "You are a secure data assistant. Users must authenticate before "
        "accessing sensitive data. Guide them through the process."
    ),
)

auth_thread = str(uuid.uuid4())
auth_config  = {"configurable": {"thread_id": auth_thread}}

print(f"\n  Unauthenticated request:")
r = auth_agent.invoke(
    {"messages": [HumanMessage("Show me the financial records.")]},
    config=auth_config,
)
print(f"  🤖 {r['messages'][-1].content}")

print(f"\n  Authentication attempt:")
r = auth_agent.invoke(
    {"messages": [HumanMessage("My password is secret123")]},
    config=auth_config,
)
print(f"  🤖 {r['messages'][-1].content}")

print(f"\n  Authenticated request:")
r = auth_agent.invoke(
    {"messages": [HumanMessage("Now show me the financial records.")]},
    config=auth_config,
)
print(f"  🤖 {r['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 4. SUMMARY: WHEN TO USE DYNAMIC TOOLS
# ════════════════════════════════════════════════════════════════════

print("\n── 4. When to use dynamic tool selection ─────────────────")
print("""
  Use dynamic tool selection when:
  ✅ Different users have different permissions (RBAC)
  ✅ Features are gated behind authentication or feature flags
  ✅ Too many tools overwhelm the model (>15 tools → filter down)
  ✅ Tools should only appear at specific conversation stages
  ✅ Tools depend on user plan (free vs. pro vs. enterprise)

  Approaches:
  ┌────────────────────────┬──────────────────────────────────────┐
  │ Approach               │ When to use                          │
  ├────────────────────────┼──────────────────────────────────────┤
  │ Filter pre-registered  │ All tools known at startup time      │
  │ (wrap_model_call)      │ Most common approach                 │
  ├────────────────────────┼──────────────────────────────────────┤
  │ Runtime registration   │ Tools loaded from MCP/API at runtime │
  │                        │ Tools generated from user data       │
  └────────────────────────┴──────────────────────────────────────┘
""")
