"""
06_context_and_runtime.py
=========================
Demonstrates per-run CONTEXT — passing runtime data (user ID, API keys,
feature flags) to tools and middleware without hardcoding them.

Concepts covered:
  - context_schema= — declare the shape of runtime context as a dataclass
  - context= parameter on invoke() — supply actual context at call time
  - runtime.context inside a tool — read context values during execution
  - thread_id vs context — thread_id scopes history, context carries per-run data

Use context for anything that changes per request:
  • Current user's ID / permissions
  • Tenant-specific API keys
  • Feature flags
  • Locale / timezone

Use thread_id for anything that spans multiple turns of a conversation.
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.agents.runtime import get_runtime     # access context inside a tool
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ── 1. Define the Context schema ──────────────────────────────────────────────
# Use a plain Python dataclass.  All fields are available in tools via runtime.

@dataclass
class UserContext:
    user_id: str
    username: str
    role: str                            = "viewer"         # admin | editor | viewer
    preferred_language: str             = "English"
    api_key_for_external_service: str   = ""
    tenant_id: Optional[str]            = None


# ── 2. Define tools that READ context ────────────────────────────────────────

@tool
def get_user_profile() -> str:
    """Return the current user's profile information."""
    runtime = get_runtime()
    ctx: UserContext = runtime.context
    return (
        f"User profile:\n"
        f"  ID:       {ctx.user_id}\n"
        f"  Name:     {ctx.username}\n"
        f"  Role:     {ctx.role}\n"
        f"  Language: {ctx.preferred_language}\n"
        f"  Tenant:   {ctx.tenant_id or 'default'}"
    )


@tool
def perform_admin_action(action: str) -> str:
    """Perform an administrative action (only allowed for admin users).

    Args:
        action: The action to perform (e.g. 'reset_database', 'clear_cache')
    """
    runtime = get_runtime()
    ctx: UserContext = runtime.context

    # ── Role-based access control inside a tool ───────────────────────────────
    if ctx.role != "admin":
        return (
            f"❌ Access denied. User '{ctx.username}' (role={ctx.role}) "
            f"does not have permission to perform '{action}'. "
            f"Admin role required."
        )
    return f"✅ Admin action '{action}' executed successfully by {ctx.username}"


@tool
def fetch_personalised_data(topic: str) -> str:
    """Fetch data personalised to the current user's language and preferences.

    Args:
        topic: Topic to fetch data about
    """
    runtime = get_runtime()
    ctx: UserContext = runtime.context

    return (
        f"Personalised data for '{topic}' "
        f"(user={ctx.username}, language={ctx.preferred_language}):\n"
        f"Here is curated content about {topic} in {ctx.preferred_language}."
    )


# ── 3. Create the agent with context_schema ───────────────────────────────────

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_user_profile, perform_admin_action, fetch_personalised_data],
    context_schema=UserContext,          # ← declares shape of context
    checkpointer=MemorySaver(),
    system_prompt=(
        "You are a personalised assistant. Always greet the user by name. "
        "Use tools to fetch data specific to the current user."
    ),
)


# ── 4. Invoke with different contexts ────────────────────────────────────────

def run_as(user: UserContext, question: str) -> str:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        context=user,                    # ← per-run context
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    print("=" * 60)
    print("Context & Runtime Demo")
    print("=" * 60)

    # ── Scenario 1: Admin user ────────────────────────────────────────────────
    admin_user = UserContext(
        user_id="u-001",
        username="Vinod",
        role="admin",
        preferred_language="English",
        tenant_id="acme-corp",
    )

    print("\n👤 Running as ADMIN user (Vinod) …")
    print(f"\n🧑 What's my profile?")
    print(f"🤖 {run_as(admin_user, 'What is my profile?')}")

    print(f"\n🧑 Reset the database cache.")
    print(f"🤖 {run_as(admin_user, 'Please perform the action: reset_database')}")

    # ── Scenario 2: Viewer user (no admin rights) ─────────────────────────────
    viewer_user = UserContext(
        user_id="u-042",
        username="Priya",
        role="viewer",
        preferred_language="Hindi",
    )

    print("\n" + "─" * 60)
    print("\n👤 Running as VIEWER user (Priya) …")
    print(f"\n🧑 Fetch data about LangChain.")
    print(f"🤖 {run_as(viewer_user, 'Fetch personalised data about LangChain for me.')}")

    print(f"\n🧑 Try admin action (should be denied).")
    print(f"🤖 {run_as(viewer_user, 'Perform the admin action: delete_all_logs')}")

    # ── Key takeaway ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY POINT:")
    print("  thread_id  → scopes conversation HISTORY across turns")
    print("  context    → carries PER-RUN data (user, keys, flags)")
    print("  Both can be used together on every invoke() call.")
    print("=" * 60)
