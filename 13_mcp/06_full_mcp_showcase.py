"""
06_full_mcp_showcase.py
========================
Production-ready showcase: a SMART DATA ASSISTANT that combines
MCP tools, interceptors, callbacks, and context engineering.

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │                   Smart Data Assistant                           │
  ├─────────────────────────────────────────────────────────────────┤
  │  MCP Servers:                                                    │
  │    math (stdio) — arithmetic tools                               │
  │                                                                 │
  │  Interceptors:                                                   │
  │    audit_interceptor   — log every MCP tool call to store       │
  │    auth_interceptor    — gate sensitive tools by state           │
  │    context_interceptor — inject user_id from runtime context     │
  │                                                                 │
  │  Callbacks:                                                      │
  │    on_progress         — stream analysis progress to console     │
  │                                                                 │
  │  Context Schema (DataCtx):                                       │
  │    user_id, role, tenant                                         │
  │                                                                 │
  │  LangChain Middleware:                                           │
  │    @dynamic_prompt     — role-aware system prompt               │
  │    @before_model audit — log model calls to store               │
  │                                                                 │
  │  Scenarios:                                                      │
  │    A. Authenticated power user — full tool access               │
  │    B. Unauthenticated user — restricted access                  │
  │    C. Multi-step calculation — chained MCP tool calls           │
  │    D. Store-based audit trail — cross-session log retrieval     │
  └─────────────────────────────────────────────────────────────────┘
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest, before_model
from langchain.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

load_dotenv()

MATH_SERVER = str(Path(__file__).parent / "servers" / "math_server.py")

print("=" * 60)
print("Smart Data Assistant — Full MCP Showcase")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# CONTEXT SCHEMA
# ════════════════════════════════════════════════════════════════════

@dataclass
class DataCtx:
    user_id: str
    role:    str    # "analyst", "viewer", "admin"
    tenant:  str


# ════════════════════════════════════════════════════════════════════
# MCP INTERCEPTORS
# ════════════════════════════════════════════════════════════════════

shared_store = InMemoryStore()

RESTRICTED_TOOLS = {"divide"}   # treat as "privileged" for the demo


async def audit_interceptor(request: MCPToolCallRequest, handler):
    """Log every MCP tool call to long-term store for audit trail."""
    runtime = request.runtime
    user_id = runtime.context.user_id if (runtime and runtime.context) else "anonymous"
    store   = runtime.store if runtime else None

    start = time.time()
    try:
        result = await handler(request)
        status = "success"
    except Exception as e:
        status = f"error:{e}"
        raise
    finally:
        elapsed = round(time.time() - start, 3)
        if store:
            existing = store.get(("mcp_audit",), user_id)
            log      = existing.value.get("calls", []) if existing else []
            log.append({
                "tool":    request.name,
                "args":    str(request.args)[:80],
                "status":  status,
                "elapsed": elapsed,
                "ts":      time.time(),
            })
            store.put(("mcp_audit",), user_id, {"calls": log[-100:]})
            print(f"  [MCPAudit] {request.name} → {status} ({elapsed}s)")

    return result


async def auth_interceptor(request: MCPToolCallRequest, handler):
    """Block restricted MCP tools for unauthenticated sessions."""
    runtime = request.runtime
    if runtime and request.name in RESTRICTED_TOOLS:
        state = runtime.state
        if not state.get("authenticated", False):
            print(f"  [Auth] 🚫 Blocked '{request.name}' — not authenticated")
            return ToolMessage(
                content=f"Tool '{request.name}' requires authentication.",
                tool_call_id=runtime.tool_call_id,
            )
        print(f"  [Auth] ✅ '{request.name}' — authenticated")
    return await handler(request)


async def context_interceptor(request: MCPToolCallRequest, handler):
    """Forward user_id from runtime context to MCP tool args."""
    runtime = request.runtime
    if runtime and runtime.context:
        user_id = runtime.context.user_id
        modified = request.override(args={**request.args, "_user": user_id})
        return await handler(modified)
    return await handler(request)


# ════════════════════════════════════════════════════════════════════
# MCP CALLBACKS
# ════════════════════════════════════════════════════════════════════

async def progress_callback(progress, total, message, context: CallbackContext):
    pct = f"{(progress / total * 100):.0f}%" if total else f"{progress}"
    print(f"  [Progress] {pct} — {message or ''}")


# ════════════════════════════════════════════════════════════════════
# LANGCHAIN MIDDLEWARE
# ════════════════════════════════════════════════════════════════════

@dynamic_prompt
def data_assistant_prompt(request: ModelRequest) -> str:
    """Role-aware system prompt."""
    ctx = request.runtime.context
    role_hints = {
        "analyst": "You have full access to all math tools.",
        "viewer":  "You can read data but cannot perform division operations.",
        "admin":   "You have full system access.",
    }
    return (
        f"You are a data assistant for tenant '{ctx.tenant}'. "
        f"{role_hints.get(ctx.role, '')} "
        f"Always be concise and precise in your calculations."
    )


_model_call_log: list[dict] = []

@before_model
def model_audit(state: AgentState, runtime: Runtime[DataCtx]) -> dict | None:
    """Persist model call metadata."""
    info = runtime.execution_info
    _model_call_log.append({
        "user":   runtime.context.user_id,
        "role":   runtime.context.role,
        "thread": info.thread_id,
        "msgs":   len(state.get("messages", [])),
    })
    return None


# ════════════════════════════════════════════════════════════════════
# BUILD AGENT
# ════════════════════════════════════════════════════════════════════

async def build_agent():
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command":   "python",
                "args":      [MATH_SERVER],
            }
        },
        tool_interceptors=[
            audit_interceptor,       # outermost — always runs
            auth_interceptor,        # gates restricted tools
            context_interceptor,     # injects user context
        ],
        callbacks=Callbacks(on_progress=progress_callback),
    )

    tools = await client.get_tools()
    print(f"\nLoaded {len(tools)} MCP tools: {[t.name for t in tools]}")

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        context_schema=DataCtx,
        store=shared_store,
        checkpointer=MemorySaver(),
        middleware=[data_assistant_prompt, model_audit],
    )
    return agent


# ════════════════════════════════════════════════════════════════════
# SCENARIOS
# ════════════════════════════════════════════════════════════════════

async def main():
    agent = await build_agent()

    # ── SCENARIO A: Authenticated analyst — full access ───────────
    print("\n" + "─" * 60)
    print("SCENARIO A — Authenticated Analyst (full access)")
    print("─" * 60)

    ctx_a  = DataCtx(user_id="ANL-001", role="analyst", tenant="acme")
    cfg_a  = {"configurable": {"thread_id": "mcp-analyst-1"}}

    r_a = await agent.ainvoke(
        {
            "messages":      [{"role": "user", "content":
                "Add 250 and 175, then divide the result by 5."}],
            "authenticated": True,
        },
        context=ctx_a,
        config=cfg_a,
    )
    print(f"\nA: {r_a['messages'][-1].content}")

    # ── SCENARIO B: Unauthenticated viewer — divide blocked ───────
    print("\n" + "─" * 60)
    print("SCENARIO B — Unauthenticated Viewer (restricted)")
    print("─" * 60)

    ctx_b = DataCtx(user_id="VWR-001", role="viewer", tenant="acme")
    cfg_b = {"configurable": {"thread_id": "mcp-viewer-1"}}

    r_b = await agent.ainvoke(
        {
            "messages":      [{"role": "user", "content":
                "Add 100 and 50. Then try to divide 200 by 4."}],
            "authenticated": False,
        },
        context=ctx_b,
        config=cfg_b,
    )
    print(f"\nB: {r_b['messages'][-1].content[:200]}")

    # ── SCENARIO C: Multi-step chained calculation ────────────────
    print("\n" + "─" * 60)
    print("SCENARIO C — Multi-Step Chained Calculation")
    print("─" * 60)

    ctx_c = DataCtx(user_id="ANL-002", role="analyst", tenant="beta")
    cfg_c = {"configurable": {"thread_id": "mcp-chain-1"}}

    r_c = await agent.ainvoke(
        {
            "messages":      [{"role": "user", "content":
                "Calculate: ((15 + 35) × 4) − 50, then divide by 10."}],
            "authenticated": True,
        },
        context=ctx_c,
        config=cfg_c,
    )
    print(f"\nC: {r_c['messages'][-1].content}")

    # ── SCENARIO D: Multi-turn conversation ──────────────────────
    print("\n" + "─" * 60)
    print("SCENARIO D — Multi-Turn Conversation (same thread)")
    print("─" * 60)

    ctx_d = DataCtx(user_id="ANL-001", role="analyst", tenant="acme")
    cfg_d = {"configurable": {"thread_id": "mcp-multi-turn"}}

    for q in ["What is 8 × 7?", "Subtract 10 from that result.", "Now divide by 6."]:
        r = await agent.ainvoke(
            {"messages": [{"role": "user", "content": q}], "authenticated": True},
            context=ctx_d,
            config=cfg_d,
        )
        print(f"  Q: {q}")
        print(f"  A: {r['messages'][-1].content[:80]}\n")

    # ── AUDIT SUMMARY ─────────────────────────────────────────────
    print("═" * 60)
    print("MCP Audit Trail:")
    for user_id in ("ANL-001", "ANL-002", "VWR-001"):
        audit = shared_store.get(("mcp_audit",), user_id)
        if audit:
            calls = audit.value.get("calls", [])
            print(f"  {user_id}: {len(calls)} MCP tool call(s)")
            for c in calls:
                print(f"    {c['tool']} → {c['status']} ({c['elapsed']}s)")

    print("\nModel Call Log:")
    for entry in _model_call_log:
        print(f"  user={entry['user']}, role={entry['role']}, "
              f"thread={entry['thread']}, msgs={entry['msgs']}")

    print("\n" + "═" * 60)
    print("Full MCP Showcase — Components Used:")
    print("  MultiServerMCPClient   — manages math stdio server")
    print("  audit_interceptor      — logs every MCP call to store")
    print("  auth_interceptor       — gates divide tool by auth state")
    print("  context_interceptor    — injects user_id from runtime ctx")
    print("  progress_callback      — streams server progress events")
    print("  @dynamic_prompt        — role-aware system prompt")
    print("  @before_model audit    — LangChain model call logging")
    print("  MemorySaver            — multi-turn conversation memory")
    print("  InMemoryStore          — cross-session MCP audit trail")
    print("═" * 60)
    print("\n✅ Full MCP showcase complete.")


if __name__ == "__main__":
    asyncio.run(main())
