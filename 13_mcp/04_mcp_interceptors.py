"""
04_mcp_interceptors.py
=======================
Demonstrates MCP tool interceptors — middleware-like functions that
wrap tool execution to add cross-cutting logic without modifying servers.

Concepts covered:
  - Basic interceptor pattern (async function + handler)
  - Injecting runtime context into MCP tool arguments
  - Reading from Store via interceptor
  - Gating tools based on State (authentication)
  - request.override() — immutable request modification
  - Modifying HTTP headers at runtime
  - Command return — update State or control graph flow
  - Composing multiple interceptors ("onion" order)
  - Retry with exponential backoff
  - Fallback on error

All interceptors use stdio math server (no HTTP server needed).
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

load_dotenv()

MATH_SERVER = str(Path(__file__).parent / "servers" / "math_server.py")

print("=" * 60)
print("MCP Interceptors")
print("=" * 60)


def make_math_client(**kwargs) -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {"math": {"transport": "stdio", "command": "python", "args": [MATH_SERVER]}},
        **kwargs,
    )


# ════════════════════════════════════════════════════════════════════
# 1. LOGGING INTERCEPTOR — basic before/after pattern
# ════════════════════════════════════════════════════════════════════

async def logging_interceptor(request: MCPToolCallRequest, handler):
    """Log every MCP tool call with its args and result."""
    print(f"  [LOG] → {request.name}({request.args})")
    result = await handler(request)
    content = result.content[0].text if result.content else "(no content)"
    print(f"  [LOG] ← {request.name} = {content[:60]}")
    return result


async def demo_logging():
    print("\n── 1. Logging Interceptor ───────────────────────────────────")

    client = make_math_client(tool_interceptors=[logging_interceptor])
    tools  = await client.get_tools()
    agent  = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt="You are a math assistant.",
    )
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Add 10 and 25."}]
    })
    print(f"  Response: {result['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 2. RUNTIME CONTEXT INTERCEPTOR
#    Inject user_id from runtime context into MCP tool arguments.
#    MCP servers can't access LangGraph runtime directly — interceptors
#    bridge the gap.
# ════════════════════════════════════════════════════════════════════

@dataclass
class UserCtx:
    user_id: str
    api_key: str = "sk-demo"


async def inject_user_context(request: MCPToolCallRequest, handler):
    """Forward user_id from runtime context to every MCP tool call."""
    runtime = request.runtime
    if runtime and runtime.context:
        user_id = runtime.context.user_id
        modified = request.override(args={**request.args, "user_id": user_id})
        print(f"  [ContextInterceptor] injected user_id={user_id} into {request.name}")
        return await handler(modified)
    return await handler(request)


async def demo_runtime_context():
    print("\n── 2. Runtime Context Interceptor (inject user_id) ──────────")

    client = make_math_client(tool_interceptors=[inject_user_context])
    tools  = await client.get_tools()
    agent  = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        context_schema=UserCtx,
        system_prompt="You are a math assistant.",
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Multiply 6 by 7."}]},
        context=UserCtx(user_id="USR-001", api_key="sk-real"),
    )
    print(f"  Response: {result['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 3. STORE INTERCEPTOR
#    Read user preferences from Store and apply them to tool calls.
# ════════════════════════════════════════════════════════════════════

@dataclass
class StoreCtx:
    user_id: str


async def personalize_from_store(request: MCPToolCallRequest, handler):
    """Apply user preference (precision) from Store."""
    runtime = request.runtime
    if runtime and runtime.store and runtime.context:
        user_id = runtime.context.user_id
        prefs   = runtime.store.get(("preferences",), user_id)
        if prefs and request.name == "divide":
            precision = prefs.value.get("decimal_precision", 2)
            print(f"  [StoreInterceptor] user={user_id}, precision={precision}")
            # We can't change server behavior, but we log the intent
            # In a real app the server might accept a precision arg
    return await handler(request)


async def demo_store_interceptor():
    print("\n── 3. Store Interceptor (user preferences) ──────────────────")

    store = InMemoryStore()
    store.put(("preferences",), "USR-STORE", {"decimal_precision": 4})

    client = make_math_client(tool_interceptors=[personalize_from_store])
    tools  = await client.get_tools()
    agent  = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        context_schema=StoreCtx,
        store=store,
        system_prompt="You are a math assistant.",
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Divide 22 by 7."}]},
        context=StoreCtx(user_id="USR-STORE"),
    )
    print(f"  Response: {result['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 4. STATE-BASED AUTH INTERCEPTOR
#    Block sensitive tools when state shows user is unauthenticated.
# ════════════════════════════════════════════════════════════════════

SENSITIVE_TOOLS = {"divide"}  # treat divide as "sensitive" for demo

async def require_auth(request: MCPToolCallRequest, handler):
    """Block sensitive MCP tools if user is not authenticated in State."""
    runtime = request.runtime
    if runtime and request.name in SENSITIVE_TOOLS:
        state         = runtime.state
        authenticated = state.get("authenticated", False)
        tool_call_id  = runtime.tool_call_id
        if not authenticated:
            print(f"  [AuthInterceptor] 🚫 Blocked {request.name} — not authenticated")
            return ToolMessage(
                content="Authentication required. Please log in before using this tool.",
                tool_call_id=tool_call_id,
            )
        print(f"  [AuthInterceptor] ✅ {request.name} — authenticated")
    return await handler(request)


async def demo_state_auth():
    print("\n── 4. State-Based Auth Interceptor ──────────────────────────")

    client = make_math_client(tool_interceptors=[require_auth])
    tools  = await client.get_tools()
    agent  = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt="You are a math assistant.",
    )

    # Unauthenticated — divide is blocked
    r_unauth = await agent.ainvoke({
        "messages":      [{"role": "user", "content": "Divide 100 by 4."}],
        "authenticated": False,
    })
    print(f"  Unauth: {r_unauth['messages'][-1].content[:100]}")

    # Authenticated — divide is allowed
    r_auth = await agent.ainvoke({
        "messages":      [{"role": "user", "content": "Divide 100 by 4."}],
        "authenticated": True,
    })
    print(f"  Auth:   {r_auth['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 5. MODIFYING REQUESTS — request.override()
#    Create modified requests immutably. Original is unchanged.
# ════════════════════════════════════════════════════════════════════

async def double_numeric_args(request: MCPToolCallRequest, handler):
    """Double all numeric arguments before calling the tool."""
    doubled = {k: v * 2 if isinstance(v, (int, float)) else v
               for k, v in request.args.items()}
    print(f"  [DoubleInterceptor] {request.name}: {request.args} → {doubled}")
    return await handler(request.override(args=doubled))


async def demo_request_override():
    print("\n── 5. Request Override (double numeric args) ────────────────")

    client = make_math_client(tool_interceptors=[double_numeric_args])
    tools  = await client.get_tools()
    agent  = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt=(
            "You are a math assistant. Always call tools with the exact numbers the user provides."
        ),
    )
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Add 3 and 4. Show me what you called."}]
    })
    print(f"  Response: {result['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 6. COMPOSING INTERCEPTORS — onion order
#    First interceptor in list = outermost layer (runs first before,
#    last after). Think middleware stack.
# ════════════════════════════════════════════════════════════════════

async def outer_interceptor(request: MCPToolCallRequest, handler):
    print("  [Outer] before")
    result = await handler(request)
    print("  [Outer] after")
    return result


async def inner_interceptor(request: MCPToolCallRequest, handler):
    print("  [Inner] before")
    result = await handler(request)
    print("  [Inner] after")
    return result


async def demo_composition():
    print("\n── 6. Composing Interceptors (onion order) ──────────────────")
    print("  Order: [outer, inner] → outer.before → inner.before → tool → inner.after → outer.after")

    client = make_math_client(tool_interceptors=[outer_interceptor, inner_interceptor])
    tools  = await client.get_tools()
    agent  = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt="You are a math assistant.",
    )
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Add 1 and 1."}]
    })
    print(f"  Response: {result['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 7. RETRY WITH EXPONENTIAL BACKOFF
# ════════════════════════════════════════════════════════════════════

_fail_count = {}

async def retry_interceptor(request: MCPToolCallRequest, handler,
                             max_retries: int = 3, delay: float = 0.05):
    """Retry failed MCP tool calls with exponential backoff."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return await handler(request)
        except Exception as e:
            last_error = e
            wait = delay * (2 ** attempt)
            print(f"  [Retry] {request.name} failed attempt {attempt+1}/{max_retries}, "
                  f"retry in {wait:.2f}s: {e}")
            await asyncio.sleep(wait)
    raise last_error


async def demo_retry():
    print("\n── 7. Retry Interceptor (exponential backoff) ───────────────")

    client = make_math_client(tool_interceptors=[retry_interceptor])
    tools  = await client.get_tools()
    agent  = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt="You are a math assistant.",
    )
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Subtract 5 from 20."}]
    })
    print(f"  Response: {result['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

async def main():
    await demo_logging()
    await demo_runtime_context()
    await demo_store_interceptor()
    await demo_state_auth()
    await demo_request_override()
    await demo_composition()
    await demo_retry()

    print("\n" + "═" * 60)
    print("Interceptor Summary:")
    print("  async def fn(request, handler) → result")
    print("  request.runtime.context  — injected runtime context")
    print("  request.runtime.store    — long-term store access")
    print("  request.runtime.state    — current agent state")
    print("  request.runtime.tool_call_id — for ToolMessage responses")
    print("  request.override(args=..) — immutable arg modification")
    print("  [a, b] tool_interceptors  — onion composition (a outermost)")
    print("  Return ToolMessage        — short-circuit tool execution")
    print("  Return Command(update=..) — update state / control flow")
    print("═" * 60)
    print("\n✅ MCP interceptors demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
