"""
mcp_overview.py — Model Context Protocol (MCP): all key concepts in one file
Covers: MultiServerMCPClient, stdio transport, get_tools, stateless vs stateful sessions,
        resources, prompts, interceptors, callbacks, async usage
"""

import asyncio
from pathlib import Path
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

load_dotenv()

MATH_SERVER = str(Path(__file__).parent / "servers" / "math_server.py")
FILES_SERVER = str(Path(__file__).parent / "servers" / "files_server.py")

def section(title): print(f"\n{'─'*55}\n{title}\n{'─'*55}")


# ════════════════════════════════════════════════════════════════════
# 1. SINGLE SERVER — stdio transport (subprocess)
# ════════════════════════════════════════════════════════════════════

async def demo_single_server():
    section("1. SINGLE SERVER (stdio transport)")
    client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command":   "python",
            "args":      [MATH_SERVER],
        }
    })
    tools = await client.get_tools()
    print(f"Tools loaded: {[t.name for t in tools]}")
    for t in tools:
        print(f"  {t.name}: {t.description[:60]}")

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt="You are a math assistant. Use tools to compute.",
    )
    r = await agent.ainvoke({"messages": [{"role": "user", "content": "What is (3 + 5) × 12?"}]})
    print(f"Answer: {r['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 2. MULTIPLE SERVERS — each adds its own tool namespace
# ════════════════════════════════════════════════════════════════════

async def demo_multi_server():
    section("2. MULTIPLE SERVERS")
    # Register two servers at once; tools from all are merged
    client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": [MATH_SERVER],
        },
        # Add another server if available:
        # "files": {"transport": "stdio", "command": "python", "args": [FILES_SERVER]},
    })
    tools = await client.get_tools()
    print(f"All tools from all servers: {[t.name for t in tools]}")

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt="You are an assistant with math and file tools.",
    )
    r = await agent.ainvoke({"messages": [{"role": "user", "content": "Add 7 + 8, then multiply by 3."}]})
    print(f"Result: {r['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 3. STATELESS vs STATEFUL SESSIONS
# ════════════════════════════════════════════════════════════════════

async def demo_sessions():
    section("3. STATELESS vs STATEFUL SESSIONS")

    client = MultiServerMCPClient({
        "math": {"transport": "stdio", "command": "python", "args": [MATH_SERVER]}
    })

    # STATELESS (default) — fresh connection per tool call
    tools = await client.get_tools()
    agent_sl = create_agent(model="openai:gpt-4o-mini", tools=tools, system_prompt="Math assistant.")
    r = await agent_sl.ainvoke({"messages": [{"role": "user", "content": "Add 7 + 8."}]})
    print(f"Stateless: {r['messages'][-1].content}")

    # STATEFUL — explicit session keeps connection alive across tool calls
    from langchain_mcp_adapters.tools import load_mcp_tools
    async with client.session("math") as session:
        session_tools = await load_mcp_tools(session)
        agent_sf = create_agent(model="openai:gpt-4o-mini", tools=session_tools, system_prompt="Math assistant.")
        r = await agent_sf.ainvoke({"messages": [{"role": "user", "content": "What is 100 / 4?"}]})
        print(f"Stateful: {r['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 4. RESOURCES — reading contextual data from MCP server
# ════════════════════════════════════════════════════════════════════

async def demo_resources():
    section("4. RESOURCES")

    client = MultiServerMCPClient({
        "math": {"transport": "stdio", "command": "python", "args": [MATH_SERVER]}
    })

    async with client.session("math") as session:
        # List available resources
        resources = await session.list_resources()
        print(f"Available resources: {[r.uri for r in resources]}")

        # Read a specific resource
        if resources:
            content = await session.read_resource(resources[0].uri)
            print(f"Resource content: {str(content)[:200]}")


# ════════════════════════════════════════════════════════════════════
# 5. PROMPTS — pre-built prompt templates from MCP server
# ════════════════════════════════════════════════════════════════════

async def demo_prompts():
    section("5. PROMPTS")

    client = MultiServerMCPClient({
        "math": {"transport": "stdio", "command": "python", "args": [MATH_SERVER]}
    })

    async with client.session("math") as session:
        prompts = await session.list_prompts()
        print(f"Available prompts: {[p.name for p in prompts]}")

        if prompts:
            # Get a prompt with arguments
            prompt_result = await session.get_prompt(prompts[0].name, arguments={})
            for msg in prompt_result.messages:
                print(f"  [{msg.role}]: {str(msg.content)[:100]}")


# ════════════════════════════════════════════════════════════════════
# 6. INTERCEPTORS — transform requests/responses mid-flight
# ════════════════════════════════════════════════════════════════════

async def demo_interceptors():
    section("6. INTERCEPTORS")

    async def logging_interceptor(request, call_next):
        """Log every tool call before and after execution."""
        print(f"  [Interceptor] → {request.get('method', '?')}")
        response = await call_next(request)
        print(f"  [Interceptor] ← response received")
        return response

    async def error_interceptor(request, call_next):
        """Catch tool errors and return fallback values."""
        try:
            return await call_next(request)
        except Exception as e:
            print(f"  [ErrorInterceptor] Tool failed: {e}")
            return {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}

    client = MultiServerMCPClient(
        servers={
            "math": {"transport": "stdio", "command": "python", "args": [MATH_SERVER]}
        },
        interceptors=[logging_interceptor, error_interceptor],
    )

    tools = await client.get_tools()
    agent = create_agent(model="openai:gpt-4o-mini", tools=tools, system_prompt="Math assistant.")
    r = await agent.ainvoke({"messages": [{"role": "user", "content": "Multiply 9 by 9."}]})
    print(f"Result: {r['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 7. CALLBACKS — lifecycle hooks on tool events
# ════════════════════════════════════════════════════════════════════

async def demo_callbacks():
    section("7. CALLBACKS")

    call_log = []

    class AuditCallback:
        async def on_tool_call_start(self, server: str, tool: str, args: dict):
            call_log.append({"event": "start", "server": server, "tool": tool})
            print(f"  [Callback] start: {server}.{tool}({args})")

        async def on_tool_call_end(self, server: str, tool: str, result):
            call_log.append({"event": "end", "server": server, "tool": tool})
            print(f"  [Callback] end: {server}.{tool} → {str(result)[:60]}")

        async def on_tool_call_error(self, server: str, tool: str, error: Exception):
            print(f"  [Callback] error: {server}.{tool} → {error}")

    client = MultiServerMCPClient(
        servers={
            "math": {"transport": "stdio", "command": "python", "args": [MATH_SERVER]}
        },
        callbacks=[AuditCallback()],
    )

    tools = await client.get_tools()
    agent = create_agent(model="openai:gpt-4o-mini", tools=tools, system_prompt="Math assistant.")
    await agent.ainvoke({"messages": [{"role": "user", "content": "Add 5 + 15, then multiply by 4."}]})
    print(f"\nAudit log ({len(call_log)} events):")
    for entry in call_log:
        print(f"  {entry}")


# ════════════════════════════════════════════════════════════════════
# 8. SSE TRANSPORT — connect to remote MCP server over HTTP
# ════════════════════════════════════════════════════════════════════

async def demo_sse_transport():
    section("8. SSE TRANSPORT (HTTP remote server)")
    print("  # SSE transport for remote MCP servers:")
    print("  client = MultiServerMCPClient({")
    print("      'remote_server': {")
    print("          'transport': 'sse',")
    print("          'url': 'http://localhost:8000/sse',")
    print("          'headers': {'Authorization': 'Bearer TOKEN'},")
    print("      }")
    print("  })")
    print("  # All other APIs remain the same: get_tools(), session(), etc.")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

async def main():
    await demo_single_server()
    await demo_multi_server()
    await demo_sessions()
    try:
        await demo_resources()
    except Exception:
        print("  (resources demo: server may not expose resources)")
    try:
        await demo_prompts()
    except Exception:
        print("  (prompts demo: server may not expose prompts)")
    await demo_interceptors()
    await demo_callbacks()
    await demo_sse_transport()

    print("""
MCP Quick Reference:
  MultiServerMCPClient(servers={...}, interceptors=[...], callbacks=[...])
  client.get_tools()               → load all tools (stateless by default)
  client.session("server_name")    → explicit stateful session context manager
  load_mcp_tools(session)          → load tools from an open session
  session.list_resources()         → list contextual data
  session.read_resource(uri)       → read a resource by URI
  session.list_prompts()           → list prompt templates
  session.get_prompt(name, args)   → render a prompt with arguments

  Transports:  stdio (subprocess)  |  sse (HTTP remote)
  Sessions:    stateless (default) |  stateful (with client.session())
""")


if __name__ == "__main__":
    asyncio.run(main())
