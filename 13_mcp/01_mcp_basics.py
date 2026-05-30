"""
01_mcp_basics.py
=================
Introduction to Model Context Protocol (MCP) with LangChain.

Concepts covered:
  - MultiServerMCPClient — connect to one or more MCP servers
  - stdio transport — local subprocess communication
  - client.get_tools() — load MCP tools as LangChain tools
  - create_agent() with MCP tools
  - Stateless vs stateful sessions
  - Running async agents with asyncio

Prerequisites:
  pip install langchain-mcp-adapters fastmcp

Usage:
  python 13_mcp/01_mcp_basics.py
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

load_dotenv()

# Absolute path to our math server
MATH_SERVER = str(Path(__file__).parent / "servers" / "math_server.py")

print("=" * 60)
print("MCP Basics — MultiServerMCPClient + stdio transport")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. SINGLE SERVER — STDIO TRANSPORT
#    The simplest setup: one local MCP server as a subprocess.
# ════════════════════════════════════════════════════════════════════

async def demo_single_server():
    print("\n── 1. Single Server (stdio transport) ───────────────────────")

    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command":   "python",
                "args":      [MATH_SERVER],
            }
        }
    )

    # Load all tools from all registered servers
    tools = await client.get_tools()
    print(f"Tools loaded: {[t.name for t in tools]}")

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt="You are a math assistant. Use available tools to compute answers.",
    )

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is (3 + 5) × 12?"}]
    })
    print(f"Answer: {result['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 2. INSPECTING LOADED TOOLS
#    Each MCP tool becomes a standard LangChain tool with name,
#    description, and args_schema auto-populated from the server.
# ════════════════════════════════════════════════════════════════════

async def demo_inspect_tools():
    print("\n── 2. Inspecting Loaded MCP Tools ───────────────────────────")

    client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command":   "python",
            "args":      [MATH_SERVER],
        }
    })

    tools = await client.get_tools()
    for t in tools:
        print(f"\n  Tool:        {t.name}")
        print(f"  Description: {t.description[:80]}")
        if hasattr(t, "args_schema") and t.args_schema:
            schema = t.args_schema.model_json_schema()
            print(f"  Args:        {list(schema.get('properties', {}).keys())}")


# ════════════════════════════════════════════════════════════════════
# 3. STATELESS vs STATEFUL SESSIONS
#    By default MultiServerMCPClient is STATELESS — each tool call
#    creates a fresh session. For stateful servers, create an
#    explicit session with client.session().
# ════════════════════════════════════════════════════════════════════

async def demo_stateless_vs_stateful():
    print("\n── 3. Stateless vs Stateful Sessions ────────────────────────")

    client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command":   "python",
            "args":      [MATH_SERVER],
        }
    })

    # STATELESS (default) — each tool call is independent
    print("  Stateless (default): each tool invocation gets a fresh session.")
    tools = await client.get_tools()
    agent_stateless = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt="You are a math assistant.",
    )
    r = await agent_stateless.ainvoke({
        "messages": [{"role": "user", "content": "Add 7 + 8, then multiply result by 3."}]
    })
    print(f"  Stateless result: {r['messages'][-1].content}")

    # STATEFUL — explicit session for servers that maintain state
    print("\n  Stateful (explicit session): tools share a single session.")
    from langchain_mcp_adapters.tools import load_mcp_tools
    async with client.session("math") as session:
        session_tools = await load_mcp_tools(session)
        agent_stateful = create_agent(
            model="openai:gpt-4o-mini",
            tools=session_tools,
            system_prompt="You are a math assistant.",
        )
        r2 = await agent_stateful.ainvoke({
            "messages": [{"role": "user", "content": "What is 100 divided by 4?"}]
        })
        print(f"  Stateful result: {r2['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 4. MULTIPLE TOOL CALLS IN ONE CONVERSATION
# ════════════════════════════════════════════════════════════════════

async def demo_multiple_calls():
    print("\n── 4. Chained Tool Calls ────────────────────────────────────")

    client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command":   "python",
            "args":      [MATH_SERVER],
        }
    })

    tools = await client.get_tools()
    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt=(
            "You are a math assistant. Solve problems step by step, "
            "using the available tools for each calculation."
        ),
    )

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content":
            "I have 24 apples. I give 6 to Alice. "
            "Then I multiply what's left by 3. What's the final count?"}]
    })
    print(f"Result: {result['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

async def main():
    await demo_single_server()
    await demo_inspect_tools()
    await demo_stateless_vs_stateful()
    await demo_multiple_calls()

    print("\n" + "═" * 60)
    print("MCP Basics Summary:")
    print("  MultiServerMCPClient  — connects to one or more MCP servers")
    print("  transport='stdio'     — local subprocess communication")
    print("  client.get_tools()    — loads MCP tools as LangChain tools")
    print("  Stateless (default)   — fresh session per tool call")
    print("  client.session()      — explicit stateful session context")
    print("═" * 60)
    print("\n✅ MCP basics demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
