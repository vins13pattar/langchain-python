"""
02_mcp_transports.py
=====================
Demonstrates the two main MCP transport mechanisms and connection options.

Concepts covered:
  - stdio transport — local subprocess, best for dev/local tools
  - http (streamable-http) transport — remote servers over HTTP
  - Passing custom headers (auth, tracing)
  - Connecting to multiple servers simultaneously
  - Per-server transport configuration

Note on running HTTP examples:
  Start the weather server first in a separate terminal:
    python 13_mcp/servers/weather_server.py
  Then run this file:
    python 13_mcp/02_mcp_transports.py
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

load_dotenv()

MATH_SERVER = str(Path(__file__).parent / "servers" / "math_server.py")

print("=" * 60)
print("MCP Transports — stdio, HTTP, multi-server")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. STDIO TRANSPORT
#    Client launches server as a subprocess and communicates via
#    stdin/stdout. Best for local tools. The subprocess runs for
#    the lifetime of the client connection.
# ════════════════════════════════════════════════════════════════════

async def demo_stdio():
    print("\n── 1. stdio Transport (local subprocess) ────────────────────")
    print("  Client launches the math server as a Python subprocess.")

    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",        # ← local subprocess
                "command":   "python",
                "args":      [MATH_SERVER],  # absolute path required
                # Optional: set env vars for the subprocess
                # "env": {"SOME_VAR": "value"},
            }
        }
    )

    tools = await client.get_tools()
    print(f"  Loaded {len(tools)} tools: {[t.name for t in tools]}")

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt="You are a math assistant.",
    )

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is 144 divided by 12?"}]
    })
    print(f"  Result: {result['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 2. HTTP TRANSPORT
#    Communicates with a remote MCP server over HTTP.
#    The server must be running before the client connects.
#    Also known as "streamable-http" in the MCP spec.
# ════════════════════════════════════════════════════════════════════

async def demo_http():
    print("\n── 2. HTTP Transport (remote server) ────────────────────────")
    print("  Requires weather server running: python servers/weather_server.py")

    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "http",                      # ← HTTP
                "url":       "http://localhost:8000/mcp", # server endpoint
            }
        }
    )

    try:
        tools = await client.get_tools()
        print(f"  Loaded {len(tools)} tools: {[t.name for t in tools]}")

        agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=tools,
            system_prompt="You are a weather assistant.",
        )

        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
        })
        print(f"  Result: {result['messages'][-1].content}")

    except Exception as e:
        print(f"  ⚠️  HTTP server not running — skipping ({type(e).__name__})")
        print("     Start it with: python 13_mcp/servers/weather_server.py")


# ════════════════════════════════════════════════════════════════════
# 3. HTTP WITH CUSTOM HEADERS
#    Pass authentication tokens, correlation IDs, or any custom
#    headers when connecting to an HTTP MCP server.
# ════════════════════════════════════════════════════════════════════

async def demo_http_with_headers():
    print("\n── 3. HTTP with Custom Headers ──────────────────────────────")

    client = MultiServerMCPClient(
        {
            "weather": {
                "transport": "http",
                "url":       "http://localhost:8000/mcp",
                "headers": {                              # ← custom headers
                    "Authorization": "Bearer demo-token-123",
                    "X-Correlation-ID": "req-abc-456",
                    "X-Client-Version": "1.0.0",
                },
            }
        }
    )

    try:
        tools = await client.get_tools()
        print(f"  Connected with auth headers. Tools: {[t.name for t in tools]}")
    except Exception as e:
        print(f"  ⚠️  Server not running — headers demo skipped ({type(e).__name__})")


# ════════════════════════════════════════════════════════════════════
# 4. MULTI-SERVER — STDIO + HTTP SIMULTANEOUSLY
#    Connect to multiple servers in one client. Tools from all
#    servers are merged and available to the agent.
# ════════════════════════════════════════════════════════════════════

async def demo_multi_server():
    print("\n── 4. Multi-Server (stdio math + HTTP weather) ──────────────")

    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command":   "python",
                "args":      [MATH_SERVER],
            },
            "weather": {
                "transport": "http",
                "url":       "http://localhost:8000/mcp",
            },
        }
    )

    try:
        tools = await client.get_tools()
        print(f"  Merged tools from both servers: {[t.name for t in tools]}")

        agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=tools,
            system_prompt=(
                "You are a helpful assistant with access to math and weather tools. "
                "Use both as needed."
            ),
        )

        # Uses math tool from stdio server
        r1 = await agent.ainvoke({
            "messages": [{"role": "user", "content": "Calculate 15 × 8, then tell me Tokyo weather."}]
        })
        print(f"  Combined response: {r1['messages'][-1].content[:180]}")

    except Exception as e:
        # Fall back to math-only if weather server isn't running
        print(f"  Weather server unavailable — running math-only demo")

        client_math = MultiServerMCPClient({
            "math": {
                "transport": "stdio",
                "command":   "python",
                "args":      [MATH_SERVER],
            }
        })
        tools = await client_math.get_tools()
        agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=tools,
            system_prompt="You are a math assistant.",
        )
        r = await agent.ainvoke({
            "messages": [{"role": "user", "content": "Calculate 15 × 8."}]
        })
        print(f"  Math result: {r['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 5. TRANSPORT CONFIGURATION REFERENCE
#    Shows all configuration options for each transport type.
# ════════════════════════════════════════════════════════════════════

def show_transport_reference():
    print("\n── 5. Transport Configuration Reference ─────────────────────")

    print("""
  stdio transport:
    {
        "transport": "stdio",
        "command":   "python",            # executable to run
        "args":      ["/path/to/server"], # command arguments
        "env":       {"KEY": "value"},    # optional env vars
    }

  http (streamable-http) transport:
    {
        "transport": "http",
        "url":       "http://host:port/mcp",  # server URL
        "headers": {                           # optional headers
            "Authorization": "Bearer TOKEN",
        },
        "auth":    auth_object,                # optional httpx.Auth
    }

  Key differences:
    stdio  — local subprocess, best for dev tools, stateful by nature
    http   — remote server, production-ready, supports load balancing
    """)


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

async def main():
    await demo_stdio()
    await demo_http()
    await demo_http_with_headers()
    await demo_multi_server()
    show_transport_reference()
    print("\n✅ MCP transports demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
