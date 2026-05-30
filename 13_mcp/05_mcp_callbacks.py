"""
05_mcp_callbacks.py
====================
Demonstrates MCP callbacks for progress notifications, server logging,
and elicitation (interactive user input during tool execution).

Concepts covered:
  - Callbacks class for subscribing to server events
  - on_progress — real-time progress updates from long-running tools
  - on_logging_message — log messages from the MCP server
  - on_elicitation — handle server requests for user input
  - CallbackContext — server_name and tool_name metadata
  - Elicitation response actions: accept, decline, cancel

Note: Progress and logging demos require the rich server:
  python 13_mcp/servers/rich_server.py  (port 8001)
"""

import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext
from langchain.agents import create_agent

load_dotenv()

MATH_SERVER = str(Path(__file__).parent / "servers" / "math_server.py")
RICH_URL     = "http://localhost:8001/mcp"

print("=" * 60)
print("MCP Callbacks — Progress, Logging & Elicitation")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. PROGRESS NOTIFICATIONS
#    Subscribe to real-time progress updates from long-running tools.
#    Server calls ctx.report_progress(progress, total, message).
# ════════════════════════════════════════════════════════════════════

async def on_progress(
    progress: float,
    total: float | None,
    message: str | None,
    context: CallbackContext,
):
    """Handle progress updates — called by the server during long ops."""
    percent = (progress / total * 100) if total else progress
    tool    = f" [{context.tool_name}]" if context.tool_name else ""
    print(f"  [Progress]{tool} {percent:.0f}% — {message or ''}")


async def demo_progress():
    print("\n── 1. Progress Notifications ────────────────────────────────")

    client = MultiServerMCPClient(
        {"rich": {"transport": "http", "url": RICH_URL}},
        callbacks=Callbacks(on_progress=on_progress),
    )

    try:
        tools = await client.get_tools()
        agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=tools,
            system_prompt="You are a data analysis assistant.",
        )
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content":
                "Run an analysis on the 'sales_q4' dataset."}]
        })
        print(f"  Result: {result['messages'][-1].content[:120]}")

    except Exception as e:
        print(f"  ⚠️  Rich server not running — skipping ({type(e).__name__})")
        print("     Start it: python 13_mcp/servers/rich_server.py")


# ════════════════════════════════════════════════════════════════════
# 2. LOGGING CALLBACKS
#    Subscribe to log messages emitted by the MCP server.
#    Useful for debugging server-side behaviour without modifying code.
# ════════════════════════════════════════════════════════════════════

async def on_logging_message(params, context: CallbackContext):
    """Handle structured log messages from MCP servers."""
    level   = getattr(params, "level", "info")
    data    = getattr(params, "data", "")
    logger  = getattr(params, "logger", "server")
    print(f"  [MCP:{context.server_name}] [{level.upper()}] {logger}: {data}")


async def demo_logging_callback():
    print("\n── 2. Server Logging Callback ───────────────────────────────")

    client = MultiServerMCPClient(
        {"rich": {"transport": "http", "url": RICH_URL}},
        callbacks=Callbacks(on_logging_message=on_logging_message),
    )

    try:
        tools = await client.get_tools()
        agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=tools,
            system_prompt="You are a financial assistant.",
        )
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content":
                "Get the stock price for MSFT."}]
        })
        print(f"  Result: {result['messages'][-1].content[:120]}")

    except Exception as e:
        print(f"  ⚠️  Rich server not running — skipping ({type(e).__name__})")


# ════════════════════════════════════════════════════════════════════
# 3. COMBINING PROGRESS + LOGGING
#    Pass multiple callbacks simultaneously.
# ════════════════════════════════════════════════════════════════════

async def demo_combined_callbacks():
    print("\n── 3. Combined Callbacks (progress + logging) ───────────────")

    client = MultiServerMCPClient(
        {"rich": {"transport": "http", "url": RICH_URL}},
        callbacks=Callbacks(
            on_progress=on_progress,
            on_logging_message=on_logging_message,
        ),
    )

    try:
        tools = await client.get_tools()
        agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=tools,
            system_prompt="You are an assistant.",
        )
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content":
                "Run analysis on 'user_activity' dataset."}]
        })
        print(f"  Result: {result['messages'][-1].content[:100]}")

    except Exception as e:
        print(f"  ⚠️  Rich server not running — skipping ({type(e).__name__})")


# ════════════════════════════════════════════════════════════════════
# 4. ELICITATION — interactive server requests
#    An MCP server can use ctx.elicit() to ask the client for
#    additional structured input DURING tool execution.
#    The client handles this via on_elicitation callback.
# ════════════════════════════════════════════════════════════════════

async def handle_elicitation_accept(mcp_context, params, context: CallbackContext):
    """Accept the elicitation request and provide data."""
    from mcp.types import ElicitResult
    print(f"  [Elicitation] Server '{context.server_name}' requests: {params.message}")
    print("  [Elicitation] Action: accept — providing email + age")
    return ElicitResult(
        action="accept",
        content={"email": "user@example.com", "age": 28},
    )


async def handle_elicitation_decline(mcp_context, params, context: CallbackContext):
    """Decline the elicitation — server continues with partial info."""
    from mcp.types import ElicitResult
    print(f"  [Elicitation] Request: {params.message}")
    print("  [Elicitation] Action: decline — user didn't provide info")
    return ElicitResult(action="decline")


async def handle_elicitation_cancel(mcp_context, params, context: CallbackContext):
    """Cancel the operation entirely."""
    from mcp.types import ElicitResult
    print(f"  [Elicitation] Request: {params.message}")
    print("  [Elicitation] Action: cancel — aborting operation")
    return ElicitResult(action="cancel")


async def demo_elicitation():
    print("\n── 4. Elicitation (interactive user input) ──────────────────")
    print("  Note: Requires a server that calls ctx.elicit()")
    print("  See servers/rich_server.py — create_profile tool uses elicitation")

    # Demonstrate each elicitation action
    for action, handler in [
        ("accept",  handle_elicitation_accept),
        ("decline", handle_elicitation_decline),
        ("cancel",  handle_elicitation_cancel),
    ]:
        print(f"\n  ─ Elicitation with '{action}' ─")
        client = MultiServerMCPClient(
            {"rich": {"transport": "http", "url": RICH_URL}},
            callbacks=Callbacks(on_elicitation=handler),
        )
        try:
            tools = await client.get_tools()
            agent = create_agent(
                model="openai:gpt-4o-mini",
                tools=tools,
                system_prompt="You are a profile management assistant.",
            )
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content":
                    "Create a profile for user 'Alice'."}]
            })
            print(f"  Response: {result['messages'][-1].content[:100]}")

        except Exception as e:
            print(f"  ⚠️  Rich server not running / no elicitation tool — skipping")
            break


# ════════════════════════════════════════════════════════════════════
# 5. CALLBACK WITH MATH SERVER (works without HTTP server)
#    Demonstrate callbacks that work with stdio transport.
# ════════════════════════════════════════════════════════════════════

_progress_events: list[dict] = []

async def collect_progress(progress, total, message, context: CallbackContext):
    _progress_events.append({
        "server": context.server_name,
        "tool":   context.tool_name,
        "pct":    round((progress / total * 100) if total else progress, 1),
    })


async def demo_stdio_callbacks():
    print("\n── 5. Callbacks with stdio Transport ────────────────────────")

    client = MultiServerMCPClient(
        {"math": {"transport": "stdio", "command": "python", "args": [MATH_SERVER]}},
        callbacks=Callbacks(on_progress=collect_progress),
    )

    tools = await client.get_tools()
    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt="You are a math assistant.",
    )
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is 99 + 1?"}]
    })
    print(f"  Result: {result['messages'][-1].content}")
    print(f"  Progress events captured: {len(_progress_events)}")
    # Math server doesn't emit progress — this shows 0 (expected)


# ════════════════════════════════════════════════════════════════════
# ELICITATION RESPONSE REFERENCE
# ════════════════════════════════════════════════════════════════════

def show_elicitation_reference():
    print("\n── Elicitation Response Actions ─────────────────────────────")
    print("""
  from mcp.types import ElicitResult

  # Accept — provide the requested data
  ElicitResult(action="accept", content={"field1": "value1", "field2": 42})

  # Decline — user doesn't want to provide info, server continues
  ElicitResult(action="decline")

  # Cancel — abort the entire operation
  ElicitResult(action="cancel")

  CallbackContext fields:
    context.server_name  — name of the MCP server
    context.tool_name    — name of the tool being executed
    """)


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

async def main():
    await demo_stdio_callbacks()
    await demo_progress()
    await demo_logging_callback()
    await demo_combined_callbacks()
    await demo_elicitation()
    show_elicitation_reference()

    print("\n" + "═" * 60)
    print("MCP Callbacks Summary:")
    print("  Callbacks(on_progress=fn)         — progress updates")
    print("  Callbacks(on_logging_message=fn)  — server log messages")
    print("  Callbacks(on_elicitation=fn)      — interactive user input")
    print("  CallbackContext.server_name        — which server")
    print("  CallbackContext.tool_name          — which tool (if in tool call)")
    print("═" * 60)
    print("\n✅ MCP callbacks demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
