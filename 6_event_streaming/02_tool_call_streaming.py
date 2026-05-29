"""
02_tool_call_streaming.py
==========================
Demonstrates streaming TOOL CALLS — both while the model is generating them
(argument chunks) and during execution (output deltas).

Concepts covered:
  - message.tool_calls              — tool-call arg chunks while model generates
  - message.tool_calls.get()        — finalized tool calls after model done
  - stream.tool_calls               — execution lifecycle per tool call
  - call.tool_name, call.input      — tool name and parsed args
  - call.output_deltas              — streaming tool output (if tool streams)
  - call.output, call.error         — final result or error after execution
  - Handling parallel tool calls    — multiple tools called at once

TWO projections for tool calls:
  1. message.tool_calls  → ARGUMENT streaming (model generating call args)
  2. stream.tool_calls   → EXECUTION streaming (tool running and returning)
"""

import os
import time
import asyncio
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

print("=" * 60)
print("Tool Call Streaming Demo")
print("=" * 60)


# ── Tools (some simulate slow execution for demo purposes) ────────

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name
    """
    time.sleep(0.3)   # simulate network call
    data = {
        "london": "☁️  Cloudy, 14°C",
        "tokyo":  "☀️  Sunny, 28°C",
        "paris":  "🌤️  Partly cloudy, 22°C",
        "sydney": "⛅  Breezy, 20°C",
    }
    return data.get(city.lower(), f"No data for '{city}'")


@tool
def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic expression.

    Args:
        expression: e.g. '100 * 1.18' or '500 / 4'
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: unsupported characters"
        return f"Result: {round(eval(expression), 4)}"  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


@tool
def search_articles(topic: str, limit: int = 3) -> str:
    """Search for recent articles on a topic.

    Args:
        topic: Topic to search for
        limit: Max number of results (default 3)
    """
    time.sleep(0.2)
    return f"Found {limit} articles about '{topic}': [article 1], [article 2], [article 3]"


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, calculate, search_articles],
    system_prompt="You are a helpful assistant. Use tools for weather, math, and search.",
)


# ════════════════════════════════════════════════════════════════════
# 1. message.tool_calls — arg chunks while model generates
# ════════════════════════════════════════════════════════════════════

print("\n── 1. message.tool_calls (argument streaming) ───────────")

INPUT = {"messages": [{"role": "user", "content": "What's the weather in Tokyo and calculate 150 * 3.5?"}]}

stream = agent.stream_events(INPUT, version="v3")

for message in stream.messages:
    # Stream argument chunks as the model generates them
    for chunk in message.tool_calls:
        name = chunk.get("name", "")
        args = chunk.get("args", "")
        if name:
            print(f"\n  🔧 Tool being called: {name}")
        if args:
            print(f"     Args fragment: {args!r}", end="", flush=True)

    # After all chunks, get the finalized complete tool calls
    finalized = message.tool_calls.get()
    if finalized:
        print(f"\n\n  ✅ Finalized tool calls:")
        for tc in finalized:
            print(f"     → {tc['name']}({tc['args']})")


# ════════════════════════════════════════════════════════════════════
# 2. stream.tool_calls — execution lifecycle
# ════════════════════════════════════════════════════════════════════

print("\n── 2. stream.tool_calls (execution lifecycle) ───────────")

stream = agent.stream_events(INPUT, version="v3")

# Drain messages silently (we only care about tool_calls here)
for _ in stream.messages:
    pass

print()
for call in stream.tool_calls:
    print(f"\n  🔧 Tool: {call.tool_name}")
    print(f"     Input: {call.input}")

    # output_deltas — stream the tool's return value as it arrives
    output = ""
    for delta in call.output_deltas:
        output += str(delta)
        print(f"     Delta: {delta!r}", end="", flush=True)

    print(f"\n     Final output: {call.output}")
    if call.error:
        print(f"     ❌ Error: {call.error}")


# ════════════════════════════════════════════════════════════════════
# 3. BOTH PROJECTIONS TOGETHER
#    See arg streaming AND execution streaming in one pass
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Both projections together ──────────────────────────")

INPUT2 = {"messages": [{"role": "user", "content": "Get weather in London and search for LangChain articles."}]}
stream = agent.stream_events(INPUT2, version="v3")

print("\n  --- Model generating tool calls ---")
for message in stream.messages:
    for chunk in message.tool_calls:
        name = chunk.get("name", "")
        if name:
            print(f"  📝 Generating: {name}(…)")

    finalized = message.tool_calls.get()
    if finalized:
        for tc in finalized:
            print(f"  ✅ Complete:   {tc['name']}({tc['args']})")

print("\n  --- Tool execution ---")
for call in stream.tool_calls:
    print(f"  ⚡ Executing:  {call.tool_name}")
    # Drain output deltas silently
    for _ in call.output_deltas:
        pass
    status = "✅" if not call.error else "❌"
    print(f"  {status} Result:    {str(call.output)[:80]}")


# ════════════════════════════════════════════════════════════════════
# 4. PARALLEL TOOL CALLS
#    Model calls multiple tools at once → multiple entries in stream.tool_calls
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Parallel tool calls ────────────────────────────────")

INPUT3 = {
    "messages": [{
        "role": "user",
        "content": "Get weather for London, Tokyo, and Paris simultaneously, and also calculate 500 * 0.08."
    }]
}

stream = agent.stream_events(INPUT3, version="v3")

# Drain messages
for _ in stream.messages:
    pass

tool_calls_seen = []
for call in stream.tool_calls:
    tool_calls_seen.append(call.tool_name)
    for _ in call.output_deltas:
        pass
    print(f"  🔧 {call.tool_name}({call.input}) → {str(call.output)[:60]}")

print(f"\n  {len(tool_calls_seen)} tool calls executed in this run:")
print(f"  {tool_calls_seen}")


# ════════════════════════════════════════════════════════════════════
# 5. ERROR HANDLING IN stream.tool_calls
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Error detection in stream.tool_calls ──────────────")


@tool
def risky_tool(value: int) -> str:
    """A tool that sometimes fails.

    Args:
        value: Any integer. If negative, raises an error.
    """
    if value < 0:
        raise ValueError(f"Value must be non-negative, got {value}")
    return f"Processed value: {value * 2}"


error_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[risky_tool],
    system_prompt="Use the risky_tool for any number processing tasks.",
)

stream = error_agent.stream_events(
    {"messages": [{"role": "user", "content": "Process the value 5."}]},
    version="v3",
)

for _ in stream.messages:
    pass

for call in stream.tool_calls:
    for _ in call.output_deltas:
        pass
    if call.error:
        print(f"\n  ❌ Tool '{call.tool_name}' failed: {call.error}")
    else:
        print(f"\n  ✅ Tool '{call.tool_name}' succeeded: {call.output}")


# ════════════════════════════════════════════════════════════════════
# 6. ASYNC TOOL CALL STREAMING
# ════════════════════════════════════════════════════════════════════

print("\n── 6. Async tool call streaming ──────────────────────────")

async def demo_async_tool_calls():
    stream = await agent.astream_events(INPUT, version="v3")

    async for message in stream.messages:
        async for chunk in message.tool_calls:
            name = chunk.get("name", "")
            if name:
                print(f"  📝 Async: {name}(…)")

    print("  ✅ Async tool call streaming complete")

asyncio.run(demo_async_tool_calls())
