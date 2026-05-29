"""
01_stream_events_basics.py
===========================
Demonstrates the BASICS of stream_events() with the v3 protocol.

Concepts covered:
  - agent.stream_events(..., version="v3")  — the modern streaming API
  - stream.messages projection              — one per model call
  - message.text                            — text deltas live, str(message.text) for final
  - message.output                          — finalized AIMessage with usage metadata
  - stream.output                           — final agent state after the run completes
  - Raw protocol events                     — for loop with full envelope access

stream_events(version="v3") returns a run object with TYPED PROJECTIONS.
Instead of parsing low-level tuples, you consume named channels:
  stream.messages, stream.tool_calls, stream.values, stream.output, …

This is the recommended API for frontends and application code.
"""

import os
import asyncio
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

print("=" * 60)
print("stream_events() Basics Demo")
print("=" * 60)


# ── Shared tools ─────────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name
    """
    data = {
        "london": "Cloudy, 14°C",
        "tokyo":  "Sunny, 28°C",
        "mumbai": "Rainy, 30°C",
        "paris":  "Clear, 22°C",
    }
    return data.get(city.lower(), f"No weather data for '{city}'")


@tool
def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic expression.

    Args:
        expression: e.g. '2 + 2', '100 * 1.08'
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: unsupported characters"
        return str(round(eval(expression), 4))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, calculate],
    system_prompt="You are a helpful assistant with weather and calculation tools.",
)

INPUT = {"messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]}


# ════════════════════════════════════════════════════════════════════
# 1. SIMPLEST USAGE — stream text tokens
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Stream text tokens (stream.messages) ───────────────")
print("\n  Tokens: ", end="", flush=True)

stream = agent.stream_events(INPUT, version="v3")

for message in stream.messages:
    for delta in message.text:
        print(delta, end="", flush=True)

print(f"\n\n  ✅ Done. Final output:")
final_state = stream.output
print(f"  {final_state['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 2. message.output — finalized AIMessage with usage metadata
# ════════════════════════════════════════════════════════════════════

print("\n── 2. message.output with usage metadata ─────────────────")

stream = agent.stream_events(INPUT, version="v3")

for message in stream.messages:
    # Collect all text tokens
    text = ""
    for delta in message.text:
        text += delta

    # message.output is the finalized AIMessage
    full_msg = message.output
    usage    = full_msg.usage_metadata if full_msg else None

    if text:
        print(f"\n  [{message.node}] Text: {text[:100]}…" if len(text) > 100 else f"\n  [{message.node}] Text: {text}")

    if usage:
        print(f"  [{message.node}] Usage:")
        print(f"    input_tokens:  {usage.get('input_tokens')}")
        print(f"    output_tokens: {usage.get('output_tokens')}")
        print(f"    total_tokens:  {usage.get('total_tokens')}")


# ════════════════════════════════════════════════════════════════════
# 3. str(message.text) — get final text without iterating deltas
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Final text without iterating (str shortcut) ────────")

stream = agent.stream_events(INPUT, version="v3")

for message in stream.messages:
    # Drain the stream silently (iterate without printing)
    # Then get the final accumulated text
    for _ in message.text:
        pass
    final_text = str(message.text)   # str() returns the full accumulated text
    if final_text:
        print(f"\n  Node: {message.node}")
        print(f"  Text: {final_text}")


# ════════════════════════════════════════════════════════════════════
# 4. stream.output — final agent state after run completes
# ════════════════════════════════════════════════════════════════════

print("\n── 4. stream.output — final agent state ──────────────────")

stream     = agent.stream_events(INPUT, version="v3")
_          = list(stream.messages)   # drain all projections

final      = stream.output
last_msg   = final["messages"][-1]

print(f"\n  Final message type:    {type(last_msg).__name__}")
print(f"  Final message content: {last_msg.content}")
print(f"  Total messages in run: {len(final['messages'])}")


# ════════════════════════════════════════════════════════════════════
# 5. RAW PROTOCOL EVENTS — full envelope for every event
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Raw protocol events (for loop) ─────────────────────")

stream = agent.stream_events(INPUT, version="v3")

event_types = {}
for event in stream:
    method    = event.get("method", "unknown")
    namespace = event.get("params", {}).get("namespace", [])
    event_types[method] = event_types.get(method, 0) + 1

print(f"\n  Raw events by method:")
for method, count in sorted(event_types.items()):
    print(f"    {method}: {count} events")
print(f"\n  (Use 'for event in stream' for full envelope access)")


# ════════════════════════════════════════════════════════════════════
# 6. ASYNC — astream_events for async contexts
# ════════════════════════════════════════════════════════════════════

print("\n── 6. Async astream_events ───────────────────────────────")

async def demo_async_stream():
    stream = await agent.astream_events(INPUT, version="v3")

    print("\n  Async tokens: ", end="", flush=True)
    async for message in stream.messages:
        async for delta in message.text:
            print(delta, end="", flush=True)
    print()

asyncio.run(demo_async_stream())


# ════════════════════════════════════════════════════════════════════
# 7. PROJECTION REFERENCE
# ════════════════════════════════════════════════════════════════════

print("\n── 7. Projection reference ───────────────────────────────")
print("""
  stream = agent.stream_events(input, version="v3")

  ┌──────────────────────────┬────────────────────────────────────┐
  │ Projection               │ What you get                       │
  ├──────────────────────────┼────────────────────────────────────┤
  │ for event in stream      │ Raw protocol events (full envelope)│
  │ stream.messages          │ One ChatModelStream per LLM call   │
  │   message.text           │ Text deltas (iterable) / str()     │
  │   message.reasoning      │ Reasoning deltas (if supported)    │
  │   message.tool_calls     │ Tool-call arg chunks + finalized   │
  │   message.output         │ Final AIMessage with usage         │
  │ stream.tool_calls        │ Tool execution lifecycle           │
  │ stream.values            │ State snapshots after each step    │
  │ stream.output            │ Final agent state dict             │
  │ stream.subgraphs         │ Nested agent/subgraph runs         │
  │ stream.extensions        │ Custom transformer projections     │
  └──────────────────────────┴────────────────────────────────────┘
""")
