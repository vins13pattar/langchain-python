"""
event_streaming_overview.py — LangChain Event Streaming: all key concepts in one file
Covers: stream_events v3, message projections, tool call streaming, async, raw events
"""

import asyncio
import time
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

def section(title): print(f"\n{'─'*55}\n{title}\n{'─'*55}")


# ── Shared tools ──────────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """Get weather for a city. Args: city: City name."""
    time.sleep(0.1)  # simulate network
    return {"london": "Cloudy 14°C", "tokyo": "Sunny 28°C", "paris": "Clear 22°C"}.get(city.lower(), "No data")

@tool
def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic expression. Args: expression: e.g. '2 + 2'"""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression): return "Error: unsupported chars"
        return str(round(eval(expression), 4))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"

@tool
def search_articles(topic: str, limit: int = 3) -> str:
    """Search articles on a topic. Args: topic: Topic, limit: Max results."""
    return f"Found {limit} articles about '{topic}'"

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, calculate, search_articles],
    system_prompt="You are a helpful assistant with weather, math, and search tools.",
)

INPUT = {"messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]}


# ════════════════════════════════════════════════════════════════════
# 1. STREAM TEXT TOKENS — stream.messages + message.text
# ════════════════════════════════════════════════════════════════════
section("1. STREAM TEXT TOKENS")

stream = agent.stream_events(INPUT, version="v3")

print("Live tokens: ", end="", flush=True)
for message in stream.messages:
    for delta in message.text:
        print(delta, end="", flush=True)

final = stream.output
print(f"\nFinal: {final['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 2. message.output — finalised AIMessage + usage metadata
# ════════════════════════════════════════════════════════════════════
section("2. USAGE METADATA (message.output)")

stream = agent.stream_events(INPUT, version="v3")
for message in stream.messages:
    for _ in message.text: pass   # drain silently
    full_msg = message.output
    if full_msg and full_msg.usage_metadata:
        u = full_msg.usage_metadata
        print(f"Node: {message.node}  in={u.get('input_tokens')} out={u.get('output_tokens')} total={u.get('total_tokens')}")


# ════════════════════════════════════════════════════════════════════
# 3. str(message.text) — get full text without iterating deltas
# ════════════════════════════════════════════════════════════════════
section("3. FULL TEXT SHORTCUT")

stream = agent.stream_events(INPUT, version="v3")
for message in stream.messages:
    for _ in message.text: pass
    text = str(message.text)
    if text:
        print(f"Node [{message.node}]: {text[:120]}")


# ════════════════════════════════════════════════════════════════════
# 4. TOOL CALL STREAMING — argument chunks + execution lifecycle
# ════════════════════════════════════════════════════════════════════
section("4. TOOL CALL STREAMING")

INPUT2 = {"messages": [{"role": "user", "content": "Weather in Tokyo AND calculate 150 * 3.5"}]}

# 4a. message.tool_calls — argument chunks while model generates
stream = agent.stream_events(INPUT2, version="v3")
print("Argument chunks:")
for message in stream.messages:
    for chunk in message.tool_calls:
        name = chunk.get("name", "")
        args = chunk.get("args", "")
        if name: print(f"  🔧 Calling: {name}")
        if args: print(f"     args fragment: {args!r}", end="", flush=True)
    finalized = message.tool_calls.get()
    if finalized:
        print(f"\n  ✅ Finalized: {[(tc['name'], tc['args']) for tc in finalized]}")

# 4b. stream.tool_calls — execution lifecycle (input, output_deltas, error)
stream = agent.stream_events(INPUT2, version="v3")
for _ in stream.messages: pass  # drain messages first
print("Execution lifecycle:")
for call in stream.tool_calls:
    print(f"  ⚡ {call.tool_name}({call.input})")
    for delta in call.output_deltas: pass  # consume deltas
    status = "✅" if not call.error else "❌"
    print(f"  {status} Result: {str(call.output)[:80]}")


# ════════════════════════════════════════════════════════════════════
# 5. PARALLEL TOOL CALLS
# ════════════════════════════════════════════════════════════════════
section("5. PARALLEL TOOL CALLS")

INPUT3 = {"messages": [{"role": "user", "content": "Get weather in London, Tokyo, Paris AND calculate 500 * 0.08"}]}
stream = agent.stream_events(INPUT3, version="v3")
for _ in stream.messages: pass
calls_seen = []
for call in stream.tool_calls:
    calls_seen.append(call.tool_name)
    for _ in call.output_deltas: pass
    print(f"  {call.tool_name}: {str(call.output)[:50]}")
print(f"  Total parallel calls: {len(calls_seen)}")


# ════════════════════════════════════════════════════════════════════
# 6. stream.output — final agent state
# ════════════════════════════════════════════════════════════════════
section("6. FINAL STATE (stream.output)")

stream = agent.stream_events(INPUT, version="v3")
for _ in stream.messages: pass
final = stream.output
print(f"Messages in run: {len(final['messages'])}")
print(f"Last message:    {final['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 7. RAW PROTOCOL EVENTS — full event envelope
# ════════════════════════════════════════════════════════════════════
section("7. RAW EVENTS")

stream = agent.stream_events(INPUT, version="v3")
event_counts: dict = {}
for event in stream:
    method = event.get("method", "unknown")
    event_counts[method] = event_counts.get(method, 0) + 1
print("Raw event counts by method:")
for method, cnt in sorted(event_counts.items()):
    print(f"  {method}: {cnt}")


# ════════════════════════════════════════════════════════════════════
# 8. ASYNC — astream_events for async contexts
# ════════════════════════════════════════════════════════════════════
section("8. ASYNC STREAMING")

async def demo_async():
    stream = await agent.astream_events(INPUT, version="v3")
    print("Async tokens: ", end="", flush=True)
    async for message in stream.messages:
        async for delta in message.text:
            print(delta, end="", flush=True)
    print("\nAsync tool calls:")
    async for call in stream.tool_calls:
        async for _ in call.output_deltas: pass
        print(f"  {call.tool_name} → {str(call.output)[:60]}")

asyncio.run(demo_async())


# ════════════════════════════════════════════════════════════════════
# PROJECTION REFERENCE
# ════════════════════════════════════════════════════════════════════
print("""
Stream projections (version="v3"):
  stream.messages        → one ChatModelStream per LLM call
    message.text         → token deltas / str() for full text
    message.tool_calls   → arg chunks + .get() for finalized
    message.output       → final AIMessage with usage metadata
  stream.tool_calls      → per-tool: input, output_deltas, output, error
  stream.values          → state snapshots after each step
  stream.output          → final agent state dict
  for event in stream    → raw protocol events (full envelope)
""")
