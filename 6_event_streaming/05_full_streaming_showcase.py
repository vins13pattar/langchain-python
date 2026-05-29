"""
05_full_streaming_showcase.py
==============================
A COMPLETE showcase combining ALL event streaming concepts.

Simulates a REAL-TIME RESEARCH ASSISTANT that:
  ✅ Streams text tokens live as they arrive
  ✅ Shows tool-call argument chunks while model generates them
  ✅ Tracks tool execution lifecycle (start, deltas, complete/error)
  ✅ Emits state snapshots after each step
  ✅ Uses stream.interleave() for a unified event loop
  ✅ Emits a final summary from stream.output
  ✅ Demonstrates async streaming with asyncio
  ✅ Implements a simple "live UI" renderer using stream events

This file is the capstone for the 6_event_streaming module.
Run it to see streaming in action from a full agent run.
"""

import os
import asyncio
import time
from typing import Iterator
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

print("=" * 60)
print("Full Event Streaming Showcase — Research Assistant")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════

@tool
def web_search(query: str, num_results: int = 3) -> str:
    """Search the web for information on any topic.

    Args:
        query:       Search query (2-6 keywords)
        num_results: Number of results to return (default 3)
    """
    # Simulated results with a brief delay
    time.sleep(0.2)
    results = [
        f"Result {i+1}: Comprehensive overview of '{query}' — source: research-db.example.com"
        for i in range(num_results)
    ]
    return "\n".join(results)


@tool
def get_statistics(topic: str) -> str:
    """Get key statistics and data points about a topic.

    Args:
        topic: The topic to retrieve statistics for
    """
    time.sleep(0.15)
    stats_db = {
        "python":       "Python usage: 48% of developers (2024). Ranked #1 on TIOBE index.",
        "langchain":    "LangChain: 92k GitHub stars. 1M+ monthly downloads. 500k+ projects.",
        "ai":           "AI market: $200B in 2023, projected $2T by 2030. 70% YoY growth.",
        "llm":          "LLM tokens per day: 10 trillion+ (2024). GPT-4 context: 128k tokens.",
    }
    for key, value in stats_db.items():
        if key in topic.lower():
            return value
    return f"Statistics for '{topic}': Growing rapidly with strong industry adoption."


@tool
def generate_summary(topic: str, findings: str) -> str:
    """Generate a structured research summary from raw findings.

    Args:
        topic:    The research topic
        findings: Raw findings to summarise
    """
    time.sleep(0.1)
    return (
        f"📋 Research Summary: {topic}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Key Findings:\n"
        f"  • {findings[:100]}…\n"
        f"  • Growing adoption across industry verticals\n"
        f"  • Strong open-source community support\n"
        f"Recommendation: High priority for further research"
    )


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[web_search, get_statistics, generate_summary],
    system_prompt=(
        "You are a research assistant. For any research topic:\n"
        "1. Search the web for information\n"
        "2. Get relevant statistics\n"
        "3. Generate a structured summary\n"
        "Be thorough but concise."
    ),
)

RESEARCH_INPUT = {
    "messages": [{
        "role": "user",
        "content": "Research the topic: LangChain and LLMs in 2024. Give me a comprehensive summary."
    }]
}


# ════════════════════════════════════════════════════════════════════
# 1. LIVE TERMINAL "UI" — real-time stream rendering
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Live Terminal UI (stream.messages + stream.tool_calls) ──")
print("\n  Running research agent…\n")
print("  " + "─" * 50)

stream = agent.stream_events(RESEARCH_INPUT, version="v3")

message_count = 0
for message in stream.messages:
    message_count += 1
    node_label = f"[{message.node}]"

    # Collect text tokens and display live
    text_buffer = ""
    for delta in message.text:
        text_buffer += delta

    # After message, check for tool calls
    finalized_tcs = message.tool_calls.get()
    if finalized_tcs:
        for tc in finalized_tcs:
            print(f"  🔧 Model calling: {tc['name']}({tc['args']})")
    elif text_buffer:
        print(f"\n  💬 {node_label} Agent response:")
        # Print line-wrapped at 60 chars
        words = text_buffer.split()
        line  = "     "
        for word in words:
            if len(line) + len(word) > 65:
                print(line)
                line = "     "
            line += word + " "
        if line.strip():
            print(line)

print(f"\n  {'─' * 50}")

# Show tool execution results
print(f"\n  Tool executions:")
for call in stream.tool_calls:
    start_time = time.time()
    for _ in call.output_deltas:
        pass
    elapsed = (time.time() - start_time) * 1000

    status = "✅" if not call.error else "❌"
    result = str(call.output)[:80] if call.output else str(call.error)[:80]
    print(f"  {status} {call.tool_name:<20} {elapsed:5.0f}ms  {result!r}")


# ════════════════════════════════════════════════════════════════════
# 2. STATE PROGRESSION — watching agent advance step by step
# ════════════════════════════════════════════════════════════════════

print("\n── 2. State progression ──────────────────────────────────")

stream  = agent.stream_events(RESEARCH_INPUT, version="v3")
prev_n  = 0
step    = 0

print(f"\n  {'Step':<6} {'Msgs':<6} {'New':<6} {'Last message type'}")
print(f"  {'─'*6} {'─'*6} {'─'*6} {'─'*25}")

for snapshot in stream.values:
    msgs  = snapshot.get("messages", [])
    n     = len(msgs)
    diff  = n - prev_n

    if diff > 0:
        step    += 1
        last     = msgs[-1]
        msg_type = type(last).__name__
        print(f"  {step:<6} {n:<6} +{diff:<5} {msg_type}")
    prev_n = n

final = stream.output
print(f"\n  Final answer:\n  {final['messages'][-1].content[:300]}")


# ════════════════════════════════════════════════════════════════════
# 3. INTERLEAVED STREAM — one unified event loop
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Unified interleaved stream ────────────────────────")

stream = agent.stream_events(RESEARCH_INPUT, version="v3")
events = []

for name, item in stream.interleave("messages", "tool_calls", "values"):
    if name == "messages":
        for _ in item.text:
            pass   # drain
        text = str(item.text)
        if text:
            events.append(("💬 message",  text[:50] + "…" if len(text) > 50 else text))
        else:
            tcs = item.tool_calls.get()
            if tcs:
                for tc in tcs:
                    events.append(("📝 tool_gen", f"{tc['name']}({tc['args']})"))

    elif name == "tool_calls":
        for _ in item.output_deltas:
            pass
        events.append(("🔧 tool_exec", f"{item.tool_name} → {str(item.output)[:40]}"))

    elif name == "values":
        n = len(item.get("messages", []))
        events.append(("📊 snapshot", f"{n} messages"))

print(f"\n  Unified event log ({len(events)} events):")
for kind, detail in events:
    print(f"    {kind:<15} {detail}")


# ════════════════════════════════════════════════════════════════════
# 4. ASYNC STREAMING
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Async streaming ────────────────────────────────────")

async def stream_research_async():
    """Stream a research run asynchronously."""
    stream = await agent.astream_events(RESEARCH_INPUT, version="v3")

    tokens_count = 0
    tool_calls   = []

    async def count_tokens():
        nonlocal tokens_count
        async for message in stream.messages:
            async for delta in message.text:
                tokens_count += 1

    async def track_tools():
        async for call in stream.tool_calls:
            tool_calls.append(call.tool_name)

    # Consume both projections concurrently
    await asyncio.gather(count_tokens(), track_tools())

    return tokens_count, tool_calls

token_n, tools_run = asyncio.run(stream_research_async())
print(f"\n  Async run complete:")
print(f"    Text token deltas received: {token_n}")
print(f"    Tools executed:             {tools_run}")


# ════════════════════════════════════════════════════════════════════
# 5. PROJECTION QUICK REFERENCE
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Quick reference ────────────────────────────────────")
print("""
  stream = agent.stream_events(input, version="v3")

  # Text tokens live
  for message in stream.messages:
      for delta in message.text:
          print(delta, end="", flush=True)
      full_text = str(message.text)          # final text

  # Tool argument chunks + finalized
  for message in stream.messages:
      for chunk in message.tool_calls:       # arg chunks
          print(chunk.get("name"), chunk.get("args"))
      finalized = message.tool_calls.get()  # complete tool calls

  # Tool execution lifecycle
  for call in stream.tool_calls:
      print(call.tool_name, call.input)
      for delta in call.output_deltas:
          print(delta, end="")
      print(call.output, call.error)

  # State snapshots
  for snapshot in stream.values:
      print(len(snapshot["messages"]), "messages")

  # Final state
  final = stream.output                     # wait for completion

  # Multi-projection (sync)
  for name, item in stream.interleave("messages", "tool_calls"):
      ...

  # Multi-projection (async)
  stream = await agent.astream_events(input, version="v3")
  await asyncio.gather(consume_messages(), consume_tools())
""")
