"""
01_updates_mode.py
==================
Demonstrates stream_mode="updates" — state updates after EACH AGENT STEP.

Concepts covered:
  - agent.stream(..., stream_mode="updates")  — one chunk per agent step
  - chunk["type"] == "updates"                — filter for update events
  - chunk["data"]                             — dict of {node_name: state_update}
  - Sequence of steps: model → tools → model
  - Content blocks in streamed messages
  - thread_id with checkpointer               — persist + resume conversation
  - version="v2"                              — required for structured chunks

stream_mode="updates" gives you a COARSE view of agent progress:
  "What did the model just decide?" and "What did the tools return?"
Each step produces one chunk. Use this when you want step-level visibility
rather than token-by-token streaming.

NOTE: This is the lower-level streaming API (agent.stream).
      For new apps, prefer stream_events() — see 6_event_streaming.
"""

import os
import uuid
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

print("=" * 60)
print("stream_mode='updates' Demo")
print("=" * 60)


# ── Tools ──────────────────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name
    """
    data = {
        "london": "☁️  Cloudy, 14°C",
        "tokyo":  "☀️  Sunny, 28°C",
        "mumbai": "🌧️  Rainy, 30°C",
    }
    return data.get(city.lower(), f"No weather data for '{city}'")


@tool
def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic expression.

    Args:
        expression: e.g. '2 + 2' or '100 * 1.08'
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: unsupported characters"
        return f"Result: {round(eval(expression), 4)}"  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, calculate],
    checkpointer=MemorySaver(),         # required for thread_id
    system_prompt="You are a helpful assistant with weather and math tools.",
)


# ════════════════════════════════════════════════════════════════════
# 1. BASIC updates STREAMING — one chunk per step
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Basic stream_mode='updates' ────────────────────────")

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

INPUT = {
    "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
}

chunk_count = 0
for chunk in agent.stream(INPUT, config=config, stream_mode="updates", version="v2"):
    chunk_count += 1

    if chunk["type"] == "updates":
        print(f"\n  Chunk #{chunk_count} — type: updates")
        for step_name, step_data in chunk["data"].items():
            msgs    = step_data.get("messages", [])
            last    = msgs[-1] if msgs else None
            content = ""

            if last:
                # content_blocks is a normalized list of typed content blocks
                blocks = getattr(last, "content_blocks", None) or last.content
                if isinstance(blocks, list):
                    for block in blocks:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                content += block.get("text", "")
                            elif block.get("type") == "tool_call":
                                content += f"[tool_call: {block.get('name')}({block.get('args')})]"
                else:
                    content = str(blocks)

            print(f"    step: {step_name!r}")
            print(f"    type: {type(last).__name__ if last else 'None'}")
            print(f"    content: {content[:100]}")

print(f"\n  Total chunks: {chunk_count}")


# ════════════════════════════════════════════════════════════════════
# 2. STEP SEQUENCE — model → tools → model
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Typical step sequence ──────────────────────────────")

config2 = {"configurable": {"thread_id": str(uuid.uuid4())}}

print(f"\n  Query: 'What's the weather in London and Tokyo?'\n")

steps_seen = []
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather in London and Tokyo?"}]},
    config=config2,
    stream_mode="updates",
    version="v2",
):
    if chunk["type"] == "updates":
        for step_name, step_data in chunk["data"].items():
            msgs = step_data.get("messages", [])
            if msgs:
                last     = msgs[-1]
                msg_type = type(last).__name__
                steps_seen.append((step_name, msg_type))
                print(f"    ▶ step={step_name!r:10}  msg_type={msg_type}")

print(f"\n  Step sequence: {' → '.join(s for s, _ in steps_seen)}")
print(f"  (model decides → tools execute → model responds)")


# ════════════════════════════════════════════════════════════════════
# 3. EXTRACTING TOOL CALLS from updates
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Extracting tool calls from updates ─────────────────")

config3 = {"configurable": {"thread_id": str(uuid.uuid4())}}

from langchain_core.messages import AIMessage, ToolMessage

print()
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Calculate 99 * 88."}]},
    config=config3,
    stream_mode="updates",
    version="v2",
):
    if chunk["type"] == "updates":
        for step, data in chunk["data"].items():
            msgs = data.get("messages", [])
            for msg in msgs:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  📝 Tool call:    {tc['name']}({tc['args']})")
                elif isinstance(msg, ToolMessage):
                    print(f"  🔧 Tool result:  {msg.content}")
                elif isinstance(msg, AIMessage) and msg.content:
                    c = msg.content if isinstance(msg.content, str) else str(msg.content)
                    print(f"  💬 Final reply:  {c[:100]}")


# ════════════════════════════════════════════════════════════════════
# 4. MULTI-TURN WITH thread_id — resuming conversation
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Multi-turn with thread_id ──────────────────────────")

persistent_config = {"configurable": {"thread_id": "vinod-session-demo"}}

def stream_reply(question: str) -> str:
    """Stream a query and return the final reply text."""
    final_text = ""
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=persistent_config,
        stream_mode="updates",
        version="v2",
    ):
        if chunk["type"] == "updates":
            for _, data in chunk["data"].items():
                msgs = data.get("messages", [])
                if msgs and isinstance(msgs[-1], AIMessage):
                    c = msgs[-1].content
                    text = c if isinstance(c, str) else ""
                    if text:
                        final_text = text
    return final_text

r1 = stream_reply("Hi! My name is Vinod and I live in Bengaluru.")
print(f"\n  T1: {r1[:100]}")

r2 = stream_reply("What is my name and where do I live?")
print(f"  T2: {r2[:100]}")
print(f"\n  ✅ Agent remembered 'Vinod' and 'Bengaluru' across turns.")


# ════════════════════════════════════════════════════════════════════
# 5. ASYNC updates STREAMING
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Async updates streaming ────────────────────────────")

import asyncio

async def stream_async():
    config4 = {"configurable": {"thread_id": str(uuid.uuid4())}}
    steps   = []
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "What is 144 / 12?"}]},
        config=config4,
        stream_mode="updates",
        version="v2",
    ):
        if chunk["type"] == "updates":
            for step in chunk["data"]:
                steps.append(step)
    return steps

steps = asyncio.run(stream_async())
print(f"\n  Async steps seen: {steps}")


# ════════════════════════════════════════════════════════════════════
# 6. CHUNK STRUCTURE REFERENCE
# ════════════════════════════════════════════════════════════════════

print("\n── 6. Chunk structure reference ──────────────────────────")
print("""
  Each chunk has this structure:
  {
    "type": "updates",
    "ns":   ["namespace"],        # subgraph namespace
    "data": {
      "model": {                  # or "tools", or any node name
        "messages": [...]         # messages added this step
      }
    }
  }

  Access pattern:
    for chunk in agent.stream(input, stream_mode="updates", version="v2"):
        if chunk["type"] == "updates":
            for step_name, step_data in chunk["data"].items():
                last_msg = step_data["messages"][-1]

  Step names in a typical create_agent:
    "model"  → AIMessage (with tool_calls or final reply)
    "tools"  → ToolMessage (tool execution result)
""")
