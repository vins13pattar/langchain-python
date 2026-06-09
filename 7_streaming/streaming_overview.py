"""
streaming_overview.py — LangChain Agent Streaming Modes: all key concepts in one file
Covers: stream_mode updates, values, messages; async; thread_id; tool call extraction
"""

import asyncio
import uuid
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

def section(title): print(f"\n{'─'*55}\n{title}\n{'─'*55}")


# ── Tools ──────────────────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """Get weather for a city. Args: city: City name."""
    return {"london": "Cloudy 14°C", "tokyo": "Sunny 28°C", "mumbai": "Rainy 30°C"}.get(city.lower(), "No data")

@tool
def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic expression. Args: expression: e.g. '2 + 2'"""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression): return "Error"
        return f"Result: {round(eval(expression), 4)}"  # noqa: S307
    except Exception as e:
        return f"Error: {e}"

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, calculate],
    checkpointer=MemorySaver(),
    system_prompt="You are a helpful assistant with weather and math tools.",
)

INPUT = {"messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]}


# ════════════════════════════════════════════════════════════════════
# 1. stream_mode="updates" — one chunk per agent step (coarse-grained)
# ════════════════════════════════════════════════════════════════════
section("1. stream_mode='updates'")

cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

print("Steps for 'Weather in Tokyo':")
for chunk in agent.stream(INPUT, config=cfg, stream_mode="updates", version="v2"):
    if chunk["type"] == "updates":
        for step_name, step_data in chunk["data"].items():
            msgs = step_data.get("messages", [])
            if msgs:
                last = msgs[-1]
                c = last.content if isinstance(last.content, str) else str(last.content)
                print(f"  step={step_name!r}  type={type(last).__name__}  content={c[:80]}")

# Extract tool calls + results from updates
cfg2 = {"configurable": {"thread_id": str(uuid.uuid4())}}
print("\nTool call extraction ('Calculate 99 × 88'):")
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Calculate 99 * 88."}]},
    config=cfg2, stream_mode="updates", version="v2"
):
    if chunk["type"] == "updates":
        for step, data in chunk["data"].items():
            for msg in data.get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  📝 Tool call: {tc['name']}({tc['args']})")
                elif isinstance(msg, ToolMessage):
                    print(f"  🔧 Tool result: {msg.content}")
                elif isinstance(msg, AIMessage) and msg.content:
                    print(f"  💬 Reply: {msg.content[:80]}")

# Multi-turn with thread_id
persistent_cfg = {"configurable": {"thread_id": "demo-session"}}
def stream_reply(question: str) -> str:
    final = ""
    for chunk in agent.stream({"messages": [{"role": "user", "content": question}]},
                               config=persistent_cfg, stream_mode="updates", version="v2"):
        if chunk["type"] == "updates":
            for _, data in chunk["data"].items():
                msgs = data.get("messages", [])
                if msgs and isinstance(msgs[-1], AIMessage) and isinstance(msgs[-1].content, str):
                    final = msgs[-1].content
    return final

print("\nMulti-turn memory:")
print("T1:", stream_reply("Hi! I'm Vinod, from Bengaluru.")[:80])
print("T2:", stream_reply("What's my name and city?")[:80])


# ════════════════════════════════════════════════════════════════════
# 2. stream_mode="values" — full state snapshot after each step
# ════════════════════════════════════════════════════════════════════
section("2. stream_mode='values'")

cfg3 = {"configurable": {"thread_id": str(uuid.uuid4())}}
print("Full state snapshots:")
step = 0
for chunk in agent.stream(INPUT, config=cfg3, stream_mode="values", version="v2"):
    step += 1
    state = chunk.get("data", {}) if chunk.get("type") == "values" else chunk
    msgs = state.get("messages", [])
    if msgs:
        last = msgs[-1]
        print(f"  Snapshot #{step}: {len(msgs)} msgs total  last={type(last).__name__}  {str(last.content)[:60]}")

print("""
  values vs updates:
    values  → full {'messages': [...]} each step — replace UI state directly
    updates → incremental delta {node: updates} — good for side-effects
""")


# ════════════════════════════════════════════════════════════════════
# 3. stream_mode="messages" — token-by-token streaming
# ════════════════════════════════════════════════════════════════════
section("3. stream_mode='messages'")

cfg4 = {"configurable": {"thread_id": str(uuid.uuid4())}}
full_response = None

print("Live tokens: ", end="", flush=True)
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Tell me a short joke, then calculate 25 * 4."}]},
    config=cfg4, stream_mode="messages", version="v2"
):
    if isinstance(chunk, dict) and chunk.get("type") == "messages":
        payload = chunk.get("data")
        msg_chunk = payload[0] if isinstance(payload, tuple) else payload
    else:
        msg_chunk = chunk[0] if isinstance(chunk, tuple) else chunk

    if isinstance(msg_chunk, AIMessageChunk):
        full_response = msg_chunk if full_response is None else full_response + msg_chunk
        if msg_chunk.content:
            text = msg_chunk.content if isinstance(msg_chunk.content, str) else ""
            print(text, end="", flush=True)

print("\n✅ Done streaming")
if full_response and full_response.tool_calls:
    print(f"Tool calls in accumulated msg: {[tc['name'] for tc in full_response.tool_calls]}")


# ════════════════════════════════════════════════════════════════════
# 4. ASYNC STREAMING
# ════════════════════════════════════════════════════════════════════
section("4. ASYNC STREAMING")

async def async_updates():
    cfg5 = {"configurable": {"thread_id": str(uuid.uuid4())}}
    steps = []
    async for chunk in agent.astream(INPUT, config=cfg5, stream_mode="updates", version="v2"):
        if chunk["type"] == "updates":
            for step in chunk["data"]:
                steps.append(step)
    return steps

steps = asyncio.run(async_updates())
print(f"Async steps: {steps}")


# ════════════════════════════════════════════════════════════════════
# QUICK REFERENCE
# ════════════════════════════════════════════════════════════════════
print("""
agent.stream(input, stream_mode=..., version="v2"):
  "updates"  → one chunk per step, incremental delta
  "values"   → full state dict after each step
  "messages" → AIMessageChunk tokens as they're generated

For new apps, prefer stream_events(version="v3") from 6_event_streaming.
""")
