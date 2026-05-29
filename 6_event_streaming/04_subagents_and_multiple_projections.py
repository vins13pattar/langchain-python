"""
04_subagents_and_multiple_projections.py
=========================================
Demonstrates SUBGRAPH / SUB-AGENT streaming and MULTIPLE PROJECTIONS.

Concepts covered:
  - stream.subgraphs            — nested agent/graph runs surfaced as handles
  - subagent.graph_name         — filter by agent name
  - subagent.messages           — inner agent's model output
  - subagent.tool_calls         — inner agent's tool execution
  - subagent.output             — inner agent's final state
  - asyncio.gather + astream_events — consume multiple projections concurrently
  - stream.interleave()         — synchronous multi-projection consumption
  - name= on create_agent       — label agents for filtering

When a create_agent instance invokes another create_agent via a tool,
the inner agent's events flow at a nested namespace and surface on
stream.subgraphs as individual handles — each with its own .messages,
.tool_calls, .values, and .output projections.
"""

import os
import asyncio
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv()

print("=" * 60)
print("Subagents & Multiple Projections Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. DEFINE INNER AGENTS (sub-agents)
# ════════════════════════════════════════════════════════════════════

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city.

    Args:
        city: City name
    """
    data = {
        "london": "☁️  Cloudy, 14°C",
        "tokyo":  "☀️  Sunny, 28°C",
        "paris":  "🌤️  Clear, 22°C",
    }
    return data.get(city.lower(), f"No data for '{city}'")


@tool
def calculate_travel_cost(distance_km: float, mode: str = "flight") -> str:
    """Estimate travel cost between cities.

    Args:
        distance_km: Distance in kilometres
        mode:        Transport mode: 'flight', 'train', or 'bus'
    """
    rate = {"flight": 0.15, "train": 0.08, "bus": 0.04}.get(mode, 0.15)
    cost = distance_km * rate
    return f"Estimated cost by {mode}: ${cost:.2f} USD"


# Weather sub-agent — labelled with name= for filtering in stream.subgraphs
weather_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    name="weather_agent",            # ← used as subagent.graph_name
    system_prompt="You are a weather specialist. Answer weather questions concisely.",
)

# Travel cost sub-agent
travel_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[calculate_travel_cost],
    name="travel_agent",
    system_prompt="You are a travel cost estimator. Be brief.",
)


# ── Wrapper tools to invoke sub-agents ───────────────────────────

@tool
def query_weather_agent(question: str) -> str:
    """Ask the weather specialist agent about weather conditions.

    Args:
        question: Weather question to forward to the specialist
    """
    result = weather_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


@tool
def query_travel_agent(question: str) -> str:
    """Ask the travel cost agent for pricing estimates.

    Args:
        question: Travel cost question to forward to the specialist
    """
    result = travel_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


# Supervisor that orchestrates the sub-agents
supervisor = create_agent(
    model="openai:gpt-4o-mini",
    tools=[query_weather_agent, query_travel_agent],
    name="supervisor",
    system_prompt=(
        "You are a travel planning supervisor. "
        "Use the weather agent for weather and the travel agent for cost estimates. "
        "Combine their answers into a helpful summary."
    ),
)

INPUT = {
    "messages": [{
        "role": "user",
        "content": "What's the weather in Tokyo? Also estimate the cost of flying 9000km."
    }]
}


# ════════════════════════════════════════════════════════════════════
# 2. stream.subgraphs — nested agent handles
# ════════════════════════════════════════════════════════════════════

print("\n── 2. stream.subgraphs — nested agent streams ───────────")

stream = supervisor.stream_events(INPUT, version="v3")

# Drain the supervisor's own messages first
print(f"\n  Supervisor messages:")
for message in stream.messages:
    text = ""
    for delta in message.text:
        text += delta
    if text:
        print(f"    [{message.node}]: {text[:100]}")

# Now iterate nested subgraph handles
print(f"\n  Subgraph runs:")
for subagent in stream.subgraphs:
    print(f"\n    Subgraph: {subagent.graph_name}")

    for message in subagent.messages:
        text = ""
        for delta in message.text:
            text += delta
        if text:
            print(f"      [{message.node}] Text: {text[:80]}")

    for call in subagent.tool_calls:
        for _ in call.output_deltas:
            pass
        print(f"      🔧 Tool: {call.tool_name} → {str(call.output)[:60]}")

    final = subagent.output
    if final:
        last = final.get("messages", [])
        if last:
            print(f"      ✅ Final: {last[-1].content[:80]}")


# ════════════════════════════════════════════════════════════════════
# 3. FILTER SUBGRAPHS BY NAME
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Filter subgraphs by name ───────────────────────────")

stream = supervisor.stream_events(INPUT, version="v3")

# Drain supervisor messages
for _ in stream.messages:
    pass

print(f"\n  Only weather_agent output:")
for subagent in stream.subgraphs:
    if subagent.graph_name != "weather_agent":
        continue   # skip other subgraphs

    print(f"    Subgraph: {subagent.graph_name}")
    for message in subagent.messages:
        text = ""
        for delta in message.text:
            text += delta
        if text:
            print(f"    Response: {text}")


# ════════════════════════════════════════════════════════════════════
# 4. MULTIPLE PROJECTIONS — async with asyncio.gather
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Multiple projections with asyncio.gather ──────────")

async def demo_concurrent_projections():
    stream = await supervisor.astream_events(INPUT, version="v3")

    async def consume_messages():
        """Consume message text in the background."""
        results = []
        async for message in stream.messages:
            text = await message.text
            if text:
                results.append(f"[{message.node}]: {text[:60]}")
        return results

    async def consume_tool_calls():
        """Consume tool execution results in the background."""
        results = []
        async for call in stream.tool_calls:
            results.append(f"{call.tool_name}: {str(call.output)[:50]}")
        return results

    # Run both concurrently
    messages_result, tool_calls_result = await asyncio.gather(
        consume_messages(),
        consume_tool_calls(),
    )

    print(f"\n  Messages ({len(messages_result)}):")
    for m in messages_result[:3]:
        print(f"    {m}")

    print(f"\n  Tool calls ({len(tool_calls_result)}):")
    for tc in tool_calls_result:
        print(f"    {tc}")

asyncio.run(demo_concurrent_projections())


# ════════════════════════════════════════════════════════════════════
# 5. stream.interleave() — synchronous multi-projection
# ════════════════════════════════════════════════════════════════════

print("\n── 5. stream.interleave() — sync multi-projection ────────")

simple_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    system_prompt="Use tools to answer weather questions.",
)

stream = simple_agent.stream_events(
    {"messages": [{"role": "user", "content": "What's the weather in London and Tokyo?"}]},
    version="v3",
)

print("\n  Interleaved events (messages + tool_calls + values):")
for name, item in stream.interleave("messages", "tool_calls", "values"):
    if name == "messages":
        # Drain text silently, just show node
        for _ in item.text:
            pass
        final_text = str(item.text)
        if final_text:
            print(f"  📝 message [{item.node}]: {final_text[:60]}")
        else:
            print(f"  📝 message [{item.node}]: (tool call)")

    elif name == "tool_calls":
        for _ in item.output_deltas:
            pass
        print(f"  🔧 tool_call: {item.tool_name}({item.input}) → {str(item.output)[:50]}")

    elif name == "values":
        n = len(item.get("messages", []))
        print(f"  📊 snapshot: {n} messages in state")
