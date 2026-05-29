"""
02_values_mode.py
=================
Demonstrates stream_mode="values" — full state snapshots after EACH GRAPH STEP.

Concepts covered:
  - agent.stream(..., stream_mode="values")  — full state dictionary streamed
  - How the messages list accumulates step-by-step
  - Comparing "values" vs "updates" stream modes
  - Extracting the latest message added in each step
  - Thread persistence with MemorySaver

In stream_mode="values", you receive the entire state payload. It is excellent
when you want to update your UI state directly with the latest state without
manually merging updates or diffs.
"""

import os
import uuid
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Define a simple mock tool
@tool
def check_flight_status(flight_number: str) -> str:
    """Get the current flight status for a given flight number.

    Args:
        flight_number: Flight ID (e.g. 'AA123', 'UA456')
    """
    flights = {
        "aa123": "✈️  AA123: Delayed 45m due to weather",
        "ua456": "✈️  UA456: On time and landing in 20m",
    }
    return flights.get(flight_number.lower(), f"No active flights found for '{flight_number}'")


# Create agent with memory
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[check_flight_status],
    checkpointer=MemorySaver(),
    system_prompt="You are a helpful flight assistant.",
)


print("=" * 60)
print("stream_mode='values' vs 'updates' Demo")
print("=" * 60)

INPUT = {"messages": [{"role": "user", "content": "What is the status of flight AA123?"}]}
config = {"configurable": {"thread_id": str(uuid.uuid4())}}


# ════════════════════════════════════════════════════════════════════
# 1. RUNNING WITH stream_mode="values"
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Running stream_mode='values' (Yields FULL state) ──────")

step_count = 0
for chunk in agent.stream(INPUT, config=config, stream_mode="values", version="v2"):
    step_count += 1
    # chunk is the entire current state dict (e.g. {'messages': [...]})
    # Under version="v2", chunk is {"type": "values", "data": state_dict}
    state = chunk.get("data", {}) if chunk.get("type") == "values" else chunk
    messages = state.get("messages", [])
    
    print(f"\n📍 Step #{step_count} — Full messages list length: {len(messages)}")
    # Print the last message in the state to see what was just added
    if messages:
        last_msg = messages[-1]
        print(f"   Last message type: {type(last_msg).__name__}")
        print(f"   Last content snippet: {str(last_msg.content)[:100]}")


# ════════════════════════════════════════════════════════════════════
# 2. RUNNING WITH stream_mode="updates" (For Comparison)
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("2. Running stream_mode='updates' (Yields ONLY increments)")
print("─" * 60)

config_updates = {"configurable": {"thread_id": str(uuid.uuid4())}}
step_count = 0

for chunk in agent.stream(INPUT, config=config_updates, stream_mode="updates", version="v2"):
    step_count += 1
    # chunk contains: {"type": "updates", "data": {node_name: updates}}
    if chunk["type"] == "updates":
        print(f"\n📍 Step #{step_count} — type: updates")
        for node_name, node_data in chunk["data"].items():
            new_msgs = node_data.get("messages", [])
            print(f"   Node: '{node_name}' executed")
            print(f"   Returned {len(new_msgs)} new messages")
            if new_msgs:
                print(f"   New message: {type(new_msgs[-1]).__name__} -> {str(new_msgs[-1].content)[:100]}")


# ════════════════════════════════════════════════════════════════════
# 3. SUMMARY REFERENCE
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("3. Comparative Summary")
print("─" * 60)
print("""
  Both stream_mode="values" and stream_mode="updates" stream at the graph step level
  (i.e. they yield one chunk after a node like "model" or "tools" finishes).

  ┌───────────────────────┬─────────────────────────────────────────────────┐
  │ stream_mode="values"  │ Yields: Full state dict {'messages': [...]}     │
  │                       │ Perfect for: Replacing client state directly.    │
  ├───────────────────────┼─────────────────────────────────────────────────┤
  │ stream_mode="updates" │ Yields: Incremental delta {'data': {node: ...}} │
  │                       │ Perfect for: Triggering side-effects per node.  │
  └───────────────────────┴─────────────────────────────────────────────────┘
""")
