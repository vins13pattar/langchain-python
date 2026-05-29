"""
04_debug_mode.py
================
Demonstrates stream_mode="debug" — verbose graph trace events for troubleshooting.

Concepts covered:
  - agent.stream(..., stream_mode="debug")
  - Understanding raw debug event types:
      - 'task'        — node execution details (inputs, outputs)
      - 'checkpoint'  — thread state checkpointing
  - Filtering and printing specific debug fields
  - Visualizing the exact flow under the hood (node entries/exits)

In stream_mode="debug", LangGraph emits everything that happens inside the graph.
It is highly valuable when debugging complex graphs, custom loops, or checking
why a node isn't executing as expected.
"""

import os
import uuid
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

# Define a mock tool
@tool
def calculate_factorial(n: int) -> int:
    """Calculate the factorial of an integer n.

    Args:
        n: Non-negative integer
    """
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[calculate_factorial],
    system_prompt="You are a mathematics assistant. Use tools to find factorials.",
)


print("=" * 60)
print("stream_mode='debug' Graph Trace Demo")
print("=" * 60)

INPUT = {"messages": [{"role": "user", "content": "What is the factorial of 5?"}]}
config = {"configurable": {"thread_id": str(uuid.uuid4())}}


# ════════════════════════════════════════════════════════════════════
# 1. GRAPH EXECUTION WITH DEBUG SIGNAL STREAMING
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Executing graph in stream_mode='debug' ─────────────")

event_count = 0

for chunk in agent.stream(INPUT, config=config, stream_mode="debug", version="v2"):
    event_count += 1
    
    # Under version="v2", chunk is a dictionary like {"type": "debug", "data": {...}}
    if chunk.get("type") == "debug":
        data = chunk.get("data", {})
        event_type = data.get("type")
        
        print(f"\n⚡ Event #{event_count} — type: '{event_type}'")
        
        # Node executions ('task' events)
        if event_type == "task":
            node_name = data.get("node")
            print(f"   Node:    {node_name!r}")
            print(f"   Action:  Executing task...")
            
            # Print inputs (truncated)
            inputs = data.get("input", {})
            print(f"   Inputs:  {str(inputs)[:120]}…")
            
            # Print outputs (if finalized)
            outputs = data.get("output")
            if outputs:
                print(f"   Outputs: {str(outputs)[:120]}…")
                
        # Checkpointing ('checkpoint' events)
        elif event_type == "checkpoint":
            checkpoint = data.get("checkpoint", {})
            config_meta = data.get("config", {})
            print(f"   Action:  Saving Thread Checkpoint")
            print(f"   Checkpoint ID: {config_meta.get('configurable', {}).get('checkpoint_id')}")
            print(f"   Saved Keys:    {list(checkpoint.get('channel_values', {}).keys())}")
            
        else:
            # Other raw debug events
            print(f"   Raw Data keys: {list(data.keys())}")

print(f"\n✅ Finished. Total of {event_count} debug signals captured.")
