"""
03_messages_mode.py
===================
Demonstrates stream_mode="messages" — token-by-token model output streaming from graphs.

Concepts covered:
  - agent.stream(..., stream_mode="messages")  — streams raw message deltas
  - Yields message chunk objects (e.g. AIMessageChunk) and metadata
  - Accumulating chunks in real-time using the '+' operator
  - Extracting live text deltas and tool-call argument fragments
  - Isolating model chunks from other graph signals

In stream_mode="messages", LangGraph streams token deltas as they are generated
by the underlying chat model. This is the lowest-level graph-streaming mode
for real-time UX (chatbots) without using the v3 stream_events wrapper.
"""

import os
import uuid
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessageChunk, BaseMessage

load_dotenv()

# Define a simple calculator tool
@tool
def calculate(expression: str) -> str:
    """Evaluate a safe mathematical expression.

    Args:
        expression: A safe math string (e.g. '25 * 4')
    """
    try:
        return str(eval(expression))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[calculate],
    system_prompt="You are a helpful calculation assistant.",
)


print("=" * 60)
print("stream_mode='messages' Token Streaming Demo")
print("=" * 60)

INPUT = {"messages": [{"role": "user", "content": "Tell me a joke about math, then calculate 987 * 65."}]}
config = {"configurable": {"thread_id": str(uuid.uuid4())}}


# ════════════════════════════════════════════════════════════════════
# 1. STREAMING TOKENS IN REAL-TIME
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Streaming message chunks (stream_mode='messages') ──")
print("\nTokens: ", end="", flush=True)

# Yields tuples of (message_chunk, metadata) or message chunks directly depending on graph settings.
# LangGraph standard streams (chunk, metadata) tuples under stream_mode="messages".

full_response = None

for chunk in agent.stream(INPUT, config=config, stream_mode="messages", version="v2"):
    # Under version="v2", chunk is {"type": "messages", "data": (msg_chunk, metadata)}
    if isinstance(chunk, dict) and chunk.get("type") == "messages":
        payload = chunk.get("data")
        msg_chunk = payload[0] if isinstance(payload, tuple) else payload
        metadata  = payload[1] if isinstance(payload, tuple) else {}
    else:
        # Fallback if without version="v2"
        msg_chunk = chunk[0] if isinstance(chunk, tuple) else chunk
        metadata  = chunk[1] if isinstance(chunk, tuple) else {}
    
    # We only care about AIMessageChunk for token outputs
    if isinstance(msg_chunk, AIMessageChunk):
        # Accumulate the chunks to reconstruct the full final message
        full_response = msg_chunk if full_response is None else full_response + msg_chunk
        
        # Print the text delta as it arrives
        if msg_chunk.content:
            text = msg_chunk.content if isinstance(msg_chunk.content, str) else str(msg_chunk.content)
            print(text, end="", flush=True)

print("\n\n✅ Streaming Complete.")


# ════════════════════════════════════════════════════════════════════
# 2. CHUNK STRUCTURE INSPECTION
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("2. Inspecting the raw chunks")
print("─" * 60)

print(f"\nFinal Reconstructed Message Class: {type(full_response).__name__}")
if full_response:
    print(f"Final Reconstructed Message Content:\n{full_response.content[:150]}...\n")
    
    # If the model requested tools, tool call chunks will be embedded in the AIMessageChunk
    if full_response.tool_calls:
        print("🔧 Tool calls extracted from accumulated message:")
        for tc in full_response.tool_calls:
            print(f"   → {tc['name']}({tc['args']})")


# ════════════════════════════════════════════════════════════════════
# 3. SCHEMA REFERENCE
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("3. Messages Stream Reference")
print("─" * 60)
print("""
  When running:
    for chunk, metadata in agent.stream(input, stream_mode="messages", version="v2"):

  1. Each `chunk` is a subclass of `BaseMessage` (typically `AIMessageChunk`).
  2. You can check the class with: `isinstance(chunk, AIMessageChunk)`.
  3. Reconstruct the full final message using standard addition: `full = full + chunk`.
  4. Metadata contains useful routing tags:
     - `metadata["langgraph_node"]`  — the node that emitted the message (e.g. 'model')
     - `metadata["checkpoint_id"]`  — thread checkpoint ID
""")
