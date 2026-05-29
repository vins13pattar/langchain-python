"""
03_message_content_and_streaming.py
=====================================
Demonstrates MESSAGE CONTENT types and STREAMING.

Concepts covered:
  - content attribute — string vs list of content blocks
  - content_blocks property — standardised cross-provider representation
  - TextContentBlock, ReasoningContentBlock — standard block types
  - Streaming with model.stream()
  - AIMessageChunk — accumulating chunks into a full message
  - Streaming tool calls (ToolCallChunk)
  - astream_events() — semantic event streaming

Content blocks are how LangChain represents rich, structured message
payloads (text, reasoning, tool calls) in a UNIFIED format that works
the same way regardless of provider (OpenAI, Anthropic, Gemini…).
"""

import os
import asyncio
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    AIMessageChunk,
)
from langchain_core.tools import tool

load_dotenv()

model = init_chat_model("openai:gpt-4o-mini")

print("=" * 60)
print("Message Content & Streaming Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. MESSAGE CONTENT — string vs content blocks
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Message content formats ────────────────────────────")

# 1a. Simple string content
msg_string = HumanMessage("Hello, how are you?")
print(f"\n1a. String content:")
print(f"  msg.content  = {msg_string.content!r}")
print(f"  type         = {type(msg_string.content).__name__}")

# 1b. List of provider-native content blocks (OpenAI format)
msg_native = HumanMessage(content=[
    {"type": "text", "text": "What is LangChain?"},
])
print(f"\n1b. Provider-native list content:")
print(f"  msg.content  = {msg_native.content}")
print(f"  type         = {type(msg_native.content).__name__}")

# 1c. Using content_blocks property — standardised view
response = model.invoke("Explain Python briefly.")
print(f"\n1c. AIMessage content_blocks (standardised view):")
for block in response.content_blocks:
    print(f"  block type = {block.get('type')!r}")
    if block.get("type") == "text":
        print(f"  text       = {block['text'][:100]}…")


# ════════════════════════════════════════════════════════════════════
# 2. STREAMING — real-time token output
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Basic text streaming ───────────────────────────────")
print("\nTokens (separated by │):")
print("", end="  ", flush=True)

chunks = []
for chunk in model.stream("List 5 famous scientists, one per line."):
    print(chunk.text, end="│", flush=True)
    chunks.append(chunk)

print(f"\n\nTotal chunks received: {len(chunks)}")
print(f"Chunk type:           {type(chunks[0]).__name__}")


# ════════════════════════════════════════════════════════════════════
# 3. ACCUMULATING CHUNKS → full AIMessage
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Accumulating streaming chunks ──────────────────────")

full_message: AIMessageChunk | None = None

for chunk in model.stream("What is the capital of France?"):
    full_message = chunk if full_message is None else full_message + chunk

print(f"\nFull accumulated message:")
print(f"  content: {full_message.content}")
print(f"  type:    {type(full_message).__name__}")
# AIMessageChunk has same interface as AIMessage — can be used interchangeably


# ════════════════════════════════════════════════════════════════════
# 4. STREAMING CONTENT BLOCKS (text, reasoning, tool calls)
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Streaming content blocks ───────────────────────────")

for chunk in model.stream("What color is the sky?"):
    for block in chunk.content_blocks:
        block_type = block.get("type")
        if block_type == "text" and block.get("text"):
            print(f"  [text]      {block['text']}", end="", flush=True)
        elif block_type == "reasoning" and block.get("reasoning"):
            print(f"  [reasoning] {block['reasoning'][:60]}…")
        elif block_type == "tool_call_chunk":
            print(f"  [tool_call_chunk] {block}")

print()  # newline


# ════════════════════════════════════════════════════════════════════
# 5. STREAMING TOOL CALLS
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Streaming tool calls ───────────────────────────────")

@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol.

    Args:
        ticker: Stock ticker (e.g. AAPL, GOOGL)
    """
    prices = {"AAPL": 213.45, "GOOGL": 178.30, "MSFT": 425.80}
    return f"${prices.get(ticker.upper(), 0):.2f}"


model_with_tools = model.bind_tools([get_stock_price])

print("\nStreaming tool call chunks for: 'Get prices for AAPL and MSFT'")
print()

accumulated = None
for chunk in model_with_tools.stream("Get the stock prices for AAPL and MSFT"):
    accumulated = chunk if accumulated is None else accumulated + chunk

    # Show tool call chunks as they arrive
    for tc_chunk in chunk.tool_call_chunks:
        name = tc_chunk.get("name", "")
        args = tc_chunk.get("args", "")
        if name:
            print(f"  🔧 Tool: {name}")
        if args:
            print(f"     Args fragment: {args!r}")

# Final accumulated tool calls
if accumulated and accumulated.tool_calls:
    print("\nFully assembled tool calls:")
    for tc in accumulated.tool_calls:
        print(f"  name: {tc['name']}, args: {tc['args']}, id: {tc['id']}")


# ════════════════════════════════════════════════════════════════════
# 6. astream_events() — semantic event streaming
# ════════════════════════════════════════════════════════════════════

print("\n── 6. astream_events() — semantic events ─────────────────")

async def demo_astream_events():
    """Demonstrates astream_events for fine-grained event handling."""
    print()
    async for event in model.astream_events("Say hello in 5 words.", version="v2"):
        event_type = event["event"]

        if event_type == "on_chat_model_start":
            print(f"  🚀 Model started")

        elif event_type == "on_chat_model_stream":
            token = event["data"]["chunk"].text
            if token:
                print(f"  📝 Token: {token!r}")

        elif event_type == "on_chat_model_end":
            full_text = event["data"]["output"].text
            print(f"  ✅ Done. Full response: {full_text!r}")

asyncio.run(demo_astream_events())


# ════════════════════════════════════════════════════════════════════
# 7. READING RESPONSE METADATA FROM CHUNKS
# ════════════════════════════════════════════════════════════════════

print("\n── 7. Token usage from streaming ────────────────────────")

final_chunk = None
for chunk in model.stream(
    "What is 100 ÷ 4?",
):
    final_chunk = chunk if final_chunk is None else final_chunk + chunk

if final_chunk and final_chunk.usage_metadata:
    u = final_chunk.usage_metadata
    print(f"\n  Input tokens:  {u.get('input_tokens')}")
    print(f"  Output tokens: {u.get('output_tokens')}")
    print(f"  Total tokens:  {u.get('total_tokens')}")
else:
    print(f"\n  Response: {final_chunk.content}")
    print("  (Usage metadata not available for this model/config)")
