"""
01_message_types.py
===================
Demonstrates the FOUR core message types in LangChain.

Concepts covered:
  - SystemMessage   — primes model behaviour / persona
  - HumanMessage    — user input (text, images, files)
  - AIMessage       — model response + metadata
  - ToolMessage     — result of a tool execution

Messages are the fundamental unit of context for LLMs in LangChain.
Every model.invoke() call takes a list of messages and returns an AIMessage.

The SAME message types work across ALL providers (OpenAI, Anthropic, Gemini…)
because LangChain normalises them to the provider's native format internally.
"""

import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import tool

load_dotenv()

model = init_chat_model("openai:gpt-4o-mini")

print("=" * 60)
print("LangChain Message Types Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. SYSTEM MESSAGE — instructions / persona
# ════════════════════════════════════════════════════════════════════
# SystemMessage sets the model's role, tone, and constraints.
# It is processed before any user messages.

print("\n── 1. SystemMessage ──────────────────────────────────────")

# Simple system instruction
system_basic = SystemMessage("You are a helpful coding assistant.")

# Detailed persona
system_detailed = SystemMessage("""
You are a senior Python developer with 10 years of experience.
- Always provide working code examples.
- Explain your reasoning step by step.
- Be concise but thorough.
- If unsure, say so rather than guessing.
""")

# Using system message with a question
messages = [
    system_detailed,
    HumanMessage("What is the difference between a list and a tuple in Python?"),
]
response = model.invoke(messages)
print(f"\nSystem: (senior Python dev persona)")
print(f"User:   What is the difference between a list and a tuple?")
print(f"Model:  {response.content[:300]}…")


# ════════════════════════════════════════════════════════════════════
# 2. HUMAN MESSAGE — user input
# ════════════════════════════════════════════════════════════════════

print("\n── 2. HumanMessage ───────────────────────────────────────")

# Basic text message
human_basic = HumanMessage("What is machine learning?")

# With optional metadata fields
human_with_meta = HumanMessage(
    content="Hello! Can you help me with Python?",
    name="vinod",       # identify the user (provider-specific support)
    id="msg-001",       # unique ID for tracing/debugging
)

# String shorthand — equivalent to a single HumanMessage
response_a = model.invoke("What is 2 + 2?")                            # string
response_b = model.invoke([HumanMessage("What is 2 + 2?")])            # HumanMessage
# Both produce the same result

print(f"\nString shorthand:       {response_a.content}")
print(f"HumanMessage explicit:  {response_b.content}")
print(f"\nHumanMessage with meta:")
print(f"  name = {human_with_meta.name}")
print(f"  id   = {human_with_meta.id}")


# ════════════════════════════════════════════════════════════════════
# 3. AI MESSAGE — model output
# ════════════════════════════════════════════════════════════════════

print("\n── 3. AIMessage ──────────────────────────────────────────")

response = model.invoke([
    SystemMessage("You are a concise assistant. Answer in one sentence."),
    HumanMessage("What is a neural network?"),
])

print(f"\nType:             {type(response).__name__}")
print(f"Content:          {response.content}")
print(f"Text (alias):     {response.text}")
print(f"ID:               {response.id}")

# Token usage (available on most providers)
if response.usage_metadata:
    u = response.usage_metadata
    print(f"Usage:")
    print(f"  input_tokens:  {u.get('input_tokens')}")
    print(f"  output_tokens: {u.get('output_tokens')}")
    print(f"  total_tokens:  {u.get('total_tokens')}")

# Manually creating an AIMessage (inject into conversation history)
manual_ai_msg = AIMessage(
    content="I'd be happy to help you with Python!",
    id="ai-manual-001",
)

# Build a conversation with a manually injected AI message
conversation = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("Can you help me with Python?"),
    manual_ai_msg,                              # ← injected as if model said it
    HumanMessage("Great! Explain list comprehensions."),
]
response = model.invoke(conversation)
print(f"\nConversation (injected AI msg):")
print(f"  {response.content[:200]}…")


# ════════════════════════════════════════════════════════════════════
# 4. TOOL MESSAGE — result of a tool call
# ════════════════════════════════════════════════════════════════════

print("\n── 4. ToolMessage ────────────────────────────────────────")

@tool
def get_weather(location: str) -> str:
    """Return the current weather for a city.

    Args:
        location: City name
    """
    weather_db = {
        "london": "Cloudy, 15°C",
        "paris":  "Sunny, 22°C",
        "tokyo":  "Rainy, 18°C",
    }
    return weather_db.get(location.lower(), f"No data for {location}")


model_with_tool = model.bind_tools([get_weather])

# Step 1 — model decides to call the tool
ai_response = model_with_tool.invoke("What's the weather in Paris?")
print(f"\nStep 1 — Model tool call request:")
for tc in ai_response.tool_calls:
    print(f"  tool: {tc['name']}, args: {tc['args']}, id: {tc['id']}")

# Step 2 — execute tool, create ToolMessage with matching tool_call_id
tool_result = get_weather.invoke(ai_response.tool_calls[0])
print(f"\nStep 2 — ToolMessage:")
print(f"  content:      {tool_result.content}")
print(f"  tool_call_id: {tool_result.tool_call_id}")
print(f"  type:         {type(tool_result).__name__}")

# Step 3 — pass result back to model for final answer
messages = [
    HumanMessage("What's the weather in Paris?"),
    ai_response,    # model's tool call request
    tool_result,    # tool's response
]
final = model_with_tool.invoke(messages)
print(f"\nStep 3 — Final answer: {final.content}")


# ════════════════════════════════════════════════════════════════════
# 5. THREE INPUT FORMATS — string, dict, message object
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Three equivalent input formats ────────────────────")

# Format A: string (shorthand for single HumanMessage)
r1 = model.invoke("Name 3 planets.")

# Format B: dict (OpenAI chat completions style)
r2 = model.invoke([
    {"role": "system",    "content": "Be very brief."},
    {"role": "user",      "content": "Name 3 planets."},
])

# Format C: message objects
r3 = model.invoke([
    SystemMessage("Be very brief."),
    HumanMessage("Name 3 planets."),
])

print(f"\nFormat A (string): {r1.content}")
print(f"Format B (dict):   {r2.content}")
print(f"Format C (object): {r3.content}")
print("\nAll three formats produce equivalent results ✅")
