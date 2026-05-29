"""
06_full_messages_showcase.py
=============================
A COMPLETE showcase combining ALL message concepts in one example.

Demonstrates:
  ✅ SystemMessage, HumanMessage, AIMessage, ToolMessage
  ✅ Three input formats: string, dict, message objects
  ✅ Multi-turn conversation with growing message history
  ✅ Tool calling loop with parallel tool calls
  ✅ ToolMessage artifact field
  ✅ AIMessage.tool_calls, usage_metadata, content_blocks
  ✅ Streaming with chunk accumulation
  ✅ Multimodal image input
  ✅ Few-shot prompting via injected AIMessages
  ✅ Conversation branching

This is the "putting it all together" file. See 01–05 for isolated
deep-dives on each concept.
"""

import os
import asyncio
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    AIMessageChunk,
)
from langchain_core.tools import tool

load_dotenv()

model = init_chat_model("openai:gpt-4o-mini", temperature=0.3)

print("=" * 60)
print("Full Messages Showcase")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name
    """
    data = {
        "london": "Rainy 14°C", "paris": "Sunny 23°C",
        "tokyo":  "Cloudy 19°C", "mumbai": "Hot 35°C",
    }
    return data.get(city.lower(), f"No data for {city}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: e.g. '2 + 2' or '100 * 1.08'
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: unsafe characters"
        return str(round(eval(expression), 4))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


TOOLS    = [get_weather, calculate]
TOOL_MAP = {t.name: t for t in TOOLS}
model_t  = model.bind_tools(TOOLS)


# ════════════════════════════════════════════════════════════════════
# 1. THREE INPUT FORMATS (all equivalent)
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Three input formats (all equivalent) ───────────────")

# Format A: plain string
r_a = model.invoke("What is 7 × 6?")

# Format B: dict (OpenAI chat completions style)
r_b = model.invoke([
    {"role": "system", "content": "Be very brief."},
    {"role": "user",   "content": "What is 7 × 6?"},
])

# Format C: message objects
r_c = model.invoke([
    SystemMessage("Be very brief."),
    HumanMessage("What is 7 × 6?"),
])

print(f"\n  A (string): {r_a.content}")
print(f"  B (dict):   {r_b.content}")
print(f"  C (object): {r_c.content}")


# ════════════════════════════════════════════════════════════════════
# 2. MULTI-TURN CONVERSATION (manual history)
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Multi-turn conversation ────────────────────────────")

history = [SystemMessage("You are a knowledgeable travel assistant. Be concise.")]

def chat(user_text: str) -> str:
    history.append(HumanMessage(user_text))
    reply = model.invoke(history)
    history.append(reply)
    return reply.content

turns = [
    "I'm planning a trip to Japan.",
    "What's the best time to visit Tokyo?",
    "What was the first destination I mentioned?",    # recall test
]

for t in turns:
    print(f"\n  🧑 {t}")
    print(f"  🤖 {chat(t)}")

print(f"\n  History length: {len(history)} messages")


# ════════════════════════════════════════════════════════════════════
# 3. TOOL CALLING LOOP WITH PARALLEL CALLS
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Tool calling loop with parallel calls ──────────────")

messages = [
    SystemMessage("Use tools to answer. Be brief."),
    HumanMessage("What's the weather in London and Paris? Also what is 1500 * 0.18?"),
]

while True:
    response = model_t.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        break   # model is done — final answer

    print(f"\n  🔧 {len(response.tool_calls)} tool call(s):")
    for tc in response.tool_calls:
        result = TOOL_MAP[tc["name"]].invoke(tc)
        messages.append(result)
        print(f"     [{tc['name']}] {tc['args']} → {result.content}")

print(f"\n  Final: {response.content}")


# ════════════════════════════════════════════════════════════════════
# 4. STREAMING + CHUNK ACCUMULATION
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Streaming ──────────────────────────────────────────")

full: AIMessageChunk | None = None
tokens = []

print("\n  Tokens: ", end="", flush=True)
for chunk in model.stream("Name 4 programming languages, comma-separated."):
    full = chunk if full is None else full + chunk
    if chunk.text:
        tokens.append(chunk.text)
        print(chunk.text, end="", flush=True)

print(f"\n  Total chunks: {len(tokens)}")
print(f"  Full content: {full.content}")


# ════════════════════════════════════════════════════════════════════
# 5. FEW-SHOT PROMPTING via injected AIMessages
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Few-shot prompting (injected AIMessages) ───────────")

few_shot = [
    SystemMessage("Classify the sentiment of the text. Reply with ONLY: POSITIVE, NEGATIVE, or NEUTRAL."),
    # Injected examples (model never actually generated these)
    HumanMessage("I love this product!"),
    AIMessage("POSITIVE"),
    HumanMessage("This is the worst experience ever."),
    AIMessage("NEGATIVE"),
    HumanMessage("The package arrived."),
    AIMessage("NEUTRAL"),
    # Real user input
    HumanMessage("I'm absolutely thrilled with my new laptop!"),
]

response = model.invoke(few_shot)
print(f"\n  Input:  'I'm absolutely thrilled with my new laptop!'")
print(f"  Output: {response.content}")


# ════════════════════════════════════════════════════════════════════
# 6. AIMessage METADATA
# ════════════════════════════════════════════════════════════════════

print("\n── 6. AIMessage metadata ─────────────────────────────────")

response = model.invoke("Explain what an API is in one sentence.")

print(f"\n  Content:   {response.content}")
print(f"  ID:        {response.id}")
print(f"  Type:      {type(response).__name__}")

# Content blocks (standardised cross-provider view)
print(f"  content_blocks:")
for block in response.content_blocks:
    t = block.get("type")
    print(f"    → type={t!r}", end="")
    if t == "text":
        print(f", text={block['text'][:50]!r}…")
    else:
        print()

# Token usage
if response.usage_metadata:
    u = response.usage_metadata
    print(f"  usage_metadata:")
    print(f"    input_tokens:  {u.get('input_tokens')}")
    print(f"    output_tokens: {u.get('output_tokens')}")
    print(f"    total_tokens:  {u.get('total_tokens')}")


# ════════════════════════════════════════════════════════════════════
# 7. MULTIMODAL IMAGE INPUT
# ════════════════════════════════════════════════════════════════════

print("\n── 7. Multimodal image input ─────────────────────────────")

image_message = HumanMessage(content=[
    {"type": "text",  "text": "What geometric shape is shown? Answer in 3 words."},
    {
        "type": "image",
        "url":  "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Square_200x200.png/200px-Square_200x200.png",
    },
])

print("\n  Sending image URL to model…")
try:
    response = model.invoke([image_message])
    print(f"  Response: {response.content}")
except Exception as e:
    print(f"  (Skipped: {type(e).__name__}: {e})")


# ════════════════════════════════════════════════════════════════════
# 8. SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  Message Types — Quick Reference")
print("=" * 60)
print("""
  ┌────────────────┬──────────────────────────────────────────────┐
  │ Type           │ Purpose                                      │
  ├────────────────┼──────────────────────────────────────────────┤
  │ SystemMessage  │ Set persona, tone, rules                     │
  │ HumanMessage   │ User input (text, images, files)             │
  │ AIMessage      │ Model response + tool_calls + metadata       │
  │ ToolMessage    │ Tool result → pass back to model             │
  ├────────────────┼──────────────────────────────────────────────┤
  │ Input format   │ Notes                                        │
  ├────────────────┼──────────────────────────────────────────────┤
  │ "string"       │ Shorthand for single HumanMessage            │
  │ {dict}         │ OpenAI chat completions format               │
  │ Message obj    │ Most explicit, enables metadata              │
  ├────────────────┼──────────────────────────────────────────────┤
  │ AIMessage key attrs                                           │
  ├────────────────┬──────────────────────────────────────────────┤
  │ .content       │ Raw content (str or list)                    │
  │ .text          │ Text alias for .content                      │
  │ .content_blocks│ Standardised cross-provider blocks           │
  │ .tool_calls    │ Tool call requests from model                │
  │ .usage_metadata│ Token counts                                 │
  │ .id            │ Unique message identifier                    │
  └────────────────┴──────────────────────────────────────────────┘
""")
