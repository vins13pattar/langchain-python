"""
messages_overview.py — LangChain Messages: all key concepts in one file
Covers: message types, conversation history, streaming, tool loop, multimodal
"""

import asyncio
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import tool

load_dotenv()

model = init_chat_model("openai:gpt-4o-mini")

def section(title): print(f"\n{'─'*55}\n{title}\n{'─'*55}")


# ════════════════════════════════════════════════════════════════════
# 1. MESSAGE TYPES — System, Human, AI, Tool
# ════════════════════════════════════════════════════════════════════
section("1. MESSAGE TYPES")

# SystemMessage — sets persona/instructions
resp = model.invoke([
    SystemMessage("You are a senior Python dev. Be concise."),
    HumanMessage("Difference between list and tuple?"),
])
print("System+Human:", resp.content[:200])

# HumanMessage — string shorthand vs explicit object (equivalent)
r1 = model.invoke("What is 2+2?")
r2 = model.invoke([HumanMessage("What is 2+2?")])
print("String:", r1.content, "  HumanMessage:", r2.content)

# AIMessage — inspect metadata
resp = model.invoke([SystemMessage("Answer in one sentence."), HumanMessage("What is a neural network?")])
print(f"AIMessage type: {type(resp).__name__}  content: {resp.content}")
if resp.usage_metadata:
    u = resp.usage_metadata
    print(f"Tokens — in: {u.get('input_tokens')}  out: {u.get('output_tokens')}")

# Three equivalent input formats
r_str  = model.invoke("Name 3 planets.")
r_dict = model.invoke([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Name 3 planets."}])
r_obj  = model.invoke([SystemMessage("Be brief."), HumanMessage("Name 3 planets.")])
print("Str:", r_str.content, "\nDict:", r_dict.content, "\nObj:", r_obj.content)


# ════════════════════════════════════════════════════════════════════
# 2. CONVERSATION HISTORY — stateless multi-turn
# ════════════════════════════════════════════════════════════════════
section("2. CONVERSATION HISTORY")

conversation: list = [SystemMessage("You are a friendly Python tutor. Keep answers to 2-3 sentences.")]

def chat(text: str) -> str:
    conversation.append(HumanMessage(text))
    resp = model.invoke(conversation)
    conversation.append(resp)
    return resp.content

print(chat("Hi! I'm Vinod, learning Python."))
print(chat("What is a list comprehension?"))
print(chat("What's my name?"))   # tests recall

# Few-shot: inject fake AIMessage to steer style
few_shot = [
    SystemMessage("You are a pirate assistant."),
    HumanMessage("What is Python?"),
    AIMessage("Arrr, Python be a mighty coding treasure, matey!"),
    HumanMessage("What is a list?"),
]
print("Few-shot pirate:", model.invoke(few_shot).content[:100])

# Branching from shared history
base = [
    SystemMessage("Be concise."),
    HumanMessage("Tell me about Python data structures."),
    AIMessage("Python has lists, tuples, sets, and dicts. Which interests you?"),
]
print("Branch lists:", model.invoke(base + [HumanMessage("Tell me about lists.")]).content[:120])
print("Branch dicts:", model.invoke(base + [HumanMessage("Tell me about dicts.")]).content[:120])


# ════════════════════════════════════════════════════════════════════
# 3. STREAMING — token-by-token output and chunk accumulation
# ════════════════════════════════════════════════════════════════════
section("3. STREAMING")

# Basic stream
print("Tokens: ", end="", flush=True)
chunks = []
for chunk in model.stream("List 5 famous scientists, one per line."):
    print(chunk.text, end="│", flush=True)
    chunks.append(chunk)
print(f"\n  Total chunks: {len(chunks)}")

# Accumulate chunks into one message
full: AIMessageChunk | None = None
for chunk in model.stream("Capital of France?"):
    full = chunk if full is None else full + chunk
print(f"Accumulated: {full.content}")

# astream_events — semantic events
async def demo_events():
    async for event in model.astream_events("Say hello in 5 words.", version="v2"):
        et = event["event"]
        if et == "on_chat_model_start":    print("  Started")
        elif et == "on_chat_model_stream": print(f"  Token: {event['data']['chunk'].text!r}", end="", flush=True)
        elif et == "on_chat_model_end":    print(f"\n  Done: {event['data']['output'].text!r}")

asyncio.run(demo_events())


# ════════════════════════════════════════════════════════════════════
# 4. TOOL MESSAGE LOOP — manual bind_tools → execute → ToolMessage
# ════════════════════════════════════════════════════════════════════
section("4. TOOL MESSAGE LOOP")

@tool
def get_weather(location: str) -> str:
    """Get weather for a city. Args: location: City name."""
    return {"london": "Rainy 14°C", "paris": "Sunny 23°C", "tokyo": "Cloudy 19°C"}.get(location.lower(), "No data")

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency. Args: amount, from_currency, to_currency (e.g. USD, EUR)."""
    rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "INR": 83.2}
    if from_currency.upper() not in rates or to_currency.upper() not in rates:
        return "Unsupported pair"
    result = (amount / rates[from_currency.upper()]) * rates[to_currency.upper()]
    return f"{amount} {from_currency} = {result:.2f} {to_currency}"

TOOLS = {"get_weather": get_weather, "convert_currency": convert_currency}
model_tools = model.bind_tools(list(TOOLS.values()))

def run_tool_loop(question: str, max_iter: int = 5) -> str:
    messages = [SystemMessage("Use tools when needed."), HumanMessage(question)]
    for _ in range(max_iter):
        resp = model_tools.invoke(messages)
        messages.append(resp)
        if not resp.tool_calls:
            return resp.content
        for tc in resp.tool_calls:
            result = TOOLS[tc["name"]].invoke(tc)
            messages.append(result)
    return "Max iterations reached"

print("Single tool:", run_tool_loop("What's the weather in Paris?"))
print("Multi tool:", run_tool_loop("Weather in London AND convert 100 USD to EUR."))

# tool_choice — force a specific tool
model_forced = model.bind_tools(list(TOOLS.values()), tool_choice="get_weather")
forced = model_forced.invoke("What is 2+2?")
print(f"Forced tool: {[tc['name'] for tc in forced.tool_calls]}")


# ════════════════════════════════════════════════════════════════════
# 5. MULTIMODAL MESSAGES — image URL and base64
# ════════════════════════════════════════════════════════════════════
section("5. MULTIMODAL MESSAGES")

IMAGE_URL = "https://picsum.photos/seed/langchain/600/400"
url_msg = HumanMessage(content=[
    {"type": "text", "text": "Describe the main colors and mood."},
    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
])
try:
    print("Image URL:", model.invoke([url_msg]).content[:150])
except Exception as e:
    print(f"Vision error: {e}")

# Base64 image (tiny 1×1 PNG demo)
b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
b64_msg = HumanMessage(content=[
    {"type": "text", "text": "What color is this image?"},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
])
try:
    print("Base64 image:", model.invoke([b64_msg]).content)
except Exception as e:
    print(f"Base64 error: {e}")
