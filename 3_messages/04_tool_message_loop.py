"""
04_tool_message_loop.py
========================
Demonstrates the complete TOOL CALLING LOOP using messages manually.

Concepts covered:
  - bind_tools()          — attach tools to a model
  - AIMessage.tool_calls  — list of tool call requests from the model
  - ToolMessage           — passes tool result back to the model
  - tool_call_id          — links ToolMessage to the correct AIMessage tool call
  - artifact field        — store extra data NOT sent to model
  - Manual vs agent loop  — when to use each
  - Parallel tool calls   — model requests multiple tools at once
  - Forcing tool use      — tool_choice="any" / "specific_tool"

The tool calling LOOP (without an agent):
  1. User message → model
  2. Model → AIMessage with tool_calls
  3. Execute each tool → ToolMessage (per call)
  4. Append messages → model again
  5. Model uses results → final AIMessage
  (Repeat steps 2-4 until model stops calling tools)

With create_agent(), this loop is handled automatically.
"""

import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import tool

load_dotenv()

model = init_chat_model("openai:gpt-4o-mini")

print("=" * 60)
print("Tool Message Loop Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. DEFINE TOOLS
# ════════════════════════════════════════════════════════════════════

@tool
def get_weather(location: str) -> str:
    """Get current weather for a city.

    Args:
        location: City name (e.g. 'London', 'Tokyo')
    """
    data = {
        "london":  "🌧️  Rainy, 14°C, humidity 85%",
        "paris":   "☀️  Sunny, 23°C, humidity 50%",
        "tokyo":   "⛅  Partly cloudy, 19°C, humidity 65%",
        "new york":"🌩️  Thunderstorm, 17°C, humidity 90%",
        "sydney":  "🌤️  Mostly sunny, 26°C, humidity 40%",
    }
    return data.get(location.lower(), f"Weather data unavailable for '{location}'")


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies.

    Args:
        amount:        Amount to convert
        from_currency: Source currency code (e.g. USD, EUR, GBP)
        to_currency:   Target currency code (e.g. USD, EUR, GBP)
    """
    # Simplified fixed rates relative to USD
    rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 149.5, "INR": 83.2}
    if from_currency.upper() not in rates or to_currency.upper() not in rates:
        return f"Unsupported currency pair: {from_currency} → {to_currency}"
    usd = amount / rates[from_currency.upper()]
    result = usd * rates[to_currency.upper()]
    return f"{amount} {from_currency.upper()} = {result:.2f} {to_currency.upper()}"


@tool
def search_books(query: str, max_results: int = 3) -> str:
    """Search for books by topic, author, or title.

    Args:
        query:       Search query
        max_results: Maximum number of results to return (default: 3)
    """
    # Simulated book database
    books = [
        "Clean Code by Robert Martin",
        "The Pragmatic Programmer by David Thomas",
        "Python Crash Course by Eric Matthes",
        "Design Patterns by Gang of Four",
        "Fluent Python by Luciano Ramalho",
    ]
    # Retrieve metadata separately (not sent to model)
    metadata = {"source": "library_db", "total_results": len(books), "query": query}

    results = books[:max_results]
    return ToolMessage(
        content=f"Books matching '{query}': {', '.join(results)}",
        tool_call_id="",                 # filled in below
        name="search_books",
        artifact=metadata,               # ← NOT sent to model; available downstream
    )


# ════════════════════════════════════════════════════════════════════
# 2. SINGLE TOOL CALL LOOP
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Single tool call loop ──────────────────────────────")

model_with_tools = model.bind_tools([get_weather, convert_currency, search_books])

messages = [
    SystemMessage("You are a helpful assistant. Use tools to answer questions."),
    HumanMessage("What's the weather like in Tokyo?"),
]

# Step 1: model requests tool call
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

print(f"\nStep 1 — Model tool calls:")
for tc in ai_msg.tool_calls:
    print(f"  🔧 {tc['name']}({tc['args']})  id={tc['id']}")

# Step 2: execute each tool and create ToolMessages
tool_map = {
    "get_weather":      get_weather,
    "convert_currency": convert_currency,
    "search_books":     search_books,
}

for tool_call in ai_msg.tool_calls:
    result = tool_map[tool_call["name"]].invoke(tool_call)
    messages.append(result)
    print(f"  📦 Tool result: {result.content}")

# Step 3: pass results back to model
final = model_with_tools.invoke(messages)
print(f"\nStep 3 — Final answer: {final.content}")


# ════════════════════════════════════════════════════════════════════
# 3. PARALLEL TOOL CALLS
#    Model calls multiple tools in one shot
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Parallel tool calls ────────────────────────────────")

messages_2 = [
    HumanMessage(
        "What's the weather in London and Paris? "
        "Also convert 100 USD to EUR."
    )
]

ai_msg_2 = model_with_tools.invoke(messages_2)
messages_2.append(ai_msg_2)

print(f"\nModel made {len(ai_msg_2.tool_calls)} tool calls in parallel:")
for tc in ai_msg_2.tool_calls:
    print(f"  🔧 {tc['name']}({tc['args']})")

# Execute ALL tool calls (could be done concurrently with asyncio)
for tool_call in ai_msg_2.tool_calls:
    result = tool_map[tool_call["name"]].invoke(tool_call)
    messages_2.append(result)
    print(f"  📦 [{tool_call['name']}] → {result.content}")

final_2 = model_with_tools.invoke(messages_2)
print(f"\nFinal answer: {final_2.content}")


# ════════════════════════════════════════════════════════════════════
# 4. TOOL MESSAGE ARTIFACT
#    Store extra metadata NOT sent to the model
# ════════════════════════════════════════════════════════════════════

print("\n── 4. ToolMessage artifact ───────────────────────────────")

messages_3 = [HumanMessage("Find me books about Python programming.")]
ai_msg_3 = model_with_tools.invoke(messages_3)
messages_3.append(ai_msg_3)

for tc in ai_msg_3.tool_calls:
    # search_books returns a ToolMessage directly
    tool_msg = search_books.invoke(tc)
    # Fix the tool_call_id (our tool returned one with empty id)
    tool_msg.tool_call_id = tc["id"]
    messages_3.append(tool_msg)

    print(f"\n  Content (sent to model): {tool_msg.content}")
    print(f"  Artifact (NOT sent):     {tool_msg.artifact}")
    print(f"  → You can use artifact for UI rendering, logging, etc.")

final_3 = model_with_tools.invoke(messages_3)
print(f"\nFinal answer: {final_3.content}")


# ════════════════════════════════════════════════════════════════════
# 5. FORCING TOOL CHOICE
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Forcing tool choice ────────────────────────────────")

# Force model to ALWAYS call a tool (any tool)
model_any = model.bind_tools([get_weather, convert_currency], tool_choice="any")
response_any = model_any.invoke("Hello!")
print(f"\n  tool_choice='any' — tool calls even for 'Hello!':")
for tc in response_any.tool_calls:
    print(f"    🔧 {tc['name']}({tc['args']})")

# Force a SPECIFIC tool
model_forced = model.bind_tools([get_weather, convert_currency], tool_choice="get_weather")
response_forced = model_forced.invoke("What is 2+2?")
print(f"\n  tool_choice='get_weather' — always calls get_weather:")
for tc in response_forced.tool_calls:
    print(f"    🔧 {tc['name']}({tc['args']})")


# ════════════════════════════════════════════════════════════════════
# 6. FULL AGENTIC LOOP (without create_agent)
#    Shows what create_agent() does automatically
# ════════════════════════════════════════════════════════════════════

print("\n── 6. Manual agentic loop (what create_agent does) ──────")

def run_agent_manually(question: str, max_iterations: int = 5) -> str:
    """Manually run the tool-calling loop until the model stops calling tools."""
    messages = [
        SystemMessage("Answer using tools when needed. Be concise."),
        HumanMessage(question),
    ]

    for iteration in range(max_iterations):
        response = model_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # Model is done — no more tool calls → final answer
            return response.content

        # Execute all tool calls
        print(f"  Iteration {iteration + 1}: {len(response.tool_calls)} tool call(s)")
        for tc in response.tool_calls:
            result = tool_map[tc["name"]].invoke(tc)
            messages.append(result)

    return "Max iterations reached"


answer = run_agent_manually(
    "What's the weather in Sydney and also convert 500 GBP to USD?"
)
print(f"\nFinal: {answer}")
