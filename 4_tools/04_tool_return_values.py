"""
04_tool_return_values.py
========================
Demonstrates the THREE types of tool return values and error handling.

Concepts covered:
  - Return string      — plain text result the model reads
  - Return dict/object — structured data the model inspects
  - Return Command     — update agent STATE + optionally show a ToolMessage
  - runtime.tool_call_id — required when using Command with ToolMessage
  - Error handling with wrap_tool_call middleware
  - Custom error messages back to the model

Tool return value decision guide:
  • String  → result is readable text ("London: Rainy, 14°C")
  • Dict    → result is structured data ({"city": "London", "temp": 14})
  • Command → tool needs to MUTATE agent state fields
"""

import os
import uuid
from dotenv import load_dotenv

from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from typing import Callable

load_dotenv()

print("=" * 60)
print("Tool Return Values & Error Handling Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. RETURN STRING — plain text for the model to read
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Return string ──────────────────────────────────────")


@tool
def get_weather_text(city: str) -> str:
    """Get current weather for a city as a text description.

    Args:
        city: City name
    """
    data = {
        "london": "☁️  Cloudy, 14°C, humidity 80%",
        "tokyo":  "☀️  Sunny, 28°C, humidity 55%",
        "mumbai": "🌧️  Rainy, 30°C, humidity 90%",
    }
    return data.get(city.lower(), f"No weather data available for '{city}'.")


result = get_weather_text.invoke({"city": "tokyo"})
print(f"\n  String return:  {result!r}")
print(f"  Type:           {type(result).__name__}")
print(f"  Model reads:    natural language — easy to incorporate in a reply")


# ════════════════════════════════════════════════════════════════════
# 2. RETURN DICT — structured data
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Return dict (structured data) ─────────────────────")


@tool
def get_weather_data(city: str) -> dict:
    """Get structured weather data for a city with multiple fields.

    Returns a structured object the model can reason over field-by-field.

    Args:
        city: City name
    """
    return {
        "city":          city.title(),
        "temperature_c": 28,
        "temperature_f": 82,
        "condition":     "sunny",
        "humidity_pct":  55,
        "uv_index":      7,
        "wind_kmh":      15,
    }


result = get_weather_data.invoke({"city": "tokyo"})
print(f"\n  Dict return: {result}")
print(f"  Type:        {type(result).__name__}")
print(f"  Model reads: structured fields — can reference specific values")


@tool
def calculate_shipping_cost(
    weight_kg: float,
    distance_km: float,
    priority: str = "standard",
) -> dict:
    """Calculate the shipping cost for a package.

    Args:
        weight_kg:   Package weight in kilograms
        distance_km: Shipping distance in kilometers
        priority:    'standard', 'express', or 'overnight'
    """
    base_rate = {"standard": 0.05, "express": 0.12, "overnight": 0.25}
    rate      = base_rate.get(priority, 0.05)
    cost      = weight_kg * distance_km * rate
    eta_days  = {"standard": 5, "express": 2, "overnight": 1}
    return {
        "cost_usd":      round(cost, 2),
        "currency":      "USD",
        "priority":      priority,
        "eta_days":      eta_days.get(priority, 5),
        "weight_kg":     weight_kg,
        "distance_km":   distance_km,
    }


result = calculate_shipping_cost.invoke({
    "weight_kg": 3.5, "distance_km": 500, "priority": "express"
})
print(f"\n  Shipping cost result: {result}")


# ════════════════════════════════════════════════════════════════════
# 3. RETURN Command — mutate agent state
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Return Command (update agent state) ────────────────")

# Command is used when the tool needs to WRITE to the agent's state graph.
# This is useful for:
#   • Setting user preferences (language, theme)
#   • Tracking counters or flags
#   • Updating custom state fields

# First define a custom AgentState with extra fields
class PreferencesState(AgentState):
    preferred_language:  str
    theme:               str
    notification_email:  str


@tool
def set_language(language: str, runtime: ToolRuntime) -> Command:
    """Set the user's preferred response language.

    Args:
        language: Language name (e.g. 'English', 'Hindi', 'French')
    """
    return Command(
        update={
            "preferred_language": language,       # ← writes to agent state
            "messages": [
                ToolMessage(
                    content=f"Language preference set to '{language}'.",
                    tool_call_id=runtime.tool_call_id,  # ← links back to the tool call
                )
            ],
        }
    )


@tool
def set_theme(theme: str, runtime: ToolRuntime) -> Command:
    """Set the user interface theme.

    Args:
        theme: 'light' or 'dark'
    """
    if theme not in ("light", "dark"):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Invalid theme '{theme}'. Use 'light' or 'dark'.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    return Command(
        update={
            "theme": theme,
            "messages": [
                ToolMessage(
                    content=f"Theme set to '{theme}'.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def get_current_settings(runtime: ToolRuntime) -> str:
    """Get the current user settings.

    No input needed — reads from agent state.
    """
    state = runtime.state
    lang  = state.get("preferred_language", "not set")
    theme = state.get("theme", "not set")
    return f"Current settings: language='{lang}', theme='{theme}'"


settings_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[set_language, set_theme, get_current_settings],
    checkpointer=MemorySaver(),
    state_schema=PreferencesState,
    system_prompt="You are a settings assistant. Help users configure their preferences.",
)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

r = settings_agent.invoke(
    {"messages": [HumanMessage("Set my language to Hindi and theme to dark.")]},
    config=config,
)
print(f"\n  Setting prefs:   {r['messages'][-1].content}")

r = settings_agent.invoke(
    {"messages": [HumanMessage("What are my current settings?")]},
    config=config,
)
print(f"  Recall prefs:    {r['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 4. ERROR HANDLING — wrap_tool_call middleware
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Error handling with wrap_tool_call ────────────────")


@tool
def risky_divide(numerator: float, denominator: float) -> str:
    """Divide two numbers. Will fail if denominator is zero.

    Args:
        numerator:   The number to divide
        denominator: The number to divide by (cannot be zero)
    """
    if denominator == 0:
        raise ZeroDivisionError("Cannot divide by zero!")
    return f"{numerator} ÷ {denominator} = {numerator / denominator:.4f}"


@tool
def flaky_api_call(endpoint: str) -> str:
    """Call an external API endpoint.

    Args:
        endpoint: API endpoint path (e.g. '/users', '/orders')
    """
    import random
    if random.random() < 0.5:
        raise ConnectionError(f"Network timeout calling {endpoint}")
    return f"API response from {endpoint}: {{status: 200, data: [...]}}"


# Custom error handler using wrap_tool_call
@wrap_tool_call
def graceful_error_handler(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    """Convert ANY tool exception into a graceful ToolMessage.

    Without this, an unhandled exception would crash the agent.
    With this, the error is converted to a message the model can read
    and decide how to proceed (retry, ask user, skip, etc.).
    """
    try:
        return handler(request)
    except ZeroDivisionError as e:
        return ToolMessage(
            content=f"Math error: {e}. Please use a non-zero denominator.",
            tool_call_id=request.tool_call["id"],
        )
    except ConnectionError as e:
        return ToolMessage(
            content=f"Network error: {e}. The API may be temporarily unavailable.",
            tool_call_id=request.tool_call["id"],
        )
    except Exception as e:
        return ToolMessage(
            content=f"Tool error ({type(e).__name__}): {e}. Please check your input and try again.",
            tool_call_id=request.tool_call["id"],
        )


error_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[risky_divide, flaky_api_call],
    middleware=[graceful_error_handler],      # ← error handling middleware
    system_prompt=(
        "You are a helpful assistant. If a tool fails, explain the error "
        "to the user and suggest how to fix it."
    ),
)

# Test: division by zero
r = error_agent.invoke({
    "messages": [HumanMessage("Divide 100 by 0")]
})
print(f"\n  Division by zero → {r['messages'][-1].content}")

# Test: normal division
r = error_agent.invoke({
    "messages": [HumanMessage("Divide 144 by 12")]
})
print(f"  Normal division  → {r['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 5. RETURN TYPE SUMMARY
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Return type summary ────────────────────────────────")
print("""
  ┌─────────────┬──────────────────────────────────────────────────┐
  │ Return type │ When to use                                      │
  ├─────────────┼──────────────────────────────────────────────────┤
  │ str         │ Natural language result → model incorporates      │
  │             │ directly into its reply.                         │
  ├─────────────┼──────────────────────────────────────────────────┤
  │ dict/obj    │ Structured data → model reasons over fields.      │
  │             │ Good for records with multiple named values.     │
  ├─────────────┼──────────────────────────────────────────────────┤
  │ Command     │ Tool must mutate agent state fields.              │
  │             │ Include ToolMessage so model sees the result.    │
  │             │ Always use runtime.tool_call_id.                 │
  └─────────────┴──────────────────────────────────────────────────┘
""")
