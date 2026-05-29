"""
02_advanced_schemas.py
======================
Demonstrates ADVANCED INPUT SCHEMA definitions for tools.

Concepts covered:
  - Pydantic BaseModel as args_schema    — rich validation + descriptions
  - Literal types                        — constrain choices
  - Optional fields with defaults        — flexible inputs
  - Nested Pydantic models               — complex structured inputs
  - JSON Schema dict as args_schema      — raw schema definition
  - Field() with descriptions            — guide the model on what to pass
  - Validating inputs before execution   — Pydantic auto-validates

When a tool has complex or constrained inputs, define a Pydantic schema
with args_schema=. This gives the model precise guidance on what to pass,
and automatically validates/rejects bad inputs before your function runs.
"""

import os
from typing import Literal, Optional, List
from dotenv import load_dotenv

from langchain.tools import tool
from langchain.agents import create_agent
from pydantic import BaseModel, Field, field_validator

load_dotenv()

print("=" * 60)
print("Advanced Schema Tools Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. PYDANTIC BaseModel AS args_schema
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Pydantic BaseModel schema ──────────────────────────")


class WeatherInput(BaseModel):
    """Input schema for weather queries."""
    location: str = Field(
        description="City name or 'City, Country' format (e.g. 'London' or 'Paris, FR')"
    )
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit: 'celsius' or 'fahrenheit'"
    )
    include_forecast: bool = Field(
        default=False,
        description="Set True to include a 5-day weather forecast"
    )

    @field_validator("location")
    @classmethod
    def location_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Location cannot be empty")
        return v.strip().title()


@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and an optional 5-day forecast for any city.

    Use this when the user asks about weather conditions, temperature,
    or wants a weather forecast for a specific location.
    """
    temp = 22 if units == "celsius" else 72
    symbol = "°C" if units == "celsius" else "°F"
    result = f"Weather in {location}: {temp}{symbol}, partly cloudy"
    if include_forecast:
        result += (
            "\n5-day forecast:\n"
            "  Day 1: 23°C sunny\n"
            "  Day 2: 19°C rainy\n"
            "  Day 3: 21°C cloudy\n"
            "  Day 4: 25°C sunny\n"
            "  Day 5: 24°C sunny"
        )
    return result


# The model now sees the rich Pydantic schema including field descriptions
print(f"\n  Tool: {get_weather.name}")
print(f"  Schema fields: {list(get_weather.args.keys())}")

# Direct invocation — Pydantic validates inputs
result = get_weather.invoke({
    "location": "london",
    "units": "celsius",
    "include_forecast": True,
})
print(f"\n  get_weather(london, celsius, forecast=True):\n    {result}")

# Validation error example
print("\n  Testing validation:")
try:
    get_weather.invoke({"location": "", "units": "celsius"})
except Exception as e:
    print(f"  ❌ Empty location rejected: {type(e).__name__}")


# ════════════════════════════════════════════════════════════════════
# 2. NESTED PYDANTIC MODELS
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Nested Pydantic models ─────────────────────────────")


class Address(BaseModel):
    """Physical address."""
    street: str = Field(description="Street address including house number")
    city:   str = Field(description="City name")
    country: str = Field(description="Country code (e.g. 'US', 'IN', 'UK')")


class ShipmentInput(BaseModel):
    """Input for creating a shipment."""
    from_address: Address = Field(description="Sender's address")
    to_address:   Address = Field(description="Recipient's address")
    weight_kg:    float   = Field(description="Package weight in kilograms", gt=0, le=30)
    priority:     Literal["standard", "express", "overnight"] = Field(
        default="standard",
        description="Shipping priority level"
    )
    fragile:      bool = Field(default=False, description="Mark package as fragile")


@tool(args_schema=ShipmentInput)
def create_shipment(
    from_address: dict,
    to_address: dict,
    weight_kg: float,
    priority: str = "standard",
    fragile: bool = False,
) -> str:
    """Create a new shipping request between two addresses.

    Use this when the user wants to ship a package or parcel.
    Requires both origin and destination addresses.
    """
    return (
        f"📦 Shipment created:\n"
        f"  From:     {from_address.get('city')}, {from_address.get('country')}\n"
        f"  To:       {to_address.get('city')}, {to_address.get('country')}\n"
        f"  Weight:   {weight_kg} kg\n"
        f"  Priority: {priority}\n"
        f"  Fragile:  {'Yes ⚠️' if fragile else 'No'}\n"
        f"  Tracking: SHIP-{abs(hash(str(to_address)))%100000:05d}"
    )


result = create_shipment.invoke({
    "from_address": {"street": "10 Baker St", "city": "London",   "country": "UK"},
    "to_address":   {"street": "5 Wall St",   "city": "New York", "country": "US"},
    "weight_kg": 2.5,
    "priority":  "express",
    "fragile":   True,
})
print(f"\n  {result}")


# ════════════════════════════════════════════════════════════════════
# 3. OPTIONAL FIELDS AND LISTS
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Optional fields and lists ──────────────────────────")


class SearchInput(BaseModel):
    """Input for database search."""
    query:      str                    = Field(description="Search query text")
    categories: Optional[List[str]]   = Field(
        default=None,
        description="Filter by categories (e.g. ['electronics', 'clothing']). None = search all."
    )
    max_price:  Optional[float]        = Field(
        default=None,
        description="Maximum price filter in USD. None = no price limit."
    )
    sort_by:    Literal["relevance", "price_asc", "price_desc", "newest"] = Field(
        default="relevance",
        description="How to sort results"
    )
    page:       int = Field(default=1, ge=1, description="Page number (starts at 1)")


@tool(args_schema=SearchInput)
def search_products(
    query: str,
    categories: Optional[List[str]] = None,
    max_price: Optional[float] = None,
    sort_by: str = "relevance",
    page: int = 1,
) -> str:
    """Search the product catalogue for items matching your query.

    Use this when the user wants to find, browse, or compare products.
    Supports filtering by category, price, and sorting options.
    """
    filters = []
    if categories:
        filters.append(f"categories={categories}")
    if max_price:
        filters.append(f"max_price=${max_price}")

    filter_str = f" [{', '.join(filters)}]" if filters else ""
    return (
        f"Search: '{query}'{filter_str} sorted by {sort_by}, page {page}\n"
        f"Results: 3 items found (simulated)"
    )


# Call with minimal args (all optional fields use defaults)
r1 = search_products.invoke({"query": "wireless headphones"})
print(f"\n  Minimal: {r1}")

# Call with full args
r2 = search_products.invoke({
    "query": "wireless headphones",
    "categories": ["electronics", "audio"],
    "max_price": 200.0,
    "sort_by": "price_asc",
    "page": 2,
})
print(f"\n  Full:    {r2}")


# ════════════════════════════════════════════════════════════════════
# 4. JSON SCHEMA (dict) AS args_schema
# ════════════════════════════════════════════════════════════════════

print("\n── 4. JSON Schema dict as args_schema ───────────────────")

# Use when you need interoperability with systems that speak JSON Schema,
# or when you don't want to import Pydantic.

weather_json_schema = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "City name or coordinates"
        },
        "units": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature unit preference",
            "default": "celsius"
        },
        "include_forecast": {
            "type": "boolean",
            "description": "Include 5-day forecast",
            "default": False
        },
    },
    "required": ["location"],   # only location is required
}


@tool(args_schema=weather_json_schema)
def get_weather_json(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get weather using a JSON Schema input definition.

    Args:
        location:         City name or coordinates
        units:            'celsius' or 'fahrenheit'
        include_forecast: Whether to include 5-day forecast
    """
    temp = 22 if units == "celsius" else 72
    return f"Weather in {location}: {temp}°{'C' if units == 'celsius' else 'F'}, partly cloudy"


result = get_weather_json.invoke({"location": "Tokyo", "units": "celsius"})
print(f"\n  JSON Schema tool result: {result}")
print(f"  Schema type:             {type(get_weather_json.args_schema).__name__}")


# ════════════════════════════════════════════════════════════════════
# 5. USE WITH AGENT — richer schema = better model decisions
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Agent with rich schema tools ──────────────────────")

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, search_products],
    system_prompt=(
        "You are a shopping and travel assistant. "
        "Use tools for weather and product searches. Be concise."
    ),
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "What's the weather forecast for Paris in Fahrenheit? Also find wireless headphones under $150."
    }]
})
print(f"\n  🧑 User: What's the weather forecast for Paris in Fahrenheit? Also find wireless headphones under $150.")
print(f"  🤖 Agent: {result['messages'][-1].content}")
