"""
tools_overview.py — LangChain Tools: all key concepts in one file
Covers: basic tools, advanced schemas, runtime context, return values, dynamic selection
"""

import uuid
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

load_dotenv()

def section(title): print(f"\n{'─'*55}\n{title}\n{'─'*55}")


# ════════════════════════════════════════════════════════════════════
# 1. BASIC TOOLS — @tool, docstring, type hints, direct invoke
# ════════════════════════════════════════════════════════════════════
section("1. BASIC TOOLS")

@tool
def add_numbers(a: float, b: float) -> str:
    """Add two numbers and return the result.
    Args:
        a: First number
        b: Second number
    """
    return f"{a} + {b} = {a + b}"

@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time.
    Args:
        timezone: Timezone name (e.g. UTC, Asia/Kolkata)
    """
    from datetime import datetime, timezone as tz
    return f"Current time ({timezone}): {datetime.now(tz.utc).strftime('%Y-%m-%d %H:%M:%S')}"

@tool("web_search")  # custom name
def search(query: str) -> str:
    """Search the web for current information about any topic.
    Args:
        query: Search keywords (2-8 words)
    """
    return f"Web results for '{query}': [simulated]"

print(f"Tool: {add_numbers.name}  args: {add_numbers.args}")
print(f"Direct invoke: {add_numbers.invoke({'a': 42, 'b': 8})}")
print(f"Custom name: {search.name}  result: {search.invoke({'query': 'Python 3.12'})}")

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[add_numbers, get_current_time, search],
    system_prompt="Use available tools when needed.",
)
r = agent.invoke({"messages": [HumanMessage("What is 1234 × 5.67?")]})
print("Agent:", r["messages"][-1].content[:120])


# ════════════════════════════════════════════════════════════════════
# 2. ADVANCED SCHEMAS — Pydantic args_schema, Literal, Optional, nested
# ════════════════════════════════════════════════════════════════════
section("2. ADVANCED SCHEMAS")

class WeatherInput(BaseModel):
    location: str = Field(description="City name")
    units: Literal["celsius", "fahrenheit"] = Field(default="celsius", description="Temperature unit")
    include_forecast: bool = Field(default=False, description="Include 5-day forecast")

    @field_validator("location")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip(): raise ValueError("Location cannot be empty")
        return v.strip().title()

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get weather for a city. Pydantic schema with validation."""
    temp = 22 if units == "celsius" else 72
    result = f"{location}: {temp}°{'C' if units == 'celsius' else 'F'}, partly cloudy"
    if include_forecast:
        result += "\nForecast: Day1: 23°C, Day2: 19°C, Day3: 21°C"
    return result

print(get_weather.invoke({"location": "london", "units": "celsius", "include_forecast": True}))
try:
    get_weather.invoke({"location": "", "units": "celsius"})
except Exception as e:
    print(f"Validation caught: {type(e).__name__}")

# Nested Pydantic model
class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str   = Field(description="City name")
    country: str = Field(description="Country code e.g. US, IN")

class ShipmentInput(BaseModel):
    from_address: Address = Field(description="Sender address")
    to_address: Address   = Field(description="Recipient address")
    weight_kg: float      = Field(description="Weight in kg", gt=0, le=30)
    priority: Literal["standard", "express", "overnight"] = Field(default="standard")

@tool(args_schema=ShipmentInput)
def create_shipment(from_address: dict, to_address: dict, weight_kg: float, priority: str = "standard") -> str:
    """Create a shipping request between two addresses."""
    return f"Shipment: {from_address['city']} → {to_address['city']}  {weight_kg}kg  {priority}  ID: SHIP-{abs(hash(str(to_address)))%99999:05d}"

print(create_shipment.invoke({
    "from_address": {"street": "10 Baker St", "city": "London", "country": "UK"},
    "to_address":   {"street": "5 Wall St",   "city": "New York", "country": "US"},
    "weight_kg": 2.5, "priority": "express",
}))

# Optional fields + list
class SearchInput(BaseModel):
    query: str                         = Field(description="Search query")
    categories: Optional[List[str]]    = Field(default=None, description="Filter by categories")
    max_price: Optional[float]         = Field(default=None, description="Max price in USD")
    sort_by: Literal["relevance", "price_asc", "price_desc", "newest"] = Field(default="relevance")

@tool(args_schema=SearchInput)
def search_products(query: str, categories: Optional[List[str]] = None, max_price: Optional[float] = None, sort_by: str = "relevance") -> str:
    """Search product catalogue. Supports category and price filters."""
    filters = []
    if categories: filters.append(f"cat={categories}")
    if max_price:  filters.append(f"max=${max_price}")
    return f"'{query}' [{', '.join(filters)}] sorted by {sort_by}: 3 results"

print(search_products.invoke({"query": "headphones", "max_price": 150.0, "sort_by": "price_asc"}))


# ════════════════════════════════════════════════════════════════════
# 3. RUNTIME CONTEXT — state, context, store via ToolRuntime
# ════════════════════════════════════════════════════════════════════
section("3. RUNTIME CONTEXT")

@dataclass
class UserCtx:
    user_id: str
    username: str
    role: str = "viewer"

USER_DB = {
    "u-001": {"name": "Alice", "account_type": "Premium", "balance": 5000.0},
    "u-002": {"name": "Bob",   "account_type": "Standard", "balance": 1200.0},
}

@tool
def get_message_count(runtime: ToolRuntime) -> str:
    """Return number of messages in the current conversation."""
    return f"This conversation has {len(runtime.state['messages'])} messages."

@tool
def get_my_account(runtime: ToolRuntime[UserCtx]) -> str:
    """Get current user's account details."""
    u = USER_DB.get(runtime.context.user_id, {})
    return f"Account: {u.get('name')}  Type: {u.get('account_type')}  Balance: ${u.get('balance',0):,.2f}"

@tool
def perform_admin_task(task: str, runtime: ToolRuntime[UserCtx]) -> str:
    """Perform admin task (admin role required).
    Args:
        task: Task to perform
    """
    if runtime.context.role != "admin":
        return f"❌ Access denied for role '{runtime.context.role}'"
    return f"✅ Admin task '{task}' done by {runtime.context.username}"

ctx_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_message_count, get_my_account, perform_admin_task],
    context_schema=UserCtx,
    checkpointer=MemorySaver(),
    system_prompt="You are a personalised assistant.",
)

def ask_as(user: UserCtx, q: str) -> str:
    return ctx_agent.invoke(
        {"messages": [HumanMessage(q)]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
        context=user,
    )["messages"][-1].content

admin = UserCtx("u-001", "Alice", "admin")
viewer = UserCtx("u-002", "Bob",  "viewer")
print("Admin account:", ask_as(admin,  "Show my account details."))
print("Admin task:   ", ask_as(admin,  "Perform admin task: generate_report"))
print("Viewer task:  ", ask_as(viewer, "Perform admin task: delete_logs"))

# Store (long-term memory)
@tool
def save_preference(key: str, value: str, runtime: ToolRuntime[UserCtx]) -> str:
    """Save a user preference to persistent storage.
    Args:
        key: Preference key (e.g. language, theme)
        value: Preference value
    """
    prefs = (runtime.store.get(("prefs",), runtime.context.user_id) or type("", (), {"value": {}})()).value
    prefs[key] = value
    runtime.store.put(("prefs",), runtime.context.user_id, prefs)
    return f"Saved {key}={value}"

@tool
def get_preferences(runtime: ToolRuntime[UserCtx]) -> str:
    """Get all saved preferences for the current user."""
    stored = runtime.store.get(("prefs",), runtime.context.user_id)
    return str(stored.value if stored else "No preferences")

store = InMemoryStore()
store_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[save_preference, get_preferences],
    context_schema=UserCtx,
    checkpointer=MemorySaver(),
    store=store,
    system_prompt="You are a personal assistant with persistent memory.",
)

user = UserCtx("u-001", "Alice", "editor")
store_agent.invoke(
    {"messages": [HumanMessage("Set my language preference to Hindi.")]},
    config={"configurable": {"thread_id": str(uuid.uuid4())}},
    context=user,
)
r = store_agent.invoke(
    {"messages": [HumanMessage("What are my preferences?")]},
    config={"configurable": {"thread_id": str(uuid.uuid4())}},  # new thread, same store
    context=user,
)
print("Stored preference recall:", r["messages"][-1].content[:120])


# ════════════════════════════════════════════════════════════════════
# 4. RETURN VALUES — string, dict, Command (state mutation)
# ════════════════════════════════════════════════════════════════════
section("4. RETURN VALUES")

# Return string
@tool
def weather_text(city: str) -> str:
    """Get weather as plain text. Args: city: City name."""
    return {"london": "Cloudy 14°C", "tokyo": "Sunny 28°C"}.get(city.lower(), "No data")

# Return dict
@tool
def weather_dict(city: str) -> dict:
    """Get structured weather data. Args: city: City name."""
    return {"city": city.title(), "temp_c": 28, "condition": "sunny", "humidity_pct": 55}

print("String:", weather_text.invoke({"city": "tokyo"}))
print("Dict:  ", weather_dict.invoke({"city": "tokyo"}))

# Return Command (mutates agent state)
class SettingsState(AgentState):
    language: str = "English"
    theme: str    = "light"

@tool
def set_language(language: str, runtime: ToolRuntime) -> Command:
    """Set preferred response language. Args: language: e.g. English, Hindi."""
    return Command(update={
        "language": language,
        "messages": [ToolMessage(content=f"Language set to '{language}'.", tool_call_id=runtime.tool_call_id)],
    })

@tool
def get_settings(runtime: ToolRuntime) -> str:
    """Get current settings."""
    return f"language='{runtime.state.get('language','?')}' theme='{runtime.state.get('theme','?')}'"

settings_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[set_language, get_settings],
    checkpointer=MemorySaver(),
    state_schema=SettingsState,
    system_prompt="You are a settings assistant.",
)
cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
r = settings_agent.invoke({"messages": [HumanMessage("Set my language to Hindi.")]}, config=cfg)
print("Set lang:", r["messages"][-1].content[:80])
r = settings_agent.invoke({"messages": [HumanMessage("What are my current settings?")]}, config=cfg)
print("Get settings:", r["messages"][-1].content[:80])

# Error handling via wrap_tool_call
@tool
def risky_divide(numerator: float, denominator: float) -> str:
    """Divide two numbers. Args: numerator, denominator (cannot be zero)."""
    if denominator == 0: raise ZeroDivisionError("Cannot divide by zero!")
    return f"{numerator} / {denominator} = {numerator/denominator:.4f}"

@wrap_tool_call
def error_handler(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage]) -> ToolMessage:
    try:
        return handler(request)
    except ZeroDivisionError as e:
        return ToolMessage(content=f"Math error: {e}", tool_call_id=request.tool_call["id"])
    except Exception as e:
        return ToolMessage(content=f"Error: {e}", tool_call_id=request.tool_call["id"])

err_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[risky_divide],
    middleware=[error_handler],
    system_prompt="If a tool fails, explain the error.",
)
r = err_agent.invoke({"messages": [HumanMessage("Divide 100 by 0")]})
print("Div by zero handled:", r["messages"][-1].content[:100])


# ════════════════════════════════════════════════════════════════════
# 5. DYNAMIC TOOL SELECTION — filter tools at runtime via middleware
# ════════════════════════════════════════════════════════════════════
section("5. DYNAMIC TOOL SELECTION")

@tool
def read_public_docs(topic: str) -> str:
    """Read public documentation. Args: topic: Topic to look up."""
    return f"Public docs for '{topic}'"

@tool
def create_document(title: str, content: str) -> str:
    """Create a document. Args: title, content."""
    return f"Document created: '{title}'"

@tool
def delete_document(doc_id: str) -> str:
    """Delete a document (admin only). Args: doc_id: Document ID."""
    return f"Document {doc_id} deleted."

ALL_TOOLS = [read_public_docs, create_document, delete_document]
ROLE_TOOLS = {
    "viewer": {"read_public_docs"},
    "editor": {"read_public_docs", "create_document"},
    "admin":  {t.name for t in ALL_TOOLS},
}

@dataclass
class RoleCtx:
    role: str = "viewer"

@wrap_model_call
def role_filter(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    ctx = request.runtime.context if request.runtime else None
    if ctx and hasattr(ctx, "role"):
        allowed = ROLE_TOOLS.get(ctx.role, ROLE_TOOLS["viewer"])
        filtered = [t for t in request.tools if t.name in allowed]
        print(f"  [role={ctx.role}] {len(filtered)}/{len(request.tools)} tools exposed")
        request = request.override(tools=filtered)
    return handler(request)

role_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=ALL_TOOLS,
    middleware=[role_filter],
    context_schema=RoleCtx,
    checkpointer=MemorySaver(),
    system_prompt="You are a document assistant. Use only your available tools.",
)

def ask_role(role: str, q: str) -> str:
    return role_agent.invoke(
        {"messages": [HumanMessage(q)]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
        context=RoleCtx(role=role),
    )["messages"][-1].content

print("Admin delete:", ask_role("admin",  "Delete document DOC-001."))
print("Editor create:", ask_role("editor", "Create doc 'Meeting Notes' with content 'Discussed Q3 goals.'"))
print("Viewer delete:", ask_role("viewer", "Delete document DOC-001."))  # should fail gracefully
