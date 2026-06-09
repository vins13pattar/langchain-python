"""
agents_overview.py — LangChain Agents: all key concepts in one file
Covers: basic agent, memory, structured output, streaming, middleware, context/runtime
"""

import uuid
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    PIIMiddleware,
)
from langchain.agents.runtime import get_runtime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

def section(title): print(f"\n{'─'*55}\n{title}\n{'─'*55}")


# ════════════════════════════════════════════════════════════════════
# 1. BASIC AGENT — create_agent + @tool + invoke
# ════════════════════════════════════════════════════════════════════
section("1. BASIC AGENT")

@tool
def calculator(operation: str, num1: float, num2: float) -> str:
    """Perform basic arithmetic: add, subtract, multiply, divide."""
    ops = {"add": num1+num2, "subtract": num1-num2, "multiply": num1*num2,
           "divide": num1/num2 if num2 != 0 else "Error: div by zero"}
    return f"{num1} {operation} {num2} = {ops.get(operation, 'unknown op')}"

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return {"london": "Cloudy 15°C", "tokyo": "Sunny 28°C"}.get(city.lower(), f"No data for {city}")

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[calculator, get_weather],
    system_prompt="You are a helpful assistant. Use tools when needed.",
)

def ask(q): return agent.invoke({"messages": [{"role": "user", "content": q}]})["messages"][-1].content

print("Q:", "What is 1234 × 56?")
print("A:", ask("What is 1234 multiplied by 56?"))
print("Q:", "Weather in Tokyo?")
print("A:", ask("What's the weather in Tokyo?"))


# ════════════════════════════════════════════════════════════════════
# 2. MEMORY — MemorySaver + thread_id for multi-turn conversations
# ════════════════════════════════════════════════════════════════════
section("2. MEMORY (multi-turn)")

@tool
def get_date() -> str:
    """Return today's date."""
    from datetime import date
    return str(date.today())

memory_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_date],
    checkpointer=MemorySaver(),
    system_prompt="You are a personal assistant. Remember everything the user tells you.",
)

THREAD = str(uuid.uuid4())
cfg = {"configurable": {"thread_id": THREAD}}

def chat(msg): return memory_agent.invoke({"messages": [{"role": "user", "content": msg}]}, config=cfg)["messages"][-1].content

print(chat("My name is Vinod and I love Python."))
print(chat("What's my name?"))        # agent should remember
print(chat("What do I love?"))        # agent should remember

# New thread = fresh memory
new_cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
fresh = memory_agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config=new_cfg)
print("Fresh thread (no memory):", fresh["messages"][-1].content)


# ════════════════════════════════════════════════════════════════════
# 3. STRUCTURED OUTPUT — response_format= Pydantic schema
# ════════════════════════════════════════════════════════════════════
section("3. STRUCTURED OUTPUT")

class WeatherReport(BaseModel):
    city: str                = Field(description="City name")
    temperature_celsius: float = Field(description="Temperature in °C")
    condition: str           = Field(description="Weather condition")
    recommendation: str      = Field(description="What to wear/bring")

@tool
def fetch_weather(city: str) -> str:
    """Fetch raw weather data for a city."""
    return {"tokyo": "temp=28, condition=sunny", "london": "temp=15, condition=cloudy"}.get(city.lower(), "No data")

weather_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[fetch_weather],
    response_format=WeatherReport,
    system_prompt="Fetch weather and fill the WeatherReport schema.",
)

result = weather_agent.invoke({"messages": [{"role": "user", "content": "Weather in Tokyo?"}]})
report: WeatherReport = result["structured_response"]
print(f"City: {report.city}  Temp: {report.temperature_celsius}°C  Cond: {report.condition}")
print(f"Tip: {report.recommendation}")


# ════════════════════════════════════════════════════════════════════
# 4. STREAMING — agent.stream() with stream_mode="values"
# ════════════════════════════════════════════════════════════════════
section("4. STREAMING")

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for '{query}': [LangChain news, AI updates, coding tips]"

stream_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_web],
    system_prompt="You are a research assistant. Search the web when needed.",
)

print("Streaming agent steps:")
for chunk in stream_agent.stream(
    {"messages": [{"role": "user", "content": "Latest AI news?"}]},
    stream_mode="values",
):
    latest = chunk["messages"][-1]
    if isinstance(latest, AIMessage) and latest.content:
        print(f"  Agent: {latest.content[:120]}")
    elif isinstance(latest, AIMessage) and latest.tool_calls:
        print(f"  Tool calls: {[tc['name'] for tc in latest.tool_calls]}")
    elif isinstance(latest, ToolMessage):
        print(f"  Tool result [{latest.name}]: {latest.content[:80]}")


# ════════════════════════════════════════════════════════════════════
# 5. MIDDLEWARE — retry, PII, HITL
# ════════════════════════════════════════════════════════════════════
section("5. MIDDLEWARE")

@tool
def read_file(path: str) -> str:
    """Read file contents."""
    return f"[Contents of {path}]: sample content"

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file (destructive)."""
    return f"Written {len(content)} bytes to {path}"

# Fault tolerance: retry on model/tool errors
fault_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[read_file],
    middleware=[ModelRetryMiddleware(max_retries=3), ToolRetryMiddleware(max_retries=2)],
    system_prompt="You are a file assistant.",
)
r = fault_agent.invoke({"messages": [{"role": "user", "content": "Read /docs/readme.txt"}]})
print("Fault-tolerant:", r["messages"][-1].content[:100])

# PII scrubbing
pii_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[PIIMiddleware()],
    system_prompt="Summarise text. Never repeat raw personal data.",
)
r = pii_agent.invoke({"messages": [{"role": "user", "content": "Process: Name=John, SSN=123-45-6789, Phone=555-867-5309"}]})
print("PII scrubbed response:", r["messages"][-1].content[:120])

# HITL — agent pauses before destructive action (skip interactive input in demo)
hitl_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[read_file, write_file],
    checkpointer=MemorySaver(),
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
    system_prompt="You are a file assistant.",
)
hitl_cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
try:
    r = hitl_agent.invoke({"messages": [{"role": "user", "content": "Write a summary to /reports/out.txt"}]}, config=hitl_cfg)
    print("HITL result:", r["messages"][-1].content[:100])
except Exception as interrupt:
    print(f"HITL paused — agent is waiting for human approval")
    # Resume with approval:
    r = hitl_agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=hitl_cfg)
    print("Resumed:", r["messages"][-1].content[:100])


# ════════════════════════════════════════════════════════════════════
# 6. CONTEXT & RUNTIME — per-run data via context_schema
# ════════════════════════════════════════════════════════════════════
section("6. CONTEXT & RUNTIME")

@dataclass
class UserContext:
    user_id: str
    username: str
    role: str = "viewer"   # admin | editor | viewer

@tool
def get_user_profile() -> str:
    """Return the current user's profile."""
    ctx: UserContext = get_runtime().context
    return f"User: {ctx.username}  ID: {ctx.user_id}  Role: {ctx.role}"

@tool
def admin_action(action: str) -> str:
    """Perform an admin action (admin role required).
    Args:
        action: Action to perform
    """
    ctx: UserContext = get_runtime().context
    if ctx.role != "admin":
        return f"❌ Access denied for role '{ctx.role}'"
    return f"✅ Admin action '{action}' done by {ctx.username}"

ctx_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_user_profile, admin_action],
    context_schema=UserContext,
    checkpointer=MemorySaver(),
    system_prompt="You are a personalised assistant. Greet users by name.",
)

def run_as(user: UserContext, question: str) -> str:
    return ctx_agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
        context=user,
    )["messages"][-1].content

admin = UserContext("u-001", "Vinod", role="admin")
viewer = UserContext("u-002", "Priya", role="viewer")

print("Admin profile:", run_as(admin, "Show my profile."))
print("Admin action: ", run_as(admin, "Perform admin action: reset_cache"))
print("Viewer action:", run_as(viewer, "Perform admin action: delete_logs"))  # denied
