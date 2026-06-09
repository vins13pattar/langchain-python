"""
short_term_memory_overview.py — LangChain Short-Term Memory: all key concepts in one file
Covers: checkpointer basics, custom state, trim/delete messages, summarization, dynamic prompt
"""

import uuid
from typing import Any, TypedDict
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    before_model, after_model, SummarizationMiddleware, dynamic_prompt, ModelRequest
)
from langchain_core.messages import RemoveMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command

load_dotenv()

def section(title): print(f"\n{'─'*55}\n{title}\n{'─'*55}")


# ════════════════════════════════════════════════════════════════════
# 1. CHECKPOINTER BASICS — MemorySaver + thread_id
# ════════════════════════════════════════════════════════════════════
section("1. CHECKPOINTER BASICS")

@tool
def get_user_profile(user_id: str) -> str:
    """Fetch a user profile by ID. Args: user_id: User's unique ID."""
    return {"u001": "Alice, Engineer, London"}.get(user_id, "Not found")

# Without checkpointer — stateless
stateless = create_agent(model="openai:gpt-4o-mini", tools=[], system_prompt="You are a helpful assistant.")
r1 = stateless.invoke({"messages": [{"role": "user", "content": "Hi! I'm Vinod."}]})
r2 = stateless.invoke({"messages": [{"role": "user", "content": "What's my name?"}]})
print("No checkpointer T1:", r1["messages"][-1].content[:80])
print("No checkpointer T2:", r2["messages"][-1].content[:80], "(no memory)")

# With MemorySaver — remembers within thread
memory_agent = create_agent(
    model="openai:gpt-4o-mini", tools=[get_user_profile],
    checkpointer=MemorySaver(),
    system_prompt="Remember everything the user tells you.",
)

cfg = {"configurable": {"thread_id": "session-vinod"}}
memory_agent.invoke({"messages": [{"role": "user", "content": "Hi! I'm Vinod from Bengaluru."}]}, cfg)
r = memory_agent.invoke({"messages": [{"role": "user", "content": "What's my name and city?"}]}, cfg)
print("With checkpointer:", r["messages"][-1].content[:100])

# Multiple isolated threads
alice_cfg = {"configurable": {"thread_id": "alice"}}
bob_cfg   = {"configurable": {"thread_id": "bob"}}
memory_agent.invoke({"messages": [{"role": "user", "content": "I'm Alice, a software engineer."}]}, alice_cfg)
memory_agent.invoke({"messages": [{"role": "user", "content": "I'm Bob, a graphic designer."}]}, bob_cfg)
ra = memory_agent.invoke({"messages": [{"role": "user", "content": "What's my name and job?"}]}, alice_cfg)
rb = memory_agent.invoke({"messages": [{"role": "user", "content": "What's my name and job?"}]}, bob_cfg)
print("Alice thread:", ra["messages"][-1].content[:80])
print("Bob thread:  ", rb["messages"][-1].content[:80])


# ════════════════════════════════════════════════════════════════════
# 2. CUSTOM STATE — extend AgentState with your own fields
# ════════════════════════════════════════════════════════════════════
section("2. CUSTOM STATE")

from langchain.tools import ToolRuntime

class UserAgentState(AgentState):
    user_name: str = ""
    language:  str = "English"
    theme:     str = "light"
    query_count: int = 0

@tool
def get_session_stats(runtime: ToolRuntime) -> str:
    """Return current session statistics."""
    s = runtime.state
    return f"User: {s.get('user_name','?')}  Lang: {s.get('language')}  Theme: {s.get('theme')}  Queries: {s.get('query_count',0)}"

@tool
def set_preference(key: str, value: str, runtime: ToolRuntime) -> Command:
    """Set a user preference (language or theme).
    Args:
        key: 'language' or 'theme'
        value: New value
    """
    allowed = {"language": ["English","Hindi","French"], "theme": ["light","dark"]}
    if key not in allowed or value not in allowed[key]:
        return Command(update={"messages": [ToolMessage(content=f"Invalid {key}={value}", tool_call_id=runtime.tool_call_id)]})
    return Command(update={key: value, "messages": [ToolMessage(content=f"✅ {key}={value}", tool_call_id=runtime.tool_call_id)]})

state_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_session_stats, set_preference],
    state_schema=UserAgentState,
    checkpointer=MemorySaver(),
    system_prompt="You are a personalised assistant.",
)

cfg2 = {"configurable": {"thread_id": str(uuid.uuid4())}}
r = state_agent.invoke(
    {"messages": [{"role": "user", "content": "Show my stats."}], "user_name": "Vinod", "language": "English"},
    config=cfg2,
)
print("Session stats:", r["messages"][-1].content[:100])
r = state_agent.invoke({"messages": [{"role": "user", "content": "Set my theme to dark."}]}, config=cfg2)
print("Set theme:", r["messages"][-1].content[:80], "  theme in state:", r.get("theme"))


# ════════════════════════════════════════════════════════════════════
# 3. TRIM & DELETE MESSAGES — manage context window
# ════════════════════════════════════════════════════════════════════
section("3. TRIM & DELETE MESSAGES")

# Trim: before_model — keep first + last 3 messages
@before_model
def trim_to_last_3(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    msgs = state["messages"]
    if len(msgs) <= 3: return None
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), msgs[0], *msgs[-3:]]}

trim_agent = create_agent(
    model="openai:gpt-4o-mini", tools=[], middleware=[trim_to_last_3],
    checkpointer=MemorySaver(),
    system_prompt="Remember facts the user shares.",
)
trim_cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
def send_trim(msg):
    r = trim_agent.invoke({"messages": [{"role": "user", "content": msg}]}, trim_cfg)
    print(f"  [{len(r['messages'])} msgs] {r['messages'][-1].content[:60]}")

send_trim("My name is Vinod.")
send_trim("I live in Bengaluru.")
send_trim("I work as a software engineer.")
send_trim("My hobby is photography.")    # trim kicks in
send_trim("Do you remember my name?")

# Delete specific messages: after_model — remove oldest pair
@after_model
def delete_oldest_pair(state: AgentState, runtime: Runtime) -> dict | None:
    msgs = state["messages"]
    if len(msgs) <= 4: return None
    print(f"      [DELETE 2 oldest]")
    return {"messages": [RemoveMessage(id=m.id) for m in msgs[:2]]}

# Wipe all messages on keyword
@before_model
def wipe_on_keyword(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    msgs = state["messages"]
    if msgs and hasattr(msgs[-1], "content") and "clear history" in msgs[-1].content.lower():
        print("      [WIPE all messages]")
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
    return None

wipe_agent = create_agent(
    model="openai:gpt-4o-mini", tools=[], middleware=[wipe_on_keyword],
    checkpointer=MemorySaver(),
    system_prompt="Tell me how many messages you can see when asked.",
)
wcfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
wipe_agent.invoke({"messages": [{"role": "user", "content": "My name is Vinod."}]}, wcfg)
wipe_agent.invoke({"messages": [{"role": "user", "content": "I work at TechCorp."}]}, wcfg)
r_before = wipe_agent.invoke({"messages": [{"role": "user", "content": "How many messages?"}]}, wcfg)
print(f"Before wipe ({len(r_before['messages'])} msgs): {r_before['messages'][-1].content[:60]}")
wipe_agent.invoke({"messages": [{"role": "user", "content": "Please clear history now."}]}, wcfg)
r_after = wipe_agent.invoke({"messages": [{"role": "user", "content": "How many messages now?"}]}, wcfg)
print(f"After wipe  ({len(r_after['messages'])} msgs): {r_after['messages'][-1].content[:60]}")


# ════════════════════════════════════════════════════════════════════
# 4. SUMMARIZATION MIDDLEWARE — compress old messages into a summary
# ════════════════════════════════════════════════════════════════════
section("4. SUMMARIZATION")

@tool
def get_current_time() -> str:
    """Get current date/time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

sum_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_current_time],
    middleware=[SummarizationMiddleware(
        model="openai:gpt-4o-mini",
        trigger=("messages", 8),   # summarise after 8 messages
        keep=("messages", 4),      # keep 4 recent after summary
    )],
    checkpointer=MemorySaver(),
    system_prompt="Remember everything the user tells you.",
)
sum_cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

def chat_sum(msg):
    return sum_agent.invoke({"messages": [{"role": "user", "content": msg}]}, sum_cfg)["messages"][-1].content

for msg in [
    "Hi! My name is Vinod.", "I'm a software engineer.", "I live in Bengaluru.",
    "My favourite language is Python.", "I've coded for 8 years.",
    "I love building AI apps.", "I enjoy photography.",
    "My project is a LangChain tutorial.",
    "What do you know about me?",   # recall after summarization
]:
    print(f"  → {chat_sum(msg)[:80]}")


# ════════════════════════════════════════════════════════════════════
# 5. DYNAMIC PROMPT — build system prompt from context/state at runtime
# ════════════════════════════════════════════════════════════════════
section("5. DYNAMIC PROMPT")

class UserCtx(TypedDict):
    user_name: str
    user_role: str
    language: str

@tool
def get_weather_report(city: str) -> str:
    """Get weather for a city. Args: city: City name."""
    return {"london": "Cloudy 14°C", "tokyo": "Sunny 28°C"}.get(city.lower(), "No data")

@dynamic_prompt
def personalised_prompt(request: ModelRequest) -> str:
    ctx = request.runtime.context
    name = ctx.get("user_name", "User")
    role = ctx.get("user_role", "general user")
    lang = ctx.get("language", "English")
    return (
        f"You are a personalised assistant.\n"
        f"User's name: {name}. Address them by name.\n"
        f"Their role: {role}. Adapt your tone accordingly.\n"
        f"Respond in {lang}."
    )

dyn_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather_report],
    middleware=[personalised_prompt],
    context_schema=UserCtx,
    checkpointer=MemorySaver(),
)

def ask_as(ctx: UserCtx, q: str) -> str:
    return dyn_agent.invoke(
        {"messages": [{"role": "user", "content": q}]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
        context=ctx,
    )["messages"][-1].content

print("Engineer:", ask_as({"user_name": "Vinod", "user_role": "Senior Python Engineer", "language": "English"},
                           "Weather in Tokyo and how does humidity affect deployments?")[:120])
print("Executive:", ask_as({"user_name": "Anita", "user_role": "Business Executive", "language": "Hindi"},
                            "What is the weather in London?")[:120])
