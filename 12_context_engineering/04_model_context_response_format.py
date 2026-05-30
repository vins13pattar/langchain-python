"""
04_model_context_response_format.py
=====================================
Demonstrates dynamic RESPONSE FORMAT (structured output) selection —
adapting the output schema based on State, Store, and Runtime Context.

Concepts covered:
  - request.override(response_format=Schema) via wrap_model_call
  - State-based format selection (conversation stage → simple vs detailed)
  - Store-based format (user's preferred verbosity)
  - Runtime Context–based format (role/environment → admin vs user schema)
  - Pydantic schema design for clear LLM guidance
  - Combining dynamic format with dynamic tools in one agent
"""

import os
from dataclasses import dataclass
from typing import Callable
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool
from langgraph.store.memory import InMemoryStore

load_dotenv()

print("=" * 60)
print("Model Context — Dynamic Response Format")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# RESPONSE SCHEMAS
# ════════════════════════════════════════════════════════════════════

class SimpleResponse(BaseModel):
    """Concise answer for early or casual interactions."""
    answer: str = Field(description="A brief, direct answer to the user's question.")


class DetailedResponse(BaseModel):
    """Detailed answer with reasoning and confidence for complex queries."""
    answer:     str   = Field(description="A thorough answer to the question.")
    reasoning:  str   = Field(description="Step-by-step reasoning used to arrive at the answer.")
    confidence: float = Field(description="Confidence score from 0.0 (uncertain) to 1.0 (certain).")
    sources:    list[str] = Field(description="List of data sources or tools consulted.", default_factory=list)


class AdminResponse(BaseModel):
    """Full technical response for admin users including system internals."""
    answer:        str        = Field(description="Answer to the question.")
    debug_info:    dict       = Field(description="Technical debug information.", default_factory=dict)
    system_status: str        = Field(description="Current system status: 'healthy', 'degraded', or 'down'.")
    action_log:    list[str]  = Field(description="List of actions taken.", default_factory=list)


class UserResponse(BaseModel):
    """Friendly, simplified response for regular end users."""
    answer: str = Field(description="Clear, user-friendly answer.")
    next_steps: list[str] = Field(description="Suggested follow-up actions for the user.", default_factory=list)


class VerboseResponse(BaseModel):
    """Rich response for users who prefer detailed explanations."""
    answer:     str       = Field(description="Detailed answer.")
    background: str       = Field(description="Background context and explanation.")
    sources:    list[str] = Field(description="Data sources consulted.", default_factory=list)
    confidence: float     = Field(description="Confidence level 0.0–1.0.")


class ConciseResponse(BaseModel):
    """Minimal response for users who prefer brevity."""
    answer: str = Field(description="One or two sentence answer.")


# ════════════════════════════════════════════════════════════════════
# SHARED TOOL
# ════════════════════════════════════════════════════════════════════

@tool
def get_system_metrics() -> str:
    """Return current system performance metrics."""
    return "CPU: 34%, Memory: 62%, Disk: 45%, API Latency: 120ms."


# ════════════════════════════════════════════════════════════════════
# 1. STATE-BASED RESPONSE FORMAT
#    Early conversations → simple schema.
#    Established conversations → detailed schema with reasoning.
# ════════════════════════════════════════════════════════════════════

@wrap_model_call
def state_based_output_format(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Select output schema based on conversation length in State."""
    msg_count = len(request.messages)
    print(f"  [FormatSelect/state] msg_count={msg_count}")

    if msg_count < 3:
        schema = SimpleResponse
        print("  [FormatSelect/state] → SimpleResponse")
    else:
        schema = DetailedResponse
        print("  [FormatSelect/state] → DetailedResponse")

    return handler(request.override(response_format=schema))


print("\n── 1. State-Based Response Format ───────────────────────────")

agent_state_fmt = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_system_metrics],
    middleware=[state_based_output_format],
    system_prompt="You are a helpful technical assistant.",
)

# New conversation — SimpleResponse
result_simple = agent_state_fmt.invoke({
    "messages": [{"role": "user", "content": "What are the current system metrics?"}]
})
last = result_simple["messages"][-1]
if hasattr(last, "parsed"):
    print(f"Simple schema:   {last.parsed}")
else:
    print(f"Simple:   {last.content[:120]}")

# Long conversation — DetailedResponse
history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
           for i in range(4)]
history.append({"role": "user", "content": "Explain the system metrics in detail."})

result_detailed = agent_state_fmt.invoke({"messages": history})
last_d = result_detailed["messages"][-1]
if hasattr(last_d, "parsed"):
    print(f"Detailed schema: {last_d.parsed}")
else:
    print(f"Detailed: {last_d.content[:150]}")


# ════════════════════════════════════════════════════════════════════
# 2. STORE-BASED RESPONSE FORMAT
#    Honour the user's preferred response verbosity from long-term memory.
# ════════════════════════════════════════════════════════════════════

@dataclass
class UserCtx:
    user_id: str


@wrap_model_call
def store_based_output_format(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Select schema based on the user's saved verbosity preference."""
    user_id = request.runtime.context.user_id
    store   = request.runtime.store
    schema  = ConciseResponse  # default

    if store:
        prefs = store.get(("preferences",), user_id)
        if prefs:
            style = prefs.value.get("response_style", "concise")
            schema = VerboseResponse if style == "verbose" else ConciseResponse
            print(f"  [FormatSelect/store] user={user_id}, style={style}")
        else:
            print(f"  [FormatSelect/store] user={user_id}, no prefs → concise")

    return handler(request.override(response_format=schema))


print("\n── 2. Store-Based Response Format (user verbosity pref) ──────")

pref_store = InMemoryStore()
pref_store.put(("preferences",), "USR-V", {"response_style": "verbose"})
pref_store.put(("preferences",), "USR-C", {"response_style": "concise"})

agent_store_fmt = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    context_schema=UserCtx,
    store=pref_store,
    middleware=[store_based_output_format],
    system_prompt="You are a knowledgeable assistant.",
)

result_verbose = agent_store_fmt.invoke(
    {"messages": [{"role": "user", "content": "Explain REST APIs."}]},
    context=UserCtx(user_id="USR-V"),
)
last_v = result_verbose["messages"][-1]
print(f"Verbose user: {(last_v.parsed if hasattr(last_v, 'parsed') else last_v.content)!s:.160}")

result_concise = agent_store_fmt.invoke(
    {"messages": [{"role": "user", "content": "Explain REST APIs."}]},
    context=UserCtx(user_id="USR-C"),
)
last_c = result_concise["messages"][-1]
print(f"Concise user: {(last_c.parsed if hasattr(last_c, 'parsed') else last_c.content)!s:.160}")


# ════════════════════════════════════════════════════════════════════
# 3. RUNTIME CONTEXT–BASED RESPONSE FORMAT
#    Admin users in production → AdminResponse (with debug info).
#    Regular users → UserResponse (friendly, minimal).
# ════════════════════════════════════════════════════════════════════

@dataclass
class RoleCtx:
    user_role:   str   # "admin", "user"
    environment: str   # "production", "staging"


@wrap_model_call
def context_based_output_format(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Select output schema from Runtime Context role + environment."""
    ctx = request.runtime.context
    print(f"  [FormatSelect/context] role={ctx.user_role}, env={ctx.environment}")

    if ctx.user_role == "admin" and ctx.environment == "production":
        schema = AdminResponse
        print("  [FormatSelect/context] → AdminResponse")
    else:
        schema = UserResponse
        print("  [FormatSelect/context] → UserResponse")

    return handler(request.override(response_format=schema))


print("\n── 3. Context-Based Response Format (role + env) ────────────")

agent_ctx_fmt = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_system_metrics],
    context_schema=RoleCtx,
    middleware=[context_based_output_format],
    system_prompt="You are a system management assistant.",
)

result_admin = agent_ctx_fmt.invoke(
    {"messages": [{"role": "user", "content":
        "Check system metrics and report status."}]},
    context=RoleCtx(user_role="admin", environment="production"),
)
last_a = result_admin["messages"][-1]
print(f"Admin response: {(last_a.parsed if hasattr(last_a, 'parsed') else last_a.content)!s:.200}")

result_user = agent_ctx_fmt.invoke(
    {"messages": [{"role": "user", "content":
        "Is the system working okay?"}]},
    context=RoleCtx(user_role="user", environment="production"),
)
last_u = result_user["messages"][-1]
print(f"User response:  {(last_u.parsed if hasattr(last_u, 'parsed') else last_u.content)!s:.200}")


# ════════════════════════════════════════════════════════════════════
# 4. COMBINED — dynamic format + dynamic tools in one agent
# ════════════════════════════════════════════════════════════════════

@dataclass
class FullCtx:
    user_role: str
    user_id:   str


@wrap_model_call
def combined_format_and_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Apply both tool filtering AND output format selection."""
    role = request.runtime.context.user_role

    # Tool filter
    if role == "admin":
        tools  = request.tools
        schema = AdminResponse
    else:
        tools  = [t for t in request.tools if t.name != "get_system_metrics"]
        schema = UserResponse

    print(f"  [Combined] role={role}, tools={[t.name for t in tools]}, schema={schema.__name__}")
    request = request.override(tools=tools, response_format=schema)
    return handler(request)


print("\n── 4. Combined Format + Tool Filtering ──────────────────────")

@tool
def get_user_summary() -> str:
    """Return a user-facing summary of account status."""
    return "Your account is active with 5 recent orders."


agent_combined = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_system_metrics, get_user_summary],
    context_schema=FullCtx,
    middleware=[combined_format_and_tools],
    system_prompt="You are a platform assistant.",
)

for role in ("admin", "user"):
    r = agent_combined.invoke(
        {"messages": [{"role": "user", "content": "Give me a status report."}]},
        context=FullCtx(user_role=role, user_id=f"ID-{role}"),
    )
    last_r = r["messages"][-1]
    preview = str(last_r.parsed if hasattr(last_r, "parsed") else last_r.content)
    print(f"  {role:6}: {preview[:120]}")

print("\n" + "═" * 60)
print("Response Format Key Points:")
print("  - Schema field descriptions guide the LLM on what to fill.")
print("  - request.override(response_format=Schema) is TRANSIENT.")
print("  - The agent runs the tool loop first, then coerces the FINAL")
print("    response into the schema — tools run unformatted.")
print("  - Combine with dynamic tools for full model context control.")
print("═" * 60)
print("\n✅ Dynamic response format demo complete.")
