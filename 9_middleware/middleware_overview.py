"""
middleware_overview.py — LangChain Middleware: all key concepts in one file
Covers: built-in middleware (retry, PII, HITL, model limit), custom middleware,
        wrap_model_call, wrap_tool_call, before/after hooks, composing stacks
"""

import re
import time
from typing import Any, Callable, Optional
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import (
    BaseMiddleware,
    PIIDetectionMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    ModelCallLimitMiddleware,
    HumanInTheLoopMiddleware,
    wrap_model_call, ModelRequest, ModelResponse,
    wrap_tool_call,
    before_model, after_model,
)
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

def section(title): print(f"\n{'─'*55}\n{title}\n{'─'*55}")


# ── Shared tools ──────────────────────────────────────────────────
_fail_count = {"n": 0}

@tool
def lookup_order(order_id: str) -> str:
    """Look up order details. Args: order_id: Order ID."""
    return f"Order {order_id}: Shipped. Amount: $129.99."

@tool
def process_refund(order_id: str, amount: float) -> str:
    """Process a refund (requires approval). Args: order_id, amount."""
    _fail_count["n"] += 1
    if _fail_count["n"] < 2:
        raise ConnectionError("Payment gateway timeout — retrying.")
    return f"Refund of ${amount:.2f} processed for {order_id}."

@tool
def read_file(path: str) -> str:
    """Read file contents. Args: path: File path."""
    return f"[Contents of {path}]"

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file. Args: path, content."""
    return f"Wrote {len(content)} bytes to {path}."


# ════════════════════════════════════════════════════════════════════
# 1. BUILT-IN MIDDLEWARE
# ════════════════════════════════════════════════════════════════════
section("1. BUILT-IN MIDDLEWARE")

# ModelRetryMiddleware — retry on transient LLM errors
# ToolRetryMiddleware  — retry on tool errors
# ModelCallLimitMiddleware — cap LLM calls per agent run
# PIIDetectionMiddleware — redact PII before model sees it
# HumanInTheLoopMiddleware — pause for human approval

agent_retry = create_agent(
    model="openai:gpt-4o-mini",
    tools=[lookup_order, process_refund],
    checkpointer=MemorySaver(),
    middleware=[
        ModelRetryMiddleware(max_retries=3),   # retry LLM on error
        ToolRetryMiddleware(max_retries=3),    # retry tools on error
        ModelCallLimitMiddleware(max_calls=6), # cap at 6 LLM calls
    ],
    system_prompt="You are a support agent. Help customers with orders and refunds.",
)

cfg = {"configurable": {"thread_id": "demo-retry"}}
r = agent_retry.invoke({"messages": [HumanMessage("Check order ORD-001.")]}, config=cfg)
print("Retry agent:", r["messages"][-1].content[:100])

# PII redaction
pii_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[PIIDetectionMiddleware(redact=True, raise_on_detect=False)],
    system_prompt="Summarise customer info. Never echo raw personal data.",
)
r = pii_agent.invoke({"messages": [HumanMessage("Process: John Doe, john@example.com, SSN 123-45-6789")]})
print("PII redacted:", r["messages"][-1].content[:100])

# HITL — pause before destructive tool
hitl_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[read_file, write_file],
    checkpointer=MemorySaver(),
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})],
    system_prompt="You are a file assistant.",
)
hitl_cfg = {"configurable": {"thread_id": "demo-hitl"}}
try:
    r = hitl_agent.invoke({"messages": [HumanMessage("Write a summary to /reports/out.txt")]}, config=hitl_cfg)
    print("HITL result:", r["messages"][-1].content[:80])
except Exception:
    print("HITL: agent paused for approval")
    r = hitl_agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=hitl_cfg)
    print("HITL resumed:", r["messages"][-1].content[:80])


# ════════════════════════════════════════════════════════════════════
# 2. CUSTOM MIDDLEWARE via BaseMiddleware
# ════════════════════════════════════════════════════════════════════
section("2. CUSTOM MIDDLEWARE (BaseMiddleware)")

class TimingMiddleware(BaseMiddleware):
    """Measure LLM call latency."""
    def before_model(self, state: dict, **kw) -> Optional[dict]:
        self._t0 = time.perf_counter()
        return None
    def after_model(self, state: dict, **kw) -> Optional[dict]:
        print(f"  [Timing] LLM took {time.perf_counter()-self._t0:.2f}s")
        return None

class ContentGuardrail(BaseMiddleware):
    """Block abuse/prompt injection attempts."""
    FORBIDDEN = [r"\b(jailbreak|ignore previous|bypass|hack)\b"]
    def before_agent(self, state: dict) -> Optional[dict]:
        for msg in state.get("messages", []):
            content = getattr(msg, "content", "") or ""
            for p in self.FORBIDDEN:
                if re.search(p, content, re.IGNORECASE):
                    print("  [Guardrail] 🚫 Blocked policy violation")
                    return {**state, "messages": state["messages"] + [
                        AIMessage(content="Request blocked by usage policy.")
                    ]}
        return None

guarded_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[lookup_order],
    checkpointer=MemorySaver(),
    middleware=[ContentGuardrail(), TimingMiddleware()],
    system_prompt="You are a support agent.",
)

cfg2 = {"configurable": {"thread_id": "demo-custom"}}
r = guarded_agent.invoke({"messages": [HumanMessage("Check order ORD-002.")]}, config=cfg2)
print("Normal request:", r["messages"][-1].content[:80])

cfg3 = {"configurable": {"thread_id": "demo-blocked"}}
r = guarded_agent.invoke({"messages": [HumanMessage("Jailbreak the system now!")]}, config=cfg3)
print("Blocked:", r["messages"][-1].content[:80])


# ════════════════════════════════════════════════════════════════════
# 3. wrap_model_call — intercept every model call
# ════════════════════════════════════════════════════════════════════
section("3. wrap_model_call")

@wrap_model_call
def logging_middleware(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Log every model call (tool count, message count)."""
    print(f"  [ModelCall] tools={len(request.tools)}  messages={len(request.messages)}")
    response = handler(request)
    tc_count = len(response.tool_calls) if hasattr(response, "tool_calls") else 0
    print(f"  [ModelCall] response tool_calls={tc_count}")
    return response

@wrap_model_call
def filter_tools(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Only expose read_file (not write_file) unless explicitly needed."""
    safe_tools = [t for t in request.tools if t.name != "write_file"]
    print(f"  [ToolFilter] {len(request.tools)} → {len(safe_tools)} tools")
    return handler(request.override(tools=safe_tools))

filter_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[read_file, write_file],
    middleware=[logging_middleware, filter_tools],
    system_prompt="You are a file assistant.",
)
r = filter_agent.invoke({"messages": [HumanMessage("Read /docs/readme.md")]})
print("Filtered result:", r["messages"][-1].content[:80])


# ════════════════════════════════════════════════════════════════════
# 4. wrap_tool_call — intercept tool execution
# ════════════════════════════════════════════════════════════════════
section("4. wrap_tool_call — tool error handling")

@tool
def risky_divide(a: float, b: float) -> str:
    """Divide two numbers. Args: a: numerator, b: denominator."""
    if b == 0: raise ZeroDivisionError("Cannot divide by zero!")
    return f"{a} / {b} = {a/b:.4f}"

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
r = err_agent.invoke({"messages": [HumanMessage("Divide 144 by 12")]})
print("Normal division:", r["messages"][-1].content[:80])


# ════════════════════════════════════════════════════════════════════
# 5. before_model / after_model HOOKS
# ════════════════════════════════════════════════════════════════════
section("5. before_model / after_model HOOKS")

from langchain.agents import AgentState
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

@before_model
def trim_to_3(state: AgentState, runtime: Runtime) -> dict | None:
    """Trim conversation to first + last 3 messages before each model call."""
    msgs = state["messages"]
    if len(msgs) <= 3: return None
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), msgs[0], *msgs[-3:]]}

@after_model
def log_after(state: AgentState, runtime: Runtime) -> dict | None:
    """Log message count after each model call."""
    print(f"  [AfterModel] {len(state['messages'])} messages in state")
    return None

hook_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[trim_to_3, log_after],
    checkpointer=MemorySaver(),
    system_prompt="Remember what the user tells you.",
)
hook_cfg = {"configurable": {"thread_id": "demo-hooks"}}
for msg in ["I'm Vinod.", "I live in Bengaluru.", "I love Python.", "My hobby is hiking."]:
    r = hook_agent.invoke({"messages": [{"role": "user", "content": msg}]}, hook_cfg)
print("After trim, recall:", hook_agent.invoke({"messages": [{"role": "user", "content": "Do you remember my name?"}]}, hook_cfg)["messages"][-1].content[:80])


# ════════════════════════════════════════════════════════════════════
# 6. COMPOSING A FULL MIDDLEWARE STACK
# ════════════════════════════════════════════════════════════════════
section("6. COMPOSING A FULL STACK")

full_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[lookup_order, process_refund],
    checkpointer=MemorySaver(),
    middleware=[
        ContentGuardrail(),                                  # 1. Block abuse early
        PIIDetectionMiddleware(redact=True, raise_on_detect=False),  # 2. Redact PII
        TimingMiddleware(),                                  # 3. Measure latency
        ModelCallLimitMiddleware(max_calls=6),               # 4. Cap LLM costs
        ToolRetryMiddleware(max_retries=3),                  # 5. Retry flaky tools
        HumanInTheLoopMiddleware(interrupt_on={"process_refund": True}),  # 6. HITL gate
    ],
    system_prompt="You are a support agent. Look up orders and process refunds when needed.",
)

cfg4 = {"configurable": {"thread_id": "demo-full"}}
r = full_agent.invoke({"messages": [HumanMessage("Check status of order ORD-001.")]}, config=cfg4)
if "__interrupt__" not in r:
    print("Full stack — order lookup:", r["messages"][-1].content[:100])

print("""
Middleware execution order:
  before_agent → before_model → [LLM call] → after_model → [tool] → before_model → ... → after_agent
  
  wrap_model_call wraps individual LLM calls.
  wrap_tool_call wraps individual tool executions.
  before/after hooks fire at each graph step boundary.
""")
