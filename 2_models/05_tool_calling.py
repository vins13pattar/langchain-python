"""
05_tool_calling.py
==================
Demonstrates model-level TOOL BINDING and manual loop orchestration.

Concepts covered:
  - model.bind_tools()             — letting the model know about available tools
  - Inspecting message.tool_calls  — detecting when a model wants to call a tool
  - Executing tools manually       — matching tool names and executing Python functions
  - ToolMessage                    — returning tool outputs back to the model
  - Manual Agent Loop             — understanding what create_agent() does under the hood

This file shows the lower-level mechanics of tool calling, allowing you to build
completely custom loops or debug existing agent frameworks.
"""

import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

# ── 1. Define Tools ──────────────────────────────────────────────────────────
# Standard tools declared with the @tool decorator.

@tool
def get_tax_rate(state: str) -> float:
    """Get the sales tax rate for a specific state in the US.

    Args:
        state: The 2-letter state code (e.g. 'NY', 'CA', 'TX')
    """
    tax_rates = {
        "CA": 0.0825,  # California 8.25%
        "NY": 0.08875, # New York 8.875%
        "TX": 0.0625,  # Texas 6.25%
    }
    return tax_rates.get(state.upper(), 0.0)


@tool
def calculate_subtotal(items: list[dict]) -> float:
    """Calculate the subtotal of a list of shopping items.

    Args:
        items: List of item dicts, where each dict has 'name', 'price', and 'quantity' keys.
    """
    subtotal = 0.0
    for item in items:
        price = float(item.get("price", 0.0))
        qty   = int(item.get("quantity", 1))
        subtotal += price * qty
    return subtotal


# Register tools in a lookup dictionary for easy manual execution
TOOLS_LOOKUP = {
    "get_tax_rate": get_tax_rate,
    "calculate_subtotal": calculate_subtotal,
}


print("=" * 60)
print("Manual Model-Level Tool Calling Demo")
print("=" * 60)

# Initialize standard model
model = init_chat_model("openai:gpt-4o-mini")

# ── 2. Bind Tools to Model ────────────────────────────────────────────────────
# Binding tells the model that these tools exist and schemas are passed to the model API.
model_with_tools = model.bind_tools(list(TOOLS_LOOKUP.values()))


# ── 3. Manual Orchestration Loop ──────────────────────────────────────────────
# We will step through the conversation manually.

# Conversation history list
messages = [
    HumanMessage(
        content=(
            "I want to buy 2 shirts for $25.00 each and 1 pair of shoes for $80.00. "
            "Please calculate my subtotal, find the tax rate for California (CA), "
            "and tell me my final total including tax."
        )
    )
]

print(f"\n🧑 User: {messages[0].content}\n")

# ── Step 1: Model receives query and decides to call tools ───────────
print("--- [Step 1: Invoking model with tools bound] ---")
response_1 = model_with_tools.invoke(messages)
messages.append(response_1)

print(f"🤖 Model response type: {type(response_1).__name__}")
print(f"🤖 Has tool calls?      {bool(response_1.tool_calls)}")

if response_1.tool_calls:
    for i, tc in enumerate(response_1.tool_calls, 1):
        print(f"    Tool Call #{i}:")
        print(f"      Name: {tc['name']}")
        print(f"      Args: {tc['args']}")
        print(f"      ID:   {tc['id']}")

# ── Step 2: Manually execute the tools that the model requested ─────
print("\n--- [Step 2: Executing tools manually] ---")

for tc in response_1.tool_calls:
    tool_name = tc["name"]
    tool_args = tc["args"]
    tool_id   = tc["id"]

    if tool_name in TOOLS_LOOKUP:
        print(f"🔧 Invoking python tool: '{tool_name}' with args {tool_args}...")
        
        # Execute tool function
        actual_tool = TOOLS_LOOKUP[tool_name]
        
        # Call tool with unpacked keyword arguments
        result = actual_tool.invoke(tool_args)
        
        print(f"🔧 Tool Output: {result}")

        # Create a ToolMessage. It MUST contain:
        # - content: the string representation of the tool output
        # - tool_call_id: the exact ID matched from the model's tool call
        tool_message = ToolMessage(content=str(result), tool_call_id=tool_id)
        messages.append(tool_message)
    else:
        print(f"❌ Unknown tool requested: '{tool_name}'")

# ── Step 3: Call model again with the tool results ─────────────────
print("\n--- [Step 3: Invoking model with tool results] ---")
print(f"Total messages in history: {len(messages)}")

response_2 = model_with_tools.invoke(messages)
messages.append(response_2)

# If the model has MORE tool calls, we would continue looping.
# Let's inspect if it decided to call more or returned final text.
if response_2.tool_calls:
    print("\n🤖 Model requested MORE tools:")
    for tc in response_2.tool_calls:
        print(f"    - {tc['name']}({tc['args']})")
else:
    print(f"\n🤖 Model Final Response:\n{response_2.content}")
