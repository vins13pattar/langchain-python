"""
05_error_handling_and_retries.py
=================================
Demonstrates ERROR HANDLING and automatic validation retries in structured output.

Concepts covered:
  - Default retry behavior (handle_errors=True)
  - Validation failures (e.g. constraints ge=1, le=5 violated)
  - Custom error string prompts
  - Custom error handler functions (Callable[[Exception], str])
  - Exception types (StructuredOutputValidationError, MultipleStructuredOutputsError)
"""

import os
from typing import Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.structured_output import (
    ToolStrategy,
    StructuredOutputValidationError,
    MultipleStructuredOutputsError
)

load_dotenv()

print("=" * 60)
print("Structured Output Error Handling & Retries Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. AUTO-RETRY ON CONSTRAINT VIOLATION
# ════════════════════════════════════════════════════════════════════

print("── 1. Default Retry on Pydantic Validation Error ───────────")

class ProductRating(BaseModel):
    rating: int = Field(description="Rating from 1-5 inclusive", ge=1, le=5)
    comment: str = Field(description="Review comment")

# By default, handle_errors=True
tool_strategy_default = ToolStrategy(
    schema=ProductRating,
    handle_errors=True
)

agent_default = create_agent(
    model="openai:gpt-4o-mini",
    response_format=tool_strategy_default,
    system_prompt=(
        "You parse reviews. Extract the rating EXACTLY as stated by the user, "
        "even if it violates the 1-5 boundary first."
    )
)

# User says "10/10", which violates the "le=5" constraint
INPUT_VIOLATING = "Amazing product, 10/10!"
print(f"Query: '{INPUT_VIOLATING}'\n")

print("Invoking agent...")
result_default = agent_default.invoke({
    "messages": [{"role": "user", "content": INPUT_VIOLATING}]
})

print("\nConversation history trace:")
for msg in result_default["messages"]:
    msg_type = type(msg).__name__
    if msg_type == "ToolMessage":
        # Check if this ToolMessage was an error
        if "Error" in msg.content or "Failed" in msg.content:
            print(f"  ❌ [ToolMessage Error]: {msg.content}")
        else:
            print(f"  ✅ [ToolMessage Return]: {msg.content}")
    elif msg_type == "AIMessage" and msg.tool_calls:
        print(f"  🤖 [AIMessage ToolCalls]: {msg.tool_calls[0]['name']}({msg.tool_calls[0]['args']})")

print(f"\nFinal Validated Structured Output: {result_default['structured_response']}\n")


# ════════════════════════════════════════════════════════════════════
# 2. CUSTOM ERROR HANDLER FUNCTION
# ════════════════════════════════════════════════════════════════════

print("─" * 60)
print("2. Custom Error Handler Function (Callable)")
print("─" * 60)

def custom_error_handler(error: Exception) -> str:
    """A custom function that formats the validation error for the model."""
    if isinstance(error, StructuredOutputValidationError):
        return (
            "🚨 custom_error_handler: The extracted data does not match the Pydantic constraints. "
            "Please ensure rating is strictly between 1 and 5. Correct the values and try again."
        )
    return f"Error: {error}"

tool_strategy_custom = ToolStrategy(
    schema=ProductRating,
    handle_errors=custom_error_handler
)

agent_custom = create_agent(
    model="openai:gpt-4o-mini",
    response_format=tool_strategy_custom,
    system_prompt="You parse reviews. Extract the rating as stated by the user."
)

print("Invoking agent with custom error handler...")
result_custom = agent_custom.invoke({
    "messages": [{"role": "user", "content": INPUT_VIOLATING}]
})

print("\nConversation history trace:")
for msg in result_custom["messages"]:
    msg_type = type(msg).__name__
    if msg_type == "ToolMessage":
        if "custom_error_handler" in msg.content:
            print(f"  🚨 [Custom ToolMessage Error]: {msg.content}")
        else:
            print(f"  ✅ [ToolMessage Return]: {msg.content}")
    elif msg_type == "AIMessage" and msg.tool_calls:
        print(f"  🤖 [AIMessage ToolCalls]: {msg.tool_calls[0]['name']}({msg.tool_calls[0]['args']})")

print(f"\nFinal Validated Structured Output: {result_custom['structured_response']}")
print("─" * 60)
