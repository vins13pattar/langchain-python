"""
04_custom_tool_message.py
=========================
Demonstrates how to customize the history representation of tool-calling
structured output using the tool_message_content parameter.

Concepts covered:
  - Customizing the generated ToolMessage content
  - Viewing the chat history steps to compare
"""

import os
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

load_dotenv()

print("=" * 60)
print("Custom Tool Message Content Demo")
print("=" * 60)

TEXT_TO_PARSE = "From our meeting: Sarah needs to update the project timeline as soon as possible"
print(f"Input text: '{TEXT_TO_PARSE}'\n")


# ════════════════════════════════════════════════════════════════════
# 1. DEFINE SCHEMA AND TOOLSTRATEGY
# ════════════════════════════════════════════════════════════════════

class MeetingAction(BaseModel):
    """Action items extracted from a meeting transcript."""
    task: str = Field(description="The specific task to be completed")
    assignee: str = Field(description="Person responsible for the task")
    priority: Literal["low", "medium", "high"] = Field(description="Priority level")


# ToolStrategy with a customized tool_message_content
tool_strategy_custom = ToolStrategy(
    schema=MeetingAction,
    tool_message_content="Action item captured and added to meeting notes!"
)

agent = create_agent(
    model="openai:gpt-4o-mini",
    response_format=tool_strategy_custom,
    system_prompt="You extract action items from meetings."
)

print("Invoking agent...")
result = agent.invoke({
    "messages": [{"role": "user", "content": TEXT_TO_PARSE}]
})

# Let's print out the exact messages in the conversation history
print("\n── Inspection of Conversation History ──")
for msg in result["messages"]:
    msg_type = type(msg).__name__
    print(f"\n[{msg_type}]:")
    
    # Show tool calls in AIMessage
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"  Tool Calls:")
        for tc in msg.tool_calls:
            print(f"    - ID: {tc['id']}")
            print(f"      Name: {tc['name']}")
            print(f"      Args: {tc['args']}")
            
    # Show ToolMessage content (our customized message!)
    elif msg_type == "ToolMessage":
        print(f"  Content: {msg.content}")
        print(f"  Tool Call ID: {msg.tool_call_id}")
        
    else:
        print(f"  Content: {msg.content}")

print(f"\nStructured Response Output: {result['structured_response']}")
print("─" * 60)
