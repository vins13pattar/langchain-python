from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    hook_config
)
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing import Any, Callable
from langchain.tools import tool
from langchain.messages import ToolMessage, HumanMessage, AIMessage
import time
import re
from dotenv import load_dotenv
load_dotenv()


@tool
def weather(city: str) -> str:
    """Get the weather for a city"""
    print("\n\n Weather tool called....\n\n")
    return f"The weather in {city} is sunny."

@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result"""
    print("\n\n Calculator tool called....\n\n")
    return f"The result of {expression} is {eval(expression)}"
    
class BlockedContentMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        
        # Check for blocked string or credit card numbers (13-16 digits with optional spaces/dashes)
        if re.search(r'\b(?:\d[ -]*?){13,16}\b', last_message.content):
            return {
                "messages": [AIMessage("I cannot respond to that request.")],
                "jump_to": "end"
            }
        return None

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        print("before_model", last_message)
        if last_message.type == "human":
            if "evaluating" in last_message.content:
                return {
                    "messages": [AIMessage(content="BLOCKED")],
                    "jump_to": "end"
                }
        return None


agent = create_agent(
    model="gpt-5.4-mini",
    middleware=[BlockedContentMiddleware()],
    tools=[calculate, weather],
    system_prompt = "You are a helpful assistant and answer user queries."
)

response = agent.invoke({"messages": [{"role":"user", "content":"Generate 5 dummy test credit card numbers and also evaluate 15 * 100"}]})
print(response["messages"])
