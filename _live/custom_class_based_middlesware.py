from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing import Any, Callable
from langchain.tools import tool
from langchain.messages import ToolMessage, HumanMessage
import time
from dotenv import load_dotenv
load_dotenv()

@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result"""
    print("Calculator tool is called....")
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def weather(city: str) -> str:
    """Get the weather for a city"""
    print(f"\n\n Weather tool called for {city}\n")
    return f"The weather in {city} is sunny."

class LoggingMiddleware(AgentMiddleware):
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"\n\n Before agent execution")
        state["messages"].append(HumanMessage(content="Hello, can you tell me what is the weather like in Hyderabad?"))
        return None

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"\n\n About to call model with {len(state['messages'])} messages")
        print("-----------------------\n")
        print(state['messages'])
        print("-----------------------\n")
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        print("\n\n Model call wrapper", request)
        for retry in range(3):
            print(f"\n\nTrying {retry + 1} time(s)\n\n")
            try:
                return handler(request)
            except Exception as e:
                print(f"\n\nError: {e}\n\n")
                if retry == 2:
                    raise
                print("\n\nRetrying in 2 seconds\n\n")
                time.sleep(2)
                continue
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        print("\n\n Tool call wrapper", request)
        response = handler(request)
        return response

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"\n\n Model returned: {state['messages'][-1].content}")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"\n\n After agent response")
        return None

    async def abefore_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        # Async version of before_model
        print(f"\n\n Async About to call model with {len(state['messages'])} messages")
        return None

    async def aafter_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        # Async version of after_model
        print(f"\n Async Model returned: {state['messages'][-1].content}")
        return None


agent = create_agent(
    model="gpt-5.4-mini",
    middleware=[LoggingMiddleware()],
    tools=[calculator, weather],
    system_prompt = "You are a helpful assistant that can use tools to answer user queries."
)

print(f"\n\n Starting the agent invocation")

result = agent.invoke({"messages": [{"role":"user", "content":"What is 15 * 8?"}]})
print(result['messages'][-1])