from typing import Callable
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, before_agent, AgentState, wrap_model_call, wrap_tool_call, ModelRequest, ModelResponse, after_agent, after_model, ExtendedModelResponse
from langgraph.runtime import Runtime
from langchain.messages import  ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()


@tool
def weather(city: str) -> str:
    """Get the weather for a city"""
    print("\n")
    print("\nThis is weather tool", city)
    print("\n")
    return f"The weather in {city} is sunny."

# A first hook to be called and it is called once per a agent invokation
@before_agent
def print_a_message_before_agent_executes(state: AgentState, runtime: Runtime):
    print("\n")
    print("\nThis is before agent starts",state)
    print("\n")


# Hook to be called before model invokation and it is called for every step in the agent execution
@before_model
def print_a_message_before_model_executes(state: AgentState, runtime: Runtime):
    print("\n")
    print("\nThis is before model starts",state)
    print("\n")

# Hook to be called after model invokation and it is called for every step in the agent execution
@after_model
def print_a_message_after_model_executes(state: AgentState, runtime: Runtime):
    print("\n")
    print("\nThis is after model response",state)
    print("\n")

# Hook to be called after model invokation and it is called once per a agent invokation
@after_agent
def print_a_message_after_agent_executes(state: AgentState, runtime: Runtime):
    print("\n")
    print("\nThis is after agent response",state)
    print("\n")


@wrap_model_call
def print_a_message_before_model_call(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ExtendedModelResponse:
    print("\n")
    print("\nThis is wrapper model call",request,handler)
    print("\n")
    response = handler(request)
    return response


@wrap_tool_call
def print_a_message_before_tool_call(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage]):
    print("\n")
    print("\nThis is wrapper tool call",request,handler)
    print("\n")
    response = handler(request)
    return response


agent = create_agent(
    model = "gpt-5.4-mini",
    tools=[weather],
    middleware=[print_a_message_before_agent_executes, print_a_message_before_model_executes,print_a_message_after_model_executes,print_a_message_after_agent_executes, print_a_message_before_model_call,print_a_message_before_tool_call ],
    system_prompt = "You are a helpful assistant that can use tools to answer user queries."
)

response = agent.invoke({"messages": [{"role":"user", "content":"Hi, What is the weather like in Hyderabad?"}]})
print(response)

