from langchain.agents import create_agent
from langchain.agents.middleware import ProviderToolSearchMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.messages import HumanMessage
import random
load_dotenv()


# Tools
@tool(extras={"defer_loading": True})
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
    
@tool(extras={"defer_loading": True})
def discount_calculator(price: float, discount: float) -> str:
    """Calculate the discounted price."""
    try:
        discounted_price = price * (1 - discount / 100)
        return f"The discounted price is: {discounted_price:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"
    

@tool(extras={"defer_loading": True})
def get_weather(location: str) -> str:
    """Get the current weather for a location."""

    return f"The current weather in {location} is {random.choice([30, 35, 40])} degree Celcius"


agent = create_agent(
    model="anthropic:claude-opus-4-8",
    tools=[get_weather, calculate, discount_calculator],
    checkpointer=InMemorySaver(),
    middleware=[
        ProviderToolSearchMiddleware(),
    ],
    system_prompt="""
    You are a helpful assistant that use only tools to answer user queries.
    """,
)

CHAT_THREAD = {"configurable": {"thread_id": "thread_003"}}


result = agent.invoke({
    "messages": HumanMessage(content="What is the weather in Bangalore?")
}, config=CHAT_THREAD)

print(result)