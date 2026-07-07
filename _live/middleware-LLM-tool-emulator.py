from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
import random
import os

load_dotenv()

# Tools
@tool()
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    print("calculate tool is called")
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
    
@tool()
def discount_calculator(price: float, discount: float) -> str:
    """Calculate the discounted price."""
    print("discount_calculator tool is called")
    try:
        discounted_price = price * (1 - discount / 100)
        return f"The discounted price is: {discounted_price:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"
    

@tool()
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    print("get_weather tool is called")
    return f"The current weather in {location} is {random.choice([30, 35, 40])} degree Celcius"


# model = init_chat_model(
#     model="gemma-4-e2b",
#     model_provider="openai",
#     base_url=os.getenv("BASE_URL"),
#     api_key=os.getenv("OPENAI_API_KEY")
# )

agent = create_agent(
    model="anthropic:claude-haiku-4-5-20251001",
    tools=[calculate, discount_calculator, get_weather],
    checkpointer=InMemorySaver(),
    middleware=[
        LLMToolEmulator(),  # Emulate all tools
    ],
    system_prompt="""
    You are a helpful assistant that use only tools to answer user queries.
    """
)

CHAT_THREAD = {"configurable": {"thread_id": "thread_005"}}

result = agent.invoke({"messages": [{"role": "user", "content": "What is the weather in Bangalore?"}]}, config=CHAT_THREAD)

print(result)