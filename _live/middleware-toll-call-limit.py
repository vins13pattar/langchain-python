from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
import random
from dotenv import load_dotenv
load_dotenv()


# Tools

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
    
@tool
def discount_calculator(price: float, discount: float) -> str:
    """Calculate the discounted price."""
    try:
        discounted_price = price * (1 - discount / 100)
        return f"The discounted price is: {discounted_price:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"
    

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""

    return f"The current weather in {location} is {random.choice([30, 35, 40])} degree Celcius"

IS_PIAD_USER = True
DYNAMIC_LIMIT = 5 if IS_PIAD_USER else 2

agent = create_agent(
    model="gpt-5.5",
    tools=[calculate, discount_calculator, get_weather],
    checkpointer=InMemorySaver(),
    middleware=[
        # Global limit
        ToolCallLimitMiddleware(thread_limit=2, run_limit=DYNAMIC_LIMIT, exit_behavior="end"),
        # Tool-specific limit
        ToolCallLimitMiddleware(
            tool_name="get_weather",
            thread_limit=2,
            run_limit=DYNAMIC_LIMIT,
            exit_behavior="end"
        ),
    ],
    system_prompt="You are a helpful assistant. use tools to answer user queries"
)

CHAT_THREAD = { "configurable": { "thread_id": "thread_001"}}

result1 = agent.invoke({
    "messages": HumanMessage(content="What is the weather in Bangalore?")
}, config=CHAT_THREAD)
print("\n Result 1: ", result1)

result2 = agent.invoke({
    "messages": HumanMessage(content="What is the weather in Mysore?")
}, config=CHAT_THREAD)
print("\n Result 2: ", result2)


result3 = agent.invoke({
    "messages": HumanMessage(content="What is the weather in Belagavi?"),
}, config=CHAT_THREAD)
print("\n Result 3: ", result3)