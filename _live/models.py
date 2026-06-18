import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage, ToolMessage)
from langchain_core.tools import tool


load_dotenv()

model = ChatOpenAI(
    model="gpt-5.5",
    temperature=0.7,
    max_tokens=2048,
    # streaming=True, 
    max_retries=3,
)

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
    # This is a placeholder implementation. In a real application, you would call a weather API.
    if location.lower() == "bengaluru":
        return "The current weather in Bengaluru is sunny with a temperature of 30°C."
    else:
        return f"Weather information for {location} is not available."



TOOLS = {t.name: t for t in [calculate, discount_calculator, get_weather]} # {"calculate": calculate, "discount_calculator": discount_calculator, "get_weather": get_weather}

models_with_tools = model.bind_tools(list(TOOLS.values()))

messages = [
    SystemMessage(content="You are a helpful assistant that can perform calculations and apply discounts using available tools only."),
    HumanMessage(content="I am purchasing 2 tshirts that cost 2000rs each and 1 pair of jeans that costs 1500rs. What is the total price? What is the discounted price of the total price if I have a 10% discount?"),
    HumanMessage(content="What is the current weather in Bengaluru?"),
]

print("Initial Messages:", messages)

while True:
    response = models_with_tools.invoke(messages)
    # print(response)
    # print("====================")

    if not response.tool_calls:
        break
        
    messages.append(response)

    for tool_call in response.tool_calls:
        # print(tool_call)
        # print("====================")

        if tool_call.get("name") in TOOLS:
            result = TOOLS[tool_call.get("name")].invoke(tool_call.get("args"))
            # print(f"Result: {result}")
            # print("====================")
            messages.append(ToolMessage(content=result, tool_call_id=tool_call.get("id")))

    print("Updated Messages:", messages)
    print("====================")

print("Final Messages:", messages)