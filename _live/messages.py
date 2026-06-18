from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage, ToolMessage)


messages = [
    SystemMessage(content="You are a helpful assistant that can perform calculations and apply discounts using available tools only."),
    HumanMessage(content="I am purchasing 2 tshirts that cost 2000rs each and 1 pair of jeans that costs 1500rs. What is the total price? What is the discounted price of the total price if I have a 10% discount? What is the weather like today in Bengaluru?"),
    AIMessage(content="To calculate the total price, we can use the following formula: Total Price = (Price of tshirts * Quantity) + (Price of jeans * Quantity). For the discounted price, we can use: Discounted Price = Total Price * (1 - Discount Percentage / 100). For the weather, I will check the current weather in Bengaluru."),
    ToolMessage(content="The current weather in Bengaluru is sunny with a temperature of 30°C.", tool_call_id="12345"),
    AIMessage(content="The total price for 2 tshirts and 1 pair of jeans is 5500rs. The discounted price with a 10% discount is 4950rs. The current weather in Bengaluru is sunny with a temperature of 30°C."),
    HumanMessage(content="What is the current weather in Mysore?"),
]

