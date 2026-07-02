from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
import os
load_dotenv()



@tool
def tool1(expression: str) -> str:
    """ This is a calculator tool, use this for calculations. """
    try:
        # Safe eval using limited builtins
        result = eval(expression, {"__builtins__": None}, {})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating {expression}: {e}"

@tool
def tool2(num_dice: int = 1) -> str:
    """ This is a dice roll tool, use this to roll dice. """
    import random
    rolls = [random.randint(1, 6) for _ in range(num_dice)]
    return f"You rolled {num_dice} dice and got: {rolls}"

@tool
def tool3(to_address: str, subject: str, body: str) -> str:
    """ This is a send email tool, use this to send email. """
    return f"Email successfully sent to {to_address} with subject '{subject}'"

@tool
def tool4(city: str) -> str:
    """ This is a weather tool, use this to get weather for a city. """
    weather_data = {
        "london": "rainy, 15°C",
        "new york": "sunny, 22°C",
        "tokyo": "cloudy, 18°C",
        "hyderabad": "humid, 32°C",
        "bengaluru": "pleasant, 24°C"
    }
    city_lower = city.lower().strip()
    return f"The weather in {city} is {weather_data.get(city_lower, 'sunny, 25°C')}."

@tool
def tool5(topic: str) -> str:
    """ This is a news tool, use this to get news for a topic. """
    return f"Top news for {topic}: A breakthrough has been reported in the field of {topic}."


@tool
def search(query: str) -> str:
    """Search the web for a query"""
    return f"Search results for {query}"

main_model = init_chat_model(
    model="gemma-4-e2b",
    model_provider="openai",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

tool_selection_model = init_chat_model(
    model="gemma-4-e2b",
    model_provider="openai",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

agent = create_agent(
    model=main_model,
    tools=[tool1, tool2, tool3, tool4, tool5, search],
    checkpointer=InMemorySaver(),
    middleware=[
        LLMToolSelectorMiddleware(
            model=tool_selection_model,
            max_tools=3,
            always_include=["search"],
            system_prompt=(
                "Your goal is to select the most relevant tools for answering the user's query.\n"
                "Here are the mappings of tools to their functions:\n"
                "- tool1 is a calculator tool for math/calculations.\n"
                "- tool2 is a dice roll tool for rolling dice.\n"
                "- tool3 is an email tool to send emails.\n"
                "- tool4 is a weather tool to get weather for a city.\n"
                "- tool5 is a news tool to get news for a topic."
            )
        ),
    ],
    system_prompt="You are a helpful agent and answer user queries using only tools."
)

CHAT_THREAD = {"configurable": {"thread_id": "thread_001"}}

queries = [
    "Roll 3 dice",
    "What is 15 * 8?",
    "Send an email to john@example.com with subject hello and body world",
    "What's the weather in Tokyo?",
    "Get the latest news about AI"
]

for idx, query in enumerate(queries):
    config = {"configurable": {"thread_id": f"thread_test_{idx}"}}
    print(f"\n--- Testing query: '{query}' ---")
    res = agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)
    for msg in res["messages"]:
        print(f"[{msg.__class__.__name__}]: {repr(msg.content)}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"  Tool Calls: {msg.tool_calls}")