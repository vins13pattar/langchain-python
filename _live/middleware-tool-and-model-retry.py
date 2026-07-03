from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware, ModelRetryMiddleware

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
import random
load_dotenv()


# Mock tool that sometimes fails
failed_attempts = 0
MAX_FAILS = 2

@tool
def flaky_tool(input: str) -> str:
    """Use this tool to roll a dice and return the result"""
    global failed_attempts
    if failed_attempts < MAX_FAILS:
        failed_attempts += 1
        print(f"Flaky tool: Attempt {failed_attempts}/{MAX_FAILS} FAILED")
        raise Exception("Simulated tool failure")
    print(f"Flaky tool: Attempt {failed_attempts}/{MAX_FAILS} SUCCESS")
    return f"{random.randint(1,6)}"

model = init_chat_model(
    model="gemma-4-e2b",
    model_provider="openai",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

agent = create_agent(
    model="gpt-5.5",
    tools=[flaky_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
        ModelRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
    ],
)

result = agent.invoke(
    {"messages":[{"role":"user", "content":"roll a dice"}]}
)

print(result)