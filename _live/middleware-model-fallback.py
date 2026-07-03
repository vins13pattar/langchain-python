from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
load_dotenv()

gemma_model = init_chat_model(
    model="gemma-4-e2b",
    model_provider="openai",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)


agent = create_agent(
    system_prompt="""You are a helpful assistant.""",
    model="gpt-5.5",
    tools=[],
    middleware=[
        ModelFallbackMiddleware(
            gemma_model
        ),
    ],
)

result = agent.invoke({"messages":[{"role":"user", "content":"Tell me about yourself"}]})
print(result)