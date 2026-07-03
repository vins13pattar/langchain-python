from langchain.agents import create_agent
from langchain.agents.middleware import FilesystemFileSearchMiddleware
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
    model=gemma_model,
    tools=[],
    middleware=[
        FilesystemFileSearchMiddleware(
            root_path="/Users/vinod/Projects/MicroDegree/Langchain/_live",
            use_ripgrep=True,
        ),
    ],
    system_prompt="You are a helpful assistant that can use tools to answer user queries."
)

result = agent.invoke({"messages":[{"role":"user", "content": "List down all the python files in the current project"}]})

print(result)