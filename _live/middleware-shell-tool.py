from langchain.agents import create_agent
from langchain.agents.middleware import (
    ShellToolMiddleware,
    HostExecutionPolicy,
    DockerExecutionPolicy,
    RedactionRule,
)
from dotenv import load_dotenv

load_dotenv()



# Basic shell tool with host execution
agent = create_agent(
    model="anthropic:claude-haiku-4-5-20251001",
    # tools=[search_tool],
    middleware=[
        ShellToolMiddleware(
            workspace_root="/Users/vinod/Projects/MicroDegree/Langchain/",
            startup_commands=["source .venv/bin/activate", "pip install requests", "export FilePath=/Users/vinod/Projects/MicroDegree/Langchain/"],
            execution_policy=HostExecutionPolicy(),
        ),
    ],
)

CHAT_THREAD = {"configurable": {"thread_id": "thread_004"}}

result = agent.invoke({"messages":[{"role":"user", "content": "List down all the python files in the current project"}]}, config=CHAT_THREAD)

print(result)

