from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
import os
load_dotenv()

@tool
def read_file(path: str) -> str:
    """Read a file from the filesystem."""
    print("Read file tool used")
    with open(path, "r") as f:
        return f.read()

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    print("Write file tool used")
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {path}"

@tool
def run_tests(path: str) -> str:
    """Run tests in a given directory."""
    print("Run tests tool used")
    return f"Ran tests in {path}"

CHAT_THREAD = {"configurable": {"thread_id": "thread_001"}}


model = init_chat_model(
    model="gemma-4-e2b",
    model_provider="openai",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

agent = create_agent(
    model=model,
    tools=[read_file, write_file, run_tests],
    checkpointer=InMemorySaver(),
    middleware=[TodoListMiddleware()],
)

result = agent.invoke({"messages": [{"role":"user", "content":"Write a function in a file named app.py and add a calulator function in it. Also, test the python function from that file."}]}, config=CHAT_THREAD)

state = agent.get_state(CHAT_THREAD)
print("\n\n State \n\n", state)
print("\n\n Todos \n\n", state.values.get("todos"))