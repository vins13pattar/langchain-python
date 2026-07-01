from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

CHAT_THREAD = {"configurable": {"thread_id": "chat_001"}}

agent = create_agent(
    model="gpt-5.5",
    checkpointer=InMemorySaver(),  # Required for thread limiting
    tools=[],
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=2,
            run_limit=1,
            exit_behavior="end",
        ),
    ],
)

# First run
result = agent.invoke(
    {"messages":[{"role":"user", "content":"What is the capital of France?"}]},
    config=CHAT_THREAD
)
print("\n Result 1: ", result["messages"][-1].content)



# Second run (should be blocked due to thread_limit=1)
result2 = agent.invoke(
    {"messages":[{"role":"user", "content":"What is the capital of India?"}]},
    config=CHAT_THREAD
)

print("\n Result 2: ", result2["messages"][-1].content)