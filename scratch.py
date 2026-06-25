from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv
import random

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """Eval"""
    return str(eval(expression))

@tool
def dice_roll(num_dice: int = 1) -> str:
    """Roll"""
    return str([random.randint(1, 6) for _ in range(num_dice)])

checkpointer = InMemorySaver()
store = InMemoryStore()
CHAT_MESSAGES = {"configurable": {"thread_id": "chat_001"}}

messages_history = [
    HumanMessage(content="Hi, can you help me with some math?"),
    AIMessage(content="Of course! I can help you with math. What do you need to calculate?"),
    HumanMessage(content="What is 15 * 8?"),
    AIMessage(content="Let me calculate that for you.", tool_calls=[{"name": "calculator", "args": {"expression": "15 * 8"}, "id": "call_123"}]),
    ToolMessage(content="The result of 15 * 8 is 120", tool_call_id="call_123"),
    AIMessage(content="The result of 15 * 8 is 120. Is there anything else you need help with?"),
    HumanMessage(content="Yes, what is 120 / 4?"),
    AIMessage(content="Calculating 120 / 4...", tool_calls=[{"name": "calculator", "args": {"expression": "120 / 4"}, "id": "call_456"}]),
    ToolMessage(content="The result of 120 / 4 is 30.0", tool_call_id="call_456"),
    AIMessage(content="The result of 120 / 4 is 30.0."),
    HumanMessage(content="Thanks! Now I have a different question."),
    AIMessage(content="Sure, what's on your mind?"),
    HumanMessage(content="Toss a dice for me and tell me what did you get")
]

agent = create_agent(
    model="gpt-5.4-mini",
    tools=[calculator, dice_roll],
    system_prompt="You are a helpful assistant that can use tools to answer user queries.",
    checkpointer=checkpointer,
    store=store,
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=("messages", 10),
            keep=("messages", 4)
        )
    ]
)

result = agent.invoke({"messages": messages_history}, config=CHAT_MESSAGES)

print("len is", len(result["messages"]))
for msg in result["messages"]:
    print(msg.type, ":", msg.content[:50])

