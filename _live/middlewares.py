# Middleware imports
from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ModelFallbackMiddleware,
    PIIMiddleware,
    TodoListMiddleware,
    LLMToolSelectorMiddleware,
    ToolRetryMiddleware,
    ModelRetryMiddleware,
    LLMToolEmulator,
    ContextEditingMiddleware,
    ClearToolUsesEdit,
    ShellToolMiddleware,
    FilesystemFileSearchMiddleware,
    HostExecutionPolicy,
)

import random
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
# import deepagents.middleware.subagents
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result"""
    print("Calculator tool is called....")
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def dice_roll(num_dice: int = 1) -> str:
    """Roll dice and return the result"""
    print("Dice roll tool is called....")
    rolls = [random.randint(1, 6) for _ in range(num_dice)]
    return f"You rolled {num_dice} dice and got: {rolls}"

@tool
def send_email(to_address:str, subject:str, body:str) -> str:
    """Send an email to the specified recipient"""
    print(f"Sending email to {to_address} with subject {subject}")
    # Send actual email
    return f"Email sent to {to_address} with subject {subject}" 


checkpointer = InMemorySaver()
store = InMemoryStore()
CHAT_MESSAGES = {"configurable": {"thread_id": "chat_001"}}

chat_template = ChatPromptTemplate.from_messages([
   MessagesPlaceholder(variable_name="messages_history"), # Will get replaced by a list of messages
])

messages = chat_template.invoke({
    "messages_history": [
        # Previous conversation history to demonstrate summarization
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
})


agent = create_agent(
    model="gpt-5.4-mini",
    tools=[calculator, dice_roll, send_email],
    checkpointer=checkpointer,
    system_prompt="You are a helpful assistant that can use tools to answer user queries.",
    store=store,
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=("messages", 10),
            keep=("messages", 4)
        )
    ]
)
print("\n Length of messages before summarization", len(messages.messages)+1, "\n")

result = agent.invoke(messages, config=CHAT_MESSAGES)

print("\n Length of messages after summarization", len(result["messages"])+1, "\n")



print("\nFinal messages post summarization\n")
for msg in result["messages"]:
    print("--------------------------------------\n")
    print(f"Message type: {msg.type}\n")
    print(f"Message content: {msg.content}\n")



        