# The tool loop

from typing import TypedDict, Annotated

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()


llm = ChatOpenAI(model="gpt-5")


class State(TypedDict):
    messages: Annotated[list, add_messages]

def multiply(a: int, b: int) -> int:
    """
    This is a function to multiply two numbers
    """
    return a * b

llm_with_tools = llm.bind_tools([multiply])

def chatbot(state: State) -> State:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools=[multiply]))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition) # tools calls -> tools, else -> END
builder.add_edge("tools", "chatbot")
graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "1"}}

response = graph.invoke({"messages": [HumanMessage(content="multiply 5 with 10")]}, config=config)
print("\n")
print(response['messages'][-1].content)

response2 = graph.invoke({"messages": [HumanMessage(content="What number did we multiply?")]}, config=config)
print("\n")
print(response2['messages'][-1].content)


# This loops - LLM->tool->LLM->tool.....->done -> END

