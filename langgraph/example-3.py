# Reducer - how state merges (add_messages)

from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages] # appends instead of overrite

def greet(state: State):
    return {"messages": [AIMessage(content="Hello")]}

def follow(state: State):
    return {"messages": [AIMessage(content="How can I help you>")]}

builder = StateGraph(State)

builder.add_node("greet", greet)
builder.add_node("follow", follow)
builder.add_edge(START, "greet")
builder.add_edge("greet", "follow")
builder.add_edge("follow", END)

graph = builder.compile()

result = graph.invoke({"messages": []})

print(result['messages'])