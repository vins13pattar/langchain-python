from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    number: int
    result: str


# Node
def double(state:State):
    print(f"Doubling the number: {state['number']} * 2")
    return {"number": state["number"] * 2}

# Node
def check(state: State):
    print(f"Checking if the number is greater than 10: {state['number']} > 10")
    return {"result": "big" if state["number"] > 10 else 'small'}

#condition route -> should return node names
def route(state: State):
    print(f"Routing based on number: {state['number']} > 10")
    return "check" if state["number"] > 10 else "double"

#Edges
builder = StateGraph(State)

builder.add_node("double", double)
builder.add_node("check", check)
builder.add_conditional_edges("double", route, ["check", "double"])
builder.add_edge(START, "double")
builder.add_edge("double", "check")
builder.add_edge("check", END)

graph = builder.compile()

# START -> double -> check -> END

result = graph.invoke({"number": 3})

print(result)