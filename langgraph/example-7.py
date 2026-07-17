
import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    notes: Annotated[list, operator.add] # reducer for parallel writes

def search_web(state):
    print("Searching web for information about the query")
    return {"notes": ["Web search results"]}

def search_docs(state):
    print("Searching docs for information about the query")
    return {"notes": ["Docs search results"]}

def summarize(state):
    print("Summarizing the notes")
    return {"notes": ["Summary of the notes"]}

builder = StateGraph(State)

builder.add_node("search_web", search_web)
builder.add_node("search_docs", search_docs)
builder.add_node("summarize", summarize)

builder.add_edge(START, "search_web")
builder.add_edge(START, "search_docs")
builder.add_edge("search_web", "summarize")
builder.add_edge("search_docs", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()

result = graph.invoke({"notes": []})
