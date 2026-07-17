# LLM inside a node
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages] # appends instead of overrite

llm = ChatOpenAI(model="gpt-5")

def chatbot(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)


graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage(content="Hello")]})

print(result["messages"])


    
