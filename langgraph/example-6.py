# HITL

from typing import TypedDict, Literal, Annotated

from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()



class State(TypedDict):
    """
    This is the state of the application.
    """
    messages: Annotated[list, add_messages]
    body: str
    to: str
    subject: str
    status: Literal["pending", "approved", "rejected"]

llm = ChatOpenAI(model="gpt-5")


def draft_email(state: State) -> State:
    result = llm.invoke(state["messages"])
    body = result.content
    return {"messages": [result], "body": body, "to": "example.example", "subject": "test"}



def ask_human(state: State) -> Command[Literal["send_email", "rejected_email"]]:
    # user_input = interrupt("Approve sending this email (yes/no)")
    user_input = input(f"Approve sending this email (yes/no):\n{state['to']}\n\n{state['subject']}\n\n{state['body']}\n\n\n: ")

    if user_input.lower() == "yes":
        return Command(update={"status": "approved"}, goto="send_email")
    else:
        return Command(update={"status": "rejected"}, goto="rejected_email")

def send_email(state: State) -> State:
    """Mock function to send an email."""
    to, subject, body = state["to"], state["subject"], state["body"]
    return {"messages": [AIMessage(content=f"Email sent to {to} with subject '{subject}' and body '{body}'")]}

def rejected_email(state: State) -> State:
    return {"messages": [AIMessage(content="Email sending rejected")]}


builder = StateGraph(State)

builder.add_node("draft_email", draft_email)
builder.add_node("ask_human", ask_human)
builder.add_node("send_email", send_email)
builder.add_node("rejected_email", rejected_email)

builder.add_edge(START, "draft_email")
builder.add_edge("draft_email", "ask_human")
builder.add_edge("send_email", END)
builder.add_edge("rejected_email", END)


graph = builder.compile(checkpointer=MemorySaver())

thread= {"configurable": {"thread_id": "1"}}

response = graph.invoke({"messages": [HumanMessage(content="Create a email draft with a dummy reports on sales of Q1 (only body content)")]}, thread)

print(response)
    