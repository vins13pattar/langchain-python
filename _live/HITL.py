from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

load_dotenv()

@tool
def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

@tool
def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

skip = True

agent = create_agent(
    model="gpt-5.4-mini",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "read_email_tool": False,
            }
        ),
    ],
)

THREAD = {"configurable": {"thread_id": "thread_001"}}

result = agent.invoke({
    "messages":[
        HumanMessage(content="Send an email to team@acme.com with subject 'Q4 Results' saying revenue grew 15% this quarter"),
    ]
}, config=THREAD, version="v2")

print("\nMessages\n", result.value.get("messages"))
print("\nInterrupts\n", result.interrupts)



while result.interrupts:
    interrupt = result.interrupts[0]
    print("\nAction Requests: ", interrupt.value["action_requests"])
    print("\nReview Configs: ", interrupt.value["review_configs"])

    allowed_decisions = []
    for rc in interrupt.value.get("review_configs", []):
        for d in rc.get("allowed_decisions", []):
            if d not in allowed_decisions:
                allowed_decisions.append(d)
                
    options_str = "/".join(allowed_decisions) if allowed_decisions else "approve/edit/reject"

    if(skip):
        decision="approve"
    else:
        decision = input(f"Enter decision ({options_str}): ").strip()


    if(decision == "reject"):
        print("Rejecting the tool call")
    

    result = agent.invoke(
        Command(
            resume={"decisions": [{"type": decision}]}
        ), config=THREAD, version="v2")

    print("\nResult after approval: ", result)
    print("\nMessages after approval: ", result.value.get("messages"))

    
