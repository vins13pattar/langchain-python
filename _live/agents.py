from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
load_dotenv()
from langgraph.stream import StreamTransformer
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder


# Tools

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
    
@tool
def discount_calculator(price: float, discount: float) -> str:
    """Calculate the discounted price."""
    try:
        discounted_price = price * (1 - discount / 100)
        return f"The discounted price is: {discounted_price:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"
    

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # This is a placeholder implementation. In a real application, you would call a weather API.
    if location.lower() == "bengaluru":
        return "The current weather in Bengaluru is sunny with a temperature of 30°C."
    else:
        return f"Weather information for {location} is not available."

import uuid

username = "microdegree_user_001"

# Scopes conversation
NEW_THREAD = {"configurable": {"thread_id": f"{username}_{str(uuid.uuid4())}"}}
NEW_THREAD2 = {"configurable": {"thread_id": f"{username}_{str(uuid.uuid4())}"}}

checkpointer = MemorySaver()

agent = create_agent(
    name="MyAgent",
    model="openai:gpt-5.5",
    tools=[calculate, discount_calculator, get_weather],
    checkpointer=checkpointer
)

chat_template = ChatPromptTemplate.from_messages([
   ("system", "You are a helpful assistant. You are a expert in subject {subject}"),
   MessagesPlaceholder(variable_name="messages_history"), # Will get replaced by a list of messages
])

messages_history = [
    ("human", "what is the capital of France"),
    ("ai", "The capital of France is Paris"),
    ("human", "What is the population?"),
   ("human", "{question}"),
]

chat_prompt_value = chat_template.invoke({
    "subject": "Physics",
    "messages_history": messages_history,
    "question": "What is the weather in bengaluru?",
})

response = agent.invoke({
    "messages": chat_prompt_value.messages,
},  config=NEW_THREAD)

for message in response['messages']:
    print("----------------")
    print(message.content)







    


