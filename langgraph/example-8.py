from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

def get_weather(city:str):
    """Get weather for a city"""
    return f"Weather in {city} is sunny"

agent = create_react_agent(
    tools=[get_weather],
    model=ChatOpenAI(model="gpt-5.4-mini"),
    prompt= ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Use the tools to answer the user's question."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ),
)

result = agent.invoke({"messages": [HumanMessage(content="What is the weather like in New York?")]})

print(result["messages"])