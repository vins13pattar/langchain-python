from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage, ToolMessage)


# prompt_template = PromptTemplate(
#     template= "What is the weather in {city}?",
#     input_variables=["city"],
# )

# prompt_value = prompt_template.invoke({"city":  "bengaluru"})
# prompt_value2 = prompt_template.invoke({"city":  "Mysore"})
# prompt_value3 = prompt_template.invoke({"city":  "Mangalore"})
# prompt_value4 = prompt_template.invoke({"city":  "Hubli"})
# prompt_value5 = prompt_template.invoke({"city":  "Belgaum"})

# print(prompt_value.text)
# print(prompt_value2.text)
# print(prompt_value3.text)
# print(prompt_value4.text)
# print(prompt_value5.text)


chat_template = ChatPromptTemplate.from_messages([
   ("system", "You are a helpful assistant. You are a expert in subject {subject}"),
   MessagesPlaceholder(variable_name="messages_history"),
   ("human", "Question: {question}"),
])


# It is from persistant memory of the previous conversation
messages_history = [
    ("human", "what is the capital of France"),
    ("ai", "The capital of France is Paris"),
    ("human", "What is the population of France?"),
    ("ai", "The population of France is 65.27 million")
]

chat_prompt_value = chat_template.invoke({
    "subject": "Physics",
    "question": "What is the weather in bengaluru?",
    "messages_history": messages_history
})

print(chat_prompt_value)

print("======")

for message in chat_prompt_value.messages:
    print(message.content)
    print("=====")
