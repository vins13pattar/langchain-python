# LCEL - Langchain Expression Language
# Construct with "|" pipe operator

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(
    model="gpt-5.5",
    temperature=0.7,
    max_tokens=2048,
    max_retries=3,
)

prompt_template = PromptTemplate(
    template= "What is the weather in {city}?",
    input_variables=["city"],
)

# print(prompt_template.invoke({"city": "Bengaluru"}))
# print(model.invoke(prompt_template.invoke({"city": "Bengaluru"})))
# print(StrOutputParser().invoke(model.invoke(prompt_template.invoke({"city": "Bengaluru"}))))


chain = prompt_template | model

response = chain.invoke({"city": "bengaluru"})

print(response)



