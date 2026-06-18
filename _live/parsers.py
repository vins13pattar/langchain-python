# LCEL - Langchain Expression Language
# Construct with "|" pipe operator

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser

load_dotenv()

model = ChatOpenAI(
    model="gpt-5.5",
    temperature=0.7,
    max_retries=3,
)

def print_banner(title: str):
    """Print banner with title."""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

# Output Parser

print_banner("String Output Parser")
prompt_template = PromptTemplate(
    template= "List 4 important concepts in {subject}.",
    input_variables=["subject"],
)


chain_str = prompt_template | model | StrOutputParser()

response = chain_str.invoke({"subject": "Langchain"})
print(response)

# JSON Output Parser

print_banner("JSON Output Parser")
prompt_template_with_json = PromptTemplate(
    template= "List 4 important concepts in {subject}. Reply output in a JSON format with title of the concept and then description of the concept.",
    input_variables=["subject"],
)
chain_json = prompt_template_with_json | model | JsonOutputParser()

response_json = chain_json.invoke({"subject": "Langchain"})
print(response_json)

for item in response_json:
    print(f"Title: {item['title']}\n")
    print(f"Description: {item['description']}\n")
    print("-" * 50 + "\n")

# Comma Separated List Output Parser

print_banner("Comma Separated List Output Parser")
prompt_template_comma_separated = PromptTemplate(
    template= "List 4 important concepts in {subject}. Reply output in a comma separated list and nothing else.",
    input_variables=["subject"],
)

chain_comma_separated = prompt_template_comma_separated | model | CommaSeparatedListOutputParser()

response_comma_separated = chain_comma_separated.invoke({"subject": "Langchain"})
print(response_comma_separated)

for item in response_comma_separated:
    print(f"{item}\n")




