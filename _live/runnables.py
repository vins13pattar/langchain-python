# Runnables - Composible layers 

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.documents import Document

# RunnablePassThrough -> pass input unchanged
# RunnableLambda -> Wraps python callables and passes output of callable to next component
# RunnableParallel -> Used run branches in parallel and combines the results in a dict


load_dotenv()

model = ChatOpenAI(
    model="gpt-5.5",
    temperature=0.7,
    max_retries=3,
)

# Mock data retrieve from vector database
def retrieve_knowledge(query: str) -> list[Document]:
    # do some processing on query and return documents from vector database
    print("Retrieved")
    return [
        Document(page_content="The capital of France is Paris.", metadata={"source": "kb1.pdf", "page_number": 1}),
        Document(page_content="The population of France is approximately 65 million.", metadata={"source": "kb2.pdf", "page_number": 2}),
        Document(page_content="LangChain is a framework for developing applications powered by language models.", metadata={"source": "langchain_docs.txt", "topic": "AI"}),
        Document(page_content="Runnables are a core abstraction in LCEL (LangChain Expression Language).", metadata={"source": "langchain_docs.txt", "topic": "AI"}),
        Document(page_content="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.", metadata={"source": "travel_guide.md", "topic": "Travel"}),
    ]


# format documents
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the context provided. If answer is not in the context, say 'I don't know'."),
    ("human", "Context: {context}\n\nQuestion: {question}"),
])

runnable_parallel = RunnableParallel({
    "context": RunnableLambda(retrieve_knowledge) | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})
    
rag_chain =  runnable_parallel | rag_prompt | model | StrOutputParser()

response = rag_chain.invoke({"question": "Who is PM of India"})

print(response)