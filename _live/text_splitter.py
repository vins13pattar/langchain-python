from langchain_text_splitters import CharacterTextSplitter
from langchain_docling.loader import DoclingLoader
from langchain_docling.loader import ExportType
from pathlib import Path

import os
from dotenv import load_dotenv
load_dotenv()


pdf_path = Path("./2408.09869v5.pdf")

loader = DoclingLoader(pdf_path, export_type=ExportType.MARKDOWN)
docs = loader.load()

text_content = []
all_text = ""
for doc in docs:
    text_content.append(doc.page_content)
    all_text += doc.page_content

print(len(all_text))
print(all_text)
print("==============")
print("==============")
print("==============")


# text_splitter = CharacterTextSplitter(
#     separator="\n\n", # the only thing splits on
#     chunk_size=600,
#     chunk_overlap=100,
#     length_function=len,
#     is_separator_regex=False,
# )

# texts = text_splitter.create_documents(text_content)

# for text in texts:
#     print("==============")
#     print(text)
#     print("==============")

# text_splitter2 = CharacterTextSplitter.from_tiktoken_encoder(
#     encoding_name="cl100k_base", chunk_size=600, chunk_overlap=0
# )
# texts = text_splitter2.split_text(all_text)

# for text in texts:
#     print("============")
#     print(text)
#     print("============")
    

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


# recursive_text_splitter = RecursiveCharacterTextSplitter(
#     separators=["\n\n", "\n", ".", " "],
#     chunk_size=600,
#     chunk_overlap=100,
#     length_function=len,
#     is_separator_regex=False,
# )

# chunks = recursive_text_splitter.split_text(all_text)

# for text in chunks:
#     print("==========")
#     print(text)
#     print("==========")


CODE_SAMPLE = '''

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

'''


# recursive_text_splitter = RecursiveCharacterTextSplitter.from_language(
#     language=Language.PYTHON,
#     chunk_size=600,
#     chunk_overlap=100,
# )

# chunks = recursive_text_splitter.split_text(CODE_SAMPLE)

# for text in chunks:
#     print("========")
#     print(text)
#     print("========")



from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)

chunks = markdown_splitter.split_text(all_text)

for text in chunks:
    print("========")
    print(text)
    print("========")


