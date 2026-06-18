from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=128
)

text = ["Gen AI Developers", "DS/ML Engineers", "Full Stack Developers", "Doctor", "Lawyer", "Farmer"]

embedding_vector = embeddings.embed_documents(text)

print(len(embedding_vector))
print(len(embedding_vector[0]))
print("=======")
print(embedding_vector[0])
print("=======")
print(embedding_vector[1])
print("=======")
print(embedding_vector[2])
