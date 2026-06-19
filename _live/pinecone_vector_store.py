from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

pc = Pinecone()

index_name = "langchain"  # change if desired

from pinecone import ServerlessSpec

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)


documents = [
    Document(
        page_content="""
        LangChain is an open-source framework designed to build applications
        powered by Large Language Models (Large Language Models (LLMs) are deep learning models trained on vast
        amounts of text data.). It provides abstractions for
        prompts, models, chains, agents, tools, memory, and retrieval systems.

        Developers use LangChain to create chatbots, RAG applications,
        AI assistants, and autonomous agents.
        """,
        metadata={
            "title": "Introduction to LangChain",
            "category": "AI Framework",
            "source": "langchain_docs",
            "id": 1
        }
    ),

    Document(
        page_content="""
        Retrieval Augmented Generation (RAG) is a technique where an LLM
        retrieves relevant information from external knowledge sources before
        generating an answer.

        A typical RAG pipeline contains document loading, text splitting,
        embedding generation, vector storage, retrieval, and LLM generation.
        """,
        metadata={
            "title": "Understanding RAG",
            "category": "Generative AI",
            "source": "ai_notes",
            "id": 2
        }
    ),

    Document(
        page_content="""
        Vector databases store high-dimensional numerical representations
        called embeddings. These embeddings capture semantic meaning of text,
        images, or other data.

        Popular vector databases include FAISS, Chroma, Pinecone, and
        Weaviate. They are commonly used in semantic search and RAG systems.
        """,
        metadata={
            "title": "Vector Databases",
            "category": "Database",
            "source": "ml_guide",
            "id": 3
        }
    ),

    Document(
        page_content="""
        LangChain agents allow language models to interact with external tools.
        Instead of only generating text, an agent can decide which tool to call,
        execute actions, observe results, and continue reasoning.

        Examples include calculator agents, web search agents, database agents,
        and customer support automation agents.
        """,
        metadata={
            "title": "LangChain Agents",
            "category": "Agents",
            "source": "langchain_training",
            "id": 4
        }
    ),

    Document(
        page_content="""
        Embeddings are mathematical representations of text where similar
        meanings are placed closer together in vector space.

        Embedding models are trained using large datasets to understand
        semantic relationships between words, sentences, and documents.
        They are widely used for search, recommendations, clustering, and RAG.
        """,
        metadata={
            "title": "Embedding Models Explained",
            "category": "Machine Learning",
            "source": "ml_training",
            "id": 5
        }
    ),
    Document(
        page_content="""
        Artificial Intelligence (AI) refers to the simulation of human intelligence
        in machines. AI systems can perform tasks such as reasoning, learning,
        problem-solving, perception, and language understanding.

        Modern AI applications include chatbots, recommendation systems,
        autonomous vehicles, and predictive analytics.
        """,
        metadata={
            "title": "Introduction to Artificial Intelligence",
            "category": "Artificial Intelligence",
            "source": "ai_handbook",
            "id": 6
        }
    ),

    Document(
        page_content="""
        Machine Learning is a subset of AI that enables systems to learn from
        data without being explicitly programmed. Algorithms identify patterns
        in historical data and make predictions on new data.

        Common types include supervised learning, unsupervised learning,
        and reinforcement learning.
        """,
        metadata={
            "title": "Machine Learning Basics",
            "category": "Machine Learning",
            "source": "ml_course",
            "id": 7
        }
    ),

    Document(
        page_content="""
        Large Language Models (LLMs) are deep learning models trained on vast
        amounts of text data. They can generate human-like text, answer
        questions, summarize content, and assist with coding tasks.

        Examples include GPT, Llama, Gemini, and Claude models.
        """,
        metadata={
            "title": "Large Language Models",
            "category": "LLM",
            "source": "llm_guide",
            "id": 8
        }
    ),

    Document(
        page_content="""
        Prompt engineering is the practice of designing effective inputs for
        language models. Well-structured prompts improve response quality,
        accuracy, and consistency.

        Techniques include role prompting, chain-of-thought prompting,
        few-shot learning, and structured output prompting.
        """,
        metadata={
            "title": "Prompt Engineering Techniques",
            "category": "Prompt Engineering",
            "source": "prompting_manual",
            "id": 9
        }
    ),

    Document(
        page_content="""
        Fine-tuning is the process of adapting a pre-trained model to a
        specific domain or task using additional training data.

        Organizations use fine-tuning to customize models for customer support,
        healthcare, finance, legal analysis, and other specialized applications.
        """,
        metadata={
            "title": "Model Fine-Tuning",
            "category": "Model Training",
            "source": "ai_research",
            "id": 10
        }
    )
]

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

pinecone_vector_store = PineconeVectorStore(
    embedding=embeddings,
    index=index,
)

pinecone_vector_store.add_documents(documents=documents, ids=[str(doc.metadata["id"]) for doc in documents])

searched_docs = pinecone_vector_store.similarity_search(
    query="What is LangChain?",
    k=3
)

for doc in searched_docs:
    print("="*50)
    print(doc.page_content)
    print("="*50)
    print(doc.metadata)
    print("="*50)
    print()
    print()