from langchain_docling.loader import DoclingLoader
from langchain_docling.loader import ExportType
from pathlib import Path

import os
from dotenv import load_dotenv
load_dotenv()


pdf_path = Path("./2408.09869v5.pdf")

FILE_PATH = "https://docs.langchain.com/oss/python/integrations/document_loaders/docling"

# loader = DoclingLoader(
#     file_path=FILE_PATH,
#     export_type=ExportType.MARKDOWN
#     # export_type=ExportType.DOC_CHUNKS -> default
# )

# docs = loader.load()

# for idx, doc in enumerate(docs):
#     print(f"Doc {idx}: {doc.page_content}")
#     print("-" * 50)


# from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader

# loader = OpenDataLoaderPDFLoader(
#     file_path=str(pdf_path),
#     format="text"
# )
# documents = loader.load()

# for doc in documents:
#     print(doc.metadata, doc.page_content[:80])

# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader(str(pdf_path))

# documents = loader.load()

# for doc in documents:
#     print("-------------------------------------------------------------------------")
#     print(doc.metadata, doc.page_content)



from langchain_community.document_loaders import FireCrawlLoader

loader = FireCrawlLoader(
    url="https://www.microdegree.work", mode="scrape",
)

docs = loader.load()

for doc in docs:
    print(doc.metadata, doc.page_content)