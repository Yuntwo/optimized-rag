import gc
import logging
import os
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from local_loader import get_document_text
from remote_loader import download_file
from splitter import split_documents
from dotenv import load_dotenv
from time import sleep

EMBED_DELAY = 0.02  # 20 milliseconds


# This is to get the Streamlit app to use less CPU while embedding documents into Chromadb.
class EmbeddingProxy:
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_query(text)


# load db from persistent db
# 14439 documents in total
def get_vector_db(collection_name="chroma"):
    openai_api_key = os.environ["OPENAI_API_KEY"]
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")

    proxy_embeddings = EmbeddingProxy(embeddings)
    db = Chroma(collection_name=collection_name,
                embedding_function=proxy_embeddings,
                persist_directory=os.path.join("store/", collection_name))
    return db


# Store into db in batch size to avoid exhausting memory
def create_vector_db(texts, embeddings=None, collection_name="chroma", batch_size=100):
    if not texts:
        logging.warning("Empty texts passed in to create vector database")
    # Select embeddings
    if not embeddings:
        # To use HuggingFace embeddings instead:
        # from langchain_community.embeddings import HuggingFaceEmbeddings
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        openai_api_key = os.environ["OPENAI_API_KEY"]
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")

    proxy_embeddings = EmbeddingProxy(embeddings)
    # Create a vectorstore from documents
    # this will be a chroma collection with a default name.
    db = Chroma(collection_name=collection_name,
                embedding_function=proxy_embeddings,
                persist_directory=os.path.join("store/", collection_name))

    # Process texts in batches
    print(f"Texts length: {len(texts)}\n", flush=True)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Use 'moduleCode' in Document metadata as id, it's ensured unique
        batch_ids = [doc.metadata['moduleCode'] for doc in batch if 'moduleCode' in doc.metadata]
        # Ensure the length of batch_ids and batch matches
        if len(batch_ids) != len(batch):
            raise ValueError("Not all documents have 'moduleCode' in their metadata.")
        # print(f"Generated embeddings: {embeddings}")
        db.add_documents(documents=batch, ids=batch_ids)
        # db.add_documents(documents=batch)
        print(f"Added batch {i}\n", flush=True)
        # db.add_texts(texts=[doc.page_content for doc in batch], ids=batch_ids)
        db.persist()
        del batch
        gc.collect()

    return db


def find_similar(vs, query):
    docs = vs.similarity_search(query)
    return docs


def main():
    load_dotenv()

    pdf_filename = "examples/mal_boole.pdf"

    if not os.path.exists(pdf_filename):
        math_analysis_of_logic_by_boole = "https://www.gutenberg.org/files/36884/36884-pdf.pdf"
        local_pdf_path = download_file(math_analysis_of_logic_by_boole, pdf_filename)
    else:
        local_pdf_path = pdf_filename

    print(f"PDF path is {local_pdf_path}")

    with open(local_pdf_path, "rb") as pdf_file:
        docs = get_document_text(pdf_file, title="Analysis of Logic")

    texts = split_documents(docs)
    vs = create_vector_db(texts)

    results = find_similar(vs, query="What is meant by the simple conversion of a proposition?")
    MAX_CHARS = 300
    print("=== Results ===")
    for i, text in enumerate(results):
        # cap to max length but split by words.
        content = text.page_content
        n = max(content.find(' ', MAX_CHARS), MAX_CHARS)
        content = text.page_content[:n]
        print(f"Result {i + 1}:\n {content}\n")


if __name__ == "__main__":
    main()
