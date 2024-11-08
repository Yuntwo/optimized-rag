import gc
import logging
import os
import pickle
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from sklearn.decomposition import PCA
from dotenv import load_dotenv
from time import sleep
import numpy as np
from langchain.embeddings.base import Embeddings

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


class LocalEmbeddings(Embeddings):
    def __init__(self, local_embeddings):
        self.local_embeddings = local_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sleep(EMBED_DELAY)
        return self.local_embeddings

    def embed_query(self, text: str) -> List[float]:
        sleep(EMBED_DELAY)
        return self.local_embeddings


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


def create_low_dimension_vector_db(low_dim_collection_name="low",
                                   high_dim_collection_name="chroma",
                                   embeddings=None,
                                   reduced_dim=10, pca_model_file="pca_model.pkl", batch_size=100):
    print("Creating low dimension vector db...")
    if not embeddings:
        # To use HuggingFace embeddings instead:
        # from langchain_community.embeddings import HuggingFaceEmbeddings
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        openai_api_key = os.environ["OPENAI_API_KEY"]
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")

    proxy_embeddings = EmbeddingProxy(embeddings)
    high_dim_db = Chroma(collection_name=high_dim_collection_name,
                         embedding_function=proxy_embeddings,
                         persist_directory=os.path.join("store/", high_dim_collection_name))

    # Retrieve all embeddings and IDs from the high-dimensional collection
    records = high_dim_db._collection.get(limit=high_dim_db._collection.count(),
                                          include=['embeddings', 'metadatas', 'documents'])
    dimension_embeddings = records['embeddings']
    ids = records['ids']
    documents = records['documents']
    metadatas = records['metadatas']

    if not dimension_embeddings:
        print("No embeddings found.")
        return
    dimension_embeddings = np.array(dimension_embeddings)

    pca = PCA(n_components=reduced_dim)
    pca.fit(dimension_embeddings)

    # Save the PCA model
    with open(pca_model_file, "wb") as f:
        pickle.dump(pca, f)
    print(f"PCA model saved to {pca_model_file}")

    # Apply PCA transformation to reduce the dimensionality
    reduced_embeddings = pca.fit_transform(dimension_embeddings).tolist()

    local_embeddings = LocalEmbeddings(local_embeddings=reduced_embeddings)

    # Create a new low-dimensional vector database
    low_dim_db = Chroma(collection_name=low_dim_collection_name,
                        embedding_function=local_embeddings,
                        persist_directory=os.path.join("store/", low_dim_collection_name))

    # Process reduced embeddings in batches
    print(f"Total documents: {len(documents)}", flush=True)
    low_dim_db.add_texts(texts=documents, metadatas=metadatas, ids=ids)

    # Persist changes periodically
    low_dim_db.persist()

    print(f"Reduced embeddings added to the low-dimensional database '{low_dim_collection_name}'")
    return low_dim_db


# Store into db in batch size to avoid exhausting memory
def create_vector_db(texts, embeddings=None, collection_name="chroma", batch_size=100):
    print("Creating original dimension vector db...")
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

    vs = get_vector_db()

    query = """
    Module Code: CS4002
    Title: Exchange CS Module
    Description:
    Credits: 4
    Department: SoC Dean's Office
    Faculty: Computing
    Workload: N/A
    Semester Data: []
    """

    results = find_similar(vs, query)
    MAX_CHARS = 300
    print("=== Results ===")
    for i, text in enumerate(results):
        # cap to max length but split by words.
        content = text.page_content
        n = max(content.find(' ', MAX_CHARS), MAX_CHARS)
        content = text.page_content[:n]
        print(f"Result {i + 1}:\n {content}\n")
    metadata = results[0].metadata
    print(metadata)


if __name__ == "__main__":
    main()
