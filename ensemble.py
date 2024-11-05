import os

from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
from langchain.retrievers import EnsembleRetriever
from vector_store import create_vector_db, get_vector_db
from dotenv import load_dotenv
from optimized_retriever.full_text_retriever import CustomTFIDFRetriever


# Hybrid search for nus mods
def ensemble_retriever_from_mods(mods, embeddings=None):
    vs = get_vector_db()
    # vs = create_vector_db(mods, embeddings)
    vs_retriever = vs.as_retriever()

    # full_text_retriever = BM25Retriever.from_texts([t.page_content for t in mods])
    # full_text_retriever = TFIDFRetriever.from_texts([t.page_content for t in mods])
    full_text_retriever = CustomTFIDFRetriever.from_texts([t.page_content for t in mods])

    ensemble_retriever = EnsembleRetriever(
        retrievers=[full_text_retriever, vs_retriever],
        weights=[0.1, 0.9])

    return ensemble_retriever


def main():
    load_dotenv()
    print("To be generated")


if __name__ == "__main__":
    # this is to quite parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
