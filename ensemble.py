import os

from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
from langchain.retrievers import EnsembleRetriever
from vector_store import create_vector_db, get_vector_db, create_low_dimension_vector_db, get_low_dimension_vector_db
from dotenv import load_dotenv
from optimized_retriever.full_text_retriever import KeywordTFIDFRetriever, TruncatedSVDTFIDFRetriever, PCATFIDFRetriever
import joblib


# Hybrid search for nus mods
def ensemble_retriever_from_mods(mods, embeddings=None):
    vs = get_vector_db(collection_name="raw")
    # vs = get_vector_db(collection_name="chroma")
    # vs = create_vector_db(mods, embeddings, collection_name="raw")
    # vs = create_low_dimension_vector_db(embeddings=embeddings, reduced_dim=30)
    # vs = get_low_dimension_vector_db()
    vs_retriever = vs.as_retriever()

    # full_text_retriever = BM25Retriever.from_texts([t.page_content for t in mods])
    # full_text_retriever = TFIDFRetriever.from_texts([t.page_content for t in mods])
    full_text_retriever = joblib.load('model/full_text_retriever_raw.pkl')
    # full_text_retriever = joblib.load('model/full_text_retriever_rewriting.pkl')
    # joblib.dump(full_text_retriever, 'model/full_text_retriever_raw.pkl')
    # full_text_retriever = KeywordTFIDFRetriever.from_texts([t.page_content for t in mods])
    # 37242
    # full_text_retriever = TruncatedSVDTFIDFRetriever.from_texts([t.page_content for t in mods], n_components=37242)
    # full_text_retriever = PCATFIDFRetriever.from_texts([t.page_content for t in mods], n_components=14439)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[full_text_retriever, vs_retriever],
        weights=[0.5, 0.5])

    # ensemble_retriever = EnsembleRetriever(
    #     retrievers=[vs_retriever],
    #     weights=[1])
    # ensemble_retriever = EnsembleRetriever(
    #     retrievers=[full_text_retriever],
    #     weights=[1])

    return ensemble_retriever


def main():
    load_dotenv()
    print("To be generated")


if __name__ == "__main__":
    # this is to quite parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
