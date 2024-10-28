import os
import time

import pandas as pd
from streamlit_app import get_retriever
from rag_chain import rerank_results, detect_module_code
import re


def load_excel_file(file_path):
    df = pd.read_excel(file_path)
    return df


def main():
    cases = load_excel_file("data/test.xlsx")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    retriever = get_retriever(openai_api_key=openai_api_key)

    start_time = time.time()
    hit_count = 0
    total = len(cases)
    for index, case in cases.iterrows():
        query = case["Evaluation Question"]
        module_code = case["Module Code"]

        print(f"---- Running {index} / {total} ----")
        print("query: " + query)
        print("expected: " + module_code)

        module_codes = []
        if len(detect_module_code(query)) > 0:
            print("Direct match")
            module_codes = detect_module_code(query)
        else:
            print("Full chain")
            docs = retriever.get_relevant_documents(query)

            # As the get_relevant_documents return all documents retrieved by each child retriever (each return 4)
            # So must filter the top 4 from all documents so the weight will really work
            docs = docs[:4]

            reranked_results = rerank_results({"docs": docs, "query": query})

            reranked_results = reranked_results[:1]

            module_codes = extract_module_codes(reranked_results)
            # module_codes = extract_module_codes(docs)
            print("result:" + str(module_codes))

        if module_code in module_codes:
            print("Match found")
            hit_count += 1

    print("---- Evaluation Summary ----")
    print(f"Hit count: {hit_count}")
    print(f"Total: {total}")
    print(f"Accuracy: {hit_count / total}")
    print(f"Time taken: {time.time() - start_time}")


def extract_module_codes(documents):
    module_codes = []
    pattern = r"The module code is (\w+)"

    for document in documents:
        match = re.search(pattern, document.page_content)
        if match:
            module_codes.append(match.group(1))

    return module_codes


if __name__ == '__main__':
    # this is to quite parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
