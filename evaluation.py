import os
import time

import pandas as pd

from knowledge_graph import retrieve_related_modules
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
    explicit_hit_count = 0
    fuzzy_hit_count = 0
    is_explicit_search = True
    total = len(cases)

    # Initialize counters for each match type
    direct_match_count = 0
    full_chain_count = 0
    explicit_doc_search_count = 0
    fuzzy_doc_search_count = 0

    for index, case in cases.iterrows():
        query = case["Evaluation Question"]
        # In this case, one query is related to only one code, not considering questions like comparison
        module_code = case["Module Code"]

        print(f"---- Running {index} / {total} ----")
        print("query: " + query)
        print("expected: " + module_code)

        module_codes = detect_module_code(query)
        # !!! Solution: Detect module code and fallback to full chain if false
        is_direct_match = False
        is_direct_match = True
        # is_knowledge_graph = False
        is_knowledge_graph = True

        if is_direct_match:
            if module_codes:
                print("Direct match")
                direct_match_count += 1
                is_explicit_search = True
                explicit_doc_search_count += 1  # Direct match is also considered a explicit doc search
            else:
                is_direct_match = False

        if not is_direct_match:
            print("Full chain")
            full_chain_count += 1
            docs = retriever.get_relevant_documents(query)

            # As the get_relevant_documents return all documents retrieved by each child retriever (each return 4)
            # So must filter the top 4 from all documents so the weight will really work
            docs = docs[:4]
            # docs = docs[:1]

            # !!! Solution: Rerank
            docs = rerank_results({"docs": docs, "query": query})

            # If asking a specific module code, then only match 1
            if len(module_codes) > 0:
                # 56
                print("Explicit Doc Search")
                is_explicit_search = True
                explicit_doc_search_count += 1
                docs = docs[:1]
            # Else give more freedom
            else:
                # 44
                print("Fuzzy Doc Search")
                is_explicit_search = False
                fuzzy_doc_search_count += 1

            module_codes = extract_module_codes(docs)
            # !!! Solution: knowledge graph
            if is_knowledge_graph:
                module_codes.extend(retrieve_related_modules(query))
            print("result:" + str(module_codes))
        if module_code in module_codes:
            print("Match found")
            if is_explicit_search:
                explicit_hit_count += 1
            else:
                fuzzy_hit_count += 1
            hit_count += 1

    print("---- Evaluation Summary ----")
    # print(f"Hit count: {hit_count}")
    # print(f"Total: {total}")
    print(f"Accuracy: {hit_count / total}")
    print(f"Time taken: {time.time() - start_time}")
    # print("---- Match Type Counts ----")
    print(f"Direct match count: {direct_match_count}")
    print(f"Full chain count: {full_chain_count}")
    print(f"Explicit doc search count: {explicit_doc_search_count}, hit count: {explicit_hit_count}, rate: {explicit_hit_count / explicit_doc_search_count:.3f}")
    print(f"Fuzzy doc search count: {fuzzy_doc_search_count}, hit count: {fuzzy_hit_count}, rate: {fuzzy_hit_count / fuzzy_doc_search_count:.3f}")


def extract_module_codes(documents):
    module_codes = []
    # pattern = r"The module code is (\w+)"
    # !!! Solution: For Rewriting
    pattern = r"Module Code: (\w+)"

    for document in documents:
        match = re.search(pattern, document.page_content)
        if match:
            module_codes.append(match.group(1))

    return module_codes


if __name__ == '__main__':
    # this is to quite parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
