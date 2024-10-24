import os

from dotenv import load_dotenv
from langchain import hub
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage

from basic_chain import basic_chain, get_model
from remote_loader import get_wiki_docs
from splitter import split_documents
from vector_store import create_vector_db
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
import json


def rerank_results(inputs):
    """Rerank retrieved documents using a reranker model (BERT)."""
    retrieved_docs = inputs['docs']  # Extract the retrieved documents
    query = inputs['query']  # Extract the query

    # Initialize the BERT model for sequence classification
    reranker_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    reranked_docs = []

    # Loop over each document and rank it based on its relevance to the query
    for doc in retrieved_docs:
        # Extract document content from the `page_content` field
        text = doc.page_content

        # Tokenize both the query and the document text
        input_tokens = tokenizer(query, text, return_tensors='pt', truncation=True)

        # Use the model to generate a relevance score
        with torch.no_grad():
            outputs = reranker_model(**input_tokens)
            score = torch.sigmoid(outputs.logits[:, 1]).item()  # Get the positive classification score

        # Append the document and its score as a tuple
        reranked_docs.append((doc, score))

        # Log the document text and its score
        print(f"Document: {text[:100]}... | Score: {score}")

    # Sort the documents based on the score in descending order
    reranked_docs = sorted(reranked_docs, key=lambda x: x[1], reverse=True)

    # Log the top document
    if reranked_docs:
        top_doc = reranked_docs[0][0].page_content
        print(f"\nTop Document: {top_doc[:500]}...")  # Print a snippet of the top document
    else:
        print("No documents were retrieved.")

    # Return the top 1 document in the list
    return [doc[0] for doc in reranked_docs]


def find_similar(vs, query):
    docs = vs.similarity_search(query)
    return docs


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_question(input):
    if not input:
        return None
    elif isinstance(input, str):
        return input
    elif isinstance(input, dict) and 'question' in input:
        return input['question']
    elif isinstance(input, BaseMessage):
        return input.content
    else:
        raise Exception("string or dict with 'question' key expected as RAG chain input.")


def make_rag_chain(model, retriever, rag_prompt=None):
    # We will use a prompt template from langchain hub.
    if not rag_prompt:
        rag_prompt = hub.pull("rlm/rag-prompt")

    # And we will use the LangChain RunnablePassthrough to add some custom processing into our chain.
    rag_chain = (
            {
                # Step 1: Extract the query from the input
                "context": RunnableLambda(get_question)
                           # Step 2: Retrieve documents using the extracted query
                           | (lambda query: {"docs": retriever.get_relevant_documents(query), "query": query})
                           # Step 3: Pass the retrieved docs and query to the reranking step
                           | RunnableLambda(rerank_results)
                           # Step 4: Format the reranked documents
                           | format_docs,
                "question": RunnablePassthrough()
            }
            | rag_prompt
            | model
    )

    return rag_chain


def fetch_module_content(query):
    codes = detect_module_code(query)
    mods = load_and_chunk_json('data/mods22_23_test.json')
    contexts = []
    for code in codes:
        query = mods.get(code)
        if query:
            contexts.append(query.page_content)

    final_context = "\n\n".join(contexts)
    print(f"Fetched context: {final_context}")
    return final_context


def load_and_chunk_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    documents_map = {}
    for module in data:
        page_content = (
            f"The module code is {module['moduleCode']}, and the title of the module is {module['title']}. "
            f"{module['description']} "
            f"It offers {module['moduleCredit']} module credit and is provided by the {module['department']}, "
            f"under the {module['faculty']}. "
            f"The workload for this module includes {module.get('workload', 'N/A')}. "
            f"Students must meet the prerequisite, which states that they should be {module.get('prerequisite', 'N/A')} "
            f"in order to enroll in this module."
        )

        # Create a Document object for each module with metadata
        document = Document(
            page_content=page_content,
            metadata={
                "moduleCode": module['moduleCode'],
                "title": module['title'],
                "moduleCredit": module['moduleCredit'],
                "department": module['department'],
                "faculty": module['faculty']
            }
        )

        # Store the document in the dictionary with moduleCode as key
        documents_map[module['moduleCode']] = document

    return documents_map


def make_direct_chain(model, rag_prompt):
    # And we will use the LangChain RunnablePassthrough to add some custom processing into our chain.
    rag_chain = (
            {
                # Step 1: Extract the query from the input
                "context": RunnableLambda(get_question)
                           | RunnableLambda(fetch_module_content),
                "question": RunnablePassthrough()
            }
            | rag_prompt
            | model
    )

    return rag_chain


pattern = re.compile(r'\b[a-zA-Z]{2}\d{4}[a-zA-Z0-9]?\b')
def detect_module_code(query):
    matches = pattern.findall(query)
    return matches


def main():
    load_dotenv()
    model = get_model("ChatGPT")
    docs = get_wiki_docs(query="Bertrand Russell", load_max_docs=5)
    texts = split_documents(docs)
    vs = create_vector_db(texts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professor who teaches philosophical concepts to beginners."),
        ("user", "{input}")
    ])
    # Besides similarly search, you can also use maximal marginal relevance (MMR) for selecting results.
    # retriever = vs.as_retriever(search_type="mmr")
    retriever = vs.as_retriever()

    output_parser = StrOutputParser()
    chain = basic_chain(model, prompt)
    base_chain = chain | output_parser
    rag_chain = make_rag_chain(model, retriever) | output_parser

    questions = [
        "What were the most important contributions of Bertrand Russell to philosophy?",
        "What was the first book Bertrand Russell published?",
        "What was most notable about \"An Essay on the Foundations of Geometry\"?",
    ]
    for q in questions:
        print("\n--- QUESTION: ", q)
        print("* BASE:\n", base_chain.invoke({"input": q}))
        print("* RAG:\n", rag_chain.invoke(q))


if __name__ == '__main__':
    # this is to quite parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
