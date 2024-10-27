import os
from pathlib import Path

from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import json


def list_txt_files(data_dir="./data"):
    paths = Path(data_dir).glob('**/*.txt')
    for path in paths:
        yield str(path)


def load_txt_files(data_dir="./data"):
    docs = []
    paths = list_txt_files(data_dir)
    for path in paths:
        print(f"Loading {path}")
        loader = TextLoader(path)
        docs.extend(loader.load())
    return docs


def load_csv_files(data_dir="./data"):
    docs = []
    paths = Path(data_dir).glob('**/*.csv')
    for path in paths:
        loader = CSVLoader(file_path=str(path))
        docs.extend(loader.load())
    return docs


# Use with result of file_to_summarize = st.file_uploader("Choose a file") or a string.
# or a file like object.
def get_document_text(uploaded_file, title=None):
    docs = []
    fname = uploaded_file.name
    if not title:
        title = os.path.basename(fname)
    if fname.lower().endswith('pdf'):
        pdf_reader = PdfReader(uploaded_file)
        for num, page in enumerate(pdf_reader.pages):
            page = page.extract_text()
            doc = Document(page_content=page, metadata={'title': title, 'page': (num + 1)})
            docs.append(doc)

    else:
        # assume text
        doc_text = uploaded_file.read().decode()
        docs.append(doc_text)

    return docs


def load_and_chunk_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    documents = []
    for module in data:
        # Combine each json object into a string and add metadata
        # page_content = (
        #     f"Module Code: {module['moduleCode']}\n"
        #     f"Title: {module['title']}\n"
        #     f"Description: {module['description']}\n"
        #     f"Credits: {module['moduleCredit']}\n"
        #     f"Department: {module['department']}\n"
        #     f"Faculty: {module['faculty']}\n"
        #     f"Workload: {module.get('workload', 'N/A')}\n"
        #     f"Semester Data: {module.get('semesterData', 'N/A')}\n"
        # )

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
        documents.append(document)

    return documents


if __name__ == "__main__":
    example_pdf_path = "examples/healthy_meal_10_tips.pdf"
    docs = get_document_text(open(example_pdf_path, "rb"))
    for doc in docs:
        print(doc)
    docs = get_document_text(open("examples/us_army_recipes.txt", "rb"))
    for doc in docs:
        print(doc)
    txt_docs = load_txt_files("examples")
    for doc in txt_docs:
        print(doc)
    csv_docs = load_csv_files("examples")
    for doc in csv_docs:
        print(doc)
