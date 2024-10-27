import os

from dotenv import load_dotenv
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from basic_chain import get_model
from filter import ensemble_retriever_from_docs
from local_loader import load_txt_files
from memory import create_memory_chain
from rag_chain import make_rag_chain, make_direct_chain
import re

system_prompt = """
You are a helpful assistant designed to provide information about university modules. You have access to detailed module data, which includes comprehensive details about each module offered by the university. Based on this information, you will respond to queries about the modules, such as prerequisites, workload, exam dates, and more.

### Module Data Structure

Here is an overview of the structure and key elements of the module data:

- **Module Basic Info**:
  - `moduleCode`: The unique code identifying the module.
  - `title`: The title of the module.
  - `description`: A brief overview of what the module covers.
  - `moduleCredit`: Number of credits awarded upon successful completion of the module.
  - `department`: The department offering the module.
  - `faculty`: The faculty under which the module is categorized.

- **Workload**:
  - `workload`: An array specifying the estimated number of hours per week students are expected to dedicate to the module, distributed as [lectures, tutorials, laboratory, project/fieldwork, preparatory work].

- **Prerequisites and Preclusions**:
  - `prerequisite`: Requirements students must meet before enrolling in the module. Descriptions may include specific prior modules and cohort criteria.
  - `preclusion`: Modules that a student cannot take if they enroll in this module, typically because of overlapping content.

- **Attributes**:
  - `attributes`: Special attributes related to the module, such as specific educational streams it may fit into (e.g., `mpes1`, `mpes2`).

- **Semester Data**:
  - `semesterData`: Array of objects each containing:
    - `semester`: The semester number.
    - `examDate`: The date of the examination.
    - `examDuration`: Duration of the examination in minutes.
    - `covidZones`: Areas designated for classroom arrangements under COVID-19 protocols.

### Timetable Structure

Modules may also include detailed timetable data specifying when and where classes are held:

- **Timetable**:
  - `classNo`: Class number for the section.
  - `lessonType`: Type of lesson (e.g., Lecture, Tutorial).
  - `weeks`: Weeks during which the lesson takes place, either as an array of week numbers or as an object specifying start and end dates.
  - `day`: Day of the week the lesson occurs.
  - `startTime`: The starting time of the lesson in HHMM format.
  - `endTime`: The ending time of the lesson in HHMM format.
  - `venue`: The location where the lesson is held.

### Example Queries

You need to provide accurate responses to questions such as:

- "What are the prerequisites for [ModuleCode]?"
- "How many credits is [ModuleCode] worth?"
- "When is the exam for [ModuleCode]?"
- "What is the workload for [ModuleCode]?"
- "Which department offers [ModuleCode]?"
- "What modules are available for first-year students in the [FacultyName] faculty?"

Use the module information provided to craft responses that are both precise and helpful to the users.

### Instructions

- Extract relevant information from the module data structure to answer user queries.
- Ensure clarity and completeness in responses.
- Provide additional context if necessary to improve user understanding.
- Default to available module data; if certain data points are missing, explicitly state so.
- Be considerate of the academic context and provide answers in a professional and educational tone.

Use the provided module context to respond efficiently and accurately to user inquiries.

Use the following context and the users' chat history to help the user:
If you don't know the answer, just say that you don't know.

Context: {context}

Question: """


def create_full_chain(retriever, openai_api_key=None, chat_memory=ChatMessageHistory()):
    model = get_model("ChatGPT", openai_api_key=openai_api_key)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = make_rag_chain(model, retriever, rag_prompt=prompt)
    chain = create_memory_chain(model, rag_chain, chat_memory)

    return chain


def create_direct_chain(openai_api_key=None, chat_memory=ChatMessageHistory()):
    model = get_model("ChatGPT", openai_api_key=openai_api_key)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = make_direct_chain(model, prompt)
    chain = create_memory_chain(model, rag_chain, chat_memory)

    return chain


def ask_question(direct_chain, chain, query):
    if len(detect_module_code(query)) > 0:
        print("Direct chain")
        response = direct_chain.invoke(
            {"question": query},
            config={"configurable": {"session_id": "foo"}}
        )
    else:
        print("Full chain")
        response = chain.invoke(
            {"question": query},
            config={"configurable": {"session_id": "foo"}}
        )

    return response


pattern = re.compile(r'\b[a-zA-Z]{2}\d{4}[a-zA-Z0-9]?\b')


def detect_module_code(query):
    matches = pattern.findall(query)
    return matches


def main():
    load_dotenv()

    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()

    docs = load_txt_files()
    ensemble_retriever = ensemble_retriever_from_docs(docs)
    chain = create_full_chain(ensemble_retriever)

    queries = [
        "Generate a grocery list for my family meal plan for the next week(following 7 days). Prefer local, in-season ingredients."
        "Create a list of estimated calorie counts and grams of carbohydrates for each meal."
    ]

    for query in queries:
        response = ask_question(chain, query)
        console.print(Markdown(response.content))


if __name__ == '__main__':
    # this is to quiet parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
