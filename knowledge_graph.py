# _*_ coding: utf-8 _*_
__author__ = 'yuntwo'
__date__ = '2024/11/10 12:01:19'

import json
import os
import re

from dotenv import load_dotenv
from py2neo import Graph, Node, Relationship
from langchain.prompts import PromptTemplate  # Updated import path
from langchain_openai import ChatOpenAI  # Updated import path
from langchain_core.messages import AIMessage

# Initialize Neo4j connection
# Port 7474 for UI
uri = "bolt://localhost:7687"
username = "neo4j"
password = "password"
graph = Graph(uri, auth=(username, password))
# Initialize LLM and database connection
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Load JSON data
with open("data/mods22_23_test.json", "r") as f:
    modules = json.load(f)


# Updated prompt template for generating Cypher queries
# Improved prompt template for generating Cypher queries
cypher_prompt = PromptTemplate(
    template="""You are an assistant specialized in Neo4j Cypher queries.
    Based on the following question, generate a Cypher query that accurately retrieves data from a knowledge graph with the following structure:

    - Nodes:
      - "Module" with properties like `moduleCode`, `title`, `description`, `moduleCredit`
      - "Department" with a `name` property
      - "Faculty" with a `name` property
      - "Requirement" with a `text` property

    - Relationships:
      - Modules are related to Departments with the relationship "PART_OF_DEPARTMENT"
      - Departments are related to Faculties with the relationship "PART_OF_FACULTY"
      - Modules have prerequisites with the relationship "HAS_PREREQUISITE" to "Requirement"
      - Modules have preclusions with the relationship "HAS_PRECLUSION" to "Requirement"

    Generate only the Cypher query based on this structure and do not include additional explanations.

    Question: {question}

    Cypher Query:""",
    input_variables=["question"],
)


# Function to generate a Cypher query from the question
def generate_cypher_query(question):
    # Use the LLM to generate the Cypher query as an AIMessage object
    prompt_input = cypher_prompt.format_prompt(question=question)
    response = llm.invoke(prompt_input)

    # Extract the text of the response
    if isinstance(response, AIMessage):
        cypher_query = response.content.strip()  # Extract the text content of the AIMessage

        # Remove code block delimiters if present (e.g., ```cypher ... ```)
        cypher_query = re.sub(r"^```(?:cypher)?|```$", "", cypher_query, flags=re.MULTILINE).strip()

        # Ensure the query starts with a Cypher keyword (e.g., MATCH) or handle errors
        if not cypher_query.startswith("MATCH"):
            raise ValueError("The generated text does not start with a valid Cypher query.")
    else:
        raise TypeError("Unexpected response type. Expected AIMessage.")

    return cypher_query


def create_module_nodes(modules):
    for module in modules:
        # Create module node
        module_node = Node("Module",
                           moduleCode=module["moduleCode"],
                           title=module["title"],
                           description=module["description"],
                           moduleCredit=module["moduleCredit"])
        graph.merge(module_node, "Module", "moduleCode")

        # Create department and faculty nodes
        department_node = Node("Department", name=module["department"])
        faculty_node = Node("Faculty", name=module["faculty"])

        # Merge department and faculty into the graph
        graph.merge(department_node, "Department", "name")
        graph.merge(faculty_node, "Faculty", "name")

        # Create relationships
        graph.merge(Relationship(module_node, "PART_OF_DEPARTMENT", department_node))
        graph.merge(Relationship(department_node, "PART_OF_FACULTY", faculty_node))

        # Handle prerequisites and preclusions if present
        if "prerequisite" in module:
            prerequisite_node = Node("Requirement", text=module["prerequisite"])
            graph.merge(prerequisite_node, "Requirement", "text")
            graph.merge(Relationship(module_node, "HAS_PREREQUISITE", prerequisite_node))

        if "preclusion" in module:
            preclusion_node = Node("Requirement", text=module["preclusion"])
            graph.merge(preclusion_node, "Requirement", "text")
            graph.merge(Relationship(module_node, "HAS_PRECLUSION", preclusion_node))


# Improved retrieve_related_modules function using generated Cypher query
def retrieve_related_modules(question):
    # Generate Cypher query from the question
    cypher_query = generate_cypher_query(question)

    # Execute the generated query on the graph
    try:
        result = graph.run(cypher_query).data()
        return [record["m.moduleCode"] for record in result if "m.moduleCode" in record]
    except Exception as e:
        print(f"Error executing Cypher query: {e}")
        return []

# Execute the function to populate the graph
create_module_nodes(modules)

# Example usage
# question = "Multi-core Architectures"
question = "Big Data Analytics"
related_modules = retrieve_related_modules(question)
print(related_modules)
