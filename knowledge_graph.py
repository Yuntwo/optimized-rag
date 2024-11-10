# _*_ coding: utf-8 _*_
__author__ = 'yuntwo'
__date__ = '2024/11/10 12:01:19'

import json
from neo4j import GraphDatabase
from py2neo import Graph, Node, Relationship

# Initialize Neo4j connection
# Port 7474 for UI
uri = "bolt://localhost:7687"
username = "neo4j"
password = "password"
graph = Graph(uri, auth=(username, password))

# Load JSON data
with open("data/mods22_23_test.json", "r") as f:
    modules = json.load(f)


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


def retrieve_related_modules(query):
    query_template = """
    MATCH (m:Module)-[:HAS_PREREQUISITE]->(r:Requirement)
    WHERE m.title CONTAINS $query OR m.description CONTAINS $query
    RETURN m.moduleCode AS code, m.title AS title, r.text AS prerequisite
    """
    result = graph.run(query_template, query=query)
    return [{"moduleCode": record["code"], "title": record["title"], "prerequisite": record["prerequisite"]}
            for record in result]


# Execute the function to populate the graph
create_module_nodes(modules)


# Example usage
related_modules = retrieve_related_modules("Big")
print(related_modules)
