"""
graph_loader.py
Loads skills.json and courses.json into a single networkx DiGraph.
Each skill is a node; prerequisite relationships are directed edges
(prereq -> skill). Courses are attached as node attributes, not
separate nodes, so the graph stays a pure skill-dependency graph.

Domain-agnostic by design: 'domain' is a node attribute, supporting
cross-domain paths (e.g. Cloud/DevOps skills as prerequisites for Data Science skills).
"""

import json
from pathlib import Path
import networkx as nx

BASE_DIR = Path(__file__).resolve().parent


def _resolve_data_path(filename: str, custom_path=None) -> Path:
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
    # Check current directory, then data/ subdirectory
    direct_path = BASE_DIR / filename
    if direct_path.exists():
        return direct_path
    data_path = BASE_DIR / "data" / filename
    if data_path.exists():
        return data_path
    return direct_path


def load_skill_graph(skills_path="skills.json", courses_path="courses.json") -> nx.DiGraph:
    skills_file = _resolve_data_path("skills.json", skills_path)
    courses_file = _resolve_data_path("courses.json", courses_path)

    with open(skills_file, "r", encoding="utf-8") as f:
        skills = json.load(f)
    with open(courses_file, "r", encoding="utf-8") as f:
        courses = json.load(f)

    # Group courses by the skill(s) they teach
    courses_by_skill = {}
    for course in courses:
        for sid in course.get("skill_ids", []):
            courses_by_skill.setdefault(sid, []).append(course)

    G = nx.DiGraph()

    # Add skill nodes with metadata + attached courses
    for skill in skills:
        G.add_node(
            skill["id"],
            id=skill["id"],
            name=skill["name"],
            domain=skill["domain"],
            difficulty=skill["difficulty"],
            description=skill.get("description", ""),
            prerequisites=skill.get("prerequisites", []),
            courses=courses_by_skill.get(skill["id"], []),
        )

    # Add prerequisite edges (prereq -> skill)
    for skill in skills:
        for prereq_id in skill.get("prerequisites", []):
            if prereq_id in G:
                G.add_edge(prereq_id, skill["id"])

    return G


def summarize_graph(G: nx.DiGraph):
    print(f"Total skills: {G.number_of_nodes()}")
    print(f"Total prerequisite edges: {G.number_of_edges()}")
    domains = {}
    for _, data in G.nodes(data=True):
        dom = data.get("domain", "Unknown")
        domains[dom] = domains.get(dom, 0) + 1
    print("Skills per domain:")
    for domain, count in domains.items():
        print(f"  - {domain}: {count}")
    is_dag = nx.is_directed_acyclic_graph(G)
    print(f"Is valid DAG (no circular prerequisites): {is_dag}")


if __name__ == "__main__":
    G = load_skill_graph()
    summarize_graph(G)
