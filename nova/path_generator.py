"""
path_generator.py
Personalized Learning Path Generator.

Given a learner profile and the skill graph, computes the TRUE skill
gap by walking backwards through prerequisite edges from each target
skill, then topologically sorts and groups them into ordered milestones.
"""

import networkx as nx


def compute_skill_gap(known_skill_ids, target_skill_ids, G: nx.DiGraph):
    """Walks backwards from each target skill through prerequisite
    edges to find every ancestor skill required, minus what's already known."""
    required = set()
    for target_id in target_skill_ids:
        if target_id not in G:
            continue
        ancestors = nx.ancestors(G, target_id)
        required.add(target_id)
        required.update(ancestors)

    gap = required - set(known_skill_ids or [])
    return gap


def build_milestones(gap_skill_ids, G: nx.DiGraph):
    """Groups the skill gap into ordered milestones using topological generations.
    Skills in the same milestone can be learned in parallel."""
    if not gap_skill_ids:
        return []

    subgraph = G.subgraph(gap_skill_ids)

    if not nx.is_directed_acyclic_graph(subgraph):
        raise ValueError("Skill gap subgraph has a cycle — check prerequisite data")

    milestones = []
    for i, generation in enumerate(nx.topological_generations(subgraph), start=1):
        skills_in_milestone = []
        for skill_id in generation:
            node = G.nodes[skill_id]
            skills_in_milestone.append({
                "id": skill_id,
                "name": node.get("name", skill_id),
                "domain": node.get("domain", "General"),
                "difficulty": node.get("difficulty", 1),
                "description": node.get("description", ""),
                "prerequisites": node.get("prerequisites", []),
                "courses_count": len(node.get("courses", [])),
            })
        # Sort within a milestone by difficulty so easier skills surface first
        skills_in_milestone.sort(key=lambda s: s["difficulty"])
        milestones.append({
            "milestone_number": i,
            "skills": skills_in_milestone,
        })
    return milestones


def generate_learning_path(profile, G: nx.DiGraph):
    """Full pipeline: profile -> skill gap -> milestones."""
    gap = compute_skill_gap(profile.known_skill_ids, profile.target_skill_ids, G)
    milestones = build_milestones(gap, G)
    return milestones


def print_path(milestones):
    if not milestones:
        print("No remaining milestones needed! All target skills acquired.")
        return
    for m in milestones:
        print(f"\nMilestone {m['milestone_number']}:")
        for s in m["skills"]:
            print(f"  - {s['name']} ({s['domain']}, difficulty {s['difficulty']})")


if __name__ == "__main__":
    from graph_loader import load_skill_graph
    from profiling import build_learner_profile

    G = load_skill_graph()
    profile = build_learner_profile(
        "I know basic Python and some SQL. I want to become a Data Analyst. "
        "I can commit about 5 hrs a week."
    )

    milestones = generate_learning_path(profile, G)
    print_path(milestones)
