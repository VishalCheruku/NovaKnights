"""
explainer.py
Explainability Layer.

Translates structured graph dependencies and recommender scoring signals
into clear, natural-language explanations for learners.
"""

import networkx as nx


def _resolve_skill_id(skill_identifier: str, G: nx.DiGraph) -> str:
    """Helper to resolve either a skill ID ('stats_basics') or a name ('Statistics Fundamentals') to a skill ID."""
    if skill_identifier in G:
        return skill_identifier
    target_lower = skill_identifier.lower().strip()
    for sid, data in G.nodes(data=True):
        if data.get("name", "").lower() == target_lower:
            return sid
    # Partial match
    for sid, data in G.nodes(data=True):
        if target_lower in data.get("name", "").lower() or target_lower in sid:
            return sid
    return skill_identifier


def why_this_milestone_first(milestone_number: int, milestones: list, G: nx.DiGraph) -> str:
    """Explains why a milestone is scheduled at its position and what it unlocks."""
    milestone = next((m for m in milestones if m["milestone_number"] == int(milestone_number)), None)
    if not milestone:
        return f"Milestone {milestone_number} is not currently in your active path."

    skill_names = [s["name"] for s in milestone["skills"]]
    lines = [f"Milestone {milestone_number} focuses on: {', '.join(skill_names)}."]

    if int(milestone_number) == 1:
        lines.append("All prerequisites for these skills are already met, so you can start immediately.")
    else:
        prior = next((m for m in milestones if m["milestone_number"] == int(milestone_number) - 1), None)
        if prior:
            prior_names = [s["name"] for s in prior["skills"]]
            lines.append(
                f"These skills build upon what you master in Milestone {int(milestone_number) - 1} ({', '.join(prior_names)})."
            )

    # Downstream unlocks
    unlocked = set()
    for skill in milestone["skills"]:
        sid = skill["id"]
        if sid in G:
            for succ in G.successors(sid):
                unlocked.add(G.nodes[succ].get("name", succ))

    if unlocked:
        lines.append(f"Completing this milestone unlocks: {', '.join(sorted(unlocked))}.")

    return " ".join(lines)


def why_this_course(skill_id: str, course_title: str, recommendations: dict) -> str:
    """Explains why a specific course recommendation was chosen."""
    recs = recommendations.get(skill_id, [])
    if not recs:
        # Check all skills if not found directly
        for sid, course_list in recommendations.items():
            for r in course_list:
                if course_title.lower() in r["course"]["title"].lower():
                    recs = course_list
                    break
            if recs:
                break

    match = next(
        (r for r in recs if course_title.lower() in r["course"]["title"].lower() or r["course"]["title"].lower() in course_title.lower()),
        None,
    )
    if not match:
        return f"Course '{course_title}' was not found in current recommendations for this skill."

    reasons = "; ".join(match.get("reasons", []))
    score = match.get("score", 0)
    provider = match["course"].get("provider", "Provider")
    return f"'{match['course']['title']}' by {provider} scored {score}/100 because: {reasons}."


def why_do_i_need_this_skill(skill_identifier: str, profile, G: nx.DiGraph) -> str:
    """Explains why a skill is required on the learner's journey."""
    skill_id = _resolve_skill_id(skill_identifier, G)

    if skill_id not in G:
        return f"'{skill_identifier}' was not found in the curriculum graph."

    skill_name = G.nodes[skill_id].get("name", skill_id)

    # 1. Direct target skill
    if skill_id in profile.target_skill_ids:
        return f"'{skill_name}' is one of your primary stated career / skill goals."

    # 2. Prerequisite for target skills
    unlocked_targets = []
    for target_id in profile.target_skill_ids:
        if target_id in G and skill_id in nx.ancestors(G, target_id):
            unlocked_targets.append(G.nodes[target_id].get("name", target_id))

    if unlocked_targets:
        targets_str = ", ".join(unlocked_targets)
        return (
            f"'{skill_name}' is a foundational prerequisite for {targets_str}, "
            f"which {'is' if len(unlocked_targets) == 1 else 'are'} required for your target goal."
        )

    return f"'{skill_name}' is not currently required for your selected path."


if __name__ == "__main__":
    from graph_loader import load_skill_graph
    from profiling import build_learner_profile
    from path_generator import generate_learning_path
    from recommender import recommend_for_skill_gap

    G = load_skill_graph()
    profile = build_learner_profile(
        "I know basic Python and some SQL. I want to become a Data Analyst. 5 hrs a week."
    )
    milestones = generate_learning_path(profile, G)
    gap_ids = [s["id"] for m in milestones for s in m["skills"]]
    recs = recommend_for_skill_gap(gap_ids, profile, G, profile.hours_per_week)

    print("Milestone 1 Explanation:")
    print(why_this_milestone_first(1, milestones, G))
    print("\nSkill Explanation:")
    print(why_do_i_need_this_skill("stats_basics", profile, G))
    print("\nCourse Explanation:")
    top_course = recs["data_viz"][0]["course"]["title"]
    print(why_this_course("data_viz", top_course, recs))
