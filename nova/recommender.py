"""
recommender.py
Recommendation Engine (interpretable, multi-signal scoring).

Scores courses for a given learner profile + skill gap using:
1. Base skill gap relevance (40 pts)
2. Difficulty fit + adaptive difficulty_bias (up to 25 pts)
3. Semantic text match bonus via TF-IDF (up to 15 pts)
4. Weekly pace/time budget fit (up to 20 pts)
5. Format preference/hands-on bonus (up to 10 pts)

Each recommendation carries a detailed `reasons` list for explainability.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def score_course(course, skill_node, profile, hours_per_week=None, semantic_score=None):
    """Returns (score, reasons[]) for a single course given the skill
    it belongs to and the learner's profile. Score is capped at 100."""
    score = 0
    reasons = []

    # Signal 1: Base relevance (teaches a required skill gap)
    score += 40
    skill_name = skill_node.get("name", "Target Skill")
    reasons.append(f"Teaches '{skill_name}', a prerequisite on your path to your goal")

    # Signal 2: Difficulty fit (adjusted by adaptive difficulty_bias)
    course_diff = course.get("difficulty", 2)
    base_diff = skill_node.get("difficulty", 2)
    bias = getattr(profile, "difficulty_bias", 0.0) if profile else 0.0
    effective_diff = base_diff + bias
    diff_gap = abs(course_diff - effective_diff)

    if diff_gap <= 0.5:
        score += 25
        reasons.append("Difficulty matches your current comfort level")
    elif diff_gap <= 1.5:
        score += 15
        reasons.append("Difficulty is close to your current comfort level")
    else:
        score += 5

    # Signal 3: Semantic match bonus (TF-IDF similarity with learner input)
    if semantic_score is not None:
        bonus = round(float(semantic_score) * 15)
        if bonus >= 3:
            score += bonus
            reasons.append(f"Content matches your stated interests/goals (+{bonus} pt boost)")

    # Signal 4: Time/pace budget fit
    hrs = hours_per_week or (getattr(profile, "hours_per_week", None) if profile else None)
    duration = course.get("duration_hrs", 10)
    if hrs:
        weeks_needed = duration / max(hrs, 1)
        if weeks_needed <= 2:
            score += 20
            reasons.append(f"Fits your {hrs} hrs/week pace (~{weeks_needed:.1f} weeks to complete)")
        elif weeks_needed <= 4:
            score += 10
            reasons.append(f"Moderate commitment at your pace (~{weeks_needed:.1f} weeks)")
        else:
            reasons.append(f"Longer commitment at your pace (~{weeks_needed:.1f} weeks)")
    else:
        score += 10  # Neutral score if hours not specified

    # Signal 5: Format diversity bonus
    course_format = course.get("format", "video")
    if course_format == "project":
        score += 10
        reasons.append("Hands-on practical project — reinforces skills through direct application")
    elif course_format == "reading":
        score += 5
        reasons.append("Concise self-paced reading and reference documentation")
    elif course_format == "video":
        score += 5
        reasons.append("Structured interactive video lectures")

    return min(int(score), 100), reasons


def recommend_courses_for_skill(skill_id, profile, G, hours_per_week=None, top_n=2, semantic_scores=None):
    """Given a single skill node from the graph, returns the top-N scored courses."""
    if skill_id not in G:
        return []
    skill_node = G.nodes[skill_id]
    courses = skill_node.get("courses", [])

    scored = []
    sem_scores = semantic_scores or {}
    for course in courses:
        sem = sem_scores.get(course["id"])
        score, reasons = score_course(
            course,
            skill_node,
            profile,
            hours_per_week=hours_per_week,
            semantic_score=sem,
        )
        scored.append({"course": course, "score": score, "reasons": reasons})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


def recommend_for_skill_gap(skill_gap_ids, profile, G, hours_per_week=None, top_n_per_skill=2, semantic_scores=None):
    """Runs recommendations across a list of skill IDs (skill gap)."""
    recommendations = {}
    for skill_id in skill_gap_ids:
        recommendations[skill_id] = recommend_courses_for_skill(
            skill_id,
            profile,
            G,
            hours_per_week=hours_per_week,
            top_n=top_n_per_skill,
            semantic_scores=semantic_scores,
        )
    return recommendations


def print_recommendations(recommendations, G):
    for skill_id, recs in recommendations.items():
        skill_name = G.nodes[skill_id]["name"] if skill_id in G else skill_id
        print(f"\n== {skill_name} ==")
        for r in recs:
            print(f"  [{r['score']}] {r['course']['title']} ({r['course']['provider']})")
            for reason in r["reasons"]:
                print(f"      - {reason}")


if __name__ == "__main__":
    from graph_loader import load_skill_graph
    from profiling import build_learner_profile

    G = load_skill_graph()
    profile = build_learner_profile(
        "I know basic Python and some SQL. I want to become a Data Analyst. "
        "I can commit about 5 hrs a week."
    )

    skill_gap = [sid for sid in profile.target_skill_ids if sid not in profile.known_skill_ids]
    recs = recommend_for_skill_gap(skill_gap, profile, G, profile.hours_per_week)
    print_recommendations(recs, G)
