"""
adaptive.py
Adaptive Feedback Loop and Semantic Matcher.

Provides two forms of adaptation:
1. Progress-based: Complete skill -> recomputes remaining path.
2. Feedback-based: Course difficulty feedback ('too_easy', 'too_hard', 'just_right') ->
   shifts difficulty_bias for personalized future recommendations.
3. TF-IDF Semantic matching: Computes cosine similarity between learner's stated goals
   and course content to boost genuinely aligned courses.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def complete_skill(profile, skill_id: str):
    """Marks a skill as acquired in the learner's profile."""
    if skill_id and skill_id not in profile.known_skill_ids:
        profile.known_skill_ids.append(skill_id)
    return profile


def record_course_feedback(profile, skill_id: str, feedback: str):
    """Adjusts difficulty bias based on learner feedback.
    feedback: 'too_easy' (+0.5 bias), 'too_hard' (-0.5 bias), 'just_right' (0.0)."""
    profile.feedback_log.append((skill_id, feedback))

    if feedback == "too_easy":
        profile.difficulty_bias += 0.5
    elif feedback == "too_hard":
        profile.difficulty_bias -= 0.5
    # Clamp bias to reasonable range [-2.0, 2.0]
    profile.difficulty_bias = max(-2.0, min(2.0, profile.difficulty_bias))
    return profile


def build_course_corpus(G):
    """Collects all courses attached to graph nodes with text descriptions."""
    courses = []
    texts = []
    seen_ids = set()
    for _, node in G.nodes(data=True):
        for course in node.get("courses", []):
            cid = course.get("id")
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)
            courses.append(course)
            texts.append(f"{course.get('title', '')} {node.get('name', '')} {node.get('description', '')}")
    return courses, texts


def fit_semantic_matcher(G):
    """Fits one TF-IDF vectorizer over the course corpus."""
    courses, texts = build_course_corpus(G)
    if not texts:
        texts = ["default learning course"]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    course_vectors = vectorizer.fit_transform(texts)
    course_id_to_row = {c["id"]: i for i, c in enumerate(courses)}
    return vectorizer, course_vectors, course_id_to_row


def semantic_scores_for_learner(profile, vectorizer, course_vectors, course_id_to_row):
    """Calculates cosine similarity between learner's raw input and each course."""
    if not profile or not profile.raw_text or not vectorizer or course_vectors is None:
        return {}
    try:
        learner_vec = vectorizer.transform([profile.raw_text])
        sims = cosine_similarity(learner_vec, course_vectors)[0]
        return {cid: float(sims[row]) for cid, row in course_id_to_row.items()}
    except Exception:
        return {}


if __name__ == "__main__":
    from graph_loader import load_skill_graph
    from profiling import build_learner_profile
    from path_generator import generate_learning_path, print_path
    from recommender import recommend_for_skill_gap

    G = load_skill_graph()
    profile = build_learner_profile(
        "I know basic Python and some SQL. I want to become a Data Analyst. "
        "I can commit about 5 hrs a week. I love visual storytelling with data."
    )

    print("=== Initial Path ===")
    milestones = generate_learning_path(profile, G)
    print_path(milestones)

    vectorizer, course_vectors, course_id_to_row = fit_semantic_matcher(G)
    sem_scores = semantic_scores_for_learner(profile, vectorizer, course_vectors, course_id_to_row)
    print("\nSemantic Scores Sample:", {k: round(v, 3) for k, v in list(sem_scores.items())[:3]})

    # Test feedback
    record_course_feedback(profile, "stats_basics", "too_easy")
    print(f"Updated difficulty bias after feedback: {profile.difficulty_bias}")

    # Test progress
    complete_skill(profile, "pandas_numpy")
    print("Path after completing pandas_numpy:")
    print_path(generate_learning_path(profile, G))
