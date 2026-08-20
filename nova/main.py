"""
main.py
FastAPI Backend for the AI-Powered Personalized Learning Path Recommender.

Wraps Graph Loader, Profiler, Recommender, Path Generator, Explainer,
Adaptive Feedback Loop, and Chat Assistant into a cohesive REST API.
Serves the modern web frontend dashboard directly at http://127.0.0.1:8000.
"""

import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from graph_loader import load_skill_graph
from profiling import build_learner_profile
from path_generator import generate_learning_path
from recommender import recommend_for_skill_gap
from explainer import why_this_milestone_first, why_this_course, why_do_i_need_this_skill
from adaptive import (
    complete_skill,
    record_course_feedback,
    fit_semantic_matcher,
    semantic_scores_for_learner,
)
from chat_assistant import chat_with_learner

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="AI Learning Path Recommender API",
    description="Adaptive curriculum path generation, multi-signal course recommendation, and AI explanation engine.",
    version="2.0.0",
)

# CORS middleware for local frontend and third-party clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared graph & semantic matcher state (loaded once at startup)
GRAPH = load_skill_graph()
VECTORIZER, COURSE_VECTORS, COURSE_ID_TO_ROW = fit_semantic_matcher(GRAPH)

# In-memory session store for prototype
SESSIONS = {}
CHAT_HISTORY = {}


def _get_profile(session_id: str):
    profile = SESSIONS.get(session_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new session.")
    return profile


def _build_recommendations(profile):
    """Computes milestones and scored course recommendations with semantic signals."""
    milestones = generate_learning_path(profile, GRAPH)
    gap_ids = [s["id"] for m in milestones for s in m["skills"]]
    sem_scores = semantic_scores_for_learner(profile, VECTORIZER, COURSE_VECTORS, COURSE_ID_TO_ROW)
    recommendations = recommend_for_skill_gap(
        gap_ids,
        profile,
        GRAPH,
        hours_per_week=profile.hours_per_week,
        top_n_per_skill=2,
        semantic_scores=sem_scores,
    )
    return milestones, recommendations


# ---------- Request/Response Models ----------

class StartSessionRequest(BaseModel):
    text: str


class CompleteSkillRequest(BaseModel):
    skill_id: str


class FeedbackRequest(BaseModel):
    skill_id: str
    feedback: str  # 'too_easy' | 'too_hard' | 'just_right'


class ChatRequest(BaseModel):
    message: str


# ---------- Endpoints ----------

@app.get("/")
def serve_frontend():
    """Serves the interactive web application."""
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file, media_type="text/html")
    return {"message": "Learning Path Recommender API is live. Visit /docs for Swagger."}


@app.post("/session/start")
def start_session(req: StartSessionRequest):
    """Initializes a new learner profile from free-text goals and background."""
    profile = build_learner_profile(req.text)
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = profile
    CHAT_HISTORY[session_id] = []
    
    return {
        "session_id": session_id,
        "known_skills": profile.known_skill_ids,
        "target_skills": profile.target_skill_ids,
        "hours_per_week": profile.hours_per_week,
        "difficulty_bias": profile.difficulty_bias,
        "note": "Goal not confidently matched — try mentioning a specific role or skill." if profile.unmatched_goal_text else None,
    }


@app.get("/session/{session_id}/profile")
def get_profile(session_id: str):
    """Returns the current profile status."""
    profile = _get_profile(session_id)
    return {
        "session_id": session_id,
        "known_skills": profile.known_skill_ids,
        "target_skills": profile.target_skill_ids,
        "hours_per_week": profile.hours_per_week,
        "difficulty_bias": profile.difficulty_bias,
        "feedback_log": profile.feedback_log,
        "raw_text": profile.raw_text,
    }


@app.get("/session/{session_id}/path")
def get_path(session_id: str):
    """Returns ordered milestones and skill breakdown."""
    profile = _get_profile(session_id)
    milestones, _ = _build_recommendations(profile)
    return {"milestones": milestones}


@app.get("/session/{session_id}/recommendations")
def get_recommendations(session_id: str):
    """Returns ranked course recommendations with scoring rationales."""
    profile = _get_profile(session_id)
    _, recommendations = _build_recommendations(profile)
    return {"recommendations": recommendations}


@app.get("/session/{session_id}/explain/milestone/{milestone_number}")
def explain_milestone(session_id: str, milestone_number: int):
    """Explains topological milestone ordering and downstream unlocks."""
    profile = _get_profile(session_id)
    milestones, _ = _build_recommendations(profile)
    explanation = why_this_milestone_first(milestone_number, milestones, GRAPH)
    return {"explanation": explanation}


@app.get("/session/{session_id}/explain/skill/{skill_id}")
def explain_skill(session_id: str, skill_id: str):
    """Explains why a skill is required on the path to the learner's goals."""
    profile = _get_profile(session_id)
    explanation = why_do_i_need_this_skill(skill_id, profile, GRAPH)
    return {"explanation": explanation}


@app.get("/session/{session_id}/explain/course")
def explain_course(session_id: str, skill_id: str = Query(...), course_title: str = Query(...)):
    """Explains why a specific course was recommended based on multi-signal scoring."""
    profile = _get_profile(session_id)
    _, recommendations = _build_recommendations(profile)
    explanation = why_this_course(skill_id, course_title, recommendations)
    return {"explanation": explanation}


@app.post("/session/{session_id}/complete-skill")
def post_complete_skill(session_id: str, req: CompleteSkillRequest):
    """Marks a skill complete and dynamically re-routes remaining milestones."""
    profile = _get_profile(session_id)
    complete_skill(profile, req.skill_id)
    milestones, recs = _build_recommendations(profile)
    return {
        "known_skills": profile.known_skill_ids,
        "updated_milestones": milestones,
        "recommendations": recs,
    }


@app.post("/session/{session_id}/feedback")
def post_feedback(session_id: str, req: FeedbackRequest):
    """Records course feedback ('too_easy' / 'too_hard' / 'just_right') and shifts difficulty bias."""
    profile = _get_profile(session_id)
    record_course_feedback(profile, req.skill_id, req.feedback)
    milestones, recs = _build_recommendations(profile)
    return {
        "difficulty_bias": profile.difficulty_bias,
        "feedback_log": profile.feedback_log,
        "recommendations": recs,
    }


@app.post("/session/{session_id}/chat")
def post_chat(session_id: str, req: ChatRequest):
    """Conversational assistant with tool execution (Claude LLM or smart local fallback)."""
    profile = _get_profile(session_id)
    history = CHAT_HISTORY.get(session_id, [])

    reply, updated_history = chat_with_learner(
        req.message, profile, GRAPH, VECTORIZER, COURSE_VECTORS, COURSE_ID_TO_ROW, history
    )

    CHAT_HISTORY[session_id] = updated_history
    return {"reply": reply}


@app.get("/skills")
def get_all_skills():
    """Returns the full catalog of skills from the curriculum graph."""
    skills = []
    for sid, data in GRAPH.nodes(data=True):
        skills.append({
            "id": sid,
            "name": data.get("name", sid),
            "domain": data.get("domain", "General"),
            "difficulty": data.get("difficulty", 1),
            "description": data.get("description", ""),
            "prerequisites": data.get("prerequisites", []),
            "course_count": len(data.get("courses", [])),
        })
    return {"skills": sorted(skills, key=lambda x: (x["domain"], x["difficulty"]))}


@app.get("/courses")
def get_all_courses():
    """Returns all available courses."""
    courses, _ = fit_semantic_matcher(GRAPH)[:2]
    # Return unique courses
    unique_courses = []
    seen = set()
    for _, data in GRAPH.nodes(data=True):
        for c in data.get("courses", []):
            if c["id"] not in seen:
                seen.add(c["id"])
                unique_courses.append(c)
    return {"courses": unique_courses}


@app.get("/graph")
def get_graph_data():
    """Returns nodes and edges for graph visualization."""
    nodes = []
    for sid, data in GRAPH.nodes(data=True):
        nodes.append({
            "id": sid,
            "label": data.get("name", sid),
            "domain": data.get("domain", "General"),
            "difficulty": data.get("difficulty", 1),
            "description": data.get("description", ""),
        })
    edges = [{"source": u, "target": v} for u, v in GRAPH.edges()]
    return {"nodes": nodes, "edges": edges}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "skills_loaded": GRAPH.number_of_nodes(),
        "prerequisite_edges": GRAPH.number_of_edges(),
        "active_sessions": len(SESSIONS),
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Learning Path Recommender on http://127.0.0.1:8000 ...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
