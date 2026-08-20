"""
profiling.py
Learner Profiling Engine.

Takes free-text learner input and produces a structured LearnerProfile
by matching against the skill graph (skills.json). Intelligently extracts
known skills, career/skill goals, and weekly time commitments while
preventing target skills from being falsely marked as known skills.
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).resolve().parent


def _resolve_data_path(filename: str, custom_path=None) -> Path:
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
    direct_path = BASE_DIR / filename
    if direct_path.exists():
        return direct_path
    data_path = BASE_DIR / "data" / filename
    if data_path.exists():
        return data_path
    return direct_path


def _load_skills(skills_path="skills.json"):
    path = _resolve_data_path("skills.json", skills_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Role -> target skill(s) mapping
ROLE_TARGETS = {
    "data analyst": ["pandas_numpy", "data_viz", "stats_basics"],
    "data scientist": ["ml_supervised", "ml_unsupervised", "deep_learning"],
    "machine learning engineer": ["ml_supervised", "ml_deploy"],
    "ml engineer": ["ml_supervised", "ml_deploy"],
    "frontend developer": ["react_basics", "html_css", "js_basics"],
    "frontend engineer": ["react_basics", "html_css", "js_basics"],
    "backend developer": ["backend_apis", "databases"],
    "backend engineer": ["backend_apis", "databases"],
    "full stack developer": ["fullstack_proj"],
    "fullstack developer": ["fullstack_proj"],
    "full stack engineer": ["fullstack_proj"],
    "devops engineer": ["ci_cd", "cloud_deploy"],
    "cloud engineer": ["docker_basics", "cloud_deploy"],
}

# Skill ID -> explicit keyword aliases (including short acronyms like sql, git, aws)
SKILL_ALIASES = {
    "py_basics": ["python", "python3", "py"],
    "stats_basics": ["statistics", "stats", "probability", "hypothesis testing"],
    "pandas_numpy": ["pandas", "numpy", "data wrangling", "dataframe", "dataframes"],
    "data_viz": ["data visualization", "visualization", "data viz", "matplotlib", "seaborn", "charts", "tableau"],
    "ml_supervised": ["supervised learning", "machine learning", "ml", "regression", "classification", "scikit-learn", "sklearn"],
    "ml_unsupervised": ["unsupervised learning", "clustering", "pca", "kmeans"],
    "deep_learning": ["deep learning", "neural networks", "neural network", "pytorch", "tensorflow", "cnn", "rnn"],
    "html_css": ["html", "css", "html5", "css3", "web design"],
    "js_basics": ["javascript", "js", "ecmascript"],
    "react_basics": ["react", "react.js", "reactjs"],
    "backend_apis": ["backend", "rest api", "apis", "api", "express", "fastapi", "flask", "node"],
    "databases": ["sql", "databases", "database", "relational database", "postgres", "mysql", "sqlite"],
    "fullstack_proj": ["full stack", "fullstack", "full-stack"],
    "linux_basics": ["linux", "shell", "bash", "command line", "terminal"],
    "git_basics": ["git", "github", "version control", "gitlab"],
    "docker_basics": ["docker", "containers", "containerization", "dockerfile"],
    "ci_cd": ["ci/cd", "ci cd", "cicd", "continuous integration", "github actions", "pipelines"],
    "cloud_deploy": ["cloud", "aws", "gcp", "azure", "cloud deployment"],
    "ml_deploy": ["ml deploy", "model deployment", "mlops", "serving models"],
}

STOPWORDS = {
    "basics", "fundamentals", "with", "learning", "data", "science",
    "development", "web", "cloud", "devops", "foundations", "integration",
    "project", "projects", "basic",
}


@dataclass
class LearnerProfile:
    raw_text: str
    known_skill_ids: list = field(default_factory=list)
    target_skill_ids: list = field(default_factory=list)
    hours_per_week: int = None
    unmatched_goal_text: str = None  # set if goal couldn't be matched
    difficulty_bias: float = 0.0  # shifts via feedback: negative = easier, positive = harder
    feedback_log: list = field(default_factory=list)  # history of (skill_id, feedback) tuples


def _split_text_into_sections(text: str):
    """Splits input into 'known' parts (skills already acquired) vs 'goal' parts."""
    text_lower = text.lower()
    
    # Common split markers
    goal_markers = [
        "i want to become", "i want to learn", "i want to be", "my goal is",
        "target:", "goal:", "aiming to", "interested in learning", "aspire to",
        "looking to learn", "wish to learn", "want to know"
    ]
    
    known_segment = text_lower
    goal_segment = text_lower
    
    for marker in goal_markers:
        if marker in text_lower:
            parts = text_lower.split(marker, 1)
            known_segment = parts[0]
            goal_segment = marker + " " + parts[1]
            break
            
    return known_segment, goal_segment


def match_skills_in_text(segment: str, skills):
    """Matches skills present in a text segment using exact names and alias dictionary."""
    matched = set()
    segment_lower = segment.lower()
    # Tokenize word-boundary-aware
    words = re.findall(r"\b[\w/+-]+\b", segment_lower)
    word_set = set(words)

    for skill in skills:
        sid = skill["id"]
        name_lower = skill["name"].lower()
        
        # 1. Exact skill name match
        if name_lower in segment_lower:
            matched.add(sid)
            continue
            
        # 2. Check curated aliases
        aliases = SKILL_ALIASES.get(sid, [])
        for alias in aliases:
            if " " in alias:
                if alias in segment_lower:
                    matched.add(sid)
                    break
            else:
                if alias in word_set:
                    matched.add(sid)
                    break
        if sid in matched:
            continue

        # 3. Check clean keywords from name (length >= 3 and not in stopwords)
        clean_name = name_lower.replace("(", " ").replace(")", " ").replace("/", " ")
        keywords = [w for w in clean_name.split() if len(w) >= 3 and w not in STOPWORDS]
        if any(kw in word_set for kw in keywords):
            matched.add(sid)

    return list(matched)


def extract_known_skills(text: str, skills, target_skill_ids=None):
    """Extracts known skills from the 'known/background' part of the text,
    ensuring target skills are not mistakenly marked as known."""
    known_seg, _ = _split_text_into_sections(text)
    known = match_skills_in_text(known_seg, skills)
    
    # If the user has target skills, exclude them from known skills
    if target_skill_ids:
        known = [sid for sid in known if sid not in target_skill_ids]
    return known


def match_goal(text: str, skills):
    """Matches a stated goal to either a role (ROLE_TARGETS) or direct skill names."""
    text_lower = text.lower()
    _, goal_seg = _split_text_into_sections(text)

    # 1. Check role targets first (longest match first)
    for role in sorted(ROLE_TARGETS.keys(), key=len, reverse=True):
        if role in text_lower:
            return list(ROLE_TARGETS[role]), None

    # 2. Check if specific skills were mentioned in the goal segment
    goal_skills = match_skills_in_text(goal_seg, skills)
    if goal_skills:
        return goal_skills, None

    # 3. Fallback: check entire text for direct skill names
    all_skills = match_skills_in_text(text_lower, skills)
    if all_skills:
        return all_skills, None

    return [], text


def extract_hours_per_week(text: str):
    text_lower = text.lower()
    # Matches "5 hrs/week", "5 hours a week", "5h/week", "5 hrs per week", "commit 5 hrs", "5 hours"
    patterns = [
        r"(\d+)\s*(?:hrs?|hours?|h)\s*(?:/|a|per)\s*(?:week|wk)",
        r"(?:commit|spend|have)\s*(\d+)\s*(?:hrs?|hours?|h)",
        r"(\d+)\s*(?:hrs?|hours?)\s*weekly",
        r"(\d+)\s*(?:hrs?|hours?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return int(match.group(1))
    return None


def build_learner_profile(text: str, skills_path="skills.json") -> LearnerProfile:
    skills = _load_skills(skills_path)
    target, unmatched = match_goal(text, skills)
    known = extract_known_skills(text, skills, target_skill_ids=target)
    hours = extract_hours_per_week(text)

    profile = LearnerProfile(
        raw_text=text,
        known_skill_ids=known,
        target_skill_ids=target,
        hours_per_week=hours,
        unmatched_goal_text=unmatched,
    )
    return profile


def profile_summary(profile: LearnerProfile, skills_path="skills.json") -> str:
    skills = {s["id"]: s["name"] for s in _load_skills(skills_path)}
    lines = []
    known_names = [skills.get(sid, sid) for sid in profile.known_skill_ids]
    target_names = [skills.get(sid, sid) for sid in profile.target_skill_ids]
    lines.append(f"Known skills: {', '.join(known_names) if known_names else 'none detected'}")
    lines.append(f"Target skills: {', '.join(target_names) if target_names else 'none detected'}")
    lines.append(f"Time commitment: {profile.hours_per_week or 'not specified'} hrs/week")
    if profile.unmatched_goal_text:
        lines.append("Note: goal not confidently matched — consider asking a clarifying question.")
    return "\n".join(lines)


if __name__ == "__main__":
    sample_text = (
        "I know basic Python and some SQL. I want to become a Data Analyst. "
        "I can commit about 5 hrs a week."
    )
    profile = build_learner_profile(sample_text)
    print(profile_summary(profile))
