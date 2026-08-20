# NOVA — AI-Powered Personalized Learning Path Recommender

An adaptive, graph-guided curriculum engine that transforms learner goals into topological milestone roadmaps, ranks courses using multi-signal interpretable scoring, adapts dynamically to learner progress and feedback, and provides conversational explanations.

---

## 🏗️ Architecture & Module Connections

```
                         skills.json & courses.json
                                     │
                                     ▼
                            graph_loader.py
                         (NetworkX DiGraph DAG)
                                     │
         Learner Goal ───────────────┼───────────────► profiling.py
                                     │               (LearnerProfile)
                                     ▼                      │
                            path_generator.py ◄─────────────┘
                         (Topological Milestones)
                                     │
                                     ▼
    adaptive.py ─────────────► recommender.py ◄────────── LearnerProfile
  (TF-IDF Matcher &        (Multi-Signal Scoring &
   Difficulty Bias)         Interpretable Reasons)
                                     │
                                     ▼
                               explainer.py ─────────► chat_assistant.py
                         (Milestone / Skill / Course    (Dual-Mode Tool
                            Reason Explanations)        Calling & Routing)
                                     │                         │
                                     ▼                         ▼
                                       main.py (FastAPI App)
                                                 │
                                                 ▼
                                     index.html (Web Dashboard)
```

---

## 📂 File Structure

| File | Role |
|---|---|
| `skills.json` | Skill graph nodes, domains, difficulties, descriptions, prerequisites. |
| `courses.json` | Course catalog with providers, durations, formats, difficulty levels. |
| `graph_loader.py` | Loads JSON files into a validated NetworkX DAG. |
| `profiling.py` | Extracts known skills, career goals, and time budgets from free text. |
| `path_generator.py` | Computes true skill gap via graph ancestors and builds ordered milestones. |
| `recommender.py` | Multi-signal course ranking with natural-language reasoning. |
| `adaptive.py` | Real-time adaptation via skill completions, difficulty feedback, and TF-IDF matching. |
| `explainer.py` | Explains milestone ordering, prerequisite chains, and course scoring. |
| `chat_assistant.py` | AI conversational advisor (Claude Sonnet or smart local intent dispatcher). |
| `main.py` | FastAPI REST API + static host serving the interactive dashboard. |
| `index.html` | High-fidelity, dark-mode glassmorphic web dashboard. |
| `requirements.txt` | Core Python dependencies. |

---

## 🚀 Running the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python main.py
```
*Or using uvicorn:*
```bash
uvicorn main:app --reload --port 8000
```

### 3. Open in Browser
- **Web App**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive API Docs (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 🧪 Interactive Features

1. **One-Click Quick Starts**: Try presets like *Data Analyst (5h/wk)*, *Data Scientist*, *Frontend Dev*, or *DevOps*.
2. **Topological Learning Trail**: See your skills grouped by milestone generation.
3. **Dynamic Path Rerouting**: Click **"✓ Mark Done"** on any skill to instantly recompute remaining milestones.
4. **Adaptive Difficulty Feedback**: Rate courses as *⚡ Too Easy*, *🎯 Just Right*, or *🔥 Too Hard* to watch your difficulty bias adjust in real time.
5. **AI Assistant & Explanations**: Ask questions like *"Why do I need statistics?"* or *"Why milestone 1?"* to get clear, graph-backed explanations.
