"""
chat_assistant.py
Conversational AI Layer with Dual-Mode Execution.

1. Cloud LLM Mode: Uses Claude with tool access (Anthropic API) when ANTHROPIC_API_KEY is present.
2. Local Rule/Intent Mode: Seamless fallback that parses intents (milestone explanations,
   skill relevance, course breakdown, completions, feedback, goal changes) and executes
   real tool logic locally without external dependencies.
"""

import os
import re
import json

from graph_loader import load_skill_graph
from path_generator import generate_learning_path
from explainer import (
    why_this_milestone_first,
    why_this_course,
    why_do_i_need_this_skill,
    _resolve_skill_id,
)
from adaptive import (
    complete_skill,
    record_course_feedback,
    fit_semantic_matcher,
    semantic_scores_for_learner,
)
from recommender import score_course, recommend_for_skill_gap


TOOLS = [
    {
        "name": "explain_milestone",
        "description": "Explain why a given milestone is positioned where it is in the learner's path, and what it unlocks.",
        "input_schema": {
            "type": "object",
            "properties": {"milestone_number": {"type": "integer"}},
            "required": ["milestone_number"],
        },
    },
    {
        "name": "explain_skill",
        "description": "Explain why a specific skill is on the learner's path (which goal it leads to).",
        "input_schema": {
            "type": "object",
            "properties": {"skill_id": {"type": "string"}},
            "required": ["skill_id"],
        },
    },
    {
        "name": "explain_course",
        "description": "Explain why a specific course was recommended for a skill.",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_id": {"type": "string"},
                "course_title": {"type": "string"},
            },
            "required": ["skill_id", "course_title"],
        },
    },
    {
        "name": "complete_skill",
        "description": "Mark a skill as completed by the learner, updating their path.",
        "input_schema": {
            "type": "object",
            "properties": {"skill_id": {"type": "string"}},
            "required": ["skill_id"],
        },
    },
    {
        "name": "record_feedback",
        "description": "Record the learner's feedback on a course's difficulty ('too_easy', 'too_hard', or 'just_right').",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_id": {"type": "string"},
                "feedback": {"type": "string", "enum": ["too_easy", "too_hard", "just_right"]},
            },
            "required": ["skill_id", "feedback"],
        },
    },
]


def dispatch_tool_call(tool_name, tool_input, profile, G, vectorizer, course_vectors, course_id_to_row):
    """Executes a tool call against backend logic and returns a clean text reply."""

    if tool_name == "explain_milestone":
        milestones = generate_learning_path(profile, G)
        m_num = int(tool_input.get("milestone_number", 1))
        return why_this_milestone_first(m_num, milestones, G)

    if tool_name == "explain_skill":
        sid = tool_input.get("skill_id", "")
        return why_do_i_need_this_skill(sid, profile, G)

    if tool_name == "explain_course":
        milestones = generate_learning_path(profile, G)
        gap_ids = [s["id"] for m in milestones for s in m["skills"]]
        sem_scores = semantic_scores_for_learner(profile, vectorizer, course_vectors, course_id_to_row)
        recs = recommend_for_skill_gap(gap_ids, profile, G, profile.hours_per_week, semantic_scores=sem_scores)
        return why_this_course(tool_input.get("skill_id", ""), tool_input.get("course_title", ""), recs)

    if tool_name == "complete_skill":
        sid = _resolve_skill_id(tool_input.get("skill_id", ""), G)
        complete_skill(profile, sid)
        skill_name = G.nodes[sid].get("name", sid) if sid in G else sid
        return f"Great job! Marked '{skill_name}' as complete. Your learning path has been recomputed."

    if tool_name == "record_feedback":
        sid = _resolve_skill_id(tool_input.get("skill_id", ""), G)
        fb = tool_input.get("feedback", "just_right")
        record_course_feedback(profile, sid, fb)
        return f"Thanks for the feedback ({fb.replace('_', ' ')}). Your difficulty bias is now {profile.difficulty_bias:+0.1f}."

    return f"Unknown tool: {tool_name}"


def _local_intent_fallback(user_message: str, profile, G, vectorizer, course_vectors, course_id_to_row):
    """Smart local intent dispatcher that handles common conversational questions without an external LLM."""
    msg = user_message.strip()
    msg_lower = msg.lower()

    # 1. Milestone explanation: "Why milestone 1?", "What is in milestone 2?"
    m_match = re.search(r"milestone\s*(\d+)", msg_lower)
    if m_match:
        m_num = int(m_match.group(1))
        return dispatch_tool_call("explain_milestone", {"milestone_number": m_num}, profile, G, vectorizer, course_vectors, course_id_to_row)

    # 2. Skill completion: "I finished Python", "Complete sql", "Mark pandas as done"
    if any(k in msg_lower for k in ["completed", "complete", "finished", "mark as done", "done with"]):
        for sid, data in G.nodes(data=True):
            if sid in msg_lower or data.get("name", "").lower() in msg_lower:
                return dispatch_tool_call("complete_skill", {"skill_id": sid}, profile, G, vectorizer, course_vectors, course_id_to_row)

    # 3. Course difficulty feedback: "Stats was too hard", "Python was too easy"
    if "too easy" in msg_lower or "too hard" in msg_lower:
        fb = "too_easy" if "too easy" in msg_lower else "too_hard"
        for sid, data in G.nodes(data=True):
            if sid in msg_lower or data.get("name", "").lower() in msg_lower:
                return dispatch_tool_call("record_feedback", {"skill_id": sid, "feedback": fb}, profile, G, vectorizer, course_vectors, course_id_to_row)

    # 4. Skill requirement: "Why do I need stats?", "Why statistics?", "What is Docker for?"
    if any(k in msg_lower for k in ["why do i need", "why is", "why learn", "why", "what is", "need"]):
        words_in_msg = set(re.findall(r"\b\w+\b", msg_lower))
        for sid, data in G.nodes(data=True):
            name = data.get("name", "").lower()
            clean_name = name.replace("(", " ").replace(")", " ")
            name_words = [w for w in re.findall(r"\b\w+\b", clean_name) if len(w) >= 3]
            # Check skill id, full name, or key terms
            if sid in words_in_msg or name in msg_lower or any(nw in words_in_msg for nw in name_words if nw not in ["basics", "fundamentals"]):
                return dispatch_tool_call("explain_skill", {"skill_id": sid}, profile, G, vectorizer, course_vectors, course_id_to_row)


    # 5. Course explanation: "Why this course", "Why was course X recommended"
    if "course" in msg_lower and "why" in msg_lower:
        milestones = generate_learning_path(profile, G)
        if milestones and milestones[0]["skills"]:
            first_skill = milestones[0]["skills"][0]["id"]
            return dispatch_tool_call("explain_course", {"skill_id": first_skill, "course_title": ""}, profile, G, vectorizer, course_vectors, course_id_to_row)

    # 6. General status summary
    milestones = generate_learning_path(profile, G)
    known = [G.nodes[s]["name"] for s in profile.known_skill_ids if s in G]
    remaining_count = sum(len(m["skills"]) for m in milestones)
    
    return (
        f"You currently have {len(known)} known skills ({', '.join(known) if known else 'none yet'}) "
        f"and {remaining_count} skill(s) remaining across {len(milestones)} milestone(s). "
        f"You can ask me questions like 'Why do I need statistics?', 'Why milestone 1?', "
        f"or click any skill in your trail to mark it complete or explore recommendations!"
    )


def chat_with_learner(user_message, profile, G, vectorizer, course_vectors, course_id_to_row, history=None):
    """Processes learner queries using Anthropic Claude if available, with intelligent local fallback."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            history = history or []

            system_prompt = (
                "You are an expert AI learning path advisor. The learner's known skills are: "
                f"{profile.known_skill_ids}. Their target goals are: {profile.target_skill_ids}. "
                "Use the provided tools to answer questions about milestone ordering, skill prerequisites, "
                "course recommendations, and to mark skills complete or record feedback. Keep replies concise and encouraging."
            )

            messages = history + [{"role": "user", "content": user_message}]

            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )

            # Handle tool calls
            while response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result_text = dispatch_tool_call(
                            block.name, block.input, profile, G, vectorizer, course_vectors, course_id_to_row
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        })

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    system=system_prompt,
                    tools=TOOLS,
                    messages=messages,
                )

            final_text = "".join(block.text for block in response.content if hasattr(block, "text"))
            return final_text, messages + [{"role": "assistant", "content": response.content}]
        except Exception:
            pass  # Fall through to local intent router

    # Fallback to local intent handler
    reply = _local_intent_fallback(user_message, profile, G, vectorizer, course_vectors, course_id_to_row)
    history = history or []
    updated_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply}
    ]
    return reply, updated_history


if __name__ == "__main__":
    from profiling import build_learner_profile

    G = load_skill_graph()
    vectorizer, course_vectors, course_id_to_row = fit_semantic_matcher(G)
    profile = build_learner_profile("I know basic Python. I want to become a Data Analyst. 5 hrs a week.")

    print("=== Testing Local Intent Router ===")
    reply1, _ = chat_with_learner("Why do I need statistics?", profile, G, vectorizer, course_vectors, course_id_to_row)
    print("Q: Why do I need statistics?\nA:", reply1)

    reply2, _ = chat_with_learner("Why milestone 1?", profile, G, vectorizer, course_vectors, course_id_to_row)
    print("\nQ: Why milestone 1?\nA:", reply2)
