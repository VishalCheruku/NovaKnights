"""
test_system.py
End-to-end verification script for Learning Path Recommender.
"""

import json
import urllib.request

BASE = "http://127.0.0.1:8000"


def main():
    print("=== 1. Checking Health Endpoint ===")
    health = json.loads(urllib.request.urlopen(f"{BASE}/health").read().decode())
    print("Health Status:", health)
    assert health["status"] == "healthy"

    print("\n=== 2. Starting Learner Session ===")
    start_payload = {
        "text": "I know basic Python and some SQL. I want to become a Data Analyst. I can commit 5 hrs a week."
    }
    req = urllib.request.Request(
        f"{BASE}/session/start",
        data=json.dumps(start_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    session_data = json.loads(urllib.request.urlopen(req).read().decode())
    sid = session_data["session_id"]
    print("Session Created:", sid)
    print("Known Skills Detected:", session_data["known_skills"])
    print("Target Skills Mapped:", session_data["target_skills"])
    print("Pace:", session_data["hours_per_week"], "hrs/week")
    assert "py_basics" in session_data["known_skills"] or "databases" in session_data["known_skills"]

    print("\n=== 3. Retrieving Generated Path ===")
    path_data = json.loads(urllib.request.urlopen(f"{BASE}/session/{sid}/path").read().decode())
    milestones = path_data["milestones"]
    print(f"Generated {len(milestones)} milestones:")
    for m in milestones:
        names = [s["name"] for s in m["skills"]]
        print(f"  Milestone {m['milestone_number']}: {names}")

    print("\n=== 4. Fetching Course Recommendations ===")
    rec_data = json.loads(urllib.request.urlopen(f"{BASE}/session/{sid}/recommendations").read().decode())
    for sid_key, rec_list in rec_data["recommendations"].items():
        print(f"Skill '{sid_key}':")
        for r in rec_list:
            print(f"  - [{r['score']}] {r['course']['title']} ({r['course']['provider']})")

    print("\n=== 5. Explaining Milestone 1 ===")
    exp_m1 = json.loads(urllib.request.urlopen(f"{BASE}/session/{sid}/explain/milestone/1").read().decode())
    print("Explanation:", exp_m1["explanation"])

    print("\n=== 6. Completing Skill 'stats_basics' ===")
    complete_req = urllib.request.Request(
        f"{BASE}/session/{sid}/complete-skill",
        data=json.dumps({"skill_id": "stats_basics"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    complete_res = json.loads(urllib.request.urlopen(complete_req).read().decode())
    print("Updated Known Skills:", complete_res["known_skills"])
    print("Updated Milestones Count:", len(complete_res["updated_milestones"]))

    print("\n=== 7. Recording Feedback ('too_easy') ===")
    fb_req = urllib.request.Request(
        f"{BASE}/session/{sid}/feedback",
        data=json.dumps({"skill_id": "pandas_numpy", "feedback": "too_easy"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    fb_res = json.loads(urllib.request.urlopen(fb_req).read().decode())
    print("Adjusted Difficulty Bias:", fb_res["difficulty_bias"])

    print("\n=== 8. Conversational Chat Query ===")
    chat_req = urllib.request.Request(
        f"{BASE}/session/{sid}/chat",
        data=json.dumps({"message": "Why milestone 1?"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    chat_res = json.loads(urllib.request.urlopen(chat_req).read().decode())
    print("Chat Assistant Reply:\n", chat_res["reply"])

    print("\n=== 9. Verifying Web Frontend Serving ===")
    html_content = urllib.request.urlopen(f"{BASE}/").read().decode()
    print("Frontend HTML Size:", len(html_content), "bytes")
    assert "NOVA" in html_content

    print("\n==========================================")
    print(" ALL END-TO-END VERIFICATION CHECKS PASSED!")
    print("==========================================")


if __name__ == "__main__":
    main()
