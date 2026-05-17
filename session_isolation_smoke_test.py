from app import ChatRequest, Message, chat


def run_stateless_smoke():
    sessions = {
        "session-a": [],
        "session-b": [],
    }

    def send(session_id, user_text):
        sessions[session_id].append(Message(role="user", content=user_text))
        req = ChatRequest(messages=sessions[session_id])
        resp = chat(req)
        # support ChatResponse instances (Pydantic) or raw dicts
        if hasattr(resp, "dict"):
            resp = resp.dict()
        print(f"SESSION={session_id}\nTEXT={user_text}\nREPLY={resp['reply'][:200]}\nRECS={len(resp.get('recommendations') or [])}\n")
        return resp

    # Initial queries
    resp_a1 = send("session-a", "Senior Java backend developer with Spring and AWS")
    assert resp_a1.get("recommendations"), "Session A shortlist should be populated"

    resp_b1 = send("session-b", "Graduate software engineer aptitude assessment")
    assert resp_b1.get("recommendations"), "Session B shortlist should be populated"
    assert resp_a1["recommendations"] != resp_b1["recommendations"], "Session A and B shortlists should differ"

    # Refinement for A (stateless: include previous messages in the request)
    resp_a2 = send("session-a", "add more similar assessments")
    assert resp_a2.get("recommendations"), "Refined shortlist must be present"
    assert resp_a2["recommendations"] != resp_b1["recommendations"], "Session B must not change when Session A is refined"

    # Removal for A
    resp_a3 = send("session-a", "remove java")
    assert len(resp_a3.get("recommendations")) <= len(resp_a2.get("recommendations")), "Removal should not increase shortlist size"

    # Confirm both
    resp_a_confirm = send("session-a", "final shortlist confirmed")
    resp_b_confirm = send("session-b", "final shortlist confirmed")
    assert resp_a_confirm["end_of_conversation"] is True
    assert resp_b_confirm["end_of_conversation"] is True

    print("Stateless session smoke test passed.")


if __name__ == "__main__":
    run_stateless_smoke()
