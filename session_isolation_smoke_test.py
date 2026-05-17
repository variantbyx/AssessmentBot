from copy import deepcopy

from app import ChatRequest, Message, chat, conversation_store


def make_request(session_id, user_text):
    return ChatRequest(
        session_id=session_id,
        messages=[Message(role="user", content=user_text)],
    )


def latest_state(session_id):
    return deepcopy(conversation_store.get(session_id, {}))


def run_case(session_id, text):
    response = chat(make_request(session_id, text))
    print(f"SESSION={session_id}\nTEXT={text}\nREPLY={response['reply']}\nRECS={len(response.get('recommendations') or [])}\n")
    return response


def assert_isolated():
    conversation_store.clear()

    session_a = "session-a"
    session_b = "session-b"

    run_case(session_a, "Senior Java backend developer with Spring and AWS")
    state_a_1 = latest_state(session_a)
    assert state_a_1["shortlist"], "Session A shortlist should be populated"

    run_case(session_b, "Graduate software engineer aptitude assessment")
    state_b_1 = latest_state(session_b)
    assert state_b_1["shortlist"], "Session B shortlist should be populated"
    assert state_a_1["shortlist"] != state_b_1["shortlist"], "Session A and B shortlists should differ"

    run_case(session_a, "add more similar assessments")
    state_a_2 = latest_state(session_a)
    state_b_2 = latest_state(session_b)
    assert state_a_2["shortlist"] != state_b_2["shortlist"], "Session B must not change when Session A is refined"

    run_case(session_a, "remove java")
    state_a_3 = latest_state(session_a)
    assert len(state_a_3["shortlist"]) <= len(state_a_2["shortlist"]), "Removal should not increase shortlist size"

    confirm_a = run_case(session_a, "final shortlist confirmed")
    confirm_b = run_case(session_b, "final shortlist confirmed")
    assert confirm_a["end_of_conversation"] is True
    assert confirm_b["end_of_conversation"] is True

    print("Session isolation smoke test passed.")


if __name__ == "__main__":
    assert_isolated()
