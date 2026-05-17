import os
from fastapi.testclient import TestClient
from app import app

os.environ.setdefault("USE_SEMANTIC_SEARCH", "0")
client = TestClient(app)


def test_confirm_sets_end_of_conversation_true():
    # Simulate a short conversation where user confirms final shortlist
    messages = [
        {"role": "user", "content": "Looking for a mid-level Java developer"},
        {"role": "user", "content": "final"}
    ]
    resp = client.post('/chat', json={"messages": messages})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body.get('end_of_conversation'), bool)
    assert body.get('end_of_conversation') is True
