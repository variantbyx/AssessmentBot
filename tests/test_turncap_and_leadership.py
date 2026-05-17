import os

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

os.environ.setdefault("USE_SEMANTIC_SEARCH", "0")

from app import ChatRequest, Message, app, chat  # noqa: E402


client = TestClient(app)


def test_chat_raises_422_for_more_than_eight_user_turns():
    request = ChatRequest(
        messages=[Message(role="user", content=f"Turn {index}") for index in range(9)]
    )

    with pytest.raises(HTTPException) as exc_info:
        chat(request)

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Conversation exceeds maximum of 8 user turns"


def test_leadership_query_returns_opq_recommendations_without_clarification():
    response = client.post(
        "/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "We need a solution for senior leadership, CXO level, selection purpose",
                }
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    names = [item.get("name", "") for item in body.get("recommendations", [])]

    assert any(name in {"OPQ32r", "OPQ Leadership Report"} for name in names)
    assert body.get("reply") != "What are the primary skills required for the role?"


def test_comparison_query_parses_plain_compare_and_entities():
    response = client.post(
        "/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Compare OPQ32r and Graduate 8.0 for hiring",
                }
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body.get("recommendations", [])) >= 2
    assert body.get("reply") != "What are the primary skills required for the role?"