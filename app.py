from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from copy import deepcopy
import logging
import time
from threading import RLock

from explanation_engine import (
    generate_comparison_explanation,
    generate_recommendation_explanations,
)

app = FastAPI()

logger = logging.getLogger("shl.session")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

SESSION_TTL_SECONDS = 60 * 60 * 6
SESSION_STATE_TEMPLATE = {
    "role_type": None,
    "seniority": None,
    "language": None,
    "industry": None,
    "skills": [],
    "shortlist": [],
    "excluded": [],
    "created_at": 0.0,
    "updated_at": 0.0,
}

conversation_lock = RLock()


def _new_session_state() -> Dict[str, object]:
    now = time.time()
    state = deepcopy(SESSION_STATE_TEMPLATE)
    state["created_at"] = now
    state["updated_at"] = now
    return state


def _cleanup_stale_sessions() -> None:
    now = time.time()
    stale_sessions = []

    for session_id, state in conversation_store.items():
        updated_at = float(state.get("updated_at", 0.0) or 0.0)
        if updated_at and now - updated_at > SESSION_TTL_SECONDS:
            stale_sessions.append(session_id)

    for session_id in stale_sessions:
        conversation_store.pop(session_id, None)
        logger.info("pruned stale session session_id=%s", session_id)


def _get_or_create_session_state(session_id: str) -> Dict[str, object]:
    with conversation_lock:
        _cleanup_stale_sessions()

        if session_id not in conversation_store:
            conversation_store[session_id] = _new_session_state()
            logger.info("created session session_id=%s", session_id)
        else:
            logger.info("loaded session session_id=%s", session_id)

        conversation_store[session_id]["updated_at"] = time.time()
        return conversation_store[session_id]


def _snapshot_session_state(session_id: str) -> Dict[str, object]:
    with conversation_lock:
        return deepcopy(conversation_store[session_id])


def _commit_session_state(session_id: str, state: Dict[str, object]) -> None:
    with conversation_lock:
        state["updated_at"] = time.time()
        conversation_store[session_id] = deepcopy(state)


def _normalize_message_text(value: Optional[str]) -> str:
    return (value or "").strip()


def _validate_messages(messages: List[BaseModel]) -> List[str]:
    user_messages = []

    for msg in messages:
        role = _normalize_message_text(getattr(msg, "role", ""))
        content = _normalize_message_text(getattr(msg, "content", ""))

        if role == "user" and content:
            user_messages.append(content)

    return user_messages


def _build_session_state_response(state: Dict[str, object]) -> Dict[str, object]:
    return {
        "role_type": state.get("role_type"),
        "seniority": state.get("seniority"),
        "language": state.get("language"),
        "industry": state.get("industry"),
        "skills": list(state.get("skills", [])),
        "shortlist": deepcopy(state.get("shortlist", [])),
        "excluded": list(state.get("excluded", [])),
        "created_at": state.get("created_at"),
        "updated_at": state.get("updated_at"),
    }


def _update_shortlist(session_id: str, recommendations: List[Dict[str, object]]) -> None:
    with conversation_lock:
        state = conversation_store.setdefault(session_id, _new_session_state())
        state["shortlist"] = deepcopy(recommendations)
        state["updated_at"] = time.time()
        logger.info(
            "updated shortlist session_id=%s size=%s",
            session_id,
            len(recommendations),
        )


def _append_unique_shortlist(session_id: str, recommendations: List[Dict[str, object]]) -> List[Dict[str, object]]:
    with conversation_lock:
        state = conversation_store.setdefault(session_id, _new_session_state())
        existing_names = {item.get("name") for item in state.get("shortlist", [])}
        added = 0

        for item in recommendations:
            name = item.get("name")
            if name not in existing_names:
                state["shortlist"].append(deepcopy(item))
                existing_names.add(name)
                added += 1

        state["updated_at"] = time.time()
        logger.info(
            "added shortlist items session_id=%s added=%s total=%s",
            session_id,
            added,
            len(state["shortlist"]),
        )
        return deepcopy(state["shortlist"])


def _remove_from_shortlist(session_id: str, query_text: str) -> List[Dict[str, object]]:
    query_words = {word for word in query_text.lower().split() if word}

    with conversation_lock:
        state = conversation_store.setdefault(session_id, _new_session_state())
        before = len(state["shortlist"])
        updated_shortlist = []

        for item in state["shortlist"]:
            item_name = str(item.get("name", "")).lower()
            should_remove = any(word in item_name for word in query_words)

            if should_remove:
                state["excluded"].append(item.get("name"))
            else:
                updated_shortlist.append(item)

        state["shortlist"] = updated_shortlist
        state["updated_at"] = time.time()

        logger.info(
            "removed shortlist items session_id=%s removed=%s remaining=%s",
            session_id,
            before - len(updated_shortlist),
            len(updated_shortlist),
        )

        return deepcopy(state["shortlist"])


def _touch_session_fields(state: Dict[str, object], query_lower: str) -> None:
    if "graduate" in query_lower:
        state["seniority"] = "graduate"
    elif "senior" in query_lower:
        state["seniority"] = "senior"
    elif "manager" in query_lower:
        state["seniority"] = "manager"

    if "healthcare" in query_lower:
        state["industry"] = "healthcare"
    elif "finance" in query_lower:
        state["industry"] = "finance"
    elif "sales" in query_lower:
        state["industry"] = "sales"

    skills = ["java", "python", "sql", "aws", "docker", "spring", "excel", "word"]
    for skill in skills:
        if skill in query_lower and skill not in state["skills"]:
            state["skills"].append(skill)


def _has_clear_query_signal(query_lower: str) -> bool:
    signal_terms = [
        "java",
        "python",
        "sql",
        "aws",
        "docker",
        "spring",
        "backend",
        "engineer",
        "developer",
        "coding",
        "aptitude",
        "leadership",
        "communication",
        "personality",
        "manager",
        "sales",
    ]

    return any(term in query_lower for term in signal_terms)


def _validate_chat_request(request: Any) -> Optional[str]:
    if not request.session_id or not request.session_id.strip():
        return "session_id is required and cannot be empty."

    if not request.messages:
        return "messages must contain at least one message."

    for index, msg in enumerate(request.messages):
        if not _normalize_message_text(msg.role):
            return f"messages[{index}].role is required."
        if not _normalize_message_text(msg.content):
            return f"messages[{index}].content is required."

    return None

conversation_store = {}

#Message Schema

class Message(BaseModel):
    role: str
    content: str

#Request Schema

# class ChatRequest(BaseModel):
#     messages: List[Message]

#user conversation has memory.
class ChatRequest(BaseModel):
    session_id: str
    messages: List[Message]

#homepage

@app.get("/")
def home():
    return {
        "message": "SHL Assessment Recommendation API Running"
    }

#Health Endpoint

@app.get("/health")
def health():
    return {"status": "ok"}

vague_queries = [
    "assessment",
    "test",
    "hiring",
    "job",
    "candidate"
]

#Chat Endpoint

@app.post("/chat")
def chat(request: ChatRequest):

    validation_error = _validate_chat_request(request)
    if validation_error:
        return {
            "reply": validation_error,
            "recommendations": [],
            "end_of_conversation": False,
        }

    session_id = request.session_id

    state = _get_or_create_session_state(session_id)
    state_view = _snapshot_session_state(session_id)

    try:
        from retriever import search
    except Exception:
        return {
            "reply": "Service is warming up. Please try again in a moment.",
            "recommendations": [],
            "end_of_conversation": False
        }

    latest_user_message = ""
    previous_user_message = ""

    #Get latest and previous user messages

    user_messages = _validate_messages(request.messages)

    if len(user_messages) >= 1:
        latest_user_message = user_messages[-1]

    if len(user_messages) >= 2:
        previous_user_message = user_messages[-2]

    #Merge refinement queries

    if (
        len(latest_user_message.split()) <= 3
        and previous_user_message != ""
    ):
        latest_user_message = (
            previous_user_message + " " + latest_user_message
        )

    query_lower = latest_user_message.lower()

    _touch_session_fields(state_view, query_lower)

    #refinement intent detection

    remove_intent = (
        "remove" in query_lower or
        "drop" in query_lower
    )

    add_intent = (
        "add" in query_lower or
        "include" in query_lower
    )

    confirm_intent = (
        "confirmed" in query_lower or
        "final" in query_lower or
        "lock" in query_lower
    )

    # ---------- Detect Seniority ----------

    _commit_session_state(session_id, state_view)

    # is_vague = (
    #     len(query_lower.split()) <= 3
    # )

    #ASK INTELLIGENT FOLLOWUPS

    needs_clarification = False
    clarification_question = ""

    clear_signal = _has_clear_query_signal(query_lower)

    if state_view["seniority"] is None and not clear_signal:

        needs_clarification = True
        clarification_question = (
            "What experience level is the role? "
            "Graduate, mid-level, or senior?"
        )

    elif len(state_view["skills"]) == 0 and not clear_signal:

        needs_clarification = True
        clarification_question = (
            "What are the primary skills required for the role?"
        )

    #Handle REMOVE intent

    # ---------- Remove Intent ----------

    if remove_intent:
        updated_shortlist = _remove_from_shortlist(session_id, query_lower)

        return {
            "reply": "Requested assessments removed from the shortlist." if updated_shortlist else "No matching assessments were found to remove.",
            "recommendations": updated_shortlist,
            "end_of_conversation": False
        }

    # ---------- Add Intent ----------

    if add_intent:

        new_results = search(
            latest_user_message,
            top_k=3
        )

        updated_shortlist = _append_unique_shortlist(session_id, new_results)

        return {
            "reply": "New assessments added to the shortlist.",
            "recommendations": updated_shortlist,
            "end_of_conversation": False
        }

    #Handle CONFIRM intent

    # ---------- Confirm Intent ----------

    if confirm_intent:

        current_shortlist = _snapshot_session_state(session_id)["shortlist"]
        logger.info(
            "confirmed shortlist session_id=%s size=%s",
            session_id,
            len(current_shortlist),
        )

        return {
            "reply": "Final shortlist confirmed.",
            "recommendations": current_shortlist,
            "end_of_conversation": True
        }

    #Run retrieval

    #Comparison Intent

    if "compare" in query_lower:

        recommendations = search(
            latest_user_message,
            top_k=2
        )

        #Add shortlist persistence after retrieval

        _update_shortlist(session_id, recommendations)

        if len(recommendations) >= 2:

            first = recommendations[0]
            second = recommendations[1]

            comparison_text = generate_comparison_explanation(
                latest_user_message,
                first,
                second,
            )

            # return {
            #     "reply": comparison_text,
            #     "recommendations": recommendations,
            #     "end_of_conversation": False
            # }

            return {
                "reply": comparison_text,
                "recommendations": recommendations,
                "end_of_conversation": False
            }

    if needs_clarification:

        return {
            "reply": clarification_question,
            "recommendations": None,
            "end_of_conversation": False
        }

    try:
        recommendations = search(
            latest_user_message,
            top_k=5
        )

        _update_shortlist(session_id, recommendations)

    except Exception:

        return {
            "reply": "Service is temporarily busy. Please retry your request.",
            "recommendations": [],
            "end_of_conversation": False
        }

    #Response

    return {
        "reply": generate_recommendation_explanations(
            latest_user_message,
            recommendations,
            limit=3,
        ),
        "recommendations": deepcopy(recommendations),
        "end_of_conversation": False
    }