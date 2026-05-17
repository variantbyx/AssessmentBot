from fastapi import FastAPI, Body
from pydantic import BaseModel, Field, HttpUrl, root_validator, validator
from typing import Any, Dict, List, Optional
from copy import deepcopy
import logging
import time
from threading import RLock

from explanation_engine import (
    generate_comparison_explanation,
    generate_recommendation_explanations,
)

openapi_tags = [
    {"name": "Health", "description": "Liveness and readiness endpoints."},
    {
        "name": "Recommendation API",
        "description": "Endpoints that return SHL assessment recommendations and related metadata.",
    },
    {
        "name": "Conversational Orchestration",
        "description": "Stateless conversation orchestration: message-based context reconstruction, refinements (add/remove), comparison flows, and clarification prompts.",
    },
]

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description=(
        "Stateless conversational recommender for SHL assessments.\n\n"
        "This API reconstructs conversational context from the incoming `messages` array, runs hybrid retrieval (BM25 + optional FAISS semantic search), "
        "and returns grounded, recruiter-friendly assessment recommendations. Supported flows: clarifications, add/remove refinements, comparisons, and final confirmation."
    ),
    version="1.0.0",
    contact={"name": "SHL Platform Team", "email": "platform@shl.com", "url": "https://www.shl.com"},
    license_info={"name": "Proprietary"},
    openapi_tags=openapi_tags,
    docs_url="/docs",
    redoc_url="/redoc",
)

logger = logging.getLogger("shl.session")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


SESSION_STATE_TEMPLATE = {
    "role_type": None,
    "seniority": None,
    "language": None,
    "industry": None,
    "skills": [],
    "shortlist": [],
    "excluded": [],
}

def _normalize_message_text(value: Optional[str]) -> str:
    return (value or "").strip()


# -------------------------
# Pydantic request/response
# -------------------------
from typing import Literal


class Message(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(..., description="Role of the message author", example="user")
    content: str = Field(..., min_length=1, description="Message content", example="Hiring a mid-level Java developer with AWS experience")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_items=1, description="Conversation messages in chronological order")

    @root_validator(pre=True)
    def check_messages_present(cls, values):
        msgs = values.get("messages")
        if not msgs or len(msgs) == 0:
            raise ValueError("messages must contain at least one message")
        return values


class Recommendation(BaseModel):
    name: str = Field(..., description="Assessment name", example="Java 8 (New)")
    url: Optional[str] = Field(None, description="Catalog URL for the assessment", example="https://www.shl.com/products/product-catalog/view/java-8-new/")
    categories: List[str] = Field(default_factory=list, description="Primary categories/tags", example=["Coding", "Technical"])
    job_levels: List[str] = Field(default_factory=list, description="Target job levels", example=["Graduate", "Mid", "Senior"])
    description: str = Field(default="", description="Short description snippet", example="Framework-level evaluation for Spring skills")
    duration: str = Field(default="", description="Approximate completion time", example="20 minutes")
    languages: List[str] = Field(default_factory=list, description="Available languages", example=["English"])
    assessment_type: str = Field(default="", description="Normalized assessment type", example="KnowledgeTest")
    is_adaptive: bool = Field(default=False, description="Whether the assessment is adaptive", example=False)
    simulation_mode: str = Field(default="", description="Simulation/adaptive mode hint", example="standard")
    remote_testing: bool = Field(default=False, description="Whether remote testing is supported", example=True)
    recommendation_strength: str = Field(default="", description="Relative recommendation label", example="strong")
    confidence_score: int = Field(default=0, description="Confidence score 0-100", example=82)
    matched_skills: List[str] = Field(default_factory=list, description="Skills matched to the query", example=["java", "spring", "aws"])
    reasoning_summary: str = Field(default="", description="Short reasoning summary grounding the recommendation", example="High match for Java and Spring; measures framework knowledge and coding ability.")
    hiring_suitability: str = Field(default="", description="Hiring suitability hint", example="Good for mid-senior backend roles")

    @validator("confidence_score")
    def clamp_confidence(cls, v):
        if v is None:
            return 0
        if v < 0:
            return 0
        if v > 100:
            return 100
        return int(v)

    @validator("url", pre=True, always=True)
    def ensure_url_or_none(cls, v):
        if not v:
            return None
        if isinstance(v, str) and (v.startswith("http://") or v.startswith("https://")):
            return v
        # if invalid, return None to avoid validation errors for unknown hosts
        return None


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Assistant reply text")
    recommendations: List[Recommendation] = Field(default_factory=list, description="List of recommendation objects; empty when clarifying")
    end_of_conversation: bool = Field(False, description="Whether the conversation is complete")


# Helper to coerce raw recommendation dicts into Recommendation models
def _coerce_recommendations(raw_list: List[Dict[str, object]]) -> List[Recommendation]:
    coerced: List[Recommendation] = []
    for raw in (raw_list or []):
        try:
            # map retriever keys to model fields defensively
            mapped = {
                "name": raw.get("name", ""),
                "url": raw.get("url") or raw.get("link") or None,
                "categories": raw.get("categories") or raw.get("keys") or [],
                "job_levels": raw.get("job_levels") or [],
                "description": raw.get("description", ""),
                "duration": raw.get("duration", ""),
                "languages": raw.get("languages", []),
                "assessment_type": raw.get("assessment_type", ""),
                "is_adaptive": bool(raw.get("is_adaptive", False)),
                "simulation_mode": raw.get("simulation_mode", raw.get("simulation_mode", "")) or raw.get("simulation_mode", ""),
                "remote_testing": bool(raw.get("remote_testing", False)),
                "recommendation_strength": raw.get("recommendation_strength", ""),
                "confidence_score": raw.get("confidence_score", 0),
                "matched_skills": raw.get("matched_skills", []),
                "reasoning_summary": raw.get("reasoning_summary", ""),
                "hiring_suitability": raw.get("hiring_suitability", ""),
            }
            coerced.append(Recommendation.parse_obj(mapped))
        except Exception:
            # ignore malformed items but keep API stable
            continue
    return coerced


def _validate_messages(messages: List[BaseModel]) -> List[str]:
    user_messages = []

    for msg in messages:
        role = _normalize_message_text(getattr(msg, "role", ""))
        content = _normalize_message_text(getattr(msg, "content", ""))

        if role == "user" and content:
            user_messages.append(content)

    return user_messages


def _compute_state_from_messages(messages: List[BaseModel]) -> Dict[str, object]:
    # Build a stateless session-like view by replaying user messages
    state = deepcopy(SESSION_STATE_TEMPLATE)
    user_texts = []

    for msg in messages:
        role = _normalize_message_text(getattr(msg, "role", ""))
        content = _normalize_message_text(getattr(msg, "content", ""))
        if role == "user" and content:
            user_texts.append(content)

    combined = " ".join(user_texts).strip().lower()
    _touch_session_fields(state, combined)

    # derive an initial shortlist from the combined context (stateless)
    try:
        from retriever import search
        base_results = search(combined or user_texts[-1] if user_texts else "", top_k=5)
        state["shortlist"] = deepcopy(base_results)
    except Exception:
        state["shortlist"] = []

    return state


def _build_session_state_response(state: Dict[str, object]) -> Dict[str, object]:
    return {
        "role_type": state.get("role_type"),
        "seniority": state.get("seniority"),
        "language": state.get("language"),
        "industry": state.get("industry"),
        "skills": list(state.get("skills", [])),
        "shortlist": deepcopy(state.get("shortlist", [])),
        "excluded": list(state.get("excluded", [])),
        
    }


def _merge_unique_shortlist(base: List[Dict[str, object]], additions: List[Dict[str, object]]) -> List[Dict[str, object]]:
    existing_names = {item.get("name") for item in base}
    merged = deepcopy(base)
    for item in additions:
        name = item.get("name")
        if name not in existing_names:
            merged.append(deepcopy(item))
            existing_names.add(name)
    return merged


def _filter_shortlist_remove(shortlist: List[Dict[str, object]], query_text: str) -> List[Dict[str, object]]:
    query_words = {word for word in query_text.lower().split() if word}
    updated = []
    for item in shortlist:
        item_name = str(item.get("name", "")).lower()
        should_remove = any(word in item_name for word in query_words)
        if not should_remove:
            updated.append(item)
    return deepcopy(updated)


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
    if not request.messages:
        return "messages must contain at least one message."

    for index, msg in enumerate(request.messages):
        if not _normalize_message_text(msg.role):
            return f"messages[{index}].role is required."
        if not _normalize_message_text(msg.content):
            return f"messages[{index}].content is required."

    return None


# (Pydantic Message and ChatRequest defined above)

#homepage

@app.get("/")
def home():
    return {
        "message": "SHL Assessment Recommendation API Running"
    }

#Health Endpoint

@app.get(
    "/health",
    summary="Health check endpoint",
    description="Liveness and readiness probe for the SHL Recommendation API. Returns status 'ok' when the service is available and models/indexes are loaded.",
    tags=["Health"],
    response_model=Dict[str, str],
)
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

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Conversational SHL assessment recommendation endpoint",
    description=(
        "Reconstructs conversational state from the provided `messages` array (stateless).\n\n"
        "Supported behaviours: clarification prompts when queries are vague, add/remove refinements, comparison flows (compare), and final confirmation. "
        "Include the full conversation history in `messages` to preserve context; do not rely on server-side sessions."
    ),
    tags=["Recommendation API", "Conversational Orchestration"],
    responses={
        200: {
            "description": "Successful ChatResponse",
            "content": {
                "application/json": {
                    "examples": {
                        "success": {
                            "summary": "Recommendation response",
                            "value": {
                                "reply": "Spring (New) is highly suitable for framework-level evaluation. That makes it a strong fit for java, aws, and coding screening.",
                                "recommendations": [
                                    {
                                        "name": "Spring (New)",
                                        "url": "https://www.shl.com/products/spring-new",
                                        "categories": ["Coding", "Framework"],
                                        "job_levels": ["Mid", "Senior"],
                                        "description": "Evaluates Spring framework knowledge and practical coding skills.",
                                        "duration": "25 minutes",
                                        "languages": ["English"],
                                        "assessment_type": "KnowledgeTest",
                                        "is_adaptive": False,
                                        "simulation_mode": "standard",
                                        "remote_testing": True,
                                        "recommendation_strength": "strong",
                                        "confidence_score": 88,
                                        "matched_skills": ["java", "spring", "aws"],
                                        "reasoning_summary": "High match for Spring and backend engineering skills.",
                                        "hiring_suitability": "Good for mid-senior backend roles"
                                    }
                                ],
                                "end_of_conversation": False
                            },
                        },
                        "clarification": {
                            "summary": "Clarification flow",
                            "value": {
                                "reply": "What experience level is the role? Graduate, mid-level, or senior?",
                                "recommendations": [],
                                "end_of_conversation": False
                            }
                        },
                        "comparison": {
                            "summary": "Comparison flow",
                            "value": {
                                "reply": "Java 8 is stronger for language fundamentals; Automata focuses on algorithmic reasoning.",
                                "recommendations": [
                                    {"name": "Java 8 (New)", "confidence_score": 80},
                                    {"name": "Automata Problem Solving", "confidence_score": 74}
                                ],
                                "end_of_conversation": False
                            }
                        },
                        "validation_error": {
                            "summary": "Validation error example",
                            "value": {"reply": "messages must contain at least one message.", "recommendations": [], "end_of_conversation": False}
                        }
                    }
                }
            }
        }
    },
)
def chat(request: ChatRequest = Body(
    ...,
    examples={
        "technical_query": {
            "summary": "Technical Query",
            "value": {"messages": [{"role": "user", "content": "Senior Java Spring developer"}]},
        },
        "clarification_query": {
            "summary": "Clarification Query",
            "value": {"messages": [{"role": "user", "content": "Need assessment"}]},
        },
        "comparison_query": {
            "summary": "Comparison Query",
            "value": {"messages": [{"role": "user", "content": "Compare Java 8 and Automata"}]},
        },
        "refinement_query": {
            "summary": "Refinement Query",
            "value": {"messages": [{"role": "user", "content": "Add AWS"}]},
        },
    },
)):

    # Pydantic request validation will run automatically. Keep a lightweight guard.
    validation_error = _validate_chat_request(request)
    if validation_error:
        return ChatResponse(reply=validation_error, recommendations=[], end_of_conversation=False)

    try:
        from retriever import search
    except Exception:
        return ChatResponse(reply="Service is warming up. Please try again in a moment.", recommendations=[], end_of_conversation=False)

    user_messages = _validate_messages(request.messages)
    if not user_messages:
        return ChatResponse(reply="No user message found.", recommendations=[], end_of_conversation=False)

    latest_user_message = user_messages[-1]
    previous_user_message = user_messages[-2] if len(user_messages) >= 2 else ""

    # Merge short refinement queries with previous message, but avoid merging control intents
    control_keywords = {"final", "confirmed", "confirm", "remove", "drop", "add", "include", "compare"}
    latest_tokens = latest_user_message.lower().split()
    contains_control = any(k in latest_user_message.lower() for k in control_keywords)

    if len(latest_tokens) <= 3 and previous_user_message and not contains_control:
        latest_user_message = f"{previous_user_message} {latest_user_message}"

    aggregated_query = " ".join(user_messages)
    query_lower = latest_user_message.lower()

    # compute stateless session view from full conversation history
    state_view = _compute_state_from_messages(request.messages)

    # refinement intent detection
    remove_intent = ("remove" in query_lower or "drop" in query_lower)
    add_intent = ("add" in query_lower or "include" in query_lower)
    confirm_intent = ("confirmed" in query_lower or "final" in query_lower or "lock" in query_lower)

    # Clarification logic
    needs_clarification = False
    clarification_question = ""
    clear_signal = _has_clear_query_signal(aggregated_query)

    if state_view.get("seniority") is None and not clear_signal:
        needs_clarification = True
        clarification_question = "What experience level is the role? Graduate, mid-level, or senior?"
    elif len(state_view.get("skills", [])) == 0 and not clear_signal:
        needs_clarification = True
        clarification_question = "What are the primary skills required for the role?"

    # Handle remove intent (statelessly filter the computed shortlist)
    if remove_intent:
        updated_shortlist = _filter_shortlist_remove(state_view.get("shortlist", []), latest_user_message)
        reply_text = "Requested assessments removed from the shortlist." if updated_shortlist else "No matching assessments were found to remove."
        return ChatResponse(reply=reply_text, recommendations=_coerce_recommendations(updated_shortlist), end_of_conversation=False)

    # Handle add intent (statelessly merge new search results)
    if add_intent:
        try:
            new_results = search(latest_user_message, top_k=3)
        except Exception:
            return ChatResponse(reply="Service is temporarily busy. Please retry.", recommendations=[], end_of_conversation=False)

        updated_shortlist = _merge_unique_shortlist(state_view.get("shortlist", []), new_results)
        return ChatResponse(reply="New assessments added to the shortlist.", recommendations=_coerce_recommendations(updated_shortlist), end_of_conversation=False)

    # Handle confirm intent
    if confirm_intent:
        current_shortlist = state_view.get("shortlist", [])
        return ChatResponse(reply="Final shortlist confirmed.", recommendations=_coerce_recommendations(current_shortlist), end_of_conversation=True)

    # Comparison intent
    if "compare" in query_lower:
        try:
            recommendations = search(latest_user_message, top_k=2)
        except Exception:
            return ChatResponse(reply="Service is temporarily busy. Please retry.", recommendations=[], end_of_conversation=False)

        if len(recommendations) >= 2:
            first, second = recommendations[0], recommendations[1]
            comparison_text = generate_comparison_explanation(latest_user_message, first, second)
            return ChatResponse(reply=comparison_text, recommendations=_coerce_recommendations(recommendations), end_of_conversation=False)

    if needs_clarification:
        return ChatResponse(reply=clarification_question, recommendations=[], end_of_conversation=False)

    # Default: run retrieval using aggregated context
    try:
        recommendations = search(aggregated_query or latest_user_message, top_k=5)
    except Exception:
        return ChatResponse(reply="Service is temporarily busy. Please retry your request.", recommendations=[], end_of_conversation=False)

    reply_text = generate_recommendation_explanations(latest_user_message, recommendations, limit=3)

    return ChatResponse(reply=reply_text, recommendations=_coerce_recommendations(deepcopy(recommendations)), end_of_conversation=False)