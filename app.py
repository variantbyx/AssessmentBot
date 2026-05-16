from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# making convo history

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

    session_id = request.session_id

    if session_id not in conversation_store:

        conversation_store[session_id] = {
            "role_type": None,
            "seniority": None,
            "language": None,
            "industry": None,
            "skills": [],
            "shortlist": [],
            "excluded": []
        }

    state = conversation_store[session_id]

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

    user_messages = []

    for msg in request.messages:

        if msg.role == "user":
            user_messages.append(msg.content)

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

    # ---------- Detect Seniority ----------

    if "graduate" in query_lower:
        state["seniority"] = "graduate"

    elif "senior" in query_lower:
        state["seniority"] = "senior"

    elif "manager" in query_lower:
        state["seniority"] = "manager"

    # ---------- Detect Industry ----------

    if "healthcare" in query_lower:
        state["industry"] = "healthcare"

    elif "finance" in query_lower:
        state["industry"] = "finance"

    elif "sales" in query_lower:
        state["industry"] = "sales"

    # ---------- Detect Skills ----------

    skills = [
        "java",
        "python",
        "sql",
        "aws",
        "docker",
        "spring",
        "excel",
        "word"
    ]

    for skill in skills:

        if skill in query_lower:

            if skill not in state["skills"]:
                state["skills"].append(skill)

    # is_vague = (
    #     len(query_lower.split()) <= 3
    # )

    #ASK INTELLIGENT FOLLOWUPS

    needs_clarification = False
    clarification_question = ""

    if state["seniority"] is None:

        needs_clarification = True
        clarification_question = (
            "What experience level is the role? "
            "Graduate, mid-level, or senior?"
        )

    elif len(state["skills"]) == 0:

        needs_clarification = True
        clarification_question = (
            "What are the primary skills required for the role?"
        )

    #Run retrieval

    #Comparison Intent

    if "compare" in query_lower:

        recommendations = search(
            latest_user_message,
            top_k=2
        )

        if len(recommendations) >= 2:

            first = recommendations[0]
            second = recommendations[1]

            comparison_text = (
                f"{first['name']} is mainly suited for "
                f"{', '.join(first['job_levels'][:2])} roles, while "
                f"{second['name']} targets "
                f"{', '.join(second['job_levels'][:2])} roles. "
                f"The first assessment focuses on "
                f"{', '.join(first['categories'])}, whereas "
                f"the second focuses on "
                f"{', '.join(second['categories'])}."
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

    except Exception:

        return {
            "reply": "Service is temporarily busy. Please retry your request.",
            "recommendations": [],
            "end_of_conversation": False
        }

    #Response

    return {
        "reply": "Here are some recommended SHL assessments.",
        "recommendations": recommendations,
        "end_of_conversation": False
    }