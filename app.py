from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

#Message Schema

class Message(BaseModel):
    role: str
    content: str

#Request Schema

class ChatRequest(BaseModel):
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

    from retriever import search

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

    is_vague = (
        len(query_lower.split()) <= 3
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

            return {
                "reply": comparison_text,
                "recommendations": recommendations,
                "end_of_conversation": False
            }

    if is_vague:

        return {
            "reply": (
                "Could you share more details about the role, "
                "skills required, and experience level?"
            ),
            "recommendations": [],
            "end_of_conversation": False
        }

    recommendations = search(
        latest_user_message,
        top_k=5
    )

    #Response

    return {
        "reply": "Here are some recommended SHL assessments.",
        "recommendations": recommendations,
        "end_of_conversation": False
    }