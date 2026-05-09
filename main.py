from fastapi import FastAPI
from pydantic import BaseModel

from src.utils import (
    load_data,
    search_all,
    generate_answer
)

app = FastAPI(
    title="RAG Evaluation API",
    description="PDF Question Answering API using RAG + Ollama",
    version="1.0"
)

# -------------------------
# LOAD RAG DATABASE
# -------------------------
index_dict = load_data()


# -------------------------
# REQUEST MODEL
# -------------------------
class Question(BaseModel):
    question: str


# -------------------------
# HOME
# -------------------------
@app.get("/")
def home():
    return {
        "message": "RAG API is running"
    }


# -------------------------
# ASK QUESTION
# -------------------------
@app.post("/ask")
def ask(q: Question):

    if not index_dict:
        return {
            "error": "No RAG database found"
        }

    results = search_all(
        index_dict=index_dict,
        query=q.question
    )

    answer = generate_answer(
        results=results,
        question=q.question,
        index_dict=index_dict
    )

    return {
        "question": q.question,
        "answer": answer
    }


# -------------------------
# EVALUATE ANSWER QUALITY
# -------------------------
@app.post("/evaluate")
def evaluate(q: Question):

    if not index_dict:
        return {
            "error": "No RAG database found"
        }

    results = search_all(
        index_dict=index_dict,
        query=q.question
    )

    # number of retrieved chunks
    retrieved_chunks = len(results)

    # relevance score
    relevance_score = min(retrieved_chunks / 3, 1.0)

    # hallucination estimation
    hallucination_risk = False

    if retrieved_chunks == 0:
        hallucination_risk = True

    # confidence score
    confidence_score = round(relevance_score * 100, 2)

    return {
        "question": q.question,
        "retrieved_chunks": retrieved_chunks,
        "relevance_score": round(relevance_score, 2),
        "confidence_score": confidence_score,
        "hallucination_risk": hallucination_risk,
        "status": "Evaluation complete"
    }
