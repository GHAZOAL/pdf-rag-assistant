from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import os

from src.utils import (
    load_data,
    save_data,
    build_index,
    search_all,
    generate_answer
)

app = FastAPI(
    title="RAG Assistant API",
    description="Dynamic PDF RAG System",
    version="2.0"
)

# -------------------------
# LOAD DATABASE
# -------------------------
index_dict = load_data()

if index_dict is None:
    index_dict = {}


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
        "message": "RAG Assistant API is running"
    }


# -------------------------
# UPLOAD PDF
# -------------------------
@app.post("/upload_pdf")
def upload_pdf(files: List[UploadFile] = File(...)):

    global index_dict

    uploaded = []

    os.makedirs("data", exist_ok=True)

    for file in files:

        path = os.path.join("data", file.filename)

        with open(path, "wb") as f:
            f.write(file.file.read())

        uploaded.append(file.filename)

    # rebuild index
    pdf_files = []

    for filename in os.listdir("data"):

        if filename.endswith(".pdf"):

            pdf_files.append(
                open(os.path.join("data", filename), "rb")
            )

    index_dict = build_index(pdf_files)

    save_data(index_dict)

    return {
        "uploaded_pdfs": uploaded,
        "message": "PDFs uploaded and indexed successfully"
    }


# -------------------------
# ASK QUESTION
# -------------------------
@app.post("/ask")
def ask(q: Question):

    if not index_dict:
        return {
            "error": "No PDFs uploaded"
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
# EVALUATE
# -------------------------
@app.post("/evaluate")
def evaluate(q: Question):

    if not index_dict:
        return {
            "error": "No PDFs uploaded"
        }

    results = search_all(
        index_dict=index_dict,
        query=q.question
    )

    retrieved_chunks = len(results)

    relevance_score = min(retrieved_chunks / 3, 1.0)

    hallucination_risk = False

    if retrieved_chunks == 0:
        hallucination_risk = True

    confidence_score = round(relevance_score * 100, 2)

    return {
        "question": q.question,
        "retrieved_chunks": retrieved_chunks,
        "relevance_score": round(relevance_score, 2),
        "confidence_score": confidence_score,
        "hallucination_risk": hallucination_risk,
        "status": "Evaluation complete"
    }
