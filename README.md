# PDF RAG Evaluation API

A Retrieval-Augmented Generation (RAG) system for answering questions from PDF documents using semantic search, FAISS vector database, FastAPI, and local LLMs.

## Features

- PDF Question Answering
- Semantic Search
- FAISS Vector Database
- FastAPI Backend
- Evaluation Endpoint
- Hallucination Risk Detection
- Confidence and Relevance Scoring
- Local LLM Integration (Ollama + Mistral)

## Tech Stack

- Python
- FastAPI
- Streamlit
- FAISS
- Sentence Transformers
- Ollama
- Mistral

## API Endpoints

### Ask Questions
POST `/ask`

### Evaluate Responses
POST `/evaluate`

## Run Locally

```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload
```bash
pip install -r requirements.txt
