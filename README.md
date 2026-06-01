# RAG-Based Multi-PDF Question Answering and Evaluation System

## Overview

This project is a lightweight Retrieval-Augmented Generation (RAG) system designed for answering questions from multiple PDF documents while providing answer quality evaluation.

The system allows users to upload multiple PDF files, retrieve relevant information using semantic search, generate answers, and evaluate answer quality using metrics inspired by the RAGAS framework.

The project was developed as an academic implementation of modern RAG architectures and retrieval evaluation techniques.

---

## Features

* Multi-PDF Upload Support
* Semantic Search using Sentence Transformers
* FAISS Vector Database
* PDF Source Identification
* Page Number Detection
* Question History
* Retrieval-Based Question Answering
* Hallucination Detection
* Faithfulness Estimation
* Relevance Scoring
* Confidence Scoring
* Streamlit User Interface
* Dynamic Document Processing

---

## System Architecture

PDF Upload

↓

Text Extraction

↓

Chunking

↓

Sentence Embeddings

↓

FAISS Vector Store

↓

Semantic Retrieval

↓

Answer Generation

↓

Evaluation Metrics

---

## Technologies Used

* Python
* Streamlit
* PyMuPDF
* Sentence Transformers
* FAISS
* NumPy
* Pandas
* Scikit-Learn
* LangChain

---

## Evaluation Components

The system includes lightweight implementations inspired by the RAGAS evaluation framework:

* Faithfulness Estimation
* Answer Relevancy
* Hallucination Detection
* Retrieval Quality Assessment
* Similarity-Based Confidence Scoring

---

## Example Output

Question:
What is machine learning?

Answer:
Machine learning is a subfield of computer science concerned with building algorithms from examples.

Source:
PDF: the-hundred-page-machine-learning-book_compress.pdf

Page: 7

Evaluation:

* Confidence Score
* Relevance Level
* Hallucination Risk
* Similarity Score

---

## Installation

pip install -r requirements.txt

---

## Run Application

streamlit run ui.py

or

streamlit run app.py

---

## Research Inspiration

This project was inspired by the paper:

RAGAS: Automated Evaluation of Retrieval Augmented Generation Systems

The implementation adopts several evaluation concepts from RAGAS while maintaining a lightweight architecture suitable for academic projects and experimentation.

---

## Author

Academic RAG Evaluation Project

Built using Retrieval-Augmented Generation, Semantic Search, FAISS, and Streamlit.

