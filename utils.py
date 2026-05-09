import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import faiss
import pickle
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

NOT_IN_PDF = "Not mentioned in this PDF"


# -------------------------
# READ PDF
# -------------------------
def read_pdf(file):
    reader = PdfReader(file)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "text": text.strip(),
                "page": i + 1
            })

    return pages


# -------------------------
# CHUNKING
# -------------------------
def split_chunks(pages, chunk_size=200):
    chunks = []

    for p in pages:
        words = p["text"].split()

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])

            if len(chunk) > 80:
                chunks.append({
                    "text": chunk,
                    "page": p["page"]
                })

    return chunks


# -------------------------
# EMBEDDING
# -------------------------
def embed(texts):
    return np.array(
        embed_model.encode(texts, normalize_embeddings=True)
    ).astype("float32")


# -------------------------
# BUILD INDEX
# -------------------------
def build_index(files):
    index_dict = {}

    for f in files:
        name = f.name

        pages = read_pdf(f)
        chunks = split_chunks(pages)

        texts = [c["text"] for c in chunks]
        pages_list = [c["page"] for c in chunks]

        embeddings = embed(texts)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        index_dict[name] = {
            "index": index,
            "texts": texts,
            "pages": pages_list
        }

    return index_dict


# -------------------------
# SEARCH
# -------------------------
def search_all(index_dict, query, k=3, threshold=0.4):
    q = embed([query])
    results = []

    for pdf, data in index_dict.items():
        scores, I = data["index"].search(q, k)

        for score, i in zip(scores[0], I[0]):
            if score < threshold:
                continue

            results.append({
                "text": data["texts"][i],
                "page": int(data["pages"][i]),
                "source": pdf
            })

    return results


# -------------------------
# ANSWER (FIXED + NO CONTRADICTION)
# -------------------------
def generate_answer(results, question, index_dict):

    all_pdfs = list(index_dict.keys())

    grouped = {}
    for r in results:
        grouped.setdefault(r["source"], []).append(r)

    final_output = ""
    found_any = False

    # -------------------------
    # PDF MODE
    # -------------------------
    for pdf in all_pdfs:

        if pdf not in grouped:
            final_output += f"\nPDF: {pdf}\nAnswer: {NOT_IN_PDF}\n"
            continue

        found_any = True

        items = grouped[pdf]

        # 🔥 FIX: top-3 retrieval (improves recall like regression case)
        top_chunks = items[:3]

        context = "\n".join([c["text"] for c in top_chunks])
        page = top_chunks[0]["page"]

        prompt = f"""
You are a strict document QA system.

RULES:
- Answer ONLY from context
- If not clearly present → say: Not mentioned in this PDF
- Do NOT mix answer with explanation
- Max 1 sentence

Context:
{context}

Question:
{question}

Answer:
"""

        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        answer = res.json()["response"].strip()

        # 🔥 FIX: contradiction remover
        if "not mentioned" in answer.lower():
            answer = NOT_IN_PDF

        final_output += f"\nPDF: {pdf}\nPage: {page}\nAnswer: {answer}\n"

    # -------------------------
    # GENERAL MODE
    # -------------------------
    if not found_any:

        prompt = f"""
Answer in 1 sentence.

Important:
- This is NOT from PDFs

Question:
{question}

Answer:
"""

        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        answer = res.json()["response"].strip()

        final_output += f"""
Answer:
{answer}

Source:
Not from PDFs
"""

    return final_output


# -------------------------
# SAVE
# -------------------------
def save_data(index_dict):
    with open("rag.pkl", "wb") as f:
        pickle.dump(index_dict, f)


# -------------------------
# LOAD
# -------------------------
def load_data():
    if os.path.exists("rag.pkl"):
        with open("rag.pkl", "rb") as f:
            return pickle.load(f)
    return None