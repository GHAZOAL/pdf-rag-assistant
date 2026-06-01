import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import faiss
import requests

from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -------------------------
# MODEL
# -------------------------
embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)

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
def split_chunks(
    pages,
    chunk_size=250
):

    chunks = []

    for p in pages:

        words = p["text"].split()

        for i in range(
            0,
            len(words),
            chunk_size
        ):

            chunk = " ".join(
                words[i:i + chunk_size]
            )

            if len(chunk) > 60:

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
        embed_model.encode(
            texts,
            normalize_embeddings=True
        )
    ).astype("float32")

# -------------------------
# BUILD INDEX
# -------------------------
def build_index(files):

    index_dict = {}

    for f in files:

        pages = read_pdf(f)

        chunks = split_chunks(pages)

        texts = [
            c["text"]
            for c in chunks
        ]

        pages_list = [
            c["page"]
            for c in chunks
        ]

        embeddings = embed(texts)

        index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        index.add(embeddings)

        index_dict[f.name] = {
            "index": index,
            "texts": texts,
            "pages": pages_list
        }

    return index_dict

# -------------------------
# SEARCH
# -------------------------
def search_all(
    index_dict,
    query,
    k=5,
    threshold=0.35
):

    q = embed([query])

    results = []

    for pdf, data in index_dict.items():

        scores, I = data["index"].search(
            q,
            k
        )

        found = False

        for score, i in zip(
            scores[0],
            I[0]
        ):

            if score < threshold:
                continue

            found = True

            results.append({
                "text": data["texts"][i],
                "page": int(
                    data["pages"][i]
                ),
                "source": pdf,
                "score": float(score)
            })

        if not found:

            results.append({
                "text": NOT_IN_PDF,
                "page": None,
                "source": pdf,
                "score": 0
            })

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    return results

# -------------------------
# GENERAL ANSWER
# -------------------------
def generate_general_answer(question):

    prompt = f"""
Answer briefly in one sentence.

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
        timeout=15
    )

    return (
        res.json()["response"]
        .strip()
    )

# -------------------------
# CLEAN ANSWER
# -------------------------
def clean_answer(answer):

    bad_patterns = [
        "can be inferred",
        "possibly",
        "might",
        "likely",
        "suggests",
        "implied",
        "not explicitly mentioned"
    ]

    for p in bad_patterns:

        if p in answer.lower():

            return NOT_IN_PDF

    if (
        "not mentioned"
        in answer.lower()
    ):

        return NOT_IN_PDF

    return answer

# -------------------------
# LLM JUDGE
# -------------------------
def llm_faithfulness_judge(
    question,
    context,
    answer
):

    prompt = f"""
You are a strict evaluator.

Determine whether the answer is fully supported by the context.

RULES:
- Reply ONLY PASS or FAIL
- PASS if answer is supported
- FAIL if answer includes unsupported info

Context:
{context}

Question:
{question}

Answer:
{answer}

Verdict:
"""

    try:

        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            },
            timeout=10
        )

        verdict = (
            res.json()["response"]
            .strip()
            .upper()
        )

        if "PASS" in verdict:
            return "PASS"

        return "FAIL"

    except:
        return "UNKNOWN"

# -------------------------
# GENERATE ANSWER
# -------------------------
def generate_answer(
    results,
    question
):

    final_output = ""

    found_real_answer = False

    processed = set()

    all_context = ""

    for r in results:

        pdf_name = r["source"]

        if pdf_name in processed:
            continue

        processed.add(pdf_name)

        if r["text"] == NOT_IN_PDF:

            final_output += (
                f"\nPDF: {pdf_name}\n"
                f"Answer: {NOT_IN_PDF}\n"
            )

            continue

        all_context += r["text"] + "\n"

        prompt = f"""
You are a strict extraction QA system.

RULES:
- Use ONLY context
- NEVER infer
- NEVER guess
- NEVER add information
- Maximum 20 words
- If answer missing say:
Not mentioned in this PDF

Context:
{r['text']}

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
            timeout=15
        )

        answer = (
            res.json()["response"]
            .strip()
        )

        answer = clean_answer(answer)

        if answer == NOT_IN_PDF:

            final_output += (
                f"\nPDF: {pdf_name}\n"
                f"Answer: {NOT_IN_PDF}\n"
            )

            continue

        found_real_answer = True

        final_output += (
            f"\nPDF: {pdf_name}\n"
            f"Page: {r['page']}\n"
            f"Answer: {answer}\n"
        )

    # -------------------------
    # GENERAL KNOWLEDGE
    # -------------------------
    if not found_real_answer:

        general_answer = (
            generate_general_answer(
                question
            )
        )

        final_output += (
            f"\nNo relevant information found in PDFs.\n\n"
            f"General Answer: {general_answer}\n\n"
            f"Source: General Knowledge\n"
        )

    return final_output

# -------------------------
# BASIC EVALUATION
# -------------------------
def evaluate_response(
    question,
    answer,
    results
):

    real_results = [
        r for r in results
        if r["score"] > 0
    ]

    retrieved_chunks = len(real_results)

    if retrieved_chunks == 0:

        return {
            "question": question,
            "retrieved_chunks": 0,
            "avg_similarity": 0,
            "confidence_score": 0,
            "relevance_level": "Low",
            "hallucination_risk": True,
            "llm_judge": "FAIL"
        }

    avg_score = sum(
        r["score"]
        for r in real_results
    ) / retrieved_chunks

    confidence_score = round(
        avg_score * 100,
        2
    )

    # -------------------------
    # RELEVANCE
    # -------------------------
    if avg_score > 0.75:

        relevance = "High"

    elif avg_score > 0.50:

        relevance = "Medium"

    else:

        relevance = "Low"

    hallucination_risk = (
        avg_score < 0.40
    )

    # -------------------------
    # LLM JUDGE
    # -------------------------
    context = "\n".join([
        r["text"]
        for r in real_results[:2]
    ])

    judge = llm_faithfulness_judge(
        question,
        context,
        answer
    )

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "avg_similarity": round(
            avg_score,
            3
        ),
        "confidence_score": confidence_score,
        "relevance_level": relevance,
        "hallucination_risk": hallucination_risk,
        "llm_judge": judge
    }

# -------------------------
# ADVANCED RAGAS-LIKE EVALUATION
# -------------------------
def advanced_ragas_evaluation(
    question,
    answer,
    results
):

    real_results = [
        r for r in results
        if r["score"] > 0
    ]

    if len(real_results) == 0:

        return {
            "context_precision": 0,
            "answer_relevancy": 0,
            "faithfulness": 0,
            "overall_score": 0
        }

    q_emb = embed([question])

    a_emb = embed([answer])

    contexts = [
        r["text"]
        for r in real_results
    ]

    c_embs = embed(contexts)

    # -------------------------
    # CONTEXT PRECISION
    # -------------------------
    context_scores = cosine_similarity(
        q_emb,
        c_embs
    )[0]

    context_precision = float(
        np.mean(context_scores)
    )

    # -------------------------
    # ANSWER RELEVANCY
    # -------------------------
    answer_relevancy = float(
        cosine_similarity(
            q_emb,
            a_emb
        )[0][0]
    )

    # -------------------------
    # FAITHFULNESS
    # -------------------------
    faithfulness_scores = cosine_similarity(
        a_emb,
        c_embs
    )[0]

    faithfulness = float(
        np.max(
            faithfulness_scores
        )
    )

    overall = (
        context_precision +
        answer_relevancy +
        faithfulness
    ) / 3

    return {
        "context_precision": round(
            context_precision,
            3
        ),
        "answer_relevancy": round(
            answer_relevancy,
            3
        ),
        "faithfulness": round(
            faithfulness,
            3
        ),
        "overall_score": round(
            overall,
            3
        )
    }