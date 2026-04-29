import streamlit as st
import numpy as np

from utils import (
    read_pdf,
    split_text,
    create_embeddings,
    build_index,
)

# -------------------------
# INIT STATE
# -------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.docs = []
    st.session_state.index = None


# -------------------------
# UI
# -------------------------
st.title("📄 Multi-PDF RAG Chatbot")

files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


# -------------------------
# PROCESS PDFs ONLY ONCE
# -------------------------
if files and st.session_state.index is None:

    all_chunks = []
    all_docs = []

    with st.spinner("📚 Reading PDFs..."):

        for f in files:
            text = read_pdf(f.read())
            chunks = split_text(text)

            all_chunks.extend(chunks)
            all_docs.extend([f.name] * len(chunks))

    st.session_state.chunks = all_chunks
    st.session_state.docs = all_docs

    with st.spinner("⚡ Creating embeddings..."):
        embeddings = create_embeddings(all_chunks)

    with st.spinner("⚙️ Building index..."):
        st.session_state.index = build_index(embeddings)

    st.success("✅ Ready! Ask your question.")


# -------------------------
# CHAT
# -------------------------
query = st.text_input("Ask something")

if query and st.session_state.index is not None:

    query_vec = create_embeddings([query])
    D, I = st.session_state.index.search(np.array(query_vec), k=4)

    chunks = st.session_state.chunks
    docs = st.session_state.docs

    context = []
    sources = []

    for idx in I[0]:
        context.append(chunks[idx])
        sources.append(docs[idx])

    context_text = "\n\n".join(context)

    st.subheader("📌 Answer (from PDFs)")

    st.write("Based on:")
    st.write(list(set(sources)))

    st.markdown("---")
    st.write(context_text)