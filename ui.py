import streamlit as st
from src.utils import build_index, search_all, generate_answer, save_data, load_data

st.set_page_config(page_title="Hybrid RAG AI", layout="centered")

st.title("🤖 Hybrid RAG AI (Full Coverage)")


# -------------------------
# SESSION
# -------------------------
if "data" not in st.session_state:
    st.session_state.data = load_data()

if "chat" not in st.session_state:
    st.session_state.chat = []


# -------------------------
# UPLOAD
# -------------------------
files = st.file_uploader(
    "📄 Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("🚀 Process PDFs"):
    if files:
        with st.spinner("Processing..."):
            index_dict = build_index(files)
            save_data(index_dict)
            st.session_state.data = index_dict

        st.success("✅ Ready!")


# -------------------------
# CHAT HISTORY
# -------------------------
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------
# INPUT
# -------------------------
query = st.chat_input("Ask anything...")

if query:

    st.session_state.chat.append({"role": "user", "content": query})

    if st.session_state.data:

        index_dict = st.session_state.data
        results = search_all(index_dict, query)

        answer = generate_answer(results, query, index_dict)

    else:
        answer = "⚠️ Please upload PDFs first."

    st.session_state.chat.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()