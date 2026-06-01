import streamlit as st

from src.utils import (
    build_index,
    search_all,
    generate_answer,
    evaluate_response
)

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="RAG Assistant",
    layout="wide"
)

# -------------------------
# TITLE
# -------------------------
st.title("RAG Assistant")

st.write(
    "Upload PDFs and chat with your documents."
)

# -------------------------
# SESSION STATE
# -------------------------
if "index_dict" not in st.session_state:

    st.session_state.index_dict = {}

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:

    st.header("PDF Upload")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process PDFs"):

        if not uploaded_files:

            st.warning(
                "Please upload PDFs."
            )

        else:

            with st.spinner(
                "Processing PDFs..."
            ):

                st.session_state.index_dict = (
                    build_index(
                        uploaded_files
                    )
                )

            st.success(
                "PDFs processed successfully!"
            )

# -------------------------
# CHAT SECTION
# -------------------------
st.subheader("Chat")

question = st.text_input(
    "Ask a question"
)

if st.button("Ask"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif not st.session_state.index_dict:

        st.warning(
            "Please process PDFs first."
        )

    else:

        with st.spinner(
            "Searching..."
        ):

            results = search_all(
                index_dict=(
                    st.session_state.index_dict
                ),
                query=question,
                k=3,
                threshold=0.45
            )

            answer = generate_answer(
                results=results,
                question=question
            )

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "results": results
        })

# -------------------------
# SHOW CHAT
# -------------------------
for i, chat in enumerate(
    reversed(
        st.session_state.chat_history
    )
):

    st.markdown(
        f"### User\n{chat['question']}"
    )

    st.markdown(
        f"### Assistant\n{chat['answer']}"
    )

    # -------------------------
    # EVALUATE BUTTON
    # -------------------------
    if st.button(
        f"Evaluate #{i+1}",
        key=f"eval_{i}"
    ):

        evaluation = evaluate_response(
            question=chat["question"],
            answer=chat["answer"],
            results=chat["results"]
        )

        st.info(
            f"""
Question: {evaluation['question']}

Retrieved Chunks:
{evaluation['retrieved_chunks']}

Average Similarity Score:
{evaluation['avg_similarity']}

Confidence Score:
{evaluation['confidence_score']}

Relevance Level:
{evaluation['relevance_level']}

Hallucination Risk:
{evaluation['hallucination_risk']}
"""
        )

    st.divider()