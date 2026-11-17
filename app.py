"""Streamlit web interface for the College Buddy admissions RAG."""

from __future__ import annotations

import os

import streamlit as st

from project import (
    DEMO_DOCUMENTS,
    TfidfRetriever,
    chunk_documents,
    llm_answer,
    local_answer,
)


MAX_UPLOAD_BYTES = 5 * 1024 * 1024

st.set_page_config(page_title="College Buddy", page_icon="🎓", layout="centered")
st.title("College Buddy")
st.caption("College admission guidance grounded in your uploaded documents.")

with st.sidebar:
    st.header("Knowledge files")
    uploads = st.file_uploader(
        "Add admission guidance files",
        type=["txt", "md"],
        accept_multiple_files=True,
        help=(
            "Upload UTF-8 text or Markdown files, such as a college prospectus, "
            "admissions checklist, or scholarship guide. Files are used for this "
            "session and are not saved permanently."
        ),
    )
    st.caption("Maximum combined upload size: 5 MB.")
    use_llm = st.checkbox(
        "Generate natural-language answers",
        help=(
            "Optional. Sends your question and retrieved passages to the "
            "configured OpenAI-compatible API."
        ),
    )

api_key: str | None = None
if use_llm:
    api_key = os.environ.get("OPENAI_API_KEY")
    try:
        api_key = st.secrets.get("OPENAI_API_KEY") or api_key
    except st.errors.StreamlitSecretNotFoundError:
        pass
    if not api_key:
        st.sidebar.warning(
            "Set OPENAI_API_KEY in the app environment or Streamlit secrets "
            "to use generated answers."
        )

documents: dict[str, str] = {}
upload_error: str | None = None
total_upload_bytes = 0

if uploads:
    for uploaded_file in uploads:
        raw_content = uploaded_file.getvalue()
        total_upload_bytes += len(raw_content)
        if total_upload_bytes > MAX_UPLOAD_BYTES:
            upload_error = "The combined upload size exceeds 5 MB."
            break
        try:
            content = raw_content.decode("utf-8").strip()
        except UnicodeDecodeError:
            upload_error = (
                f"'{uploaded_file.name}' is not valid UTF-8 text. "
                "Save it as UTF-8 and upload it again."
            )
            break
        if content:
            source = uploaded_file.name
            duplicate_number = 2
            while source in documents:
                source = f"{uploaded_file.name} ({duplicate_number})"
                duplicate_number += 1
            documents[source] = content

if upload_error:
    st.error(upload_error)
    st.stop()

if documents:
    st.success(f"Using {len(documents)} uploaded knowledge file(s).")
else:
    documents = DEMO_DOCUMENTS.copy()
    st.info(
        "No files uploaded. General demonstration guidance is active; "
        "upload official college information for institution-specific answers."
    )

retriever = TfidfRetriever(chunk_documents(documents))

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Retrieved passages and sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['label']}** — similarity {source['score']:.2f}"
                    )
                    st.write(source["text"])

question = st.chat_input("Ask about applications, documents, aid, or deadlines")
if question:
    results = retriever.search(question, top_k=3)
    if use_llm and not api_key:
        st.error("Configure OPENAI_API_KEY before requesting a generated answer.")
        st.stop()
    if use_llm and results:
        try:
            answer = llm_answer(
                question=question,
                results=results,
                api_key=api_key,
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                base_url=os.environ.get(
                    "OPENAI_BASE_URL", "https://api.openai.com/v1"
                ),
            )
        except RuntimeError as error:
            st.error(str(error))
            st.stop()
    else:
        answer = local_answer(question, results)
    st.session_state.messages.extend(
        [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": answer,
                "sources": [
                    {
                        "label": passage.source,
                        "score": score,
                        "text": passage.text,
                    }
                    for passage, score in results
                ],
            },
        ]
    )
    st.rerun()
