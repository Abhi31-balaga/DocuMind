"""
DocuMind — a GenAI-powered document assistant.

Upload a PDF, then ask questions, generate chapter summaries, or produce
study notes — all grounded in the document's actual content via
retrieval-augmented generation (RAG).

Run with:
    streamlit run app.py
"""

import io

import streamlit as st
from openai import OpenAI, OpenAIError
from groq import Groq

from src.config import get_api_key, get_provider_name, TOP_K
from src.pdf_processor import extract_pages, has_extractable_text
from src.chunking import chunk_pages
from src.vector_store import VectorStore, collection_name_for
from src.rag_pipeline import answer_question, generate_chapter_summary, generate_study_notes

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")


def init_session_state():
    defaults = {
        "messages": [],
        "collection_name": None,
        "doc_name": None,
        "indexed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def display_openai_error(error: Exception):
    if isinstance(error, OpenAIError):
        raw = str(error)
        if "insufficient_quota" in raw or "quota" in raw.lower():
            st.error(
                "OpenAI quota exceeded. Please check your OpenAI plan, billing, and API key."
            )
            return
        st.error(f"OpenAI API error: {raw}")
    else:
        st.error(f"Unexpected error: {error}")


init_session_state()

# --- Sidebar: API key + document upload ---
with st.sidebar:
    st.title("📄 DocuMind")
    st.caption("A GenAI-powered document assistant.")

    provider = get_provider_name()
    st.caption(f"Provider: {provider.upper()}")

    # Never hard-code the API key — read from a password-masked input
    # field at runtime only. It is kept in st.session_state for this
    # session and is never written to disk by this app.
    sidebar_key = st.text_input(
        f"{provider.upper()} API key",
        type="password",
        placeholder="sk-...",
        help="Your key is used only for this session and is never stored.",
    )
    api_key = get_api_key(sidebar_key)

    st.divider()
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    top_k = st.slider("Chunks retrieved per query (top-k)", min_value=2, max_value=10, value=TOP_K)

    st.divider()
    st.markdown(
        "**Known limitations**\n"
        "- No OCR: scanned/image-only PDFs return no text\n"
        "- Chat history resets on app restart (single-session state)\n"
        "- Retrieval quality depends on chunk size — tune per document type"
    )

if not api_key:
    st.info(f"Enter your {provider.upper()} API key in the sidebar to get started.")
    st.stop()

if provider == "groq":
    client = Groq(api_key=api_key)
else:
    client = OpenAI(api_key=api_key)

# --- Index the uploaded document (once per file) ---
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    collection_name = collection_name_for(file_bytes)

    if st.session_state.collection_name != collection_name:
        store = VectorStore(client)
        already_indexed = store.collection_exists(collection_name)

        if not already_indexed:
            file_like = io.BytesIO(file_bytes)
            if not has_extractable_text(file_like):
                st.error(
                    "This PDF doesn't appear to have an extractable text layer "
                    "(it may be scanned/image-only). OCR isn't supported yet."
                )
                st.stop()

            with st.spinner("Extracting and indexing document..."):
                file_like.seek(0)
                pages = extract_pages(file_like)
                chunks = chunk_pages(pages)
                store.index_chunks(collection_name, chunks)

        st.session_state.collection_name = collection_name
        st.session_state.doc_name = uploaded_file.name
        st.session_state.indexed = True
        st.session_state.messages = []  # reset chat for the new document

if not st.session_state.indexed:
    st.info("Upload a PDF in the sidebar to begin.")
    st.stop()

store = VectorStore(client)
st.success(f"Indexed: **{st.session_state.doc_name}**")

tab_chat, tab_summary, tab_notes = st.tabs(["💬 Ask questions", "📑 Chapter summary", "📝 Study notes"])

# --- Chat tab ---
with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for i, chunk in enumerate(msg["sources"], start=1):
                        st.markdown(f"**Excerpt {i}** (page(s): {chunk.page_numbers})")
                        st.text(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))

    question = st.chat_input("Ask a question about the document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = answer_question(
                        client, store, st.session_state.collection_name, question, top_k=top_k
                    )
                except OpenAIError as exc:
                    display_openai_error(exc)
                    result = None
                except Exception as exc:
                    display_openai_error(exc)
                    result = None

            if result:
                st.markdown(result.answer)
                if result.retrieved_chunks:
                    with st.expander("Sources"):
                        for i, chunk in enumerate(result.retrieved_chunks, start=1):
                            st.markdown(f"**Excerpt {i}** (page(s): {chunk.page_numbers})")
                            st.text(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))

        if result:
            st.session_state.messages.append(
                {"role": "assistant", "content": result.answer, "sources": result.retrieved_chunks}
            )

# --- Chapter summary tab ---
with tab_summary:
    st.markdown("Describe the chapter or section you want summarized (e.g. a title, heading, or topic).")
    chapter_hint = st.text_input("Chapter / section", key="chapter_hint")
    if st.button("Generate summary", key="summary_btn") and chapter_hint:
        with st.spinner("Generating summary..."):
            try:
                result = generate_chapter_summary(client, store, st.session_state.collection_name, chapter_hint)
            except OpenAIError as exc:
                display_openai_error(exc)
                result = None
            except Exception as exc:
                display_openai_error(exc)
                result = None

        if result:
            st.markdown(result.answer)
            with st.expander("Sources"):
                for i, chunk in enumerate(result.retrieved_chunks, start=1):
                    st.markdown(f"**Excerpt {i}** (page(s): {chunk.page_numbers})")
                    st.text(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))

# --- Study notes tab ---
with tab_notes:
    st.markdown("Generate structured study notes (key terms, main ideas, self-check questions).")
    topic_hint = st.text_input("Optional topic focus (leave blank for the whole document)", key="topic_hint")
    if st.button("Generate study notes", key="notes_btn"):
        with st.spinner("Generating study notes..."):
            try:
                result = generate_study_notes(
                    client, store, st.session_state.collection_name,
                    topic_hint=topic_hint or "the entire document",
                )
            except OpenAIError as exc:
                display_openai_error(exc)
                result = None
            except Exception as exc:
                display_openai_error(exc)
                result = None

        if result:
            st.markdown(result.answer)
            with st.expander("Sources"):
                for i, chunk in enumerate(result.retrieved_chunks, start=1):
                    st.markdown(f"**Excerpt {i}** (page(s): {chunk.page_numbers})")
                    st.text(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))
