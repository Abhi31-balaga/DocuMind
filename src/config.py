"""
Central configuration for DocuMind.

Fixed-size chunking is used instead of semantic chunking: predictable
chunk length keeps embedding cost/latency predictable and reliably fits
the LLM context window. Semantic chunking (splitting on headings/
paragraphs) assumes clean document structure that scanned or
inconsistently-formatted PDFs often lack, so fixed-size + overlap is the
more robust general default.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Chunking ---
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 200        # characters of overlap between consecutive chunks

# --- Retrieval ---
TOP_K = 5                  # number of chunks retrieved per query

# --- Models ---
AI_PROVIDER = os.environ.get("AI_PROVIDER", "groq").lower()
if AI_PROVIDER == "groq":
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHAT_MODEL = "llama-3.1-8b-instant"
    JUDGE_MODEL = "llama-3.1-8b-instant"  # used by eval.py for LLM-as-judge grading
else:
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-4o-mini"
    JUDGE_MODEL = "gpt-4o-mini"  # used by eval.py for LLM-as-judge grading

# --- Storage ---
# ChromaDB persists to disk with zero external infra, which suits a
# single-user Streamlit app. FAISS has no built-in persistence/metadata
# store, and Pinecone requires a hosted service and API key beyond
# OpenAI's — so ChromaDB is the simplest self-contained option here.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")

# --- Generation ---
SYSTEM_PROMPT = (
    "You are DocuMind, a document assistant. Answer the user's question "
    "using ONLY the provided context excerpts from the uploaded document. "
    "If the answer is not contained in the context, say clearly that the "
    "document doesn't appear to contain that information — do not "
    "fabricate an answer. When possible, ground your answer in specific "
    "details from the context."
)


def load_dotenv_config() -> None:
    """Load the local .env file when present so the app can use a project-scoped key."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)


load_dotenv_config()


def get_provider_name() -> str:
    return (os.environ.get("AI_PROVIDER") or "groq").strip().lower()


def get_api_key(sidebar_value: str | None = None) -> str | None:
    """
    Resolve the API key for the active provider.
    Priority:
    1. Value typed into the Streamlit sidebar this session (never persisted)
    2. Provider-specific environment variable (e.g. from a local .env)
    The key is never hard-coded and never written to disk by this app.
    """
    if sidebar_value:
        return sidebar_value.strip()

    provider = get_provider_name()
    if provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")

    return api_key.strip() if api_key else None
