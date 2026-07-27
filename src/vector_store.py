"""
ChromaDB wrapper.

ChromaDB persists to disk with zero external infra, which suits a
single-user Streamlit app; FAISS has no built-in persistence/metadata
store, and Pinecone requires a hosted service and API key beyond
OpenAI's.

Per-document collections: each uploaded PDF gets its own collection,
named by a hash of its content. This isolates one document's chunks
from another's so multiple users/documents don't leak into each
other's retrieval results.
"""

import hashlib
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from openai import OpenAI

from src.chunking import Chunk
from src.config import PERSIST_DIR, EMBEDDING_MODEL, TOP_K, get_provider_name


def collection_name_for(file_bytes: bytes) -> str:
    """Deterministic, collision-resistant collection name for a document."""
    digest = hashlib.sha256(file_bytes).hexdigest()[:16]
    return f"doc_{digest}"


class VectorStore:
    def __init__(self, client: Any, persist_dir: str = PERSIST_DIR):
        self._client = client
        self._chroma = chromadb.PersistentClient(path=persist_dir)
        self._local_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        provider = get_provider_name()
        if provider == "groq":
            return self._local_embedding_fn(texts)

        response = self._client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def collection_exists(self, name: str) -> bool:
        existing = [c.name for c in self._chroma.list_collections()]
        return name in existing

    def index_chunks(self, collection_name: str, chunks: list[Chunk]) -> None:
        """Embed and store chunks. Skips work if already indexed."""
        collection = self._chroma.get_or_create_collection(collection_name)
        if collection.count() > 0:
            return  # already indexed in a previous session

        # Batch embeddings to stay well under API request size limits.
        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            embeddings = self._embed([c.text for c in batch])
            collection.add(
                ids=[c.id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[{"page_numbers": ",".join(map(str, c.page_numbers))} for c in batch],
            )

    def query(self, collection_name: str, question: str, top_k: int = TOP_K):
        collection = self._chroma.get_or_create_collection(collection_name)
        query_embedding = self._embed([question])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(collection.count(), 1)),
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            {
                "id": ids[i],
                "text": documents[i],
                "page_numbers": metadatas[i].get("page_numbers", ""),
                "distance": distances[i],
            }
            for i in range(len(documents))
        ]
