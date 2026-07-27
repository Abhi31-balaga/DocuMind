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
import json
import math
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.chunking import Chunk
from src.config import PERSIST_DIR, EMBEDDING_MODEL, TOP_K


def collection_name_for(file_bytes: bytes) -> str:
    """Deterministic, collision-resistant collection name for a document."""
    digest = hashlib.sha256(file_bytes).hexdigest()[:16]
    return f"doc_{digest}"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _collection_path(persist_dir: Path, collection_name: str) -> Path:
    return persist_dir / f"{collection_name}.json"


def _load_collection(file_path: Path) -> dict:
    if not file_path.exists():
        return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    return json.loads(file_path.read_text(encoding="utf-8"))


def _save_collection(file_path: Path, data: dict) -> None:
    file_path.write_text(json.dumps(data), encoding="utf-8")


class VectorStore:
    def __init__(self, client: Any, persist_dir: str = PERSIST_DIR):
        self._client = client
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def collection_exists(self, name: str) -> bool:
        return _collection_path(self._persist_dir, name).exists()

    def index_chunks(self, collection_name: str, chunks: list[Chunk]) -> None:
        """Embed and store chunks in a lightweight JSON collection."""
        file_path = _collection_path(self._persist_dir, collection_name)
        data = _load_collection(file_path)
        if data["ids"]:
            return  # already indexed in a previous session

        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            embeddings = self._embed([c.text for c in batch])
            data["ids"].extend(c.id for c in batch)
            data["documents"].extend(c.text for c in batch)
            data["metadatas"].extend(
                {"page_numbers": ",".join(map(str, c.page_numbers))} for c in batch
            )
            data["embeddings"].extend(embeddings)

        _save_collection(file_path, data)

    def query(self, collection_name: str, question: str, top_k: int = TOP_K):
        file_path = _collection_path(self._persist_dir, collection_name)
        data = _load_collection(file_path)
        if not data["ids"]:
            return []

        query_embedding = self._embed([question])[0]
        scores = [
            (_cosine_similarity(query_embedding, emb), idx)
            for idx, emb in enumerate(data["embeddings"])
        ]
        scores.sort(key=lambda x: x[0], reverse=True)

        top_results = [scores[i][1] for i in range(min(top_k, len(scores)))]
        return [
            {
                "id": data["ids"][idx],
                "text": data["documents"][idx],
                "page_numbers": data["metadatas"][idx].get("page_numbers", ""),
                "distance": 1.0 - scores[i][0],
            }
            for i, idx in enumerate(top_results)
        ]
