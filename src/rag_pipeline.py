"""
Retrieval-augmented generation pipeline.

Retrieval and generation are kept as two separately callable stages
(`retrieve` and `generate`) so eval.py can measure them independently:
retrieval hit-rate isolates the embedding/retrieval stage, while answer
accuracy measures the combined pipeline.
"""

from dataclasses import dataclass

from typing import Any

from openai import OpenAI
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.vector_store import VectorStore
from src.config import CHAT_MODEL, SYSTEM_PROMPT, TOP_K, get_provider_name


@dataclass
class RetrievedChunk:
    text: str
    page_numbers: str
    distance: float


@dataclass
class RagAnswer:
    answer: str
    retrieved_chunks: list[RetrievedChunk]


def retrieve(store: VectorStore, collection_name: str, question: str,
             top_k: int = TOP_K) -> list[RetrievedChunk]:
    results = store.query(collection_name, question, top_k=top_k)
    return [
        RetrievedChunk(text=r["text"], page_numbers=r["page_numbers"], distance=r["distance"])
        for r in results
    ]


def _build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        page_hint = f" (page(s): {c.page_numbers})" if c.page_numbers else ""
        blocks.append(f"[Excerpt {i}{page_hint}]\n{c.text}")
    return "\n\n".join(blocks)


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4))
def _chat(client: Any, system: str, user: str, model: str = CHAT_MODEL) -> str:
    provider = get_provider_name()
    if provider == "groq":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def answer_question(client: OpenAI, store: VectorStore, collection_name: str,
                     question: str, top_k: int = TOP_K) -> RagAnswer:
    chunks = retrieve(store, collection_name, question, top_k=top_k)
    context = _build_context(chunks)
    user_prompt = (
        f"Context excerpts from the document:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )
    answer = _chat(client, SYSTEM_PROMPT, user_prompt)
    return RagAnswer(answer=answer, retrieved_chunks=chunks)


def generate_chapter_summary(client: OpenAI, store: VectorStore, collection_name: str,
                              chapter_hint: str, top_k: int = 8) -> RagAnswer:
    """
    Chapter summaries reuse the same retrieve-then-generate path: the
    'question' is a request to summarize a section/chapter, so relevant
    chunks are retrieved the same way a factual question would be.
    """
    query = f"Content of the chapter or section: {chapter_hint}"
    chunks = retrieve(store, collection_name, query, top_k=top_k)
    context = _build_context(chunks)
    system = (
        "You are DocuMind. Summarize the requested chapter/section using ONLY "
        "the provided excerpts. Produce a concise, well-structured summary "
        "(3-6 bullet points or short paragraphs). If the excerpts don't seem "
        "to cover that chapter, say so explicitly."
    )
    user_prompt = f"Context excerpts:\n\n{context}\n\nSummarize: {chapter_hint}"
    answer = _chat(client, system, user_prompt)
    return RagAnswer(answer=answer, retrieved_chunks=chunks)


def generate_study_notes(client: OpenAI, store: VectorStore, collection_name: str,
                          topic_hint: str = "the entire document", top_k: int = 10) -> RagAnswer:
    chunks = retrieve(store, collection_name, topic_hint, top_k=top_k)
    context = _build_context(chunks)
    system = (
        "You are DocuMind. Produce structured study notes from the provided "
        "excerpts ONLY: key terms with short definitions, main ideas as "
        "bullet points, and 3-5 self-check questions with answers. Do not "
        "invent facts not present in the excerpts."
    )
    user_prompt = f"Context excerpts:\n\n{context}\n\nTopic focus: {topic_hint}"
    answer = _chat(client, system, user_prompt)
    return RagAnswer(answer=answer, retrieved_chunks=chunks)
