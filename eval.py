"""
Evaluation harness for DocuMind.

Runs the real pipeline (same retrieve/generate code the app uses)
against a small hand-labeled test set and reports two separate numbers:

- Retrieval hit-rate: was the chunk containing the ground-truth answer
  actually in the top-k retrieved results? This isolates the
  embedding/retrieval stage from the generation stage.
- Answer accuracy: did the final generated answer correctly state the
  expected fact, graded by a second LLM call (LLM-as-judge) rather than
  exact string match, since correct answers can be phrased differently.

The two numbers are reported separately, not collapsed into one: a
wrong answer can come from either a retrieval miss or a generation
error, and that distinction matters when debugging or discussing the
system.

Usage:
    python eval.py --pdf yourdoc.pdf --testset test_qa.json --api_key sk-...

Test set format (test_qa.json):
    [
      {"question": "...", "expected_answer": "short ground-truth fact"},
      ...
    ]
"""

import argparse
import json
import sys

from openai import OpenAI

from src.pdf_processor import extract_pages
from src.chunking import chunk_pages
from src.vector_store import VectorStore, collection_name_for
from src.rag_pipeline import retrieve, answer_question
from src.config import JUDGE_MODEL, TOP_K


def load_test_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_pdf(client: OpenAI, pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    collection_name = collection_name_for(file_bytes)

    store = VectorStore(client)
    if not store.collection_exists(collection_name):
        with open(pdf_path, "rb") as f:
            pages = extract_pages(f)
        chunks = chunk_pages(pages)
        store.index_chunks(collection_name, chunks)
    return collection_name


def retrieval_hit(client: OpenAI, store: VectorStore, collection_name: str,
                   question: str, expected_answer: str, top_k: int) -> bool:
    """
    Simplified hit-rate check: was any key term from the expected answer
    present (case-insensitive substring) in the retrieved context? This
    avoids requiring a manually-labeled 'gold chunk id' for every QA
    pair, at the cost of being an approximation rather than an exact
    match against a labeled gold chunk.
    """
    chunks = retrieve(store, collection_name, question, top_k=top_k)
    context = " ".join(c.text for c in chunks).lower()
    key_terms = [t.strip(".,;:!?\"'").lower() for t in expected_answer.split() if len(t) > 3]
    if not key_terms:
        return expected_answer.strip().lower() in context
    hits = sum(1 for term in key_terms if term in context)
    return hits / len(key_terms) >= 0.5


def judge_answer(client: OpenAI, question: str, expected_answer: str, generated_answer: str) -> bool:
    """LLM-as-judge: does the generated answer correctly convey the expected fact?"""
    prompt = (
        f"Question: {question}\n"
        f"Expected (ground-truth) answer: {expected_answer}\n"
        f"Generated answer: {generated_answer}\n\n"
        "Does the generated answer correctly convey the expected fact, "
        "even if phrased differently? Reply with exactly one word: "
        "CORRECT or INCORRECT."
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict grading assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    verdict = (response.choices[0].message.content or "").strip().upper()
    return verdict.startswith("CORRECT")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DocuMind's RAG pipeline.")
    parser.add_argument("--pdf", required=True, help="Path to the test PDF")
    parser.add_argument("--testset", required=True, help="Path to test_qa.json")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Chunks retrieved per query")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)
    test_set = load_test_set(args.testset)
    if not test_set:
        print("Test set is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"Indexing {args.pdf}...")
    collection_name = index_pdf(client, args.pdf)
    store = VectorStore(client)

    hit_count = 0
    correct_count = 0
    rows = []

    for item in test_set:
        question = item["question"]
        expected = item["expected_answer"]

        hit = retrieval_hit(client, store, collection_name, question, expected, args.top_k)
        result = answer_question(client, store, collection_name, question, top_k=args.top_k)
        correct = judge_answer(client, question, expected, result.answer)

        hit_count += int(hit)
        correct_count += int(correct)
        rows.append({
            "question": question,
            "expected_answer": expected,
            "generated_answer": result.answer,
            "retrieval_hit": hit,
            "answer_correct": correct,
        })

        print(f"[{'HIT ' if hit else 'MISS'}] [{'CORRECT  ' if correct else 'INCORRECT'}] {question}")

    n = len(test_set)
    retrieval_hit_rate = hit_count / n
    answer_accuracy = correct_count / n

    print("\n--- Results ---")
    print(f"Retrieval hit-rate: {retrieval_hit_rate:.1%} ({hit_count}/{n})")
    print(f"Answer accuracy:    {answer_accuracy:.1%} ({correct_count}/{n})")
    print(
        "\nNote: a wrong final answer can come from either a retrieval miss "
        "or a generation error — that's why these are reported separately "
        "rather than as one blended number."
    )

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "retrieval_hit_rate": retrieval_hit_rate,
            "answer_accuracy": answer_accuracy,
            "rows": rows,
        }, f, indent=2)
    print("\nDetailed results written to eval_results.json")


if __name__ == "__main__":
    main()
