# DocuMind

A GenAI-powered document assistant. Upload a PDF, then ask questions, generate chapter
summaries, or produce study notes — all grounded in the document's actual content via
retrieval-augmented generation (RAG).

## Architecture

```
                 ┌─────────────┐
   Upload PDF →  │ pdf_processor│  (pypdf text extraction, per page)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  chunking   │  (fixed-size chunks + overlap)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐        ┌────────────┐
                 │ vector_store│ ──────▶│  ChromaDB  │ (persisted to disk,
                 │ (embeddings)│        │ per-doc coll.│  one collection/PDF)
                 └──────┬──────┘        └────────────┘
                        │  top-k retrieval
                        ▼
                 ┌─────────────┐
                 │ rag_pipeline│  (context-grounded generation via OpenAI)
                 └──────┬──────┘
                        ▼
                 Streamlit UI (chat / chapter summary / study notes)
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter your OpenAI API key in the sidebar (never hard-coded — it's read from a
password-masked input field at runtime only, and never written to disk).

## Evaluation methodology (how the accuracy number is produced)

`eval.py` runs the real pipeline against a small hand-labeled test set (`test_qa.json`)
and reports two separate numbers:

- **Retrieval hit-rate** — was the chunk containing the ground-truth answer actually in
  the top-k retrieved results? This isolates the embedding/retrieval stage from the
  generation stage.
- **Answer accuracy** — did the final generated answer correctly state the expected fact,
  graded by a second LLM call (LLM-as-judge) rather than exact string match, since
  correct answers can be phrased differently.

To reproduce a number for your own test PDF:

1. Pick a PDF, write 15–20 question/expected-answer pairs by hand into `test_qa.json`
   (short ground-truth facts, not full paragraphs).
2. `python eval.py --pdf yourdoc.pdf --testset test_qa.json --api_key sk-...`
3. Report both the retrieval hit-rate and answer accuracy — don't collapse them into one
   number, since a wrong answer can come from either a retrieval miss or a generation
   error, and interviewers will ask which.

This replaces a placeholder "90% accuracy" claim with a number you can actually
reproduce and defend.

## Design decisions worth knowing for an interview

- **Fixed-size chunking over semantic chunking**: predictable chunk length keeps
  embedding cost/latency predictable and reliably fits the LLM context window.
  Semantic chunking (splitting on headings/paragraphs) assumes clean document
  structure that scanned/inconsistent PDFs often lack, so fixed-size + overlap is the
  more robust general default.
- **ChromaDB over FAISS/Pinecone**: ChromaDB persists to disk with zero external
  infra, which suits a single-user Streamlit app; FAISS has no built-in
  persistence/metadata store, and Pinecone requires a hosted service and API key
  beyond OpenAI's.
- **Per-document collections**: isolates one uploaded PDF's chunks from another's, so
  multiple users/documents don't leak into each other's retrieval results.
- **Known limitations**: no OCR (scanned/image-only PDFs return no text), single-
  session state (no persistent chat history across app restarts), and retrieval quality
  depends heavily on chunk size — tuning this per document type is a natural next step.

## Project structure

```
DocuMind/
├── app.py                 # Streamlit UI (chat, chapter summary, study notes)
├── eval.py                 # Evaluation harness (retrieval hit-rate + LLM-as-judge)
├── test_qa.json             # Sample hand-labeled test set (replace with your own)
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── src/
    ├── config.py            # Models, chunk size, top-k, API key resolution
    ├── pdf_processor.py      # PDF → per-page text (pypdf)
    ├── chunking.py           # Fixed-size chunking with overlap + page tracking
    ├── vector_store.py       # ChromaDB wrapper, per-document collections
    └── rag_pipeline.py       # Retrieve + grounded generation, summaries, study notes
```

## License

MIT — see [LICENSE](LICENSE).
