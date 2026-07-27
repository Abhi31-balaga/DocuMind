"""
Fixed-size character chunking with overlap.

Chosen over semantic chunking (splitting on headings/paragraphs) because
predictable chunk length keeps embedding cost/latency predictable and
reliably fits the LLM context window, and it doesn't depend on clean
document structure that scanned/inconsistent PDFs often lack.
"""

from dataclasses import dataclass

from src.pdf_processor import PageText
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    id: str
    text: str
    page_numbers: list[int]  # pages this chunk's text spans


def chunk_pages(pages: list[PageText], chunk_size: int = CHUNK_SIZE,
                 overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """
    Concatenate page text with page markers preserved, then slide a
    fixed-size window with overlap across the full text. Each resulting
    chunk records which source page(s) it overlaps, which lets answers
    cite a page number back to the user.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Build a single text stream, remembering the character offset at
    # which each page starts so we can map chunk offsets back to pages.
    full_text = ""
    page_starts: list[tuple[int, int]] = []  # (page_number, start_offset)
    for page in pages:
        page_starts.append((page.page_number, len(full_text)))
        full_text += page.text + "\n"

    if not full_text.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    text_len = len(full_text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = full_text[start:end].strip()

        if chunk_text:
            pages_in_chunk = _pages_for_range(page_starts, start, end)
            chunks.append(
                Chunk(id=f"chunk_{idx}", text=chunk_text, page_numbers=pages_in_chunk)
            )
            idx += 1

        if end == text_len:
            break
        start = end - overlap

    return chunks


def _pages_for_range(page_starts: list[tuple[int, int]], start: int, end: int) -> list[int]:
    """Return the page numbers whose text overlaps [start, end)."""
    pages = []
    for i, (page_num, offset) in enumerate(page_starts):
        next_offset = page_starts[i + 1][1] if i + 1 < len(page_starts) else float("inf")
        if offset < end and next_offset > start:
            pages.append(page_num)
    if not pages and page_starts:
        pages = [page_starts[0][0]]
    return pages
