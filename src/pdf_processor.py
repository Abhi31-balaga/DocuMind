"""
PDF text extraction.

Known limitation: no OCR. Scanned/image-only PDFs have no embedded text
layer and will return empty or near-empty text here. Adding an OCR
fallback (e.g. pytesseract over rasterized pages) is a natural next step
but is out of scope for this baseline.
"""

from dataclasses import dataclass
from typing import BinaryIO

from pypdf import PdfReader


@dataclass
class PageText:
    page_number: int  # 1-indexed
    text: str


def extract_pages(file: BinaryIO) -> list[PageText]:
    """Extract text page-by-page from a PDF file-like object."""
    reader = PdfReader(file)
    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(PageText(page_number=i, text=text))
    return pages


def extract_full_text(file: BinaryIO) -> str:
    """Extract and concatenate all page text, separated by page breaks."""
    pages = extract_pages(file)
    return "\n\n".join(p.text for p in pages if p.text.strip())


def has_extractable_text(file: BinaryIO) -> bool:
    """
    Quick check used by the UI to warn the user that a PDF looks
    scanned/image-only (no text layer) before they waste an API call.
    """
    pages = extract_pages(file)
    total_chars = sum(len(p.text.strip()) for p in pages)
    file.seek(0)
    return total_chars > 20
