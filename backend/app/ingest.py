"""Turn uploaded files into text chunks ready for the vector store."""
from __future__ import annotations

import re
from pathlib import Path

from . import config
from .errors import IngestError

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # re-join words hyphenated across lines
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)]. Plain-text files use page 0 (meaning "no page")."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            if reader.is_encrypted:
                reader.decrypt("")
            raw = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        except Exception as exc:  # corrupt / password-protected / unsupported
            raise IngestError(f"Could not read this PDF: {exc}") from exc
    elif suffix in {".txt", ".md"}:
        raw = [(0, path.read_text(encoding="utf-8", errors="ignore"))]
    else:
        raise IngestError("Unsupported file type. Upload a PDF, TXT or MD file.")

    pages = [(n, clean_text(t)) for n, t in raw]
    pages = [(n, t) for n, t in pages if t]
    if not pages:
        raise IngestError(
            "No text could be extracted. If this is a scanned PDF, run OCR on it first "
            "(e.g. with ocrmypdf) and upload the result."
        )
    return pages


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into ~`size`-character chunks, preferring paragraph/sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            window_start = start + int(size * 0.6)
            cut = max(
                text.rfind("\n\n", window_start, end),
                text.rfind(". ", window_start, end),
                text.rfind("\n", window_start, end),
            )
            if cut != -1:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        space = text.find(" ", nxt, end)  # start the next chunk on a word boundary
        if space != -1:
            nxt = space + 1
        start = nxt
    return chunks


def build_chunks(
    source: str, doc_type: str, pages: list[tuple[int, str]], size: int, overlap: int
) -> list[dict]:
    chunks: list[dict] = []
    idx = 0
    for page, text in pages:
        for piece in chunk_text(text, size, overlap):
            chunks.append(
                {
                    "id": f"{source}::p{page}::c{idx}",
                    "text": piece,
                    "source": source,
                    "page": page,
                    "doc_type": doc_type,
                    "idx": idx,
                }
            )
            idx += 1
    return chunks


def ingest_file(path: Path, source: str, doc_type: str, store) -> dict:
    """Extract, chunk and index a file. Re-uploading the same `source` replaces the old version."""
    pages = extract_pages(Path(path))
    chunks = build_chunks(source, doc_type, pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        raise IngestError("No usable text found in this file.")
    store.delete_document(source)
    store.add(chunks)
    return {"source": source, "doc_type": doc_type, "pages": len(pages), "chunks": len(chunks)}
