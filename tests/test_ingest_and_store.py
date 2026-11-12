import pytest
from reportlab.pdfgen import canvas

from backend.app.errors import IngestError
from backend.app.ingest import chunk_text, extract_pages, ingest_file
from backend.app.store import LocalStore


def make_pdf(path, pages):
    c = canvas.Canvas(str(path))
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()


# ---------------------------------------------------------------- chunking

def test_short_text_is_one_chunk():
    assert chunk_text("Just one sentence.", 900, 150) == ["Just one sentence."]


def test_long_text_is_split_with_overlap():
    text = " ".join(f"Sentence number {i} is about scheduling and processes." for i in range(200))
    chunks = chunk_text(text, 400, 80)
    assert len(chunks) > 5
    assert all(len(c) <= 400 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert b[:20] in a  # consecutive chunks share overlapping text


def test_text_without_spaces_still_terminates():
    chunks = chunk_text("a" * 5000, 900, 150)
    assert len(chunks) >= 5
    assert all(len(c) <= 900 for c in chunks)


# ---------------------------------------------------------------- extraction

def test_extract_pdf_pages(tmp_path):
    pdf = tmp_path / "notes.pdf"
    make_pdf(pdf, ["Paging divides memory into frames.", "Deadlock needs circular wait."])
    pages = extract_pages(pdf)
    assert [n for n, _ in pages] == [1, 2]
    assert "Paging" in pages[0][1] and "Deadlock" in pages[1][1]


def test_blank_pdf_is_rejected_with_ocr_hint(tmp_path):
    pdf = tmp_path / "scan.pdf"
    make_pdf(pdf, [""])
    with pytest.raises(IngestError, match="OCR"):
        extract_pages(pdf)


def test_unsupported_extension(tmp_path):
    f = tmp_path / "slides.pptx"
    f.write_text("x")
    with pytest.raises(IngestError):
        extract_pages(f)


def test_corrupt_pdf(tmp_path):
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"this is not a pdf")
    with pytest.raises(IngestError):
        extract_pages(f)


def test_reupload_replaces_instead_of_duplicating(tmp_path, store):
    f = tmp_path / "n.txt"
    f.write_text("Paging maps pages to frames. " * 5)
    first = ingest_file(f, "n.txt", "notes", store)
    ingest_file(f, "n.txt", "notes", store)
    assert store.count() == first["chunks"]


# ---------------------------------------------------------------- local store

def test_search_ranks_relevant_chunk_first(loaded_store):
    hits = loaded_store.query("What are the conditions for a deadlock?", k=3)
    assert hits and "deadlock" in hits[0]["text"].lower()


def test_search_matches_word_variants(loaded_store):
    hits = loaded_store.query("how does scheduling work with a time quantum", k=3)
    assert any("quantum" in h["text"].lower() for h in hits)


def test_doc_type_filter(loaded_store):
    hits = loaded_store.query("deadlock", k=5, doc_type="past_paper")
    assert hits and all(h["doc_type"] == "past_paper" for h in hits)


def test_no_match_returns_empty(loaded_store):
    assert loaded_store.query("photosynthesis chlorophyll", k=3) == []


def test_delete_document(loaded_store):
    assert loaded_store.delete_document("os_notes.txt") > 0
    assert all(d["source"] != "os_notes.txt" for d in loaded_store.list_documents())
    assert loaded_store.delete_document("os_notes.txt") == 0


def test_store_persists_to_disk(loaded_store):
    reopened = LocalStore(loaded_store.path)
    assert reopened.count() == loaded_store.count()
    assert reopened.query("deadlock", k=1)


def test_get_texts_respects_limit_and_type(loaded_store):
    text = loaded_store.get_texts("syllabus", 300)
    assert text.startswith("--- os_syllabus.txt ---")
    assert len(text) <= 330
    assert loaded_store.get_texts("notes", 50000).count("--- os_notes.txt ---") == 1
