"""Vector store for study material.

Two interchangeable backends with the same interface:
  * ChromaStore - ChromaDB with semantic embeddings (default when chromadb is installed)
  * LocalStore  - dependency-free BM25 keyword search persisted to JSON (automatic fallback)

Every hit is a dict: {id, text, source, page, doc_type, score}  (higher score = better).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path

from . import config

log = logging.getLogger(__name__)


def source_label(source: str, page: int | None) -> str:
    return f"{source}, p.{page}" if page else source


def _join_texts(chunks: list[dict], max_chars: int) -> str:
    """Concatenate chunk text in document order, with a header per source, up to max_chars."""
    out: list[str] = []
    total = 0
    current = None
    for c in sorted(chunks, key=lambda c: (c["source"], c["idx"])):
        piece = c["text"]
        if c["source"] != current:
            current = c["source"]
            piece = f"--- {current} ---\n{piece}"
        if total + len(piece) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                out.append(piece[:remaining])
            break
        out.append(piece)
        total += len(piece) + 2
    return "\n\n".join(out)


def _summarise_documents(metas: list[dict]) -> list[dict]:
    docs: dict[str, dict] = {}
    for m in metas:
        d = docs.setdefault(m["source"], {"source": m["source"], "doc_type": m["doc_type"], "chunks": 0, "pages": 0})
        d["chunks"] += 1
        d["pages"] = max(d["pages"], int(m.get("page") or 0))
    return sorted(docs.values(), key=lambda d: d["source"].lower())


# --------------------------------------------------------------------------- local (BM25)

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an and are as at be by for from has have how i if in is it its of on or that the this to was "
    "were what when where which who why will with you your do does did can could should would about "
    "into than then there these those they their them we our not no so such also "
    # question / command words that say nothing about the topic
    "explain describe define discuss compare state give tell me show list write calculate "
    "difference differences between work works use used using mean means example examples "
    "please much many more most some any all each other".split()
)
K1, B = 1.5, 0.75


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in _WORD.findall(text.lower()) if len(t) > 1 and t not in _STOP]


class LocalStore:
    name = "local-bm25"

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._chunks: list[dict] = []
        self._index = None
        if self.path.exists():
            try:
                self._chunks = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("Could not read %s; starting with an empty store.", self.path)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._chunks), encoding="utf-8")
        os.replace(tmp, self.path)

    def _build_index(self) -> None:
        docs, df, total = [], Counter(), 0
        for c in self._chunks:
            toks = _tokens(c["text"])
            tf = Counter(toks)
            docs.append((tf, len(toks)))
            df.update(tf.keys())
            total += len(toks)
        self._index = (docs, df, (total / len(docs)) if docs else 0.0)

    def count(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[dict]) -> None:
        with self._lock:
            self._chunks.extend(chunks)
            self._index = None
            self._save()

    def delete_document(self, source: str) -> int:
        with self._lock:
            before = len(self._chunks)
            self._chunks = [c for c in self._chunks if c["source"] != source]
            removed = before - len(self._chunks)
            if removed:
                self._index = None
                self._save()
            return removed

    def list_documents(self) -> list[dict]:
        with self._lock:
            return _summarise_documents(self._chunks)

    def get_texts(self, doc_type: str, max_chars: int = 12000) -> str:
        with self._lock:
            return _join_texts([c for c in self._chunks if c["doc_type"] == doc_type], max_chars)

    def query(self, text: str, k: int = 5, doc_type: str | None = None) -> list[dict]:
        with self._lock:
            if not self._chunks:
                return []
            if self._index is None:
                self._build_index()
            docs, df, avgdl = self._index
            terms = set(_tokens(text))
            if not terms:
                return []
            n = len(docs)
            scored = []
            for chunk, (tf, dl) in zip(self._chunks, docs):
                if doc_type and chunk["doc_type"] != doc_type:
                    continue
                score = 0.0
                for t in terms:
                    f = tf.get(t, 0)
                    if not f:
                        continue
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    score += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / (avgdl or 1)))
                if score > 0:
                    scored.append((score, chunk))
            scored.sort(key=lambda x: -x[0])
            return [
                {k_: c[k_] for k_ in ("id", "text", "source", "page", "doc_type")} | {"score": round(s, 4)}
                for s, c in scored[:k]
            ]


# --------------------------------------------------------------------------- ChromaDB

class ChromaStore:
    name = "chroma"

    def __init__(self, path: Path):
        import chromadb  # imported lazily so the app still runs without it

        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.col = self.client.get_or_create_collection("study_material", metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self.col.count()

    def add(self, chunks: list[dict]) -> None:
        for i in range(0, len(chunks), 100):
            batch = chunks[i : i + 100]
            self.col.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[
                    {"source": c["source"], "page": c["page"], "doc_type": c["doc_type"], "idx": c["idx"]}
                    for c in batch
                ],
            )

    def delete_document(self, source: str) -> int:
        found = self.col.get(where={"source": source}, include=["metadatas"])
        removed = len(found["ids"])
        if removed:
            self.col.delete(where={"source": source})
        return removed

    def list_documents(self) -> list[dict]:
        return _summarise_documents(self.col.get(include=["metadatas"])["metadatas"] or [])

    def get_texts(self, doc_type: str, max_chars: int = 12000) -> str:
        res = self.col.get(where={"doc_type": doc_type}, include=["documents", "metadatas"])
        chunks = [
            {"text": doc, "source": m["source"], "idx": m.get("idx", 0)}
            for doc, m in zip(res["documents"] or [], res["metadatas"] or [])
        ]
        return _join_texts(chunks, max_chars)

    def query(self, text: str, k: int = 5, doc_type: str | None = None) -> list[dict]:
        available = self.col.count()
        if doc_type:
            available = len(self.col.get(where={"doc_type": doc_type}, include=["metadatas"])["ids"])
        if available == 0 or not text.strip():
            return []
        kwargs = {"query_texts": [text], "n_results": min(k, available)}
        if doc_type:
            kwargs["where"] = {"doc_type": doc_type}
        res = self.col.query(**kwargs)
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(
                {
                    "id": cid,
                    "text": doc,
                    "source": meta["source"],
                    "page": meta.get("page", 0),
                    "doc_type": meta["doc_type"],
                    "score": round(1 - dist, 4),
                }
            )
        return hits


# --------------------------------------------------------------------------- factory

@lru_cache(maxsize=1)
def get_store():
    backend = config.VECTOR_BACKEND
    if backend in ("auto", "chroma"):
        try:
            return ChromaStore(config.CHROMA_DIR)
        except Exception as exc:
            if backend == "chroma":
                raise
            log.warning("ChromaDB unavailable (%s); using the built-in keyword store instead.", exc)
    return LocalStore(config.LOCAL_STORE_PATH)
