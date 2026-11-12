"""Retrieval-augmented question answering over the student's own materials."""
from __future__ import annotations

import re

from . import config
from .llm import llm as default_llm
from .prompts import QA_SYSTEM, QA_USER
from .store import get_store, source_label

NO_MATERIAL_MSG = (
    "I couldn't find anything about that in your uploaded materials. Try rephrasing the question, "
    "or upload notes that cover this topic."
)


def retrieve(store, query: str, k: int) -> list[dict]:
    """Top-k chunks, preferring notes/syllabus. Past-paper questions are noisy context, so they are a fallback."""
    hits = store.query(query, k=k * 2)
    preferred = [h for h in hits if h["doc_type"] != "past_paper"]
    return (preferred or hits)[:k]


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i}] ({source_label(h['source'], h['page'])})\n{h['text']}" for i, h in enumerate(hits, 1))


def clean_history(history: list[dict] | None, max_turns: int = 6) -> list[dict]:
    """Keep a valid user/assistant alternation that starts with 'user' and ends with 'assistant'."""
    out: list[dict] = []
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        item = {"role": role, "content": content[:2000]}
        if out and out[-1]["role"] == role:
            out[-1] = item
        else:
            out.append(item)
    while out and out[0]["role"] != "user":
        out.pop(0)
    if out and out[-1]["role"] == "user":
        out.pop()
    return out[-max_turns:]


def answer_question(question: str, history: list[dict] | None = None, *, store=None, llm=None, k: int | None = None) -> dict:
    store = store or get_store()
    llm = llm or default_llm
    k = k or config.TOP_K
    history = clean_history(history)

    # A follow-up like "explain that more simply" has no keywords of its own, so search with the previous question too.
    query = question
    if history:
        last_user = next((t["content"] for t in reversed(history) if t["role"] == "user"), "")
        query = f"{last_user}\n{question}"

    hits = retrieve(store, query, k)
    if not hits:
        return {"answer": NO_MATERIAL_MSG, "sources": []}

    user_msg = QA_USER.format(context=format_context(hits), question=question)
    answer = llm.complete(QA_SYSTEM, history + [{"role": "user", "content": user_msg}], max_tokens=1200)

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    sources = [
        {
            "id": i,
            "source": h["source"],
            "page": h["page"],
            "doc_type": h["doc_type"],
            "snippet": h["text"][:400],
            "cited": i in cited,
        }
        for i, h in enumerate(hits, 1)
    ]
    return {"answer": answer, "sources": sources}
