"""Topic map (from syllabus + past papers) and personalised study plan."""
from __future__ import annotations

import json

from . import db
from .errors import LLMError, NoMaterialError
from .llm import llm as default_llm
from .prompts import PLAN_SYSTEM, PLAN_USER, TOPICS_SYSTEM, TOPICS_USER
from .store import get_store

TOPICS_KEY = "topics"
MAX_CONTEXT_CHARS = 12000


def get_cached_topics() -> dict:
    return db.kv_get(TOPICS_KEY) or {"topics": [], "has_past_papers": False, "generated_at": None}


def _clamp_int(value, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def analyze_topics(*, store=None, llm=None) -> dict:
    store = store or get_store()
    llm = llm or default_llm

    syllabus = store.get_texts("syllabus", MAX_CONTEXT_CHARS)
    papers = store.get_texts("past_paper", MAX_CONTEXT_CHARS)
    if not syllabus:
        syllabus = store.get_texts("notes", MAX_CONTEXT_CHARS)
    if not syllabus:
        raise NoMaterialError("Upload a syllabus (or some notes) first, then analyze your topics.")

    prompt = TOPICS_USER.format(syllabus=syllabus, papers=papers or "(none supplied)")
    data = llm.complete_json(TOPICS_SYSTEM, prompt, max_tokens=2500)

    topics = []
    seen = set()
    for t in data.get("topics", []) if isinstance(data, dict) else []:
        if not isinstance(t, dict):
            continue
        name = str(t.get("name", "")).strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        subs = t.get("subtopics")
        topics.append(
            {
                "name": name,
                "importance": _clamp_int(t.get("importance"), 1, 5, 3),
                "past_paper_questions": _clamp_int(t.get("past_paper_questions"), 0, 999, 0) if papers else 0,
                "subtopics": [str(s).strip() for s in subs if str(s).strip()][:6] if isinstance(subs, list) else [],
            }
        )
    if not topics:
        raise LLMError("The AI didn't return any topics. Please try again.")

    topics.sort(key=lambda t: (-t["importance"], -t["past_paper_questions"], t["name"].lower()))
    result = {"topics": topics, "has_past_papers": bool(papers), "generated_at": db.now()}
    db.kv_set(TOPICS_KEY, result)
    return result


def make_plan(days: int, hours_per_day: float, *, store=None, llm=None) -> dict:
    llm = llm or default_llm
    cached = db.kv_get(TOPICS_KEY)
    if not cached or not cached.get("topics"):
        cached = analyze_topics(store=store, llm=llm)

    weak = db.progress_summary()["weak_topics"]
    topics_text = "\n".join(
        f"- {t['name']} (importance {t['importance']}/5; subtopics: {', '.join(t['subtopics']) or 'n/a'})"
        for t in cached["topics"]
    )
    prompt = PLAN_USER.format(days=days, hours=hours_per_day, topics=topics_text, weak=", ".join(weak) or "none yet")
    data = llm.complete_json(PLAN_SYSTEM, prompt, max_tokens=3500)

    plan_days = []
    for i, d in enumerate(data.get("days", []) if isinstance(data, dict) else [], 1):
        if not isinstance(d, dict):
            continue
        tasks = d.get("tasks")
        focus = d.get("focus_topics")
        try:
            hours = round(float(d.get("hours", hours_per_day)), 1)
        except (TypeError, ValueError):
            hours = hours_per_day
        plan_days.append(
            {
                "day": i,
                "title": str(d.get("title", f"Day {i}")).strip() or f"Day {i}",
                "focus_topics": [str(x).strip() for x in focus if str(x).strip()] if isinstance(focus, list) else [],
                "tasks": [str(x).strip() for x in tasks if str(x).strip()] if isinstance(tasks, list) else [],
                "hours": hours,
            }
        )
    plan_days = plan_days[:days]
    if not plan_days:
        raise LLMError("The AI didn't return a usable plan. Please try again.")
    return {"summary": str(data.get("summary", "")).strip(), "weak_topics": weak, "days": plan_days}
