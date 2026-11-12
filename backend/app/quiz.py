"""Quiz generation (grounded in the student's notes) and grading."""
from __future__ import annotations

import random
import uuid

from . import db
from .errors import LLMError, NoMaterialError, NotFoundError
from .llm import llm as default_llm
from .prompts import GRADE_SYSTEM, GRADE_USER, QUIZ_SCHEMA, QUIZ_SYSTEM, QUIZ_USER
from .rag import format_context, retrieve
from .store import get_store, source_label


def _as_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_question(raw, hits: list[dict]) -> dict | None:
    """Return a clean question dict, or None if the model's output for it is unusable."""
    if not isinstance(raw, dict):
        return None
    qtype = str(raw.get("type", "")).strip().lower()
    question = str(raw.get("question", "")).strip()
    if not question:
        return None

    label = None
    src = _as_int(raw.get("source"))
    if src is not None and 1 <= src <= len(hits):
        label = source_label(hits[src - 1]["source"], hits[src - 1]["page"])

    q = {"type": qtype, "question": question, "explanation": str(raw.get("explanation", "")).strip(), "source_label": label}

    if qtype == "mcq":
        options = raw.get("options")
        answer_index = _as_int(raw.get("answer_index"))
        if not isinstance(options, list) or answer_index is None:
            return None
        options = [str(o).strip() for o in options]
        if len(options) < 2 or any(not o for o in options) or len({o.lower() for o in options}) != len(options):
            return None
        options = options[:4]
        if not 0 <= answer_index < len(options):
            return None
        correct = options[answer_index]
        random.shuffle(options)  # models love putting the right answer first
        q["options"] = options
        q["answer_index"] = options.index(correct)
        return q

    if qtype == "short":
        model_answer = str(raw.get("model_answer", "")).strip()
        if not model_answer:
            return None
        key_points = raw.get("key_points")
        q["model_answer"] = model_answer
        q["key_points"] = [str(p).strip() for p in key_points if str(p).strip()] if isinstance(key_points, list) else []
        return q

    return None


def _public(q: dict) -> dict:
    """What the client sees: no answers, no explanations."""
    out = {"id": q["id"], "type": q["type"], "question": q["question"]}
    if q["type"] == "mcq":
        out["options"] = q["options"]
    return out


def generate_quiz(topic: str, n: int = 5, difficulty: str = "medium", *, store=None, llm=None) -> dict:
    store = store or get_store()
    llm = llm or default_llm
    topic = topic.strip()

    hits = retrieve(store, topic, 8)
    if not hits:
        raise NoMaterialError(
            f"I couldn't find anything about '{topic}' in your uploaded materials. "
            "Try a topic name that appears in your notes."
        )

    mcq = 1 if n == 1 else max(1, round(n * 0.7))
    short = n - mcq
    prompt = QUIZ_USER.format(
        topic=topic, difficulty=difficulty, n=n, mcq=mcq, short=short, context=format_context(hits), schema=QUIZ_SCHEMA
    )
    data = llm.complete_json(QUIZ_SYSTEM, prompt, max_tokens=3500)

    raw_questions = data.get("questions") if isinstance(data, dict) else None
    questions = [q for q in (_validate_question(r, hits) for r in (raw_questions or [])) if q][:n]
    if not questions:
        raise LLMError("The AI didn't return usable questions. Please try again.")
    for i, q in enumerate(questions, 1):
        q["id"] = f"q{i}"

    quiz_id = uuid.uuid4().hex[:12]
    db.save_quiz(quiz_id, topic, difficulty, {"topic": topic, "difficulty": difficulty, "questions": questions})
    return {"quiz_id": quiz_id, "topic": topic, "difficulty": difficulty, "questions": [_public(q) for q in questions]}


def _snap(score) -> float | None:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    return 1.0 if s >= 0.75 else 0.5 if s >= 0.25 else 0.0


def _grade_short_answers(pending: list[tuple[dict, dict]], llm) -> None:
    """Grade all short answers in one LLM call; on failure leave them ungraded (score None)."""
    items = "\n\n".join(
        f"Item {q['id']}\nQuestion: {q['question']}\nModel answer: {q['model_answer']}\n"
        f"Key points: {'; '.join(q.get('key_points') or []) or 'n/a'}\nStudent answer: {r['your_answer']}"
        for q, r in pending
    )
    try:
        data = llm.complete_json(GRADE_SYSTEM, GRADE_USER.format(items=items), max_tokens=1500)
        grades = {str(g.get("id")): g for g in data.get("grades", []) if isinstance(g, dict)}
    except LLMError:
        grades = {}

    for q, r in pending:
        g = grades.get(q["id"])
        score = _snap(g.get("score")) if g else None
        if score is None:
            r["score"] = None
            r["feedback"] = "This answer couldn't be auto-graded. Compare it with the model answer below."
        else:
            r["score"] = score
            r["feedback"] = str(g.get("feedback", "")).strip()


def grade_quiz(quiz_id: str, answers: dict[str, str], *, llm=None) -> dict:
    llm = llm or default_llm
    rec = db.get_quiz(quiz_id)
    if rec is None:
        raise NotFoundError("Quiz not found.")
    if rec["result"]:  # already submitted: return the same result, don't double-count progress
        return rec["result"]

    results: list[dict] = []
    pending: list[tuple[dict, dict]] = []
    for q in rec["payload"]["questions"]:
        given = (answers.get(q["id"]) or "").strip()
        r = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "explanation": q.get("explanation", ""),
            "source": q.get("source_label"),
            "your_answer": None,
            "score": 0.0,
            "feedback": "",
        }
        if q["type"] == "mcq":
            correct = q["options"][q["answer_index"]]
            r["correct_answer"] = correct
            pick = _as_int(given)
            if pick is not None and 0 <= pick < len(q["options"]):
                r["your_answer"] = q["options"][pick]
                r["score"] = 1.0 if pick == q["answer_index"] else 0.0
                r["feedback"] = "Correct!" if r["score"] == 1.0 else f"Not quite. The correct answer is: {correct}"
            else:
                r["feedback"] = "No answer selected."
        else:
            r["correct_answer"] = q["model_answer"]
            if given:
                r["your_answer"] = given
                pending.append((q, r))
            else:
                r["feedback"] = "No answer given."
        results.append(r)

    if pending:
        _grade_short_answers(pending, llm)

    graded = [r for r in results if r["score"] is not None]
    total = round(sum(r["score"] for r in graded), 2)
    result = {
        "quiz_id": quiz_id,
        "topic": rec["topic"],
        "score": total,
        "out_of": len(graded),
        "percent": round(100 * total / len(graded)) if graded else 0,
        "results": results,
    }
    db.add_attempts(quiz_id, rec["topic"], [r["score"] for r in graded])
    db.save_result(quiz_id, result)
    return result
