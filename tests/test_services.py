import pytest

from backend.app import db, planner, quiz, rag
from backend.app.errors import LLMError, NoMaterialError, NotFoundError
from backend.app.llm import LLMClient, extract_json

from .conftest import FakeLLM, mcq, short


# ---------------------------------------------------------------- JSON extraction

def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('{"a": {"b": 2}}') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_complete_json_retries_once_then_fails(monkeypatch):
    client = LLMClient()
    replies = iter(["not json", '{"ok": true}'])
    monkeypatch.setattr(client, "complete", lambda *a, **k: next(replies))
    assert client.complete_json("sys", "user") == {"ok": True}

    replies = iter(["bad", "still bad"])
    monkeypatch.setattr(client, "complete", lambda *a, **k: next(replies))
    with pytest.raises(LLMError):
        client.complete_json("sys", "user")


def test_missing_api_key_gives_helpful_error(monkeypatch):
    from backend.app import config

    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        LLMClient().complete("sys", "hi")


# ---------------------------------------------------------------- RAG

def test_ask_returns_cited_sources(loaded_store):
    llm = FakeLLM(text="Four conditions must hold [1].")
    out = rag.answer_question("What are the conditions for deadlock?", store=loaded_store, llm=llm)
    assert out["answer"].startswith("Four conditions")
    assert out["sources"][0]["cited"] is True
    assert "[1]" in llm.calls[0][2][-1]["content"]  # numbered context was sent to the model


def test_ask_with_no_matching_material_skips_the_llm(loaded_store):
    llm = FakeLLM()
    out = rag.answer_question("Explain photosynthesis", store=loaded_store, llm=llm)
    assert out["sources"] == [] and llm.calls == []


def test_notes_are_preferred_over_past_paper_questions(loaded_store):
    hits = rag.retrieve(loaded_store, "what is thrashing", 3)
    assert hits and all(h["doc_type"] != "past_paper" for h in hits)
    assert "thrashing" in hits[0]["text"].lower()


def test_history_is_cleaned_and_used_for_retrieval(loaded_store):
    history = [
        {"role": "assistant", "content": "stray opening"},
        {"role": "user", "content": "What is a deadlock?"},
        {"role": "assistant", "content": "A circular wait."},
        {"role": "user", "content": "dangling question"},
    ]
    cleaned = rag.clean_history(history)
    assert [t["role"] for t in cleaned] == ["user", "assistant"]

    llm = FakeLLM()
    # "explain that more simply" has no keywords; retrieval only works thanks to the previous question
    out = rag.answer_question("explain that more simply", history, store=loaded_store, llm=llm)
    assert out["sources"]


# ---------------------------------------------------------------- quiz generation + grading

def test_generate_quiz_hides_answers_and_drops_bad_questions(loaded_store):
    bad = {"type": "mcq", "question": "Broken?", "options": ["only one"], "answer_index": 0}
    llm = FakeLLM(json_responses=[{"questions": [mcq(), bad, short()]}])
    out = quiz.generate_quiz("deadlock", 3, "medium", store=loaded_store, llm=llm)
    assert [q["type"] for q in out["questions"]] == ["mcq", "short"]
    assert all("answer_index" not in q and "model_answer" not in q for q in out["questions"])
    assert len(out["questions"][0]["options"]) == 4


def test_generate_quiz_with_no_material(loaded_store):
    with pytest.raises(NoMaterialError):
        quiz.generate_quiz("photosynthesis", 3, "easy", store=loaded_store, llm=FakeLLM())


def test_generate_quiz_when_model_returns_nothing_usable(loaded_store):
    llm = FakeLLM(json_responses=[{"questions": [{"type": "essay"}]}])
    with pytest.raises(LLMError):
        quiz.generate_quiz("deadlock", 3, "easy", store=loaded_store, llm=llm)


def _make_quiz(loaded_store):
    llm = FakeLLM(json_responses=[{"questions": [mcq(), short()]}])
    out = quiz.generate_quiz("deadlock", 2, "medium", store=loaded_store, llm=llm)
    stored = db.get_quiz(out["quiz_id"])["payload"]["questions"]
    correct_index = next(q["answer_index"] for q in stored if q["type"] == "mcq")
    return out["quiz_id"], correct_index


def test_grading_mixed_quiz_and_progress(loaded_store):
    quiz_id, correct_index = _make_quiz(loaded_store)
    llm = FakeLLM(json_responses=[{"grades": [{"id": "q2", "score": 0.5, "feedback": "Missing two conditions."}]}])
    result = quiz.grade_quiz(quiz_id, {"q1": str(correct_index), "q2": "mutual exclusion and hold and wait"}, llm=llm)
    assert result["score"] == 1.5 and result["out_of"] == 2 and result["percent"] == 75
    assert result["results"][0]["score"] == 1.0 and result["results"][1]["score"] == 0.5

    summary = db.progress_summary()
    assert summary["questions_answered"] == 2 and summary["quizzes_taken"] == 1
    assert summary["weak_topics"] == []  # 75% is above the 70% threshold


def test_wrong_and_blank_answers_mark_topic_weak(loaded_store):
    quiz_id, correct_index = _make_quiz(loaded_store)
    wrong = str((correct_index + 1) % 4)
    result = quiz.grade_quiz(quiz_id, {"q1": wrong, "q2": ""}, llm=FakeLLM())
    assert result["score"] == 0 and result["out_of"] == 2
    assert db.progress_summary()["weak_topics"] == ["deadlock"]


def test_submitting_twice_does_not_double_count(loaded_store):
    quiz_id, correct_index = _make_quiz(loaded_store)
    first = quiz.grade_quiz(quiz_id, {"q1": str(correct_index), "q2": ""}, llm=FakeLLM())
    second = quiz.grade_quiz(quiz_id, {"q1": "0", "q2": "something"}, llm=FakeLLM())
    assert first == second
    assert db.progress_summary()["questions_answered"] == 2


def test_short_answer_grading_failure_is_not_counted(loaded_store):
    quiz_id, correct_index = _make_quiz(loaded_store)
    llm = FakeLLM(json_responses=[LLMError("down")])
    result = quiz.grade_quiz(quiz_id, {"q1": str(correct_index), "q2": "my answer"}, llm=llm)
    assert result["results"][1]["score"] is None
    assert result["out_of"] == 1 and result["score"] == 1.0


def test_unknown_quiz():
    with pytest.raises(NotFoundError):
        quiz.grade_quiz("nope", {}, llm=FakeLLM())


# ---------------------------------------------------------------- topics + plan

TOPICS = {
    "topics": [
        {"name": "Memory Management", "importance": 9, "past_paper_questions": 2, "subtopics": ["Paging", "TLB"]},
        {"name": "Deadlocks", "importance": 5, "past_paper_questions": 2, "subtopics": ["Banker's algorithm"]},
        {"name": "deadlocks", "importance": 1},  # duplicate name, dropped
        {"name": "Scheduling", "importance": "x", "subtopics": "oops"},  # bad values, defaulted
    ]
}


def test_analyze_topics_normalises_and_caches(loaded_store):
    llm = FakeLLM(json_responses=[TOPICS])
    out = planner.analyze_topics(store=loaded_store, llm=llm)
    names = [t["name"] for t in out["topics"]]
    assert names == ["Deadlocks", "Memory Management", "Scheduling"] or names[0] in {"Deadlocks", "Memory Management"}
    assert max(t["importance"] for t in out["topics"]) == 5
    assert out["has_past_papers"] is True
    assert [t for t in out["topics"] if t["name"] == "Scheduling"][0]["importance"] == 3
    assert planner.get_cached_topics()["topics"] == out["topics"]
    assert "Syllabus / notes" in llm.calls[0][2] and "Past papers" in llm.calls[0][2]


def test_analyze_topics_needs_material(store):
    with pytest.raises(NoMaterialError):
        planner.analyze_topics(store=store, llm=FakeLLM())


def test_plan_uses_weak_topics_and_truncates_days(loaded_store):
    db.kv_set(planner.TOPICS_KEY, {"topics": [{"name": "Deadlocks", "importance": 5, "subtopics": []}]})
    db.add_attempts("q", "Deadlocks", [0.0, 0.0, 1.0])
    plan_json = {
        "summary": "Focus on deadlocks.",
        "days": [{"day": i, "title": f"D{i}", "focus_topics": ["Deadlocks"], "tasks": ["Practice"], "hours": 2} for i in range(5)],
    }
    llm = FakeLLM(json_responses=[plan_json])
    plan = planner.make_plan(3, 2.0, store=loaded_store, llm=llm)
    assert len(plan["days"]) == 3 and plan["weak_topics"] == ["Deadlocks"]
    assert "Deadlocks" in llm.calls[0][2]
