import json
from pathlib import Path

import pytest

from backend.app import config
from backend.app.ingest import ingest_file
from backend.app.store import LocalStore, get_store

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test gets its own database, upload folder and vector store."""
    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "LOCAL_STORE_PATH", tmp_path / "local.json")
    monkeypatch.setattr(config, "VECTOR_BACKEND", "local")
    get_store.cache_clear()
    yield
    get_store.cache_clear()


class FakeLLM:
    """Stands in for the Anthropic client. `json_responses` are returned in order by complete_json."""

    def __init__(self, text="Deadlock needs four conditions [1].", json_responses=None):
        self.text = text
        self.json_responses = list(json_responses or [])
        self.calls = []

    def complete(self, system, user, max_tokens=1500):
        self.calls.append(("complete", system, user))
        return self.text

    def complete_json(self, system, user, max_tokens=2500):
        self.calls.append(("complete_json", system, user))
        response = self.json_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "test_store.json")


@pytest.fixture
def loaded_store(store):
    """A store holding the sample syllabus, notes and past paper."""
    ingest_file(SAMPLES / "os_syllabus.txt", "os_syllabus.txt", "syllabus", store)
    ingest_file(SAMPLES / "os_notes.txt", "os_notes.txt", "notes", store)
    ingest_file(SAMPLES / "os_past_paper.txt", "os_past_paper.txt", "past_paper", store)
    return store


def mcq(question="What is a deadlock?", correct="A circular wait among processes", source=1):
    return {
        "type": "mcq",
        "question": question,
        "options": [correct, "A fast scheduler", "A page fault", "A kind of semaphore"],
        "answer_index": 0,
        "explanation": "Deadlock is a circular wait.",
        "source": source,
    }


def short(question="Name the four deadlock conditions."):
    return {
        "type": "short",
        "question": question,
        "model_answer": "Mutual exclusion, hold and wait, no preemption, circular wait.",
        "key_points": ["mutual exclusion", "hold and wait", "no preemption", "circular wait"],
        "explanation": "All four must hold.",
        "source": 1,
    }


__all__ = ["FakeLLM", "mcq", "short", "json"]
