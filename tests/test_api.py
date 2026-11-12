"""HTTP-level tests. Skipped automatically unless fastapi and httpx are installed."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.llm import llm  # noqa: E402
from backend.app.main import app  # noqa: E402

from .conftest import SAMPLES, mcq  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "Four conditions [1].")
    return TestClient(app)


def upload(client, name, doc_type):
    with open(SAMPLES / name, "rb") as f:
        return client.post("/upload", files={"file": (name, f)}, data={"doc_type": doc_type})


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["vector_backend"] == "local-bm25"


def test_upload_list_ask_delete(client):
    res = upload(client, "os_notes.txt", "notes")
    assert res.status_code == 200 and res.json()["chunks"] > 0
    assert [d["source"] for d in client.get("/documents").json()["documents"]] == ["os_notes.txt"]

    ans = client.post("/ask", json={"question": "What is a deadlock?"}).json()
    assert ans["sources"] and ans["sources"][0]["cited"]

    assert client.delete("/documents/os_notes.txt").status_code == 200
    assert client.delete("/documents/os_notes.txt").status_code == 404


def test_upload_validation(client):
    assert client.post("/upload", files={"file": ("x.exe", b"x")}).status_code == 422
    assert client.post("/upload", files={"file": ("x.txt", b"hello world")}, data={"doc_type": "bogus"}).status_code == 422
    assert client.post("/upload", files={"file": ("blank.txt", b"   ")}).status_code == 422


def test_quiz_round_trip(client, monkeypatch):
    upload(client, "os_notes.txt", "notes")
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: {"questions": [mcq()]})
    quiz = client.post("/quiz", json={"topic": "deadlock", "n": 1}).json()
    assert "answer_index" not in quiz["questions"][0]
    result = client.post(f"/quiz/{quiz['quiz_id']}/submit", json={"answers": {"q1": ""}}).json()
    assert result["out_of"] == 1 and result["score"] == 0
    assert client.get("/progress").json()["weak_topics"] == ["deadlock"]


def test_errors_are_json(client):
    assert client.post("/ask", json={"question": "x"}).status_code == 422
    res = client.post("/quiz", json={"topic": "anything at all"})
    assert res.status_code == 400 and "detail" in res.json()
