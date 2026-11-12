"""Small SQLite layer: saved quizzes, per-question attempts (for weak-topic tracking), key/value cache."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

WEAK_THRESHOLD = 0.7  # topics with average score below this are flagged as weak

SCHEMA = """
CREATE TABLE IF NOT EXISTS quizzes (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    difficulty TEXT,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    result TEXT
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    quiz_id TEXT,
    topic TEXT NOT NULL,
    score REAL NOT NULL,
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_initialized_for: str | None = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def session():
    global _initialized_for
    path = Path(config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        if _initialized_for != str(path):
            conn.executescript(SCHEMA)
            _initialized_for = str(path)
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------- quizzes ----------

def save_quiz(quiz_id: str, topic: str, difficulty: str, payload: dict) -> None:
    with session() as c:
        c.execute(
            "INSERT INTO quizzes (id, topic, difficulty, created_at, payload) VALUES (?,?,?,?,?)",
            (quiz_id, topic, difficulty, now(), json.dumps(payload)),
        )


def get_quiz(quiz_id: str) -> dict | None:
    with session() as c:
        row = c.execute("SELECT * FROM quizzes WHERE id = ?", (quiz_id,)).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "topic": row["topic"],
        "difficulty": row["difficulty"],
        "payload": json.loads(row["payload"]),
        "result": json.loads(row["result"]) if row["result"] else None,
    }


def save_result(quiz_id: str, result: dict) -> None:
    with session() as c:
        c.execute("UPDATE quizzes SET result = ? WHERE id = ?", (json.dumps(result), quiz_id))


# ---------- attempts / progress ----------

def add_attempts(quiz_id: str, topic: str, scores: list[float]) -> None:
    ts = now()
    with session() as c:
        c.executemany(
            "INSERT INTO attempts (quiz_id, topic, score, ts) VALUES (?,?,?,?)",
            [(quiz_id, topic.strip(), float(s), ts) for s in scores],
        )


def progress_summary() -> dict:
    with session() as c:
        per_topic = c.execute(
            "SELECT MAX(topic) AS topic, COUNT(*) AS attempts, AVG(score) AS avg_score, MAX(ts) AS last_ts "
            "FROM attempts GROUP BY LOWER(TRIM(topic)) ORDER BY avg_score ASC, attempts DESC"
        ).fetchall()
        per_quiz = c.execute(
            "SELECT quiz_id, MAX(topic) AS topic, AVG(score) AS avg_score, MIN(ts) AS ts "
            "FROM attempts WHERE quiz_id IS NOT NULL GROUP BY quiz_id ORDER BY ts ASC, quiz_id ASC"
        ).fetchall()
        overall = c.execute("SELECT COUNT(*) AS n, AVG(score) AS avg_score FROM attempts").fetchone()

    topics = [
        {
            "topic": r["topic"],
            "attempts": r["attempts"],
            "accuracy": round(r["avg_score"], 3),
            "last_practised": r["last_ts"],
            "weak": r["avg_score"] < WEAK_THRESHOLD,
        }
        for r in per_topic
    ]
    return {
        "questions_answered": overall["n"],
        "overall_accuracy": round(overall["avg_score"], 3) if overall["n"] else None,
        "quizzes_taken": len(per_quiz),
        "topics": topics,
        "weak_topics": [t["topic"] for t in topics if t["weak"]],
        "history": [
            {"quiz_id": r["quiz_id"], "topic": r["topic"], "accuracy": round(r["avg_score"], 3), "ts": r["ts"]}
            for r in per_quiz
        ],
    }


# ---------- key/value (cached topic map) ----------

def kv_get(key: str) -> Any | None:
    with session() as c:
        row = c.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    return json.loads(row["value"]) if row else None


def kv_set(key: str, value: Any) -> None:
    with session() as c:
        c.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )


def kv_delete(key: str) -> None:
    with session() as c:
        c.execute("DELETE FROM kv WHERE key = ?", (key,))
