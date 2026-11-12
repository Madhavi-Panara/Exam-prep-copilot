"""FastAPI backend for Exam Prep Copilot.

Run from the repository root:  uvicorn backend.app.main:app --reload
"""
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import config, db, planner, quiz as quiz_service, rag
from .errors import AppError, IngestError, NotFoundError
from .ingest import SUPPORTED_EXTENSIONS, ingest_file
from .llm import llm
from .store import get_store

DOC_TYPES = {"notes", "syllabus", "past_paper"}

app = FastAPI(title="Exam Prep Copilot", version="1.0.0")


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


# ----------------------------------------------------------------------------- schemas

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list)


class QuizRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=200)
    n: int = Field(5, ge=1, le=15)
    difficulty: Literal["easy", "medium", "hard"] = "medium"


class SubmitRequest(BaseModel):
    answers: dict[str, str] = Field(default_factory=dict)


class PlanRequest(BaseModel):
    days: int = Field(7, ge=1, le=30)
    hours_per_day: float = Field(2.0, ge=0.5, le=12)


# ----------------------------------------------------------------------------- helpers

def _safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(name or "").name).strip(" .")
    return name or "upload"


# ----------------------------------------------------------------------------- routes

@app.get("/health")
def health():
    store = get_store()
    return {
        "status": "ok",
        "model": config.MODEL_NAME,
        "llm_provider": config.LLM_PROVIDER,
        "llm_configured": llm.configured,
        "vector_backend": store.name,
        "chunks_indexed": store.count(),
    }


@app.post("/upload")
def upload(file: UploadFile = File(...), doc_type: str = Form("notes")):
    if doc_type not in DOC_TYPES:
        raise IngestError(f"doc_type must be one of: {', '.join(sorted(DOC_TYPES))}.")
    source = _safe_filename(file.filename)
    suffix = Path(source).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise IngestError("Unsupported file type. Upload a PDF, TXT or MD file.")

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    tmp = config.UPLOAD_DIR / f".{uuid.uuid4().hex}{suffix}"
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    try:
        with tmp.open("wb") as out:
            while True:
                block = file.file.read(1 << 20)
                if not block:
                    break
                size += len(block)
                if size > limit:
                    raise AppError(f"File too large (limit {config.MAX_UPLOAD_MB} MB).", 413)
                out.write(block)
        summary = ingest_file(tmp, source, doc_type, get_store())
        os.replace(tmp, config.UPLOAD_DIR / source)
    finally:
        tmp.unlink(missing_ok=True)

    db.kv_delete(planner.TOPICS_KEY)  # the cached topic map is now out of date
    return summary


@app.get("/documents")
def list_documents():
    return {"documents": get_store().list_documents()}


@app.delete("/documents/{source}")
def delete_document(source: str):
    removed = get_store().delete_document(source)
    if not removed:
        raise NotFoundError("Document not found.")
    (config.UPLOAD_DIR / _safe_filename(source)).unlink(missing_ok=True)
    db.kv_delete(planner.TOPICS_KEY)
    return {"deleted": source, "chunks_removed": removed}


@app.post("/ask")
def ask(req: AskRequest):
    return rag.answer_question(req.question, [t.model_dump() for t in req.history])


@app.get("/topics")
def get_topics():
    return planner.get_cached_topics()


@app.post("/topics/analyze")
def analyze_topics():
    return planner.analyze_topics()


@app.post("/quiz")
def create_quiz(req: QuizRequest):
    return quiz_service.generate_quiz(req.topic, req.n, req.difficulty)


@app.post("/quiz/{quiz_id}/submit")
def submit_quiz(quiz_id: str, req: SubmitRequest):
    return quiz_service.grade_quiz(quiz_id, req.answers)


@app.get("/progress")
def progress():
    return db.progress_summary()


@app.post("/plan")
def study_plan(req: PlanRequest):
    return planner.make_plan(req.days, req.hours_per_day)
