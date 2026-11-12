
# 📚 Exam Prep Copilot

An AI study assistant built on **your own** syllabus, notes and past papers. Upload them, then:

- **Ask your notes**: answers come only from your material, with `[1]`-style citations and the source passages shown (RAG)
- **Quiz me**: generates multiple-choice and short-answer questions on any topic, grounded in your notes, and grades them with explanations
- **Topic map**: reads your syllabus and past papers and ranks topics by how much they matter for the exam
- **Progress**: tracks accuracy per topic and flags your weak topics
- **Study plan**: a day-by-day revision timetable that prioritises important and weak topics

Built for the **Homingo hackathon** (track: **LLM apps**).

## How it works

```
 PDF / TXT / MD ──► extract text ──► chunk (~900 chars, overlap) ──► vector store (ChromaDB)
                                                                           │
 question ─────────────────────────────► retrieve top-k chunks ◄───────────┘
                                                │
                       numbered excerpts + strict "answer only from these" prompt
                                                │
                                           Claude API ──► cited answer
```

- **Quizzes** are generated from retrieved notes as structured JSON, validated, and stored server-side so the answers never reach the browser before you submit. Multiple-choice is marked deterministically; short answers are marked by the model against a rubric (model answer + key points).
- **Weak topics** = topics where your average quiz score is below 70%. They feed straight into the study plan.
- **Search backend**: ChromaDB (semantic) by default. If ChromaDB can't be installed on your machine, the app automatically falls back to a built-in BM25 keyword search, so it always runs.

## Tech stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.10+ |
| Backend | FastAPI, SQLite (quizzes, progress) |
| LLM | Anthropic Claude API, or Gemini / Groq / Ollama / any OpenAI-compatible API |
| Retrieval | ChromaDB embeddings, with BM25 fallback |
| PDF parsing | pypdf |
| Frontend | Streamlit |
| Tests | pytest |

## Setup

```bash
git clone <your-repo-url>
cd exam-prep-copilot

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
# open .env and set up ONE AI provider (see below)
```

### Choosing an AI provider

No Anthropic key? Any of these work. Edit the matching lines in `.env`:

| Provider | Cost | `.env` settings |
|----------|------|-----------------|
| Anthropic Claude | paid API key | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY=...` |
| Google Gemini | free tier ([get a key](https://aistudio.google.com/apikey)) | `LLM_PROVIDER=gemini`, `LLM_API_KEY=...` |
| Groq | free tier ([get a key](https://console.groq.com/keys)) | `LLM_PROVIDER=groq`, `LLM_API_KEY=...` |
| Ollama | free, runs on your computer | `LLM_PROVIDER=ollama`, `MODEL_NAME=llama3.1` |
| Any OpenAI-compatible API | varies | `LLM_PROVIDER=openai`, `LLM_BASE_URL=...`, `LLM_API_KEY=...` |

Free tiers have rate limits, so quiz generation may occasionally ask you to wait a minute. If a provider renames its models, set `MODEL_NAME` to a current one from its docs. Smaller free models write weaker quiz questions than Claude; check quality on your own notes.

## Run

Two terminals, both from the repo root with the virtualenv active:

```bash
# 1. Backend
uvicorn backend.app.main:app --reload

# 2. Frontend
streamlit run frontend/app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). API docs are at http://127.0.0.1:8000/docs.

**Quick demo:** upload the three files in `samples/` (set the type to *Syllabus*, *Notes* and *Past paper* respectively), click **Analyze my syllabus** on the Topics tab, then ask a question, take a quiz, and generate a study plan.

> The first ChromaDB upload downloads a small embedding model (~80 MB), so it needs internet and takes a moment.

## Project structure

```
backend/app/
  main.py      FastAPI routes
  ingest.py    PDF/TXT extraction and chunking
  store.py     ChromaDB store + BM25 fallback store
  llm.py       Claude API wrapper, JSON extraction and retry
  prompts.py   every prompt template
  rag.py       retrieval + cited answers
  quiz.py      quiz generation, validation, grading
  planner.py   topic map and study plan
  db.py        SQLite: quizzes, attempts, progress
frontend/app.py   Streamlit UI
samples/          demo syllabus, notes and past paper (Operating Systems)
tests/            pytest suite
docs/             build plan and hackathon submission write-up
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | status, model, search backend |
| POST | `/upload` | upload a file (`doc_type`: notes, syllabus, past_paper) |
| GET / DELETE | `/documents`, `/documents/{source}` | list / remove material |
| POST | `/ask` | cited answer `{question, history}` |
| POST / GET | `/topics/analyze`, `/topics` | build / read the topic map |
| POST | `/quiz` | generate `{topic, n, difficulty}` |
| POST | `/quiz/{id}/submit` | grade `{answers}` |
| GET | `/progress` | per-topic accuracy and weak topics |
| POST | `/plan` | `{days, hours_per_day}` |

## Tests

```bash
pip install -r backend/requirements-dev.txt
pytest
```

## Limitations

- Single user, no login. Data lives in `data/` on the machine running the backend.
- Scanned (image-only) PDFs have no text to extract; run OCR first (e.g. `ocrmypdf`).
- Short-answer marking is done by the model, so treat it as guidance, not an official grade.


