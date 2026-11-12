# Homingo submission notes

## Project
**Exam Prep Copilot**: an LLM app that turns a student's own syllabus, notes and past papers into a personal tutor. It answers questions with citations, generates and grades quizzes, ranks topics by exam importance, tracks weak topics, and builds a revision plan.

**Track:** LLM apps

## Problem
Students revise from scattered PDFs and slides, can't tell what matters most for the exam, and get no feedback on what they actually don't know. Generic chatbots don't know their course and can make things up.

## What we built
1. **Cited Q&A over your own notes**: retrieval-augmented generation, so answers come only from uploaded material and every claim points back to a passage.
2. **Quiz engine**: grounded multiple-choice and short-answer questions at three difficulties, marked with explanations.
3. **Topic map**: syllabus plus past papers analysed to rank topics by exam weight.
4. **Weak-topic tracking**: accuracy per topic across quizzes.
5. **Study plan**: day-by-day timetable that prioritises important and weak topics.

## Skills applied (for the form)
| Skill | Where it is used |
|-------|------------------|
| Python | whole backend and frontend |
| LLM APIs | Provider-switchable client: Anthropic Claude Messages API or any OpenAI-compatible API such as Gemini, Groq, Ollama (`llm.py`). Edit this row to name the provider you actually used. |
| Prompt engineering | grounded system prompts, citation rules, JSON-schema outputs, grading rubric, retry on malformed JSON (`prompts.py`, `llm.py`) |
| RAG | chunking, retrieval, numbered-context prompting, citation checking (`ingest.py`, `rag.py`) |
| Vector databases | ChromaDB with semantic embeddings, BM25 fallback (`store.py`) |
| Backend & APIs | FastAPI REST API, SQLite persistence, validation, error handling (`main.py`, `db.py`) |
| Frontend | Streamlit multi-tab UI (`frontend/app.py`) |
| Data engineering | PDF text extraction and cleaning pipeline (`ingest.py`) |

## Tools and technologies
Python, FastAPI, Uvicorn, Pydantic, an LLM API (Claude / Gemini / Groq: name the one you used), ChromaDB, pypdf, SQLite, Streamlit, pytest.

## Design decisions worth mentioning
- **Grounding over fluency**: the model may only use retrieved excerpts, must cite them, and says so when the notes don't cover a question.
- **Answers never leave the server before submission**: quizzes are stored server-side and graded there.
- **Deterministic where possible**: multiple-choice is marked in code; the model only marks free-text answers.
- **Resilient**: malformed model JSON is retried and validated; a failed short-answer grading doesn't lose the rest of the quiz; if ChromaDB is unavailable the app falls back to keyword search.
- **Tested**: automated tests cover chunking, PDF extraction, retrieval, quiz validation and grading, progress tracking and planning.

## Demo script (about 3 minutes)
1. Upload the syllabus, notes and a past paper; click **Analyze my syllabus** to show the ranked topic map.
2. Ask "What is a safe state in the Banker's algorithm?" and show the citations.
3. Take a 5-question quiz on Deadlocks; deliberately miss one; show the feedback.
4. Open **Progress** to show the weak topic, then generate a 7-day plan that prioritises it.
