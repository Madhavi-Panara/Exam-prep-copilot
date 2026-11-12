"""Central settings, loaded from environment variables / the project's .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _path(env_name: str, default: str) -> Path:
    p = Path(os.getenv(env_name, default))
    return p if p.is_absolute() else ROOT / p


# --- LLM ---
# LLM_PROVIDER: anthropic (default) | gemini | groq | ollama | openai (any OpenAI-compatible endpoint)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()  # key for gemini / groq / openai-compatible providers

# provider -> (base URL, default model). Check your provider's docs if a model name stops working.
_PRESETS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash"),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "ollama": ("http://localhost:11434/v1", "llama3.1"),
}
_preset = _PRESETS.get(LLM_PROVIDER)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", _preset[0] if _preset else "").strip().rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME") or (_preset[1] if _preset else "claude-sonnet-5-5")

# --- Storage ---
DATA_DIR = _path("DATA_DIR", "data")
UPLOAD_DIR = _path("UPLOAD_DIR", "data/uploads")
CHROMA_DIR = _path("CHROMA_DIR", "data/chroma_db")
DB_PATH = _path("DB_PATH", "data/app.db")
LOCAL_STORE_PATH = _path("LOCAL_STORE_PATH", "data/local_store.json")

# "auto" = ChromaDB if installed, otherwise the built-in keyword (BM25) store.
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "auto").lower()

# --- RAG tuning ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))

for _d in (DATA_DIR, UPLOAD_DIR, DB_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)
