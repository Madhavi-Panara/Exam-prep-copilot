"""LLM access: Anthropic's Messages API, or any OpenAI-compatible endpoint (Gemini, Groq, Ollama, ...).

Pick the provider with LLM_PROVIDER in .env. The rest of the app only calls complete() / complete_json().
"""
from __future__ import annotations

import json
import re
import time

from . import config
from .errors import LLMError

OPENAI_COMPATIBLE = {"gemini", "groq", "ollama", "openai"}


def extract_json(text: str) -> dict:
    """Pull a JSON object out of a model reply, tolerating code fences and surrounding prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("JSON is not an object")
    return obj


class LLMClient:
    def __init__(self) -> None:
        self._anthropic = None

    @property
    def configured(self) -> bool:
        provider = config.LLM_PROVIDER
        if provider == "anthropic":
            return bool(config.ANTHROPIC_API_KEY)
        if provider == "ollama":
            return True  # local, no key needed
        return provider in OPENAI_COMPATIBLE and bool(config.LLM_API_KEY) and bool(config.LLM_BASE_URL)

    # ------------------------------------------------------------------ providers

    def _complete_anthropic(self, system: str, messages: list[dict], max_tokens: int) -> str:
        if not config.ANTHROPIC_API_KEY:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Add it to .env and restart the backend "
                "(or set LLM_PROVIDER=gemini / groq / ollama to use a different provider).",
                503,
            )
        if self._anthropic is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise LLMError("The 'anthropic' package is missing. Run: pip install -r backend/requirements.txt", 503) from exc
            self._anthropic = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        try:
            msg = self._anthropic.messages.create(
                model=config.MODEL_NAME, max_tokens=max_tokens, system=system, messages=messages
            )
        except Exception as exc:
            raise LLMError(f"The AI request failed: {exc}") from exc
        return "".join(getattr(block, "text", "") for block in msg.content).strip()

    def _complete_openai_compatible(self, system: str, messages: list[dict], max_tokens: int) -> str:
        provider = config.LLM_PROVIDER
        if provider != "ollama" and not config.LLM_API_KEY:
            raise LLMError(f"LLM_API_KEY is not set for provider '{provider}'. Add it to .env and restart the backend.", 503)
        if not config.LLM_BASE_URL:
            raise LLMError("LLM_BASE_URL is not set. Add it to .env (e.g. https://api.groq.com/openai/v1).", 503)

        import requests

        headers = {"Content-Type": "application/json"}
        if config.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"
        # Gemini 2.5 models spend part of the token budget on hidden reasoning, so give them headroom.
        budget = max_tokens * 2 if provider in ("gemini", "groq") else max_tokens        
        body = {
            "model": config.MODEL_NAME,
            "max_tokens": budget,
            "messages": [{"role": "system", "content": system}] + messages,
        }

        response = None
        for attempt in range(3):  # free tiers rate-limit aggressively; back off and retry
            try:
                response = requests.post(f"{config.LLM_BASE_URL}/chat/completions", json=body, headers=headers, timeout=120)
            except requests.exceptions.RequestException as exc:
                raise LLMError(f"Could not reach the AI provider at {config.LLM_BASE_URL}: {exc}") from exc
            if response.status_code != 429 or attempt == 2:
                break
            time.sleep(2 * (attempt + 1))

        if response.status_code == 429:
            raise LLMError("The AI provider's rate limit was hit. Wait a minute and try again.", 429)
        if not response.ok:
            raise LLMError(f"The AI request failed ({response.status_code}): {response.text[:300]}")
        try:
            return (response.json()["choices"][0]["message"]["content"] or "").strip()
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("The AI provider returned an unexpected response.") from exc

    # ------------------------------------------------------------------ public API

    def complete(self, system: str, user, max_tokens: int = 1500) -> str:
        """`user` is a string, or a full list of {"role", "content"} messages."""
        messages = [{"role": "user", "content": user}] if isinstance(user, str) else user
        provider = config.LLM_PROVIDER
        if provider == "anthropic":
            return self._complete_anthropic(system, messages, max_tokens)
        if provider in OPENAI_COMPATIBLE:
            return self._complete_openai_compatible(system, messages, max_tokens)
        raise LLMError(
            f"Unknown LLM_PROVIDER '{provider}'. Use one of: anthropic, gemini, groq, ollama, openai.", 503
        )

    def complete_json(self, system: str, user: str, max_tokens: int = 2500) -> dict:
        json_system = system + "\n\nRespond with a single valid JSON object and nothing else - no prose, no code fences."
        raw = self.complete(json_system, user, max_tokens)
        try:
            return extract_json(raw)
        except ValueError:
            retry = f"{user}\n\nYour previous reply was not valid JSON. Reply again with ONLY the JSON object."
            raw = self.complete(json_system, retry, max_tokens)
            try:
                return extract_json(raw)
            except ValueError as exc:
                raise LLMError("The AI returned a malformed response. Please try again.") from exc


llm = LLMClient()
