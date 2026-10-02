"""
Everything that calls an AI model lives here. Two things happen in this
file and nowhere else in the backend:
  - Chat/reasoning calls go to Groq (fast, free-tier friendly)
  - Embeddings come from a free LOCAL model (Groq has no embeddings API)
"""

import json
import time

from groq import Groq
from sentence_transformers import SentenceTransformer

from backend.config import GROQ_API_KEY, GROQ_MODEL

_client = Groq(api_key=GROQ_API_KEY)
_embedder = None  # loaded once, on first use (see get_embedder below)

MAX_RETRIES = 5
PAUSE_BETWEEN_CALLS = 2.1  # seconds; keeps you under Groq's free-tier rate limit


def get_embedder():
    """Loads the local embedding model once and reuses it (it's ~80MB)."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_embedding(text):
    """Turns text into a 384-number vector (a numpy array) representing its meaning.

    Do NOT call .tolist() here. The pgvector adapter registered in db.py only
    recognises numpy arrays; a plain Python list is sent to Postgres as a
    numeric[] array, which breaks the `<=>` similarity operator."""
    return get_embedder().encode(text)


def llm(prompt, json_mode=True):
    """
    Sends one prompt to Groq and returns the raw text reply.

    json_mode=True tells Groq the reply MUST be a JSON object (a {...}, not
    a bare [...] list) — every prompt in this project that wants JSON back
    is written to ask for an object wrapping a list, to satisfy this.

    If Groq returns a rate-limit or temporary server error, this waits and
    retries automatically instead of crashing the whole pipeline run.
    """
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}

    # gpt-oss models "think" before answering; low effort keeps tagging/
    # summarizing fast and cheap on tokens, since these are simple tasks.
    if "gpt-oss" in GROQ_MODEL:
        extra["reasoning_effort"] = "low"

    for attempt in range(MAX_RETRIES):
        try:
            response = _client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                **extra,
            )
            time.sleep(PAUSE_BETWEEN_CALLS)
            return response.choices[0].message.content.strip()
        except Exception as e:
            message = str(e).lower()
            retryable = any(
                code in message for code in ("429", "503", "rate_limit", "overloaded")
            )
            if retryable and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt * 3)  # 3s, 6s, 12s, 24s...
                continue
            raise

    raise RuntimeError("Groq did not respond after several retries.")


def llm_json(prompt, default=None):
    """Convenience wrapper: calls llm() in JSON mode and parses the result.
    Returns `default` (an empty dict by default) if the model's reply isn't
    valid JSON, so one bad reply never crashes the whole pipeline."""
    raw = llm(prompt, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default if default is not None else {}