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


def llm(prompt, json_mode=True, max_completion_tokens=2000, system=None, history=None):
    """
    Sends one prompt to Groq and returns the raw text reply.

    system: optional system prompt (the assistant's role, tone and rules).
    Leave it as None for the pipeline steps that don't need one.

    history: optional list of earlier chat turns, each {"role": "user" or
    "assistant", "content": "..."}. The model has NO memory of its own - every
    call starts blank - so the only way it "remembers" the conversation is that
    we re-send the earlier turns every time. They go between the system prompt
    and the new question.

    json_mode=True tells Groq the reply MUST be a JSON object (a {...}, not
    a bare [...] list) — every prompt in this project that wants JSON back
    is written to ask for an object wrapping a list, to satisfy this.

    max_completion_tokens caps the reply length. This matters a lot in JSON
    mode: if the model's answer gets cut off mid-object because it ran out
    of room, Groq rejects it with "Failed to validate JSON" — raising this
    for bigger prompts (e.g. tagging a whole batch of reviews) avoids that.

    If Groq returns a rate-limit, temporary server error, or a JSON
    validation failure, this waits and retries automatically instead of
    crashing the whole pipeline run.
    """
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}

    # gpt-oss models "think" before answering; low effort keeps tagging/
    # summarizing fast and cheap on tokens, since these are simple tasks.
    if "gpt-oss" in GROQ_MODEL:
        extra["reasoning_effort"] = "low"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend({"role": m["role"], "content": m["content"]} for m in history)
    messages.append({"role": "user", "content": prompt})

    for attempt in range(MAX_RETRIES):
        try:
            response = _client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0,
                max_completion_tokens=max_completion_tokens,
                **extra,
            )
            time.sleep(PAUSE_BETWEEN_CALLS)
            return response.choices[0].message.content.strip()
        except Exception as e:
            message = str(e).lower()
            retryable = any(
                code in message
                for code in ("429", "503", "rate_limit", "overloaded", "json_validate_failed")
            )
            if retryable and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt * 3)  # 3s, 6s, 12s, 24s...
                continue
            raise

    raise RuntimeError("Groq did not respond after several retries.")


def llm_json(prompt, default=None, max_completion_tokens=2000, system=None, history=None):
    """Convenience wrapper: calls llm() in JSON mode and parses the result.
    Returns `default` (an empty dict by default) if the call fails for ANY
    reason — a bad/cut-off JSON reply, a validation error Groq rejects
    outright, a rate limit that outlasts the retries, anything — so one
    stubborn batch never crashes the whole pipeline run. The caller sees an
    empty result for that batch instead of a stack trace."""
    try:
        raw = llm(prompt, json_mode=True, max_completion_tokens=max_completion_tokens,
                  system=system, history=history)
        return json.loads(raw)
    except Exception:
        return default if default is not None else {}