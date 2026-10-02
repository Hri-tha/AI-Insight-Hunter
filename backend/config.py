"""
Loads settings from the .env file. Every other backend module imports from
here instead of reading os.environ directly, so there's one place that knows
where configuration comes from.
"""

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b").strip()
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add it to your .env file (see .env.example)."
    )
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is missing. Add it to your .env file (see .env.example)."
    )
