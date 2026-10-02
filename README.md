# AI Insight Hunter

Full pipeline: Postgres (with pgvector) + Groq + Streamlit.

## 1. Get a free Postgres database with pgvector

Easiest option: Supabase (free tier, pgvector included).

1. Go to supabase.com, sign up, create a new project (pick any name/password/region).
2. Wait for the project to finish setting up.
3. Left sidebar -> SQL Editor -> New query. Paste and run:
       create extension if not exists vector;
4. Still in SQL Editor, open `database/schema.sql` from this project, paste its
   contents, and run it. This creates the `reviews` and `analysis_runs` tables.
5. Left sidebar -> Project Settings -> Database -> Connection string -> URI.
   Copy it. It looks like:
       postgresql://postgres:[YOUR-PASSWORD]@db.xxxxxxxx.supabase.co:5432/postgres
   Replace [YOUR-PASSWORD] with the database password you set in step 1.

## 2. Get a free Groq API key

1. console.groq.com -> sign up -> API Keys -> Create API Key.
2. Copy it (starts with gsk_).

## 3. Set up the project

    pip install -r requirements.txt

Copy `.env.example` to `.env` and fill in your real values:

    GROQ_API_KEY=gsk_your-real-key
    GROQ_MODEL=openai/gpt-oss-20b
    DATABASE_URL=postgresql://postgres:your-password@db.xxxxxxxx.supabase.co:5432/postgres

## 4. Run it

    streamlit run frontend/app.py --server.fileWatcherType none

Click "Use sample data" for a quick first test, or upload your own CSV with
these required columns: review_id, review_date, rating, review_text,
product_category, returned (optional: seller, customer_segment).

## Project layout

    database/schema.sql   - run once against your Postgres database
    backend/config.py     - reads .env
    backend/db.py         - Postgres connection helper
    backend/llm.py        - all Groq calls + local embeddings live here
    backend/taxonomy.py   - fixed label lists the LLM must choose from
    backend/pipeline.py   - the 8 pipeline steps + chat (no UI code)
    frontend/app.py       - Streamlit UI (no pipeline logic, only calls backend/)

## Notes on the Groq free tier

- Tagging is batched (10 reviews per LLM call), not one-call-per-review, so a
  few hundred reviews is a few dozen calls, not a few hundred.
- Free tier limits are roughly 30 requests/minute and 1,000 requests/day on
  openai/gpt-oss-20b. If you hit "model not found" later, Groq has likely
  retired that model — check console.groq.com/docs/deprecations and update
  GROQ_MODEL in .env (no code change needed).
- Embeddings never touch Groq — they run locally via sentence-transformers,
  so they're free and don't count against any API limit.
