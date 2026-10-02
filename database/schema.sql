-- AI Insight Hunter — Postgres schema (requires the pgvector extension)
-- Run this once against your database before starting the app.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per review. embedding is 384 numbers because we use a free local
-- embedding model (all-MiniLM-L6-v2), not OpenAI's 1536-dimension model.
CREATE TABLE IF NOT EXISTS reviews (
    upload_id           UUID NOT NULL,
    review_id            TEXT NOT NULL,
    review_date          DATE NOT NULL,
    rating                INT NOT NULL,
    review_text           TEXT NOT NULL,
    product_category      TEXT NOT NULL,
    returned              BOOLEAN NOT NULL,
    seller                TEXT,
    customer_segment      TEXT,
    issue_category        TEXT,
    issue_subcategory     TEXT,
    problem               TEXT,
    sentiment             TEXT,
    severity              TEXT,
    evidence_quote        TEXT,
    embedding             VECTOR(384),
    PRIMARY KEY (upload_id, review_id)
);

-- Caches a finished pipeline run so the dashboard can reload without
-- re-tagging and re-calling the LLM every time.
CREATE TABLE IF NOT EXISTS analysis_runs (
    upload_id   UUID PRIMARY KEY,
    results     JSONB,
    created_at  TIMESTAMP DEFAULT now()
);

-- Speeds up "find similar reviews" once you have more than a few thousand
-- rows. Safe to run even on an empty table; skip it for small test uploads.
-- CREATE INDEX IF NOT EXISTS reviews_embedding_idx
--     ON reviews USING hnsw (embedding vector_cosine_ops);
