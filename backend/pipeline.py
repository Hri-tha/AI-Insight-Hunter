"""
The 8-step pipeline from the technical design doc, plus chat.
Only steps 2, 5, 7, 8 and chat call the LLM. Step 6 calls the embedding
model and runs a vector similarity search. Steps 1, 3 and 4 never touch AI.

Every function takes `conn` (a database connection from backend.db) as the
last or near-last argument, so the frontend controls when connections open
and close.
"""

import json
import uuid

import pandas as pd
import psycopg2.extras

from backend.llm import llm, llm_json, get_embedding
from backend.prompts import CHAT_SYSTEM_PROMPT
from backend.taxonomy import ISSUE_TAXONOMY, SENTIMENT, SEVERITY

REQUIRED_COLUMNS = ["review_id", "review_date", "rating", "review_text", "product_category", "returned"]
OPTIONAL_COLUMNS = ["seller", "customer_segment"]


def chunks(lst, n):
    """Splits a list into groups of at most n items — used to batch reviews
    into one LLM call instead of one call per review."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# =====================================================================================
# STEP 1 — Data Collection
# No AI. Validates the uploaded file has the required columns, then inserts
# every row into the reviews table under a new upload_id.
# =====================================================================================

def collect_data(df, conn):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Your file is missing required columns: {missing}")

    upload_id = str(uuid.uuid4())
    rows = []
    for _, r in df.iterrows():
        rows.append((
            upload_id,
            str(r["review_id"]),
            pd.to_datetime(r["review_date"]).date(),
            int(r["rating"]),
            str(r["review_text"]),
            str(r["product_category"]),
            bool(r["returned"]),
            str(r["seller"]) if "seller" in df.columns and pd.notna(r.get("seller")) else None,
            str(r["customer_segment"]) if "customer_segment" in df.columns and pd.notna(r.get("customer_segment")) else None,
        ))

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, """
            INSERT INTO reviews (upload_id, review_id, review_date, rating, review_text,
                                  product_category, returned, seller, customer_segment)
            VALUES %s
        """, rows)
    conn.commit()
    return upload_id


# =====================================================================================
# STEP 2 — Review Understanding (Tagging)
# Sends reviews to Groq in batches of 5, asks for structured labels, then
# generates an embedding for every review using the local embedding model.
# =====================================================================================

def build_tagging_prompt(review_batch, known_problems):
    reviews_block = "\n".join(f'{r["review_id"]}: "{r["review_text"]}"' for r in review_batch)
    return f"""Tag each review below.

issue_category and issue_subcategory: choose ONLY from this list:
{json.dumps(ISSUE_TAXONOMY)}

sentiment: choose ONLY from {SENTIMENT}
severity: choose ONLY from {SEVERITY}
problem: a short label for the specific issue (e.g. "Incorrect Fit", "Delivery Delay").
Reuse one of these existing labels if it fits: {known_problems}
Only invent a new label if none of these fit.
evidence_quote: the exact phrase from the review that supports your tags.

Reviews:
{reviews_block}

Return ONLY a JSON object shaped exactly like this (a list INSIDE one object — not a bare list):
{{"tags": [{{"review_id": "...", "issue_category": "...", "issue_subcategory": "...", "problem": "...", "sentiment": "...", "severity": "...", "evidence_quote": "..."}}]}}"""


def tag_reviews(conn, upload_id, batch_size=5, progress_callback=None):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT review_id, review_text FROM reviews WHERE upload_id=%s AND problem IS NULL",
            (upload_id,),
        )
        reviews = cur.fetchall()

    known_problems = []
    batches = list(chunks(reviews, batch_size))

    for idx, batch in enumerate(batches):
        prompt = build_tagging_prompt(batch, known_problems)
        result = llm_json(prompt, default={"tags": []})
        tags = result.get("tags", [])

        with conn.cursor() as cur:
            for t in tags:
                problem = t.get("problem") or "Unclear"
                if problem not in known_problems:
                    known_problems.append(problem)
                cur.execute("""
                    UPDATE reviews SET issue_category=%s, issue_subcategory=%s, problem=%s,
                           sentiment=%s, severity=%s, evidence_quote=%s
                    WHERE upload_id=%s AND review_id=%s
                """, (
                    t.get("issue_category"), t.get("issue_subcategory"), problem,
                    t.get("sentiment"), t.get("severity"), t.get("evidence_quote"),
                    upload_id, t.get("review_id"),
                ))
        conn.commit()

        if progress_callback:
            progress_callback(idx + 1, len(batches), stage="tagging")

    embed_reviews(conn, upload_id, progress_callback=progress_callback)


def embed_reviews(conn, upload_id, progress_callback=None):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT review_id, review_text FROM reviews WHERE upload_id=%s", (upload_id,))
        rows = cur.fetchall()

    with conn.cursor() as cur:
        for i, r in enumerate(rows):
            vec = get_embedding(r["review_text"])
            cur.execute(
                "UPDATE reviews SET embedding=%s WHERE upload_id=%s AND review_id=%s",
                (vec, upload_id, r["review_id"]),
            )
            if progress_callback and i % 5 == 0:
                progress_callback(i + 1, len(rows), stage="embedding")
    conn.commit()


# =====================================================================================
# STEP 3 — Analytics
# Pure SQL + Python. No LLM. Counts, rates, week-over-week change, which
# product/seller a problem concentrates in, and its return-rate impact.
# =====================================================================================

def compute_stats(conn, upload_id):
    query = """
        SELECT problem, issue_category, COUNT(*) AS reviews,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM reviews WHERE upload_id=%(id)s), 2) AS rate_pct,
               ROUND(AVG(rating), 1) AS avg_rating,
               MODE() WITHIN GROUP (ORDER BY severity) AS severity
        FROM reviews
        WHERE upload_id=%(id)s
          AND problem IS NOT NULL
          AND problem <> 'No Specific Problem'
          AND sentiment IN ('Negative', 'Mixed')
        GROUP BY problem, issue_category
        ORDER BY reviews DESC
    """
    base = pd.read_sql(query, conn, params={"id": upload_id})

    results = []
    for _, row in base.iterrows():
        p = row.to_dict()
        p["wow_pct"] = compute_wow(conn, upload_id, p["problem"])
        p["concentration"] = compute_concentration(conn, upload_id, p["problem"])
        p["business_impact"] = compute_return_rate(conn, upload_id, p["problem"])
        results.append(p)
    return results


def compute_wow(conn, upload_id, problem):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE review_date >= CURRENT_DATE - 7) AS this_week,
                COUNT(*) FILTER (WHERE review_date BETWEEN CURRENT_DATE - 14 AND CURRENT_DATE - 8) AS last_week
            FROM reviews WHERE upload_id=%s AND problem=%s
        """, (upload_id, problem))
        row = cur.fetchone()
    if not row["last_week"]:
        return None
    return round(100.0 * (row["this_week"] - row["last_week"]) / row["last_week"], 1)


def compute_concentration(conn, upload_id, problem, by="product_category"):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            SELECT {by} AS name, COUNT(*) AS cnt
            FROM reviews WHERE upload_id=%s AND problem=%s
            GROUP BY {by} ORDER BY cnt DESC LIMIT 1
        """, (upload_id, problem))
        row = cur.fetchone()
        if not row:
            return None
        cur.execute(
            "SELECT COUNT(*) AS total FROM reviews WHERE upload_id=%s AND problem=%s",
            (upload_id, problem),
        )
        total = cur.fetchone()["total"]
    return {"by": by, "name": row["name"], "share_pct": round(100 * row["cnt"] / total, 0)}


def compute_return_rate(conn, upload_id, problem):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT ROUND(100.0*AVG(returned::int),1) AS rate FROM reviews WHERE upload_id=%s AND problem=%s",
            (upload_id, problem),
        )
        problem_rate = cur.fetchone()["rate"]
        cur.execute(
            "SELECT ROUND(100.0*AVG(returned::int),1) AS rate FROM reviews WHERE upload_id=%s",
            (upload_id,),
        )
        overall_rate = cur.fetchone()["rate"]
    return {
        "return_rate_pct": float(problem_rate or 0),
        "overall_return_rate_pct": float(overall_rate or 0),
    }


# =====================================================================================
# STEP 4 — Significant Problem Identification
# A plain Python rule. No LLM.
# =====================================================================================

def decide(p):
    if p["reviews"] < 10:
        return "INSUFFICIENT_DATA", "Too few reviews to judge."
    signs = [
        p["rate_pct"] >= 5,
        (p["wow_pct"] or 0) >= 20,
        p["severity"] in ("High", "Critical"),
    ]
    count = sum(signs)
    if count >= 2:
        return "INVESTIGATE", f"{count} of 3 warning signs"
    return "MONITOR", f"{count} of 3 warning signs"


# =====================================================================================
# STEP 5 — Hypothesis Generation
# LLM proposes plausible explanations from the STATS ONLY (not raw reviews).
# =====================================================================================

def build_hypothesis_prompt(problem_stats):
    return f"""A customer feedback problem needs investigating.

Problem: {problem_stats['problem']}
Category: {problem_stats['issue_category']}
Rate: {problem_stats['rate_pct']}% of reviews, up {problem_stats['wow_pct']}% week over week
Severity: {problem_stats['severity']}
Concentrated in: {problem_stats['concentration']}

Propose 2-4 plausible hypotheses for WHY this is happening.
For each, give a short search phrase to find supporting reviews.

Return ONLY a JSON object:
{{"hypotheses": [{{"id": "H1", "statement": "...", "search_query": "..."}}]}}"""


def generate_hypotheses(problem_stats):
    result = llm_json(build_hypothesis_prompt(problem_stats), default={"hypotheses": []})
    return result.get("hypotheses", [])


# =====================================================================================
# STEP 6 — Evidence Retrieval (RAG search)
# No LLM call. Pure vector similarity search via pgvector's <=> operator.
# =====================================================================================

def search_reviews(conn, upload_id, query_text, problem_label=None, k=10):
    q_emb = get_embedding(query_text)
    sql = "SELECT review_id, review_text, rating FROM reviews WHERE upload_id=%s AND embedding IS NOT NULL"
    params = [upload_id]
    if problem_label:
        sql += " AND problem=%s"
        params.append(problem_label)
    sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
    params += [q_emb, k]

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def gather_evidence(conn, upload_id, problem_label, hypotheses):
    for h in hypotheses:
        h["evidence"] = search_reviews(conn, upload_id, h["search_query"], problem_label=problem_label)
    return hypotheses


# =====================================================================================
# STEP 7 — Verdict + Root Cause
# LLM judges each hypothesis against its retrieved evidence, then synthesizes
# a likely cause, opportunities, and data gaps.
# =====================================================================================

def build_verdict_prompt(problem, hypotheses_with_evidence):
    blocks = []
    for h in hypotheses_with_evidence:
        quotes = "\n".join(f'- {e["review_text"]}' for e in h["evidence"])
        blocks.append(f'{h["id"]}: {h["statement"]}\nEvidence:\n{quotes}')

    return f"""Problem: {problem}

Hypotheses and their evidence:
{chr(10).join(blocks)}

For EACH hypothesis, decide a verdict from ["SUPPORTED", "MIXED", "INCONCLUSIVE"] based ONLY on the evidence shown.
Never state a verdict the evidence does not support.

Then provide:
- likely_cause: one sentence, with a confidence level (Low/Medium/High)
- opportunities: a short list of product ideas
- data_gaps: what additional data would confirm this

Return ONLY a JSON object:
{{"verdicts": [{{"id": "H1", "verdict": "...", "why": "..."}}],
  "likely_cause": "...", "confidence": "...",
  "opportunities": ["..."], "data_gaps": ["..."]}}"""


def judge_hypotheses(problem, hypotheses_with_evidence):
    return llm_json(
        build_verdict_prompt(problem, hypotheses_with_evidence),
        default={"verdicts": [], "likely_cause": "", "confidence": "Low", "opportunities": [], "data_gaps": []},
    )


def investigate(problem_stats, conn, upload_id):
    hypotheses = generate_hypotheses(problem_stats)                                    # Step 5
    hypotheses = gather_evidence(conn, upload_id, problem_stats["problem"], hypotheses)  # Step 6
    result = judge_hypotheses(problem_stats["problem"], hypotheses)                      # Step 7
    return hypotheses, result


# =====================================================================================
# STEP 8 — Report Summary
# One LLM call turns already-computed numbers into a short paragraph.
# It performs no calculation of its own.
# =====================================================================================

def build_summary_prompt(results):
    facts = json.dumps([
        {k: p[k] for k in ("problem", "rate_pct", "wow_pct", "severity", "decision")}
        for p in results["problems"]
    ])
    return f"""Write a 2-3 sentence summary for a product manager, for a Monday report.

Facts (use ONLY these numbers, do not calculate or estimate anything):
{facts}

Mention the biggest or fastest-growing problem first."""


def write_summary(results):
    results["summary"] = llm(build_summary_prompt(results), json_mode=False)
    return results


# =====================================================================================
# CHAT — runs on demand whenever the person types a question
# =====================================================================================

CHART_METRICS = ("reviews", "rate_pct", "avg_rating", "wow_pct")


def build_chart(spec, results):
    """Builds chart data from the already-computed stats. The LLM only picks
    WHICH metric to plot; the numbers themselves always come from the database,
    so the model can never invent them."""
    if not isinstance(spec, dict) or spec.get("metric") not in CHART_METRICS:
        return None
    metric = spec["metric"]
    rows = [p for p in results["problems"] if p.get(metric) is not None]
    if not rows:
        return None
    data = pd.DataFrame(
        {metric: [float(p[metric]) for p in rows]},
        index=[p["problem"] for p in rows],
    )
    return {"type": "bar", "title": spec.get("title") or metric, "data": data}


def answer_question(question, results, conn, upload_id):
    """Returns (answer_text, source_review_ids, chart_or_None)."""
    relevant_reviews = search_reviews(conn, upload_id, question, k=10)
    evidence_text = "\n".join(f'- [{r["review_id"]}] {r["review_text"]}' for r in relevant_reviews)

    # Leave out each problem's bulky "evidence" list to keep the prompt small
    compact_stats = [{k: v for k, v in p.items() if k != "evidence"} for p in results["problems"]]
    stats_context = json.dumps(compact_stats, default=str)

    prompt = f"""Stats already computed:
{stats_context}

Relevant customer reviews:
{evidence_text}

User message: {question}"""

    reply = llm_json(
        prompt,
        system=CHAT_SYSTEM_PROMPT,
        default={"answer": "Sorry, I couldn't generate a reply. Please try again.", "chart": None},
    )
    answer = reply.get("answer") or "Sorry, I couldn't generate a reply. Please try again."
    chart = build_chart(reply.get("chart"), results)
    sources = [r["review_id"] for r in relevant_reviews]
    return answer, sources, chart


# =====================================================================================
# RUN CACHING — save/load a finished run so re-opening the dashboard doesn't
# require re-tagging or re-calling the LLM
# =====================================================================================

def save_run(conn, upload_id, results):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO analysis_runs (upload_id, results)
            VALUES (%s, %s)
            ON CONFLICT (upload_id) DO UPDATE SET results = EXCLUDED.results, created_at = now()
        """, (upload_id, json.dumps(results, default=str)))
    conn.commit()


def load_run(conn, upload_id):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT results FROM analysis_runs WHERE upload_id=%s", (upload_id,))
        row = cur.fetchone()
    return row["results"] if row else None


# =====================================================================================
# FULL PIPELINE, CHAINED
# =====================================================================================

def run_pipeline(df, conn, progress_callback=None):
    upload_id = collect_data(df, conn)                                   # 1
    tag_reviews(conn, upload_id, progress_callback=progress_callback)    # 2 (tags + embeddings)
    problems = compute_stats(conn, upload_id)                            # 3

    for p in problems:
        p["decision"], p["decision_reason"] = decide(p)                  # 4

    for p in problems:
        if p["decision"] == "INVESTIGATE":
            p["evidence"], p["investigation"] = investigate(p, conn, upload_id)  # 5, 6, 7
        else:
            p["evidence"], p["investigation"] = [], None

    results = {
        "upload_id": upload_id,
        "meta": {
            "total_reviews": len(df),
            "date_range": f"{df['review_date'].min()} to {df['review_date'].max()}",
        },
        "problems": problems,
    }
    results = write_summary(results)  # 8
    save_run(conn, upload_id, results)
    return results