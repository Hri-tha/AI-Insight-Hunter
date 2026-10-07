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
from backend.prompts import CHAT_SYSTEM_PROMPT, SQL_SYSTEM_PROMPT
from backend.sql_tools import run_readonly_query, dataframe_to_json
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
    """upload_id=None means "search every upload in the database"."""
    q_emb = get_embedding(query_text)
    sql = "SELECT review_id, review_text, rating FROM reviews WHERE embedding IS NOT NULL"
    params = []
    if upload_id:
        sql += " AND upload_id=%s"
        params.append(upload_id)
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
#
# For every question we gather up to THREE kinds of context, then ask the LLM:
#   1. SQL result   - the LLM writes a SELECT, we validate + run it on the WHOLE database
#   2. Similar reviews - vector search (RAG) for qualitative "why" questions
#   3. Dashboard stats - already-computed numbers for the current upload
# Earlier chat turns are passed as `history` so follow-up questions work.
# =====================================================================================

MAX_HISTORY_MESSAGES = 8   # = last 4 question/answer pairs. Bigger = more tokens per call.
MAX_SQL_ATTEMPTS = 2       # first try + one self-correction
CHART_METRICS = ("reviews", "rate_pct", "avg_rating", "wow_pct")
CHART_WORDS = ("chart", "graph", "plot", "visual", "compare", "comparison",
               "distribution", "trend", "breakdown", "histogram")


def wants_chart(question):
    q = question.lower()
    return any(w in q for w in CHART_WORDS)


def generate_sql(question, history, upload_id, scope, chart_requested,
                 previous_sql=None, previous_error=None):
    """Asks the LLM to turn the question into a SQL query (or null)."""
    notes = []
    if scope == "current" and upload_id:
        notes.append(f"Only use rows from the current upload: WHERE upload_id = '{upload_id}'.")
    else:
        notes.append("Use ALL uploads (do not filter on upload_id) unless the user asks about a specific file.")
        if upload_id:
            notes.append(f"For reference, the most recent upload_id is '{upload_id}'.")
    if chart_requested:
        notes.append("The user wants a chart: return exactly two columns (label, numeric value).")
    if previous_sql:
        notes.append(f"Your previous query failed.\nQuery: {previous_sql}\nError: {previous_error}\nWrite a corrected query.")

    prompt = "\n".join(notes) + f"\n\nUser question: {question}"
    result = llm_json(prompt, system=SQL_SYSTEM_PROMPT, history=history,
                      default={"sql": None}, max_completion_tokens=800)
    sql = result.get("sql")
    return sql if isinstance(sql, str) and sql.strip() else None


def fetch_sql_context(question, history, conn, upload_id, scope, chart_requested):
    """Generate -> validate -> run. If the query fails, show the LLM its own
    error and let it try ONE more time. Returns (dataframe_or_None, sql, error)."""
    sql = generate_sql(question, history, upload_id, scope, chart_requested)
    if not sql:
        return None, None, None   # the question doesn't need the database

    require_id = upload_id if scope == "current" else None
    error = None
    for attempt in range(MAX_SQL_ATTEMPTS):
        try:
            df, cleaned = run_readonly_query(conn, sql, require_upload_id=require_id)
            return df, cleaned, None
        except Exception as e:                       # UnsafeSQLError or a Postgres error
            conn.rollback()
            error = str(e).strip().splitlines()[0]
            if attempt + 1 < MAX_SQL_ATTEMPTS:
                retry = generate_sql(question, history, upload_id, scope, chart_requested,
                                     previous_sql=sql, previous_error=error)
                if not retry:
                    break
                sql = retry
    return None, sql, error


def build_chart_from_df(spec, df, force=False):
    """Turns a SQL result into chart data. The LLM only picks WHICH columns to
    plot; the numbers come from the database. If the LLM forgot to ask for a
    chart but the user clearly wanted one (force=True), we still build one from
    the first label column + first numeric column."""
    if df is None or df.empty or len(df.columns) < 2:
        return None
    spec = spec if isinstance(spec, dict) else {}
    if not spec and not force:
        return None

    x = spec.get("x")
    if x not in df.columns:
        x = df.columns[0]
    numeric = [c for c in df.columns if c != x and pd.api.types.is_numeric_dtype(df[c])]
    y = spec.get("y")
    if y not in numeric:
        if not numeric:
            return None
        y = numeric[0]

    data = df[[x, y]].copy()
    data[x] = data[x].astype(str)
    data[y] = pd.to_numeric(data[y], errors="coerce")
    data = data.dropna().groupby(x, sort=False)[y].sum().to_frame()   # one bar per label
    if data.empty:
        return None

    kind = spec.get("type") if spec.get("type") in ("bar", "line") else "bar"
    return {"type": kind, "title": spec.get("title") or f"{y} by {x}", "data": data}


def build_chart(spec, results):
    """Older chart builder: plots a metric from the dashboard stats (current
    upload only). Used only when there is no SQL result."""
    if not results or not isinstance(spec, dict) or spec.get("metric") not in CHART_METRICS:
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


def answer_question(question, conn, upload_id=None, results=None, history=None, scope="all"):
    """Answers one chat message.

    history: earlier turns as [{"role": "user"/"assistant", "content": "..."}], oldest first,
             NOT including `question` itself.
    scope:   "all" = whole database, "current" = only `upload_id`.
    Returns a dict: answer, sources, chart, sql, sql_error.
    """
    if scope == "current" and not upload_id:
        scope = "all"
    history = (history or [])[-MAX_HISTORY_MESSAGES:]
    chart_requested = wants_chart(question)

    # 1) SQL over the whole database
    sql_df, sql_used, sql_error = fetch_sql_context(
        question, history, conn, upload_id, scope, chart_requested)

    # 2) Vector search. A short follow-up like "and why?" means nothing on its
    #    own, so borrow the previous user question to give the search some meaning.
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    search_text = question if len(question.split()) > 6 else f"{last_user} {question}".strip()
    relevant = search_reviews(conn, upload_id if scope == "current" else None, search_text, k=8)
    evidence_text = "\n".join(f'- [{r["review_id"]}] {r["review_text"][:300]}' for r in relevant)

    # 3) Build the prompt from whatever context we have
    parts = []
    if results:
        keys = ("problem", "issue_category", "reviews", "rate_pct", "wow_pct",
                "avg_rating", "severity", "decision")
        compact = [{k: p.get(k) for k in keys} for p in results["problems"][:10]]
        parts.append("Dashboard stats for the CURRENT upload:\n" + json.dumps(compact, default=str))
    if sql_df is not None:
        parts.append(f"SQL that was run on the database:\n{sql_used}\n\n"
                     f"SQL result ({len(sql_df)} rows, columns: {list(sql_df.columns)}):\n"
                     f"{dataframe_to_json(sql_df)}")
    elif sql_error:
        parts.append(f"A database query was attempted but failed ({sql_error}). "
                     "Do not guess numbers; say you could not compute it.")
    if evidence_text:
        parts.append("Relevant customer reviews:\n" + evidence_text)
    if chart_requested and sql_df is not None and not sql_df.empty:
        parts.append("The user wants a chart. Set \"chart\" using exact column names from the SQL result.")

    prompt = "\n\n".join(parts) + f"\n\nUser message: {question}"

    fallback = "Sorry, I couldn't generate a reply. Please try again."
    reply = llm_json(prompt, system=CHAT_SYSTEM_PROMPT, history=history,
                     default={"answer": fallback, "chart": None}, max_completion_tokens=1200)
    answer = reply.get("answer") or fallback

    chart = build_chart_from_df(reply.get("chart"), sql_df, force=chart_requested)
    if chart is None and sql_df is None:
        chart = build_chart(reply.get("chart"), results)

    return {
        "answer": answer,
        "sources": [r["review_id"] for r in relevant],
        "chart": chart,
        "sql": sql_used,
        "sql_error": sql_error,
    }



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