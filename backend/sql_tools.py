"""
Safe "text-to-SQL" helpers.

The LLM writes a SQL query; this file decides whether that query is safe to run
and then runs it. NEVER execute LLM-written SQL without going through here.

Defence in depth (each layer assumes the one before it could fail):
  1. validate_sql()  - allow-list: one SELECT only, known tables, no dangerous words
  2. READ ONLY transaction - Postgres itself refuses any write
  3. statement_timeout     - a runaway query is killed after a few seconds
  4. LIMIT wrapper         - never pulls back more than MAX_ROWS rows
  5. (recommended) a Postgres user that only has SELECT permission - see the guide
"""

import re
from decimal import Decimal

import pandas as pd

ALLOWED_TABLES = {"reviews", "analysis_runs"}

# Huge / useless-to-an-LLM columns. Blocked so a query can't dump them.
BLOCKED_WORDS = {"embedding", "results"}

FORBIDDEN_WORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "grant", "revoke", "copy", "call", "execute", "do", "vacuum", "merge",
    "into", "set", "reset", "listen", "notify", "lock", "refresh",
    "analyze", "explain", "recursive",
}

MAX_ROWS = 200          # hard cap on rows pulled from the DB
TIMEOUT_MS = 5000       # kill queries that run longer than 5 seconds


class UnsafeSQLError(ValueError):
    """Raised when a query fails validation. The message is shown back to the
    LLM so it can fix its own query."""


def validate_sql(sql, require_upload_id=None):
    """Returns a cleaned version of `sql` if it is safe, otherwise raises
    UnsafeSQLError. The cleaned version is what must be executed."""
    if not sql or not str(sql).strip():
        raise UnsafeSQLError("The query is empty.")

    # One left-to-right pass: keep 'string literals' as they are, drop comments.
    # (Doing it in two separate passes would let a '--' inside a string eat real SQL.)
    def _drop_comments(m):
        return m.group(0) if m.group(0).startswith("'") else " "

    cleaned = re.sub(r"'(?:[^']|'')*'|/\*.*?\*/|--[^\n]*", _drop_comments, str(sql).strip(), flags=re.S)
    cleaned = cleaned.strip().rstrip(";").strip()

    # Blank out 'string literals' so words INSIDE them (e.g. ILIKE '%delete%')
    # are not mistaken for SQL keywords.
    no_literals = re.sub(r"'(?:[^']|'')*'", "''", cleaned)
    lowered = no_literals.lower()

    if ";" in no_literals:
        raise UnsafeSQLError("Only ONE statement is allowed (no semicolons).")
    if '"' in no_literals or "\\" in cleaned or "$" in no_literals:
        raise UnsafeSQLError("Do not use double quotes, backslashes or $ in the query.")
    if not re.match(r"^\s*(select|with)\b", lowered):
        raise UnsafeSQLError("Only SELECT queries are allowed.")

    words = set(re.findall(r"[a-z_][a-z0-9_]*", lowered))

    bad = words & FORBIDDEN_WORDS
    if bad:
        raise UnsafeSQLError(f"Forbidden keyword(s): {sorted(bad)}")
    if any(w.startswith("pg_") for w in words) or "information_schema" in words:
        raise UnsafeSQLError("System tables and functions are not allowed.")
    bad_cols = words & BLOCKED_WORDS
    if bad_cols:
        raise UnsafeSQLError(f"Do not select these columns: {sorted(bad_cols)}")

    # Every table after FROM / JOIN must be on the allow-list (CTE names are OK).
    cte_names = set(re.findall(r"(?:\bwith|,)\s*([a-z_][a-z0-9_]*)\s+as\s*\(", lowered))
    for table in re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_.]*)", lowered):
        table = table.removeprefix("public.")
        if table not in ALLOWED_TABLES and table not in cte_names:
            raise UnsafeSQLError(
                f"Table '{table}' is not allowed. Allowed tables: {sorted(ALLOWED_TABLES)}"
            )

    if require_upload_id and str(require_upload_id) not in cleaned:
        raise UnsafeSQLError(
            f"The query must filter on upload_id = '{require_upload_id}' (current-upload mode)."
        )

    return cleaned


def run_readonly_query(conn, sql, require_upload_id=None):
    """Validates then runs `sql`. Returns (DataFrame, cleaned_sql)."""
    cleaned = validate_sql(sql, require_upload_id=require_upload_id)
    wrapped = f"SELECT * FROM (\n{cleaned}\n) AS q LIMIT {MAX_ROWS}"

    conn.rollback()  # SET TRANSACTION must be the FIRST thing in a transaction
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(f"SET LOCAL statement_timeout = {int(TIMEOUT_MS)}")
            cur.execute(wrapped)   # no params => '%' inside LIKE patterns is safe
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.rollback()            # always end the transaction

    # Postgres NUMERIC arrives as Decimal; turn into float so pandas/JSON/charts behave
    rows = [[float(v) if isinstance(v, Decimal) else v for v in row] for row in rows]
    return pd.DataFrame(rows, columns=columns), cleaned


def dataframe_to_json(df, max_rows=40):
    """Compact JSON for putting a query result into an LLM prompt."""
    return df.head(max_rows).to_json(orient="records", date_format="iso", default_handler=str)
