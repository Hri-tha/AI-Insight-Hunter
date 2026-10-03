"""
=====================================================================================
AI INSIGHT HUNTER — Frontend (Streamlit)
Two-panel layout: dashboard on the left, chat on the right — matching the
technical design doc's mockup. This file only handles the UI; all the real
work (tagging, stats, hypotheses, chat) lives in backend/pipeline.py.

RUN (from the project's root folder, one level above frontend/):
    streamlit run frontend/app.py --server.fileWatcherType none
=====================================================================================
"""

import sys
from pathlib import Path

# Lets this file import the backend package when run as `streamlit run frontend/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from backend.db import get_connection
from backend.pipeline import REQUIRED_COLUMNS, run_pipeline, answer_question, load_run

st.set_page_config(page_title="AI Insight Hunter", layout="wide")
st.title("AI Insight Hunter")

# session_state remembers things between Streamlit reruns (it reruns the
# whole script on every click)
if "results" not in st.session_state:
    st.session_state.results = None
if "upload_id" not in st.session_state:
    st.session_state.upload_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def make_sample_df():
    """A tiny built-in dataset so you can try the app with no file at hand."""
    import datetime
    today = datetime.date.today()
    rows = []
    sizing_reviews = [
        "The shirt runs really small, ordered my usual medium but it barely fit.",
        "Sizing is way off — I had to return it for a larger size.",
        "The size chart on the website is inaccurate, very misleading.",
        "Ordered L but it fits like S, had to send it back.",
        "Size chart says M should fit me, but it is much smaller.",
    ]
    delivery_reviews = [
        "Delivery took over two weeks, way longer than promised.",
        "My package arrived 10 days late with no updates from support.",
        "Shipping was extremely delayed, no tracking info provided.",
    ]
    for i, text in enumerate(sizing_reviews * 3):
        rows.append({
            "review_id": f"R{1000+i}", "review_date": today - datetime.timedelta(days=i % 14),
            "rating": 2, "review_text": text, "product_category": "Women's Apparel",
            "returned": i % 2 == 0,
        })
    for i, text in enumerate(delivery_reviews * 2):
        rows.append({
            "review_id": f"R{2000+i}", "review_date": today - datetime.timedelta(days=i % 14),
            "rating": 2, "review_text": text, "product_category": "Electronics",
            "returned": i % 3 == 0,
        })
    return pd.DataFrame(rows)


# --- Upload section ---
col_upload, col_sample = st.columns([3, 1])
with col_upload:
    uploaded_file = st.file_uploader("Upload a review sheet (CSV)", type=["csv"])
with col_sample:
    st.write("")
    use_sample = st.button("Use sample data")

df = None
if use_sample:
    df = make_sample_df()
    st.info("Using built-in sample data.")
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview:")
    st.dataframe(df.head())

if df is not None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Your file is missing required columns: {missing}")
    elif st.button("Run analysis"):
        progress_bar = st.progress(0, text="Starting...")

        def update_progress(current, total, stage="tagging"):
            label = "Tagging reviews" if stage == "tagging" else "Generating embeddings"
            progress_bar.progress(current / total, text=f"{label}... ({current}/{total})")

        conn = get_connection()
        try:
            with st.spinner("Running the full pipeline — this can take a while on the free tier..."):
                results = run_pipeline(df, conn, progress_callback=update_progress)
        except Exception as e:
            progress_bar.empty()
            st.error(f"Something went wrong: {e}")
            conn.close()
            st.stop()
        conn.close()

        progress_bar.empty()
        st.session_state.results = results
        st.session_state.upload_id = results["upload_id"]
        st.session_state.chat_history = []
        st.success("Done.")


# =====================================================================================
# DASHBOARD + CHAT (two panels, side by side)
# =====================================================================================

if st.session_state.results:
    results = st.session_state.results
    dashboard_col, chat_col = st.columns([2, 1])

    with dashboard_col:
        meta = results["meta"]
        investigated = [p for p in results["problems"] if p["decision"] == "INVESTIGATE"]
        monitored = [p for p in results["problems"] if p["decision"] == "MONITOR"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Reviews analysed", meta["total_reviews"])
        m2.metric("Problems found", len(investigated))
        m3.metric("Monitoring", len(monitored))

        st.info(results.get("summary", ""))

        st.subheader("Top customer problems")
        for p in results["problems"]:
            if p["decision"] == "INSUFFICIENT_DATA":
                continue
            badge = "🔴" if p["severity"] in ("High", "Critical") else "🟡"
            with st.expander(f"{badge} {p['problem']} — {p['decision']} — {p['reviews']} reviews"):
                st.write(f"**Category:** {p['issue_category']}  |  **Rate:** {p['rate_pct']}%  "
                         f"|  **WoW change:** {p['wow_pct']}%  |  **Avg rating:** {p['avg_rating']}")
                if p.get("concentration"):
                    c = p["concentration"]
                    st.write(f"**Concentrated in:** {c['name']} ({c['share_pct']}% of these reviews)")
                if p.get("business_impact"):
                    b = p["business_impact"]
                    st.write(f"**Return rate:** {b['return_rate_pct']}% vs {b['overall_return_rate_pct']}% overall")

                if p.get("investigation"):
                    inv = p["investigation"]
                    st.markdown("**Investigation**")
                    for v in inv.get("verdicts", []):
                        st.write(f"- {v['id']} ({v['verdict']}): {v['why']}")
                    st.write(f"**Likely cause:** {inv.get('likely_cause', '')} "
                             f"(confidence: {inv.get('confidence', '')})")
                    if inv.get("opportunities"):
                        st.write("**Opportunities:** " + ", ".join(inv["opportunities"]))
                    if inv.get("data_gaps"):
                        st.write("**Data gaps:** " + ", ".join(inv["data_gaps"]))

    with chat_col:
        st.subheader("Ask a question")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("chart"):
                    st.caption(msg["chart"]["title"])
                    st.bar_chart(msg["chart"]["data"])

        question = st.chat_input("e.g. Why is Incorrect Fit rising?  or  Show a chart of reviews per problem")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            chart = None
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    conn = get_connection()
                    try:
                        answer, sources, chart = answer_question(
                            question, results, conn, st.session_state.upload_id
                        )
                    except Exception as e:
                        answer = f"Sorry, that failed: {e}"
                    conn.close()
                st.write(answer)
                if chart:
                    st.caption(chart["title"])
                    st.bar_chart(chart["data"])

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "chart": chart}
            )
else:
    st.info("Upload a file (or click 'Use sample data') and click 'Run analysis' to get started.")