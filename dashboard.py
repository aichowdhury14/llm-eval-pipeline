"""Streamlit dashboard: streamlit run dashboard.py  (open it after a run finishes)."""
import streamlit as st

from evalpipe.db import connect
from evalpipe.report import METRICS

st.set_page_config(page_title="LLM Eval Pipeline", layout="wide")
st.title("LLM evaluation runs")

con = connect(read_only=True)
runs = con.execute("SELECT * FROM run_summary").df()
if runs.empty:
    st.info("No runs yet. Run `python -m evalpipe run --label baseline` first.")
    st.stop()

runs["name"] = runs["run_id"].astype(str) + " · " + runs["label"]
st.dataframe(runs.drop(columns=["name", "created_at"]), hide_index=True, use_container_width=True)

metric = st.selectbox("Metric", METRICS, index=1)
st.bar_chart(runs.set_index("name")[metric])

st.subheader("Compare two runs, question by question")
left, right = st.columns(2)
base = left.selectbox("Baseline", runs["name"], index=0)
cand = right.selectbox("Candidate", runs["name"], index=len(runs) - 1)
ids = [int(runs.loc[runs["name"] == n, "run_id"].iloc[0]) for n in (base, cand)]

detail = con.execute(
    """
    SELECT q.question, q.answers AS reference,
           a.answer AS base_answer, sa.f1 AS base_f1, sa.retrieval_hit AS base_hit,
           b.answer AS cand_answer, sb.f1 AS cand_f1, sb.retrieval_hit AS cand_hit,
           sb.f1 - sa.f1 AS f1_change
    FROM questions q
    JOIN results a ON a.question_id = q.question_id AND a.run_id = ?
    JOIN scores sa ON sa.question_id = q.question_id AND sa.run_id = a.run_id
    JOIN results b ON b.question_id = q.question_id AND b.run_id = ?
    JOIN scores sb ON sb.question_id = q.question_id AND sb.run_id = b.run_id
    ORDER BY f1_change
    """,
    ids,
).df()
st.caption("Sorted with the biggest drops first.")
st.dataframe(detail, hide_index=True, use_container_width=True)
