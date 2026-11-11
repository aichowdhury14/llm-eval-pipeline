"""Streamlit dashboard: python -m streamlit run dashboard.py

Reads the Parquet snapshots in data/export, not the database file, so it can
stay open while a run is in progress. Press R in the browser to reload.
"""
import os
from pathlib import Path

import duckdb
import streamlit as st

from evalpipe.db import DEFAULT_DB
from evalpipe.report import METRICS

EXPORT = Path(os.environ.get("EVALPIPE_DB") or DEFAULT_DB).parent / "export"

st.set_page_config(page_title="LLM Eval Pipeline", layout="wide")
st.title("LLM evaluation runs")

if not (EXPORT / "runs.parquet").exists():
    st.info("No results yet. Run `python -m evalpipe run --label baseline`, "
            "or `python -m evalpipe export` if you already have runs.")
    st.stop()

con = duckdb.connect()
con.execute(f"CREATE VIEW runs AS SELECT * FROM read_parquet('{(EXPORT / 'runs.parquet').as_posix()}')")
con.execute(f"CREATE VIEW answers AS SELECT * FROM read_parquet('{(EXPORT / 'answers.parquet').as_posix()}')")

runs = con.execute("SELECT * FROM runs ORDER BY run_id").df()
if runs.empty:
    st.info("No finished runs yet.")
    st.stop()

runs["name"] = runs["run_id"].astype(str) + " · " + runs["label"]
st.dataframe(runs.drop(columns=["name", "created_at"]), hide_index=True, width="stretch")

metric = st.selectbox("Metric", METRICS, index=1)
st.bar_chart(runs.set_index("name")[metric])

st.subheader("Compare two runs, question by question")
left, right = st.columns(2)
base = left.selectbox("Baseline", runs["name"], index=0)
cand = right.selectbox("Candidate", runs["name"], index=len(runs) - 1)
ids = [int(runs.loc[runs["name"] == n, "run_id"].iloc[0]) for n in (base, cand)]

detail = con.execute(
    """
    SELECT a.question, a.reference,
           a.answer AS base_answer, a.f1 AS base_f1, a.retrieval_hit AS base_hit,
           b.answer AS cand_answer, b.f1 AS cand_f1, b.retrieval_hit AS cand_hit,
           b.f1 - a.f1 AS f1_change
    FROM answers a
    JOIN answers b ON b.question_id = a.question_id AND b.run_id = ?
    WHERE a.run_id = ?
    ORDER BY f1_change, a.question
    """,
    [ids[1], ids[0]],
).df()
st.caption("Sorted with the biggest drops first.")
st.dataframe(detail, hide_index=True, width="stretch")
