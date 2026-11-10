"""DuckDB storage: one local file holds the test set, every run, and its scores."""
import os
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "eval.duckdb"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id VARCHAR PRIMARY KEY, title VARCHAR, text VARCHAR);
CREATE TABLE IF NOT EXISTS questions (
    question_id VARCHAR PRIMARY KEY, question VARCHAR,
    answers VARCHAR[], doc_id VARCHAR);
CREATE TABLE IF NOT EXISTS embeddings (
    doc_id VARCHAR, model VARCHAR, vector FLOAT[], PRIMARY KEY (doc_id, model));
CREATE SEQUENCE IF NOT EXISTS run_seq;
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY, label VARCHAR, created_at TIMESTAMP,
    model VARCHAR, mode VARCHAR, top_k INTEGER, prompt_name VARCHAR,
    prompt_text VARCHAR, fault VARCHAR, embed_model VARCHAR, git_sha VARCHAR);
CREATE TABLE IF NOT EXISTS results (
    run_id INTEGER, question_id VARCHAR, retrieved_doc_ids VARCHAR[],
    answer VARCHAR, latency_ms DOUBLE, PRIMARY KEY (run_id, question_id));
CREATE TABLE IF NOT EXISTS scores (
    run_id INTEGER, question_id VARCHAR, exact_match DOUBLE, f1 DOUBLE,
    contains_answer DOUBLE, retrieval_hit DOUBLE, reciprocal_rank DOUBLE,
    PRIMARY KEY (run_id, question_id));
CREATE OR REPLACE VIEW run_summary AS
SELECT r.run_id, r.label, r.model, r.mode, r.top_k, r.prompt_name, r.fault,
       r.created_at, count(*) AS n,
       avg(s.exact_match) AS exact_match, avg(s.f1) AS f1,
       avg(s.contains_answer) AS contains_answer,
       avg(s.retrieval_hit) AS retrieval_hit, avg(s.reciprocal_rank) AS mrr,
       avg(x.latency_ms) AS latency_ms
FROM runs r
JOIN results x USING (run_id)
JOIN scores s USING (run_id, question_id)
GROUP BY ALL
ORDER BY r.run_id;
"""


def connect(path=None, read_only=False):
    path = Path(path or os.environ.get("EVALPIPE_DB") or DEFAULT_DB)
    if read_only:
        return duckdb.connect(str(path), read_only=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    return con
