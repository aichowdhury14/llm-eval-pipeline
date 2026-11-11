"""Parquet snapshots for the dashboard.

DuckDB lets only one process open the database file, so the dashboard never
opens it. It reads these snapshots instead, which are rewritten after every run.
"""
import os
import time
from pathlib import Path

SNAPSHOTS = {
    "runs": "SELECT * FROM run_summary",
    "answers": """
        SELECT x.run_id, q.question_id, q.question, q.answers AS reference, x.answer,
               x.retrieved_doc_ids, x.latency_ms, s.exact_match, s.f1,
               s.contains_answer, s.retrieval_hit
        FROM results x
        JOIN scores s USING (run_id, question_id)
        JOIN questions q USING (question_id)
    """,
}


def export_dir(con):
    path = con.execute(
        "SELECT path FROM duckdb_databases() WHERE database_name = current_database()"
    ).fetchone()[0]
    return Path(path).parent / "export"


def export_snapshot(con):
    out = export_dir(con)
    out.mkdir(parents=True, exist_ok=True)
    for name, query in SNAPSHOTS.items():
        tmp, final = out / f"{name}.tmp.parquet", out / f"{name}.parquet"
        con.execute(f"COPY ({query}) TO '{tmp.as_posix()}' (FORMAT parquet)")
        for attempt in range(5):  # the dashboard may be reading the old file
            try:
                os.replace(tmp, final)
                break
            except PermissionError:
                time.sleep(0.5)
        else:
            print(f"Warning: could not update {final}; close the dashboard and run `evalpipe export`.")
    return out
