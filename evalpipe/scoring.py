"""Scoring is a separate stage: it can be re-run on old results when metrics change."""
from . import metrics


def score_run(con, run_id):
    mode = con.execute("SELECT mode FROM runs WHERE run_id = ?", [run_id]).fetchone()[0]
    rows = con.execute(
        "SELECT x.question_id, x.answer, x.retrieved_doc_ids, q.answers, q.doc_id "
        "FROM results x JOIN questions q USING (question_id) WHERE x.run_id = ?",
        [run_id],
    ).fetchall()
    scored = []
    for question_id, answer, retrieved, answers, gold_doc in rows:
        is_rag = mode == "rag"
        scored.append((
            run_id, question_id,
            metrics.exact_match(answer, answers),
            metrics.token_f1(answer, answers),
            metrics.contains_answer(answer, answers),
            metrics.retrieval_hit(retrieved, gold_doc) if is_rag else None,
            metrics.reciprocal_rank(retrieved, gold_doc) if is_rag else None,
        ))
    con.execute("DELETE FROM scores WHERE run_id = ?", [run_id])
    con.executemany("INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?, ?)", scored)
    return len(scored)
