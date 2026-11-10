"""Run summaries, run-to-run comparison, and the regression gate."""
METRICS = ("exact_match", "f1", "contains_answer", "retrieval_hit", "mrr")
SUMMARY_COLS = ("run_id", "label", "model", "mode", "top_k", "prompt_name", "fault", "n",
                *METRICS, "latency_ms")


def summary(con):
    rows = con.execute(f"SELECT {', '.join(SUMMARY_COLS)} FROM run_summary").fetchall()
    return [dict(zip(SUMMARY_COLS, r)) for r in rows]


def resolve_run(con, ref):
    """A run is referenced by its id, or by label (the latest run with that label)."""
    ref = str(ref)
    row = con.execute(
        "SELECT max(run_id) FROM runs WHERE label = ? OR CAST(run_id AS VARCHAR) = ?", [ref, ref]
    ).fetchone()
    if row[0] is None:
        raise KeyError(f"No run found for {ref!r}")
    return row[0]


def compare(con, base_ref, cand_ref, max_drop=0.05):
    """Return (rows, regressions). A metric regresses when the candidate scores
    more than max_drop below the baseline."""
    runs = {r["run_id"]: r for r in summary(con)}
    base, cand = runs[resolve_run(con, base_ref)], runs[resolve_run(con, cand_ref)]
    rows, regressions = [], []
    for m in METRICS:
        b, c = base[m], cand[m]
        delta = None if b is None or c is None else c - b
        rows.append({"metric": m, "base": b, "candidate": c, "delta": delta})
        if delta is not None and delta < -max_drop:
            regressions.append(m)
    return rows, regressions


def _fmt(value):
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def table(rows, cols, markdown=False):
    cells = [[_fmt(r[c]) for c in cols] for r in rows]
    if markdown:
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        return "\n".join(lines + ["| " + " | ".join(c) + " |" for c in cells])
    widths = [max([len(c), *(len(row[i]) for row in cells)]) for i, c in enumerate(cols)]
    line = lambda vals: "  ".join(v.ljust(w) for v, w in zip(vals, widths))
    return "\n".join([line(cols), line(["-" * w for w in widths]), *map(line, cells)])
