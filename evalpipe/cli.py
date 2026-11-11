"""Command line: python -m evalpipe <ingest|run|score|runs|compare|report|export>"""
import argparse
import sys
from datetime import date

from . import report
from .db import ROOT, connect
from .export import export_snapshot
from .ingest import ingest
from .ollama import OllamaClient
from .runner import FAULTS, MODES, PROMPTS, run_eval
from .scoring import score_run


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="evalpipe", description="LLM evaluation pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="build/load the test set")
    p.add_argument("--refresh", action="store_true", help="re-download instead of using data/testset.json")
    p.add_argument("--questions", type=int, default=30)
    p.add_argument("--docs", type=int, default=60)

    p = sub.add_parser("run", help="run the bot over the test set and score it")
    p.add_argument("--label", required=True)
    p.add_argument("--model", default="qwen3:1.7b")
    p.add_argument("--mode", choices=MODES, default="rag")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--prompt", choices=[k for k in PROMPTS if k != "closed"], default="concise")
    p.add_argument("--fault", choices=FAULTS, default="none")
    p.add_argument("--embed-model", default="bge-m3")
    p.add_argument("--limit", type=int, help="only the first N questions")

    p = sub.add_parser("score", help="re-score an existing run")
    p.add_argument("run")

    sub.add_parser("runs", help="list all runs with their scores")

    p = sub.add_parser("compare", help="compare two runs; exit 1 on regression")
    p.add_argument("base")
    p.add_argument("candidate")
    p.add_argument("--max-drop", type=float, default=0.05)

    sub.add_parser("report", help="write reports/runs.md")
    sub.add_parser("export", help="refresh the dashboard snapshots")

    args = parser.parse_args(argv)
    con = connect()
    try:
        return _dispatch(args, con)
    except (RuntimeError, KeyError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


def _dispatch(args, con):
    if args.cmd == "ingest":
        docs, questions = ingest(con, refresh=args.refresh, n_questions=args.questions, n_docs=args.docs)
        print(f"Loaded {docs} documents and {questions} questions.")
    elif args.cmd == "run":
        run_id = run_eval(con, OllamaClient(), args.label, args.model, args.mode, args.top_k,
                          args.prompt, args.fault, args.limit, args.embed_model)
        row = [r for r in report.summary(con) if r["run_id"] == run_id]
        print(report.table(row, report.SUMMARY_COLS))
    elif args.cmd == "score":
        run_id = report.resolve_run(con, args.run)
        print(f"Scored {score_run(con, run_id)} answers for run {run_id}.")
    elif args.cmd == "runs":
        print(report.table(report.summary(con), report.SUMMARY_COLS))
    elif args.cmd == "compare":
        rows, regressions = report.compare(con, args.base, args.candidate, args.max_drop)
        print(report.table(rows, ("metric", "base", "candidate", "delta")))
        if regressions:
            print(f"\nREGRESSION: {', '.join(regressions)} dropped by more than {args.max_drop}")
            return 1
        print("\nOK: no metric dropped by more than", args.max_drop)
    elif args.cmd == "report":
        out = ROOT / "reports" / "runs.md"
        out.parent.mkdir(exist_ok=True)
        body = report.table(report.summary(con), report.SUMMARY_COLS, markdown=True)
        out.write_text(f"# Evaluation runs\n\nUpdated {date.today()}.\n\n{body}\n", encoding="utf-8")
        print(f"Wrote {out}")
    elif args.cmd == "export":
        print(f"Wrote snapshots to {export_snapshot(con)}")
    return 0
