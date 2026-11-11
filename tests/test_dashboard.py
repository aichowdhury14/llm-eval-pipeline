"""The dashboard renders from snapshots, including while the database is locked by a writer."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from evalpipe.db import connect
from evalpipe.ingest import load_testset
from evalpipe.runner import run_eval
from tests.test_pipeline import TESTSET, FakeClient

DASHBOARD = str(Path(__file__).resolve().parent.parent / "dashboard.py")


def test_dashboard_renders_while_database_is_open(tmp_path, monkeypatch):
    db = tmp_path / "eval.duckdb"
    monkeypatch.setenv("EVALPIPE_DB", str(db))
    con = connect(db)  # stays open, like a run in progress
    load_testset(con, TESTSET)
    run_eval(con, FakeClient(), "baseline", "fake", top_k=1)
    run_eval(con, FakeClient(), "broken", "fake", top_k=1, fault="random-retrieval")

    app = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    assert not app.exception
    runs_table, detail_table = (d.value for d in app.dataframe)
    assert list(runs_table["label"]) == ["baseline", "broken"]
    assert len(detail_table) == len(TESTSET["questions"])
    assert detail_table["f1_change"].min() < 0


def test_dashboard_without_results_shows_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALPIPE_DB", str(tmp_path / "eval.duckdb"))
    app = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    assert not app.exception and "No results yet" in app.info[0].value
