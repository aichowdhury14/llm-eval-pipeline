"""End-to-end run with a fake model, so CI needs no Ollama."""
import re
import zlib

import pytest

from evalpipe import report
from evalpipe.db import connect
from evalpipe.ingest import load_testset
from evalpipe.runner import run_eval

FACTS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Kenya", "Nairobi"), ("Peru", "Lima"),
    ("Norway", "Oslo"), ("Egypt", "Cairo"), ("Cuba", "Havana"), ("Nepal", "Kathmandu"),
]
TESTSET = {
    "documents": [
        {"doc_id": f"d{i}", "title": c, "text": f"The capital city of {c} is {cap}."}
        for i, (c, cap) in enumerate(FACTS)
    ],
    "questions": [
        {"question_id": f"q{i}", "question": f"What is the capital of {c}?", "answers": [cap], "doc_id": f"d{i}"}
        for i, (c, cap) in enumerate(FACTS)
    ],
}


class FakeClient:
    """Bag-of-words embeddings; the 'model' answers with the last word of the first context."""

    def embed(self, model, texts):
        vectors = []
        for text in texts:
            vec = [0.0] * 256
            for token in re.findall(r"\w+", text.lower()):
                vec[zlib.crc32(token.encode()) % 256] += 1.0
            vectors.append(vec)
        return vectors

    def chat(self, model, system, user):
        first = re.search(r"\[1\] (.*)", user)
        return first.group(1).rstrip(".").split()[-1] if first else "unknown"


@pytest.fixture
def con(tmp_path):
    con = connect(tmp_path / "test.duckdb")
    load_testset(con, TESTSET)
    return con


def test_rag_run_is_logged_and_scored(con):
    run_id = run_eval(con, FakeClient(), "baseline", "fake", top_k=2)
    row = report.summary(con)[0]
    assert (row["run_id"], row["n"]) == (run_id, len(FACTS))
    assert row["retrieval_hit"] == 1.0 and row["exact_match"] == 1.0


def test_planted_fault_is_flagged_as_regression(con):
    run_eval(con, FakeClient(), "baseline", "fake", top_k=1)
    run_eval(con, FakeClient(), "broken", "fake", top_k=1, fault="random-retrieval")
    _, regressions = report.compare(con, "baseline", "broken")
    assert "retrieval_hit" in regressions and "exact_match" in regressions
    assert report.compare(con, "baseline", "baseline")[1] == []


def test_closed_and_gold_modes_have_no_retrieval_score(con):
    run_eval(con, FakeClient(), "closed", "fake", mode="closed")
    run_eval(con, FakeClient(), "gold", "fake", mode="gold")
    closed, gold = report.summary(con)
    assert closed["retrieval_hit"] is None and closed["exact_match"] == 0.0
    assert gold["retrieval_hit"] is None and gold["exact_match"] == 1.0


def test_label_resolves_to_latest_run(con):
    run_eval(con, FakeClient(), "x", "fake", limit=2)
    second = run_eval(con, FakeClient(), "x", "fake", limit=2)
    assert report.resolve_run(con, "x") == second == report.resolve_run(con, second)
