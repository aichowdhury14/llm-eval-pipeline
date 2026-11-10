"""Build a small frozen test set from SQuAD v1.1 and load it into DuckDB.

The test set is saved to data/testset.json and committed, so every run is
scored against exactly the same questions.
"""
import json
import random
import time
from pathlib import Path

import requests

from .db import ROOT

API = "https://datasets-server.huggingface.co/rows"
DATASET = {"dataset": "rajpurkar/squad", "config": "plain_text", "split": "validation"}
SPLIT_ROWS = 10570
TESTSET_PATH = ROOT / "data" / "testset.json"


def fetch_rows(offset, length=100):
    for attempt in range(4):
        resp = requests.get(API, params={**DATASET, "offset": offset, "length": length}, timeout=60)
        if resp.status_code == 200:
            return [r["row"] for r in resp.json()["rows"]]
        time.sleep(2 * (attempt + 1))
    resp.raise_for_status()


def build_testset(n_questions=30, n_docs=60, seed=7, docs_per_page=6):
    """Pick n_docs distinct paragraphs spread across topics, and one question
    for n_questions of them. Paragraphs without a question act as distractors."""
    pages = -(-n_docs // docs_per_page)
    docs, first_question = {}, {}
    for page in range(pages):
        taken = 0
        for row in fetch_rows(page * (SPLIT_ROWS // pages)):
            text = row["context"]
            if text in docs:
                continue
            if taken == docs_per_page or len(docs) == n_docs:
                break
            docs[text] = {"doc_id": f"d{len(docs) + 1:03d}", "title": row["title"], "text": text}
            first_question[text] = row
            taken += 1
    chosen = random.Random(seed).sample(sorted(docs), min(n_questions, len(docs)))
    questions = [
        {
            "question_id": first_question[t]["id"],
            "question": first_question[t]["question"],
            "answers": sorted(set(first_question[t]["answers"]["text"])),
            "doc_id": docs[t]["doc_id"],
        }
        for t in chosen
    ]
    return {
        "source": "SQuAD v1.1 validation (rajpurkar/squad), CC BY-SA 4.0",
        "documents": list(docs.values()),
        "questions": sorted(questions, key=lambda q: q["doc_id"]),
    }


def load_testset(con, testset):
    """Replace the test set tables. Embeddings are kept only for unchanged documents."""
    con.execute("DELETE FROM questions")
    con.execute("DELETE FROM documents")
    con.executemany(
        "INSERT INTO documents VALUES (?, ?, ?)",
        [(d["doc_id"], d["title"], d["text"]) for d in testset["documents"]],
    )
    con.executemany(
        "INSERT INTO questions VALUES (?, ?, ?, ?)",
        [(q["question_id"], q["question"], q["answers"], q["doc_id"]) for q in testset["questions"]],
    )


def ingest(con, refresh=False, path=TESTSET_PATH, **build_args):
    path = Path(path)
    if refresh or not path.exists():
        testset = build_testset(**build_args)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(testset, ensure_ascii=False, indent=1), encoding="utf-8")
        con.execute("DELETE FROM embeddings")
    testset = json.loads(path.read_text(encoding="utf-8"))
    load_testset(con, testset)
    return len(testset["documents"]), len(testset["questions"])
