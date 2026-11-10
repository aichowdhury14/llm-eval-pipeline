"""The stand-in bot and the run loop: retrieve, answer, log every row."""
import random
import subprocess
import time
from datetime import datetime

import numpy as np

from .db import ROOT
from .scoring import score_run

PROMPTS = {
    "concise": (
        "Answer the question using only the given context. Reply with the shortest "
        "possible answer: a name, number or short phrase. No full sentences, no explanation."
    ),
    "verbose": (
        "You are a helpful assistant. Answer the question using the given context, "
        "in one or two full sentences, and explain your reasoning."
    ),
    "closed": "Reply with the shortest possible answer: a name, number or short phrase.",
}
MODES = ("rag", "gold", "closed")
FAULTS = ("none", "random-retrieval")


def ensure_embeddings(con, client, embed_model):
    missing = con.execute(
        "SELECT doc_id, text FROM documents WHERE doc_id NOT IN "
        "(SELECT doc_id FROM embeddings WHERE model = ?) ORDER BY doc_id",
        [embed_model],
    ).fetchall()
    if missing:
        print(f"Embedding {len(missing)} documents with {embed_model}...")
        vectors = client.embed(embed_model, [text for _, text in missing])
        con.executemany(
            "INSERT INTO embeddings VALUES (?, ?, ?)",
            [(doc_id, embed_model, vec) for (doc_id, _), vec in zip(missing, vectors)],
        )


def _unit(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def retrieve_all(con, client, embed_model, questions, top_k):
    """Embed every question in one pass, so the embedding and chat models are
    never loaded at the same time on a memory-limited machine."""
    ensure_embeddings(con, client, embed_model)
    rows = con.execute(
        "SELECT doc_id, vector FROM embeddings WHERE model = ? ORDER BY doc_id", [embed_model]
    ).fetchall()
    doc_ids = [r[0] for r in rows]
    scores = _unit(client.embed(embed_model, questions)) @ _unit([r[1] for r in rows]).T
    return [[doc_ids[i] for i in np.argsort(-row)[:top_k]] for row in scores]


def build_user_message(question, contexts):
    if not contexts:
        return f"Question: {question}"
    numbered = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, 1))
    return f"Context:\n{numbered}\n\nQuestion: {question}"


def _git_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def run_eval(con, client, label, model, mode="rag", top_k=3, prompt="concise",
             fault="none", limit=None, embed_model="bge-m3"):
    if mode not in MODES or fault not in FAULTS or prompt not in PROMPTS:
        raise ValueError("unknown mode, fault or prompt")
    questions = con.execute(
        "SELECT question_id, question, doc_id FROM questions ORDER BY question_id"
    ).fetchall()[:limit]
    if not questions:
        raise RuntimeError("No questions loaded. Run `python -m evalpipe ingest` first.")
    docs = dict(con.execute("SELECT doc_id, text FROM documents").fetchall())

    if mode == "closed":
        retrieved = [[] for _ in questions]
        prompt = "closed"
    elif mode == "gold":
        retrieved = [[doc_id] for _, _, doc_id in questions]
    elif fault == "random-retrieval":
        rng = random.Random(0)
        retrieved = [rng.sample(sorted(docs), top_k) for _ in questions]
    else:
        retrieved = retrieve_all(con, client, embed_model, [q[1] for q in questions], top_k)

    run_id = con.execute(
        "INSERT INTO runs VALUES (nextval('run_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING run_id",
        [label, datetime.now(), model, mode, top_k if mode == "rag" else None, prompt,
         PROMPTS[prompt], fault, embed_model if mode == "rag" else None, _git_sha()],
    ).fetchone()[0]

    for n, ((question_id, question, _), doc_ids) in enumerate(zip(questions, retrieved), 1):
        started = time.perf_counter()
        answer = client.chat(model, PROMPTS[prompt], build_user_message(question, [docs[d] for d in doc_ids]))
        latency_ms = (time.perf_counter() - started) * 1000
        con.execute(
            "INSERT INTO results VALUES (?, ?, ?, ?, ?)",
            [run_id, question_id, doc_ids, answer, latency_ms],
        )
        print(f"  [{n}/{len(questions)}] {latency_ms / 1000:5.1f}s  {answer[:70]!r}", flush=True)

    score_run(con, run_id)
    return run_id
