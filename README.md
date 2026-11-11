# LLM Eval Pipeline

A small data pipeline that measures whether a question-answering bot got better or worse after a change. It runs a fixed set of questions through a local model, logs every question, retrieved passage and answer to DuckDB, scores them, and fails loudly when a score drops.

Everything is free and runs offline on a CPU-only PC: Python, DuckDB, and [Ollama](https://ollama.com).

## How it works

```
SQuAD sample ──► data/testset.json ──► DuckDB ──► run (retrieve + answer) ──► score ──► compare / dashboard
   ingest          frozen, in git      storage      stand-in bot               metrics    regression gate
```

| Stage | Code | What it does |
|---|---|---|
| Ingest | `evalpipe/ingest.py` | Samples 30 questions and 58 passages from SQuAD v1.1 and freezes them in `data/testset.json` |
| Store | `evalpipe/db.py` | One DuckDB file with tables for documents, questions, embeddings, runs, results and scores |
| Run | `evalpipe/runner.py` | A stand-in bot: embeds the question, retrieves the top passages, asks the model, logs the row |
| Score | `evalpipe/scoring.py`, `metrics.py` | Exact match, token F1, contains-answer, retrieval hit rate, mean reciprocal rank |
| Compare | `evalpipe/report.py` | Compares two runs and exits with code 1 if any metric dropped too far |
| Export | `evalpipe/export.py` | Writes Parquet snapshots of the scores after every run, for the dashboard |
| Dashboard | `dashboard.py` | Streamlit page with run scores and a question-by-question diff |

The bot has three modes, so you can tell retrieval problems apart from model problems:

- `rag`: retrieves passages by embedding similarity (the realistic case).
- `gold`: is handed the correct passage, which shows the best the model can do.
- `closed`: gets no passage at all, which shows what the model already knows.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
ollama pull qwen3:1.7b
ollama pull bge-m3
```

## Usage

```powershell
# 1. Load the test set (uses the committed data/testset.json)
python -m evalpipe ingest

# 2. Baseline run
python -m evalpipe run --label baseline

# 3. Change one thing and run again
python -m evalpipe run --label verbose-prompt --prompt verbose
python -m evalpipe run --label top1 --top-k 1
python -m evalpipe run --label bigger-model --model qwen3:4b

# 4. Compare (exit code 1 means a regression)
python -m evalpipe compare baseline verbose-prompt

# 5. See everything
python -m evalpipe runs
python -m evalpipe report        # writes reports/runs.md
python -m streamlit run dashboard.py
```

### Prove the pipeline catches a fault

Plant a known bug (the retriever returns random passages) and check that the comparison flags it:

```powershell
python -m evalpipe run --label broken --fault random-retrieval
python -m evalpipe compare baseline broken
```

## Tests

```powershell
python -m pytest
```

The tests swap Ollama for a fake model, so they run in seconds and in GitHub Actions on every push.

## Notes

- **Memory:** the two models need roughly 1.5 GB free each, and are never loaded at the same time. If Ollama reports "failed to allocate buffer", close some applications and run again.
- **Speed:** expect a few seconds to half a minute per question on CPU. Use `--limit 5` for a quick check.
- **Dashboard:** DuckDB lets only one process open the database file, so the dashboard reads Parquet snapshots in `data/export` instead. It can stay open during a run and shows each run once it finishes; press R in the browser to reload.
- **One run at a time:** for the same reason, a second `evalpipe` command fails with a "file is being used by another process" error while a run is in progress.
- **Metrics:** there is no LLM judge yet. All scores are deterministic string and ranking metrics, which are fast and repeatable but strict on wording.

## Data

The test set is a sample of [SQuAD v1.1](https://huggingface.co/datasets/rajpurkar/squad) (Rajpurkar et al.), licensed CC BY-SA 4.0.

---

Built by Atiqul Islam Chowdhury.
