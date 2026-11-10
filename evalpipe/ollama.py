"""Thin client for a local Ollama server. Tests swap this for a fake.

Context windows are kept small on purpose: Ollama allocates memory for the whole
window up front, and the defaults do not fit on a memory-limited PC.
"""
import os

import requests

EMBED_CTX = int(os.environ.get("EVALPIPE_EMBED_CTX", "1024"))
CHAT_CTX = int(os.environ.get("EVALPIPE_CHAT_CTX", "2048"))


def _post(url, payload):
    try:
        resp = requests.post(url, json=payload, timeout=900)
    except requests.ConnectionError:
        raise RuntimeError(f"Cannot reach Ollama at {url}. Is it running?") from None
    if not resp.ok:  # surface Ollama's own message, e.g. out-of-memory or model not found
        raise RuntimeError(f"Ollama {resp.status_code}: {resp.text[:400]}")
    return resp.json()


class OllamaClient:
    def __init__(self, url=None):
        self.url = (url or os.environ.get("EVALPIPE_OLLAMA_URL") or "http://localhost:11434").rstrip("/")

    def embed(self, model, texts, batch=8):
        vectors = []
        for i in range(0, len(texts), batch):
            out = _post(
                f"{self.url}/api/embed",
                {
                    "model": model,
                    "input": texts[i : i + batch],
                    "truncate": True,
                    "options": {"num_ctx": EMBED_CTX},
                    # unload after the last batch so the chat model has the memory
                    "keep_alive": 0 if i + batch >= len(texts) else "5m",
                },
            )
            vectors.extend(out["embeddings"])
        return vectors

    def chat(self, model, system, user, max_tokens=96):
        out = _post(
            f"{self.url}/api/chat",
            {
                "model": model,
                "stream": False,
                "think": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": 0, "seed": 0, "num_predict": max_tokens,
                            "num_ctx": CHAT_CTX},
            },
        )
        return out["message"]["content"].strip()
