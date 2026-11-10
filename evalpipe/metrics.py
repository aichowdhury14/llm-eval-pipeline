"""Cheap, deterministic metrics (no LLM judge), following the SQuAD conventions."""
import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text):
    text = _ARTICLES.sub(" ", text.lower().translate(_PUNCT))
    return " ".join(text.split())


def exact_match(prediction, answers):
    return float(any(normalize(prediction) == normalize(a) for a in answers))


def _f1(prediction, answer):
    pred, gold = normalize(prediction).split(), normalize(answer).split()
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if not pred or not gold or not overlap:
        return float(pred == gold)
    precision, recall = overlap / len(pred), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction, answers):
    return max(_f1(prediction, a) for a in answers)


def contains_answer(prediction, answers):
    """Lenient: a reference answer appears somewhere in a longer reply."""
    pred = f" {normalize(prediction)} "
    return float(any(f" {normalize(a)} " in pred for a in answers if normalize(a)))


def retrieval_hit(retrieved, gold_doc_id):
    return float(gold_doc_id in retrieved)


def reciprocal_rank(retrieved, gold_doc_id):
    return 1.0 / (retrieved.index(gold_doc_id) + 1) if gold_doc_id in retrieved else 0.0
