from evalpipe import metrics


def test_normalize_strips_case_punctuation_articles():
    assert metrics.normalize("The  Denver Broncos!") == "denver broncos"


def test_exact_match_uses_any_reference():
    assert metrics.exact_match("denver broncos", ["Carolina", "The Denver Broncos"]) == 1.0
    assert metrics.exact_match("Broncos won", ["Denver Broncos"]) == 0.0


def test_token_f1_partial_overlap():
    assert metrics.token_f1("Denver Broncos", ["Denver Broncos"]) == 1.0
    assert metrics.token_f1("Broncos", ["Denver Broncos"]) == 2 * 1.0 * 0.5 / 1.5
    assert metrics.token_f1("Panthers", ["Denver Broncos"]) == 0.0
    assert metrics.token_f1("", ["Denver Broncos"]) == 0.0


def test_contains_answer_matches_whole_words_only():
    assert metrics.contains_answer("It was the Denver Broncos who won.", ["Denver Broncos"]) == 1.0
    assert metrics.contains_answer("In 1066 exactly", ["106"]) == 0.0


def test_retrieval_metrics():
    assert metrics.retrieval_hit(["d2", "d1"], "d1") == 1.0
    assert metrics.reciprocal_rank(["d2", "d1"], "d1") == 0.5
    assert metrics.retrieval_hit(["d2"], "d1") == 0.0
    assert metrics.reciprocal_rank(["d2"], "d1") == 0.0
