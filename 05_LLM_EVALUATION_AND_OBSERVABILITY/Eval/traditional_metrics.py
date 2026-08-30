"""Dependency-free traditional NLP metrics for educational use."""

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(tokens[index : index + order])
        for index in range(len(tokens) - order + 1)
    )


def _precision_recall_f1(overlap: int, predicted: int, reference: int) -> dict:
    precision = overlap / predicted if predicted else 0.0
    recall = overlap / reference if reference else 0.0
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def token_f1_score(response: str, reference: str) -> float:
    response_tokens = tokenize(response)
    reference_tokens = tokenize(reference)
    overlap = sum((Counter(response_tokens) & Counter(reference_tokens)).values())
    return _precision_recall_f1(
        overlap,
        len(response_tokens),
        len(reference_tokens),
    )["f1"]


def bleu_score(response: str, reference: str, max_order: int = 4) -> float:
    response_tokens = tokenize(response)
    reference_tokens = tokenize(reference)
    precisions = []

    for order in range(1, max_order + 1):
        response_ngrams = _ngrams(response_tokens, order)
        if not response_ngrams:
            break
        reference_ngrams = _ngrams(reference_tokens, order)
        overlap = sum((response_ngrams & reference_ngrams).values())
        precisions.append(overlap / sum(response_ngrams.values()))

    if not precisions or any(precision == 0 for precision in precisions):
        return 0.0

    geometric_mean = math.exp(
        sum(math.log(precision) for precision in precisions) / len(precisions)
    )
    brevity_penalty = (
        1.0
        if len(response_tokens) >= len(reference_tokens)
        else math.exp(1 - len(reference_tokens) / max(len(response_tokens), 1))
    )
    return brevity_penalty * geometric_mean


def gleu_score(response: str, reference: str, max_order: int = 4) -> float:
    response_tokens = tokenize(response)
    reference_tokens = tokenize(reference)
    scores = []

    for order in range(1, max_order + 1):
        response_ngrams = _ngrams(response_tokens, order)
        reference_ngrams = _ngrams(reference_tokens, order)
        if not response_ngrams or not reference_ngrams:
            break
        overlap = sum((response_ngrams & reference_ngrams).values())
        precision = overlap / sum(response_ngrams.values())
        recall = overlap / sum(reference_ngrams.values())
        scores.append(min(precision, recall))

    return sum(scores) / len(scores) if scores else 0.0


def meteor_style_score(response: str, reference: str) -> float:
    """Approximate METEOR using exact unigram matches and chunk penalty."""
    response_tokens = tokenize(response)
    reference_tokens = tokenize(reference)
    if not response_tokens or not reference_tokens:
        return 0.0

    available_positions: dict[str, list[int]] = {}
    for index, token in enumerate(reference_tokens):
        available_positions.setdefault(token, []).append(index)

    matched_positions = []
    for token in response_tokens:
        positions = available_positions.get(token, [])
        if positions:
            matched_positions.append(positions.pop(0))

    matches = len(matched_positions)
    if matches == 0:
        return 0.0

    precision = matches / len(response_tokens)
    recall = matches / len(reference_tokens)
    weighted_f_mean = (10 * precision * recall) / (recall + 9 * precision)

    chunks = 1 + sum(
        current != previous + 1
        for previous, current in zip(matched_positions, matched_positions[1:])
    )
    penalty = 0.5 * (chunks / matches) ** 3
    return weighted_f_mean * (1 - penalty)


def rouge_n(response: str, reference: str, order: int) -> dict:
    response_ngrams = _ngrams(tokenize(response), order)
    reference_ngrams = _ngrams(tokenize(reference), order)
    overlap = sum((response_ngrams & reference_ngrams).values())
    return _precision_recall_f1(
        overlap,
        sum(response_ngrams.values()),
        sum(reference_ngrams.values()),
    )


def rouge_l(response: str, reference: str) -> dict:
    response_tokens = tokenize(response)
    reference_tokens = tokenize(reference)
    rows = len(response_tokens) + 1
    columns = len(reference_tokens) + 1
    lengths = [[0] * columns for _ in range(rows)]

    for row in range(1, rows):
        for column in range(1, columns):
            if response_tokens[row - 1] == reference_tokens[column - 1]:
                lengths[row][column] = lengths[row - 1][column - 1] + 1
            else:
                lengths[row][column] = max(
                    lengths[row - 1][column],
                    lengths[row][column - 1],
                )

    return _precision_recall_f1(
        lengths[-1][-1],
        len(response_tokens),
        len(reference_tokens),
    )
