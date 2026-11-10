"""Span-level attribution utilities."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class SpanScore:
    start: int
    end: int
    score: float


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def score_spans(
    attention: Sequence[float],
    contrast: Sequence[float],
    threshold: float = 0.15,
) -> List[SpanScore]:
    """Combine attention and contrastive scores into ranked spans."""
    a = np.asarray(attention, dtype=float)
    c = np.asarray(contrast, dtype=float)
    if a.shape != c.shape:
        raise ValueError("attention and contrast vectors must have the same length")
    combined = 0.6 * _softmax(a) + 0.4 * _softmax(c)
    spans: List[SpanScore] = []
    i = 0
    while i < combined.size:
        if combined[i] >= threshold:
            j = i
            while j + 1 < combined.size and combined[j + 1] >= threshold:
                j += 1
            spans.append(SpanScore(i, j + 1, float(combined[i : j + 1].mean())))
            i = j + 1
        else:
            i += 1
    return sorted(spans, key=lambda s: s.score, reverse=True)


def align_spans_to_tokens(offsets: Sequence[Tuple[int, int]], span: SpanScore):
    """Map token spans back to character offsets."""
    if not offsets:
        return []
    start_char = offsets[span.start][0]
    end_char = offsets[span.end - 1][1]
    return [start_char, end_char]
