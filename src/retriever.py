"""Tiny retriever used for the reproducibility experiments."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Document:
    doc_id: str
    text: str


class Retriever:
    def __init__(self, documents: List[Document], dim: int = 128):
        self.documents = documents
        rng = np.random.default_rng(7)
        self.matrix = rng.normal(size=(len(documents), dim))

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        scores = self.matrix @ query
        order = np.argsort(-scores)[:top_k]
        return [(self.documents[i].doc_id, float(scores[i])) for i in order]
