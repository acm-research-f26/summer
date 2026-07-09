"""
TF-IDF + cosine similarity scorer for task–element relevance.

This is the core ML layer for Implementation 2. Given a natural-language task
and a list of interactive DOM elements (as text blobs), it ranks elements by
semantic overlap with the task. No GPU or external API required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ScoredElement:
    index: int
    text: str
    tag: str
    score: float


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _element_blob(tag: str, text: str, attrs: dict[str, str] | None = None) -> str:
    """Build a feature string from element metadata."""
    parts = [tag, text]
    if attrs:
        for key in ("aria-label", "placeholder", "title", "role", "name"):
            if key in attrs and attrs[key]:
                parts.append(attrs[key])
    return _normalize_text(" ".join(parts))


class ActionRelevanceScorer:
    """
    Lightweight relevance model: TF-IDF vectors for task + elements, cosine rank.

    Parameters
    ----------
    top_k : int
        Number of top elements to return from rank().
    min_score : float
        Elements below this cosine score are filtered out.
    ngram_range : tuple
        Passed to TfidfVectorizer; (1,2) captures bigrams like "apply now".
    """

    def __init__(
        self,
        top_k: int = 5,
        min_score: float = 0.05,
        ngram_range: tuple[int, int] = (1, 2),
    ):
        self.top_k = top_k
        self.min_score = min_score
        self._vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
            max_features=8000,
        )

    def rank(
        self,
        task: str,
        elements: list[dict],
    ) -> list[ScoredElement]:
        """
        Rank interactive elements by relevance to task.

        Each element dict should have: index, text, tag, and optional attrs.
        """
        if not elements:
            return []

        task_norm = _normalize_text(task)
        blobs = [_element_blob(el.get("tag", "unknown"), el.get("text", ""), el.get("attrs")) for el in elements]

        # Fit on task + all elements so IDF is page-local (adapts to each page)
        corpus = [task_norm] + blobs
        matrix = self._vectorizer.fit_transform(corpus)
        task_vec = matrix[0:1]
        elem_matrix = matrix[1:]

        scores = cosine_similarity(task_vec, elem_matrix).flatten()

        ranked: list[ScoredElement] = []
        for el, score in zip(elements, scores):
            if score < self.min_score:
                continue
            ranked.append(
                ScoredElement(
                    index=el["index"],
                    text=el.get("text", ""),
                    tag=el.get("tag", "unknown"),
                    score=float(score),
                )
            )

        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked[: self.top_k]

    def is_in_top_k(self, element_index: int, ranked: list[ScoredElement]) -> bool:
        return any(r.index == element_index for r in ranked)

    def score_for_index(self, element_index: int, ranked: list[ScoredElement]) -> float | None:
        for r in ranked:
            if r.index == element_index:
                return r.score
        return None
