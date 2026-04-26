"""Simple extractive summarizer based on TF-IDF sentence scoring."""

from __future__ import annotations

import math
import re
from collections import Counter

from src.preprocessing import split_sentences


class TFIDFExtractor:
    def __init__(self) -> None:
        self.idf: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        return re.sub(r"[^a-z\s]", " ", text.lower()).split()

    def fit(self, corpus: list[str]) -> None:
        num_docs = len(corpus)
        doc_freq = Counter()
        for doc in corpus:
            doc_freq.update(set(self._tokenize(doc)))
        self.idf = {word: math.log((num_docs + 1) / (freq + 1)) + 1 for word, freq in doc_freq.items()}

    def _tf(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = len(tokens) or 1
        return {word: count / total for word, count in counts.items()}

    def score_sentences(self, sentences: list[str]) -> list[float]:
        scores: list[float] = []
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            if not tokens:
                scores.append(0.0)
                continue
            tf = self._tf(tokens)
            score = sum(tf.get(word, 0.0) * self.idf.get(word, 0.0) for word in tokens)
            scores.append(score / len(tokens))
        return scores

    def summarize(self, text: str, ratio: float = 0.4) -> str:
        sentences = split_sentences(text)
        if len(sentences) <= 2:
            return text
        scores = self.score_sentences(sentences)
        num_keep = max(1, int(len(sentences) * ratio))
        top_indices = sorted(sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:num_keep])
        return " ".join(sentences[idx] for idx in top_indices)
