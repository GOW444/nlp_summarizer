"""Custom vocabulary and tokenizer."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from src.preprocessing import normalize_text


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]


class Vocabulary:
    def __init__(self, min_freq: int = 2) -> None:
        self.min_freq = min_freq
        self.word2idx = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self.idx2word = {idx: token for token, idx in self.word2idx.items()}
        self.freq = Counter()

    def tokenize(self, text: str) -> list[str]:
        text = normalize_text(text)
        text = re.sub(r"(\d+\.?\d*)", r" \1 ", text)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)
        return [token for token in text.split() if token]

    def build(self, texts: Iterable[str]) -> None:
        for text in texts:
            self.freq.update(self.tokenize(text))

        for word, count in self.freq.items():
            if count >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(
        self,
        text: str,
        max_len: int = 512,
        add_special_tokens: bool = False,
    ) -> list[int]:
        tokens = self.tokenize(text)[:max_len]
        indices = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
        if add_special_tokens:
            indices = [self.word2idx["<SOS>"]] + indices[: max(0, max_len - 2)] + [self.word2idx["<EOS>"]]
        return indices

    def decode(self, indices: Sequence[int], skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        special_ids = {self.word2idx[token] for token in SPECIAL_TOKENS}

        for idx in indices:
            idx = int(idx)
            if skip_special_tokens and idx in special_ids:
                continue
            tokens.append(self.idx2word.get(idx, "<UNK>"))
        return " ".join(tokens).strip()

    def batch_decode(self, batch_indices: Sequence[Sequence[int]]) -> list[str]:
        return [self.decode(indices) for indices in batch_indices]

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "min_freq": self.min_freq,
                    "word2idx": self.word2idx,
                    "freq": dict(self.freq),
                },
                handle,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        vocab = cls(min_freq=data["min_freq"])
        vocab.word2idx = {token: int(idx) for token, idx in data["word2idx"].items()}
        vocab.idx2word = {idx: token for token, idx in vocab.word2idx.items()}
        vocab.freq = Counter(data.get("freq", {}))
        return vocab

    def __len__(self) -> int:
        return len(self.word2idx)
    @property
    def pad_idx(self): return self.word2idx["<PAD>"]

    @property
    def bos_idx(self): return self.word2idx["<SOS>"]

    @property
    def eos_idx(self): return self.word2idx["<EOS>"]

    @property
    def unk_idx(self): return self.word2idx["<UNK>"]
