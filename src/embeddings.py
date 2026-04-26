"""Embedding strategy factory for static and contextual source encoders."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


STATIC_EMBEDDING_TYPES = {"random", "glove", "gensim", "word2vec", "fasttext"}


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def _random_matrix(vocab, embed_dim: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.1, (len(vocab), embed_dim)).astype(np.float32)
    matrix[0] = 0.0
    return torch.tensor(matrix, dtype=torch.float32)


def load_glove(glove_path: str, vocab, embed_dim: int = 100, seed: int = 42) -> torch.Tensor:
    """Load GloVe vectors and align them to the project's custom vocabulary."""

    path = Path(glove_path)
    if not path.exists():
        raise FileNotFoundError(f"GloVe file not found: {glove_path}")

    glove: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split()
            if len(parts) != embed_dim + 1:
                continue
            glove[parts[0]] = np.asarray(parts[1:], dtype=np.float32)

    matrix = _random_matrix(vocab=vocab, embed_dim=embed_dim, seed=seed).numpy()
    hits = 0
    for word, idx in vocab.word2idx.items():
        vector = glove.get(word)
        if vector is not None:
            matrix[idx] = vector
            hits += 1

    print(f"GloVe coverage: {hits}/{len(vocab)} ({100 * hits / max(1, len(vocab)):.1f}%)")
    return torch.tensor(matrix, dtype=torch.float32)


def load_gensim_vectors(
    vector_path: str,
    vocab,
    seed: int = 42,
    binary: bool = False,
    vector_format: str = "word2vec",
) -> torch.Tensor:
    """Load Gensim Word2Vec/FastText vectors and align them to the custom vocabulary.

    ``vector_format`` supports ``word2vec`` text/bin files via
    ``KeyedVectors.load_word2vec_format`` and native ``keyed_vectors`` files via
    ``KeyedVectors.load``.
    """

    path = Path(vector_path)
    if not path.exists():
        raise FileNotFoundError(f"Gensim vector file not found: {vector_path}")

    try:
        from gensim.models import KeyedVectors
    except ImportError as exc:
        raise ImportError("Install gensim to use Gensim-based embeddings.") from exc

    if vector_format == "keyed_vectors":
        vectors = KeyedVectors.load(str(path), mmap="r")
    else:
        vectors = KeyedVectors.load_word2vec_format(str(path), binary=binary)

    keyed_vectors = getattr(vectors, "wv", vectors)
    embed_dim = int(keyed_vectors.vector_size)
    matrix = _random_matrix(vocab=vocab, embed_dim=embed_dim, seed=seed).numpy()

    hits = 0
    for word, idx in vocab.word2idx.items():
        if word in keyed_vectors:
            matrix[idx] = keyed_vectors.get_vector(word)
            hits += 1

    print(f"Gensim coverage: {hits}/{len(vocab)} ({100 * hits / max(1, len(vocab)):.1f}%)")
    return torch.tensor(matrix, dtype=torch.float32)


class EmbeddingWrapper(nn.Module):
    """Strategy-aware source embedding module plus source text vectorizer.

    Static strategies consume the project's custom vocabulary IDs. BERT consumes
    Hugging Face tokenizer IDs, then returns contextual token features.
    """

    def __init__(
        self,
        embedding_type: str,
        embedding_dim: int,
        padding_idx: int = 0,
        embedding: nn.Module | None = None,
        tokenizer: Any | None = None,
        bert_model: nn.Module | None = None,
        fine_tune: bool = False,
        model_name: str | None = None,
        approach_name: str | None = None,
        static_matrix: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.embedding_type = embedding_type.lower()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embedding = embedding
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.fine_tune = fine_tune
        self.model_name = model_name
        self.approach_name = approach_name or self._default_approach_name()
        self._static_matrix = static_matrix

        if self.bert_model is not None:
            for parameter in self.bert_model.parameters():
                parameter.requires_grad = fine_tune

    @property
    def is_contextual(self) -> bool:
        return self.embedding_type == "bert"

    @property
    def is_static(self) -> bool:
        return self.embedding_type in STATIC_EMBEDDING_TYPES

    def _default_approach_name(self) -> str:
        suffix = self.model_name or f"{self.embedding_dim}d"
        if self.embedding_type == "bert":
            suffix = f"{suffix}-finetune-{self.fine_tune}"
        return _safe_name(f"{self.embedding_type}_{suffix}")

    def static_matrix(self) -> torch.Tensor | None:
        """Return a detached copy of the static matrix for target embeddings."""

        if self._static_matrix is None:
            return None
        return self._static_matrix.detach().clone()

    def encode_source(self, text: str, max_len: int, vocab) -> list[int]:
        """Vectorize raw source text for the selected embedding strategy."""

        if self.is_contextual:
            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=max_len,
                truncation=True,
                padding=False,
            )
            return list(encoded["input_ids"])

        indices = vocab.encode(text, max_len=max_len)
        if not indices:
            indices = [vocab.word2idx["<UNK>"]]
        return indices

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return token features shaped ``[batch, seq_len, embedding_dim]``."""

        if self.is_contextual:
            attention_mask = (input_ids != self.padding_idx).long()
            if self.fine_tune:
                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.last_hidden_state

            with torch.no_grad():
                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state.detach()

        return self.embedding(input_ids)


def build_embedding_wrapper(config: dict, vocab) -> EmbeddingWrapper:
    """Build the configured embedding strategy.

    Supported ``config["embeddings"]["type"]`` values are ``random``, ``glove``,
    ``gensim``/``word2vec``/``fasttext``, and ``bert``.
    """

    embedding_cfg = config.get("embeddings", {})
    model_cfg = config.get("model", {})
    path_cfg = config.get("paths", {})
    seed = config.get("training", {}).get("seed", 42)

    embedding_type = embedding_cfg.get("type", "glove").lower()
    freeze_static = bool(embedding_cfg.get("freeze_static", False))
    fallback_to_random = bool(embedding_cfg.get("fallback_to_random", True))
    approach_override = embedding_cfg.get("approach_name")

    if embedding_type == "bert":
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers to use BERT embeddings.") from exc

        model_name = embedding_cfg.get("bert_model_name", "bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)
        padding_idx = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        hidden_size = int(bert_model.config.hidden_size)
        return EmbeddingWrapper(
            embedding_type="bert",
            embedding_dim=hidden_size,
            padding_idx=padding_idx,
            tokenizer=tokenizer,
            bert_model=bert_model,
            fine_tune=bool(embedding_cfg.get("bert_fine_tune", False)),
            model_name=model_name,
            approach_name=approach_override,
        )

    if embedding_type == "glove":
        glove_path = embedding_cfg.get("glove_path") or path_cfg.get("glove_path", "data/glove.6B.100d.txt")
        embed_dim = int(embedding_cfg.get("embed_dim", model_cfg.get("embed_dim", 100)))
        try:
            matrix = load_glove(glove_path, vocab=vocab, embed_dim=embed_dim, seed=seed)
        except FileNotFoundError:
            if not fallback_to_random:
                raise
            print(f"Skipping GloVe initialization because {glove_path} does not exist.")
            embedding_type = "random"
            matrix = _random_matrix(vocab=vocab, embed_dim=embed_dim, seed=seed)

    elif embedding_type in {"gensim", "word2vec", "fasttext"}:
        vector_path = embedding_cfg.get("gensim_path") or embedding_cfg.get("vector_path")
        if not vector_path:
            raise ValueError("Set embeddings.gensim_path or embeddings.vector_path for Gensim embeddings.")
        try:
            matrix = load_gensim_vectors(
                vector_path=vector_path,
                vocab=vocab,
                seed=seed,
                binary=bool(embedding_cfg.get("gensim_binary", False)),
                vector_format=embedding_cfg.get("gensim_format", "word2vec"),
            )
        except FileNotFoundError:
            if not fallback_to_random:
                raise
            embed_dim = int(embedding_cfg.get("embed_dim", model_cfg.get("embed_dim", 100)))
            print(f"Skipping Gensim initialization because {vector_path} does not exist.")
            embedding_type = "random"
            matrix = _random_matrix(vocab=vocab, embed_dim=embed_dim, seed=seed)

    elif embedding_type == "random":
        embed_dim = int(embedding_cfg.get("embed_dim", model_cfg.get("embed_dim", 100)))
        matrix = _random_matrix(vocab=vocab, embed_dim=embed_dim, seed=seed)

    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

    embedding = nn.Embedding.from_pretrained(matrix, freeze=freeze_static, padding_idx=0)
    return EmbeddingWrapper(
        embedding_type=embedding_type,
        embedding_dim=int(matrix.size(1)),
        padding_idx=0,
        embedding=embedding,
        approach_name=approach_override or _safe_name(f"{embedding_type}_{int(matrix.size(1))}d"),
        static_matrix=matrix,
    )


def maybe_load_glove(glove_path: str, vocab, embed_dim: int = 100, seed: int = 42) -> torch.Tensor | None:
    """Backward-compatible helper kept for older scripts."""

    try:
        return load_glove(glove_path, vocab=vocab, embed_dim=embed_dim, seed=seed)
    except FileNotFoundError:
        print(f"Skipping GloVe initialization because {glove_path} does not exist.")
        return None
