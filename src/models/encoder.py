"""Transformer encoder implemented from scratch using PyTorch primitives."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.attention import MultiHeadAttention


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / model_dim))

        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        model_dim: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        max_positions: int = 512,
        dropout: float = 0.1,
        padding_idx: int = 0,
        pretrained_embeddings: torch.Tensor | None = None,
        embedding_wrapper: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.embedding = embedding_wrapper or nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if embedding_wrapper is not None:
            embed_dim = int(getattr(embedding_wrapper, "embedding_dim", embed_dim))
            padding_idx = int(getattr(embedding_wrapper, "padding_idx", padding_idx))
        elif pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)

        self.input_projection = nn.Linear(embed_dim, model_dim) if embed_dim != model_dim else nn.Identity()
        self.position_encoding = SinusoidalPositionalEncoding(model_dim=model_dim, max_len=max_positions)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.padding_idx = padding_idx

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(src)
        x = self.input_projection(x) * self.scale
        x = self.position_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return self.norm(x)
