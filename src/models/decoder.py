"""Transformer decoder implemented from scratch using PyTorch primitives."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.attention import MultiHeadAttention
from src.models.encoder import SinusoidalPositionalEncoding


class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),  # Changed from nn.ReLU()
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-Norm implementation
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory, mask=memory_mask))
        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x


class TransformerDecoder(nn.Module):
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
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings.clone())

        self.input_projection = nn.Linear(embed_dim, model_dim) if embed_dim != model_dim else nn.Identity()
        self.position_encoding = SinusoidalPositionalEncoding(model_dim=model_dim, max_len=max_positions)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
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

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(tgt)
        x = self.input_projection(x) * self.scale
        x = self.position_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.norm(x)
