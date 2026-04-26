"""From-scratch attention layers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.size()
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = tensor.size()
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, seq_len, self.model_dim)

    def _prepare_mask(self, mask: torch.Tensor | None, scores: torch.Tensor) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.dim() == 2:
            mask = mask[:, None, None, :]
        elif mask.dim() == 3:
            mask = mask[:, None, :, :]
        return mask.to(device=scores.device, dtype=torch.bool)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ):
        query_states = self._split_heads(self.q_proj(query))
        key_states = self._split_heads(self.k_proj(key))
        value_states = self._split_heads(self.v_proj(value))

        scores = torch.matmul(query_states, key_states.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        attn_mask = self._prepare_mask(mask, scores)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, value_states)
        output = self.out_proj(self._merge_heads(context))

        if need_weights:
            return output, attn_weights
        return output
