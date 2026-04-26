"""Dataset and collate utilities."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.preprocessing import normalize_text, prepare_source_text


class DischargeSummaryDataset(Dataset):
    def __init__(
        self,
        data: list[dict[str, Any]],
        vocab,
        max_src: int = 300,
        max_tgt: int = 128,
        normalize: bool = True,
        source_encoder=None,
    ) -> None:
        self.data = data
        self.vocab = vocab
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.normalize = normalize
        self.source_encoder = source_encoder

    def __len__(self) -> int:
        return len(self.data)

    def source_text(self, idx: int) -> str:
        """Return the normalized source text for evaluation/readability metrics."""

        row = self.data[idx]
        src_text = prepare_source_text(row["note"], row.get("question"), row["task"])
        return normalize_text(src_text) if self.normalize else src_text

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        row = self.data[idx]
        answer = row["answer"]

        src_text = self.source_text(idx)
        if self.normalize:
            answer = normalize_text(answer)

        if self.source_encoder is not None:
            src = self.source_encoder.encode_source(src_text, max_len=self.max_src, vocab=self.vocab)
        else:
            src = self.vocab.encode(src_text, max_len=self.max_src)
            if not src:
                src = [self.vocab.word2idx["<UNK>"]]
        tgt = self.vocab.encode(answer, max_len=self.max_tgt - 2, add_special_tokens=True)

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
            len(src),
            len(tgt),
        )


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, int, int]]):
    srcs, tgts, src_lengths, tgt_lengths = zip(*batch)
    max_src = max(tensor.size(0) for tensor in srcs)
    max_tgt = max(tensor.size(0) for tensor in tgts)

    src_batch = torch.stack([F.pad(tensor, (0, max_src - tensor.size(0))) for tensor in srcs])
    tgt_batch = torch.stack([F.pad(tensor, (0, max_tgt - tensor.size(0))) for tensor in tgts])

    return (
        src_batch,
        tgt_batch,
        torch.tensor(src_lengths, dtype=torch.long),
        torch.tensor(tgt_lengths, dtype=torch.long),
    )


def load_serialized_split(path: str) -> list[dict[str, Any]]:
    data = torch.load(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of rows in {path}, found {type(data)!r}")
    return data
