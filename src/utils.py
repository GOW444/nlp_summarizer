"""Shared utility helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(data, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_checkpoint(state: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, target)


def load_checkpoint(path: str | Path, map_location: str | torch.device | None = None) -> dict:
    return torch.load(path, map_location=map_location)


def load_model_state(model: torch.nn.Module, state_dict: dict) -> None:
    """Load model weights while accepting pre-EmbeddingWrapper checkpoints."""

    expected_keys = model.state_dict().keys()
    remapped = dict(state_dict)
    old_encoder_key = "encoder.embedding.weight"
    new_encoder_key = "encoder.embedding.embedding.weight"
    if old_encoder_key in remapped and new_encoder_key in expected_keys:
        remapped[new_encoder_key] = remapped.pop(old_encoder_key)
    model.load_state_dict(remapped)
