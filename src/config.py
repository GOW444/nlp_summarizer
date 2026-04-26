"""Project configuration helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "paths": {
        "processed_dir": "data/processed",
        "glove_path": "",
        "checkpoint_dir": "checkpoints",
        "results_dir": "results",
    },
    "data": {
        "relevant_tasks": ["Paraphrasing", "Summarization"],
        "min_word_freq": 2,
        "max_src_len": 300,
        "max_tgt_len": 128,
        "summarization_ratio": 0.4,
    },
    "model": {
        "embed_dim": 100,
        "model_dim": 256,
        "ff_dim": 1024,
        "num_heads": 8,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "dropout": 0.1,
        "max_positions": 512,
        "classifier_hidden_dim": 128,
    },
    "embeddings": {
        "type": "glove",
        "embed_dim": 100,
        "glove_path": "data/glove.6B.100d.txt",
        "gensim_path": "",
        "gensim_binary": False,
        "gensim_format": "word2vec",
        "bert_model_name": "bert-base-uncased",
        "bert_fine_tune": False,
        "freeze_static": False,
        "apply_to_decoder": True,
        "fallback_to_random": True,
        "approach_name": "",
    },
    "training": {
        "batch_size": 32,
        "epochs": 20,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "clip": 1.0,
        "label_smoothing": 0.1,
        "warmup_pct": 0.1,
        "num_workers": 0,
        "seed": 42,
        "selection_metric": "rouge_l",
        "fk_penalty_weight": 0.3,  # Added: FK penalty weight for composite score
        "max_val_generation_batches": 10,
    },
    "decoding": {
        "beam_size": 6,  # Increased from 4
        "max_gen_len": 128,
        "length_penalty": 1.2,  # Increased from 0.7 to penalize short outputs
        "no_repeat_ngram_size": 3,
        "min_gen_len": 10,  # Added: prevents degenerate 1-2 word outputs
    },
    "pipeline": {
        "complexity_threshold": 0.5,
        "long_note_sentence_threshold": 6,
    },
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | None = None) -> Dict[str, Dict[str, Any]]:
    if not config_path:
        return copy.deepcopy(DEFAULT_CONFIG)

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        user_config = json.load(handle)
    return _deep_update(DEFAULT_CONFIG, user_config)


def save_config(config: Dict[str, Dict[str, Any]], config_path: str) -> None:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
