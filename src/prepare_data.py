"""Prepare discharge simplifier data artifacts from multiple sources.

Supported datasets:
    1. medalpaca/medical_meadow_medical_flashcards  (original)
    2. ccdv/pubmed-summarization                   (biomedical abstractive summaries)
    3. abisee/cnn_dailymail                        (general summarization, 3.0.0)
    4. custom CSV/JSONL                            (--custom-path, see --custom-src-col / --custom-tgt-col)

Usage examples:
    # Original only (backward compatible)
    python -m src.prepare_data

    # All built-in datasets merged
    python -m src.prepare_data --datasets medalpaca pubmed cnndm

    # Add your own CSV/JSONL on top
    python -m src.prepare_data --datasets medalpaca pubmed --custom-path my_data.jsonl

    # Cap rows per dataset for quick experiments
    python -m src.prepare_data --datasets medalpaca pubmed --max-examples 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Callable

import torch
from datasets import load_dataset

from src.config import load_config
from src.preprocessing import prepare_source_text
from src.vocab import Vocabulary


# ---------------------------------------------------------------------------
# Row schema
# ---------------------------------------------------------------------------
# Every internal row is:
#   {"note": <source text>, "question": "", "answer": <target text>, "task": "simplification"}


def _make_row(note: str, answer: str) -> dict:
    note = note.strip()
    answer = answer.strip()
    if not note or not answer:
        return {}
    return {"note": note, "question": "", "answer": answer, "task": "simplification"}


# ---------------------------------------------------------------------------
# Dataset loaders — each returns list[dict] in internal row format
# ---------------------------------------------------------------------------

def _load_medalpaca(max_examples: int | None = None) -> list[dict]:
    """medalpaca/medical_meadow_medical_flashcards — Q&A + paraphrase pairs."""
    print("  Loading medalpaca/medical_meadow_medical_flashcards ...")
    ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    rows = []
    for item in ds:
        row = _make_row(
            note=item.get("input", ""),
            answer=item.get("output", ""),
        )
        if row:
            rows.append(row)
    print(f"    → {len(rows)} rows")
    return rows


def _load_pubmed(max_examples: int | None = None) -> list[dict]:
    """ccdv/pubmed-summarization — article→abstract pairs (biomedical).

    Fields: 'article' (full paper body), 'abstract' (summary target).
    We use the first 1024 chars of the article as source to stay within
    max_src_len and keep the abstract as the simplification target.
    """
    print("  Loading ccdv/pubmed-summarization ...")
    ds = load_dataset("ccdv/pubmed-summarization", split="train", trust_remote_code=True)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    rows = []
    for item in ds:
        # Truncate article body to ~first 3 sentences worth (~600 chars) to
        # keep source lengths reasonable for a small model.
        article = item.get("article", "").strip()[:600]
        abstract = item.get("abstract", "").strip()
        row = _make_row(note=article, answer=abstract)
        if row:
            rows.append(row)
    print(f"    → {len(rows)} rows")
    return rows


def _load_cnndm(max_examples: int | None = None) -> list[dict]:
    """abisee/cnn_dailymail — news article→highlight pairs (general summarization).

    Fields: 'article', 'highlights'.
    CNN/DM highlights are naturally shorter and somewhat simpler than the
    full article, making them decent FK-reduction training signal.
    """
    print("  Loading abisee/cnn_dailymail (3.0.0) ...")
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", trust_remote_code=True)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    rows = []
    for item in ds:
        article = item.get("article", "").strip()[:600]
        highlights = item.get("highlights", "").strip()
        row = _make_row(note=article, answer=highlights)
        if row:
            rows.append(row)
    print(f"    → {len(rows)} rows")
    return rows


def _load_custom(path: str, src_col: str, tgt_col: str, max_examples: int | None = None) -> list[dict]:
    """Load a custom CSV or JSONL file.

    For CSV: expects a header row with at least `src_col` and `tgt_col`.
    For JSONL: expects one JSON object per line with those keys.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Custom dataset not found: {path}")
    print(f"  Loading custom dataset: {path} ...")
    rows = []
    if p.suffix.lower() == ".csv":
        with p.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, item in enumerate(reader):
                if max_examples and i >= max_examples:
                    break
                row = _make_row(note=item.get(src_col, ""), answer=item.get(tgt_col, ""))
                if row:
                    rows.append(row)
    elif p.suffix.lower() in (".jsonl", ".json"):
        with p.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                item = json.loads(line.strip())
                row = _make_row(note=item.get(src_col, ""), answer=item.get(tgt_col, ""))
                if row:
                    rows.append(row)
    else:
        raise ValueError(f"Unsupported custom file type: {p.suffix}. Use .csv or .jsonl")
    print(f"    → {len(rows)} rows")
    return rows


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_LOADERS: dict[str, Callable] = {
    "medalpaca": _load_medalpaca,
    "pubmed": _load_pubmed,
    "cnndm": _load_cnndm,
}


# ---------------------------------------------------------------------------
# Vocab helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _collect_vocab_texts(rows: list[dict]) -> list[str]:
    texts: list[str] = []
    for row in rows:
        src_text = prepare_source_text(row["note"], row.get("question"), row["task"])
        texts.extend([src_text, row["answer"]])
    return texts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare discharge simplifier data artifacts.")
    parser.add_argument("--config", default=None, help="Path to JSON config file.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_LOADERS.keys()),
        default=["medalpaca"],
        help="Which built-in datasets to load and merge. Default: medalpaca only.",
    )
    parser.add_argument("--output-dir", default=None, help="Override the processed output directory.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Row cap PER DATASET for quick experiments.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splits.")
    # Custom dataset args
    parser.add_argument("--custom-path", default=None, help="Path to a custom CSV or JSONL file.")
    parser.add_argument("--custom-src-col", default="source", help="Column name for source text in custom file.")
    parser.add_argument("--custom-tgt-col", default="target", help="Column name for target text in custom file.")
    # Legacy arg kept for backward compatibility
    parser.add_argument("--dataset-name", default=None, help="[Legacy] HuggingFace dataset identifier (ignored if --datasets is set).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def _split_rows(rows: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    """80/10/10 train-val-test split."""
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir or config["paths"]["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    print(f"\nLoading datasets: {args.datasets}")
    for name in args.datasets:
        loader = DATASET_LOADERS[name]
        rows = loader(max_examples=args.max_examples)
        all_rows.extend(rows)

    if args.custom_path:
        custom_rows = _load_custom(
            path=args.custom_path,
            src_col=args.custom_src_col,
            tgt_col=args.custom_tgt_col,
            max_examples=args.max_examples,
        )
        all_rows.extend(custom_rows)

    print(f"\nTotal rows before split: {len(all_rows)}")

    train_rows, val_rows, test_rows = _split_rows(all_rows, seed=args.seed)
    print(f"Split sizes — train: {len(train_rows)}, val: {len(val_rows)}, test: {len(test_rows)}")

    # Sanity check
    print("\nSample pair (train[0]):")
    print(f"  SRC : {train_rows[0]['note'][:120]}")
    print(f"  TGT : {train_rows[0]['answer'][:120]}")

    # Build vocabulary from training data only
    vocab = Vocabulary(min_freq=config["data"]["min_word_freq"])
    vocab.build(_collect_vocab_texts(train_rows))
    vocab.save(str(output_dir / "vocab.json"))
    print(f"\nVocabulary size: {len(vocab)}")

    # Persist splits
    torch.save(train_rows, output_dir / "train.pt")
    torch.save(val_rows,   output_dir / "val.pt")
    torch.save(test_rows,  output_dir / "test.pt")
    print(f"All artifacts saved to {output_dir}/")
    print("\nNext steps:")
    print("  1. python audit_fk.py --filter        # remove pairs where target is harder than source")
    print("  2. python -m src.train                 # train on filtered splits")


if __name__ == "__main__":
    main()
