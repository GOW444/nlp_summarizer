"""Evaluation utilities and CLI."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.dataset import DischargeSummaryDataset, collate_fn, load_serialized_split
from src.embeddings import build_embedding_wrapper
from src.factory import build_transformer_model
from src.preprocessing import split_sentences
from src.utils import get_device, load_checkpoint, load_model_state, save_json
from src.vocab import Vocabulary


def ngrams(tokens: list[str], n: int):
    return Counter(tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1))


def rouge_n(hypothesis: str, reference: str, n: int = 1) -> dict[str, float]:
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    hyp_ngrams = ngrams(hyp_tokens, n)
    ref_ngrams = ngrams(ref_tokens, n)
    overlap = sum((hyp_ngrams & ref_ngrams).values())
    precision = overlap / max(sum(hyp_ngrams.values()), 1)
    recall = overlap / max(sum(ref_ngrams.values()), 1)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_l(hypothesis: str, reference: str) -> dict[str, float]:
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    m, n = len(hyp), len(ref)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    precision = lcs / max(m, 1)
    recall = lcs / max(n, 1)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


def count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def flesch_kincaid_grade(text: str) -> float:
    sentences = split_sentences(text)
    words = text.split()
    if not sentences or not words:
        return 0.0

    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum(count_syllables(word) for word in words)
    asl = num_words / num_sentences
    asw = num_syllables / num_words
    return round(0.39 * asl + 11.8 * asw - 15.59, 2)


def flesch_reading_ease(text: str) -> float:
    sentences = split_sentences(text)
    words = text.split()
    if not sentences or not words:
        return 0.0

    num_syllables = sum(count_syllables(word) for word in words)
    asl = len(words) / len(sentences)
    asw = num_syllables / len(words)
    return round(206.835 - 1.015 * asl - 84.6 * asw, 2)


def evaluate_semantic(predictions: list[str], references: list[str]) -> dict[str, float]:
    try:
        from bert_score import score as bert_score
    except ImportError:
        return {
            "bertscore_precision": float("nan"),
            "bertscore_recall": float("nan"),
            "bertscore_f1": float("nan"),
        }

    precision, recall, f1 = bert_score(
        predictions,
        references,
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False,
    )
    return {
        "bertscore_precision": precision.mean().item(),
        "bertscore_recall": recall.mean().item(),
        "bertscore_f1": f1.mean().item(),
    }


def evaluate_model(
    model,
    dataloader,
    vocab,
    device,
    config,
    max_batches: int | None = None,
    compute_semantic: bool = True,
) -> dict[str, float]:
    model.eval()
    predictions: list[str] = []
    references: list[str] = []
    fk_before: list[float] = []
    fk_after: list[float] = []

    decoding_cfg = config["decoding"]
    dataset = getattr(dataloader, "dataset", None)
    row_offset = 0

    with torch.no_grad():
        for batch_idx, (src, tgt, _, _) in enumerate(tqdm(dataloader, desc="evaluating", leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch_size = src.size(0)
            src = src.to(device)
            batch_predictions = model.generate(
                src=src,
                vocab=vocab,
                beam_size=decoding_cfg["beam_size"],
                max_len=decoding_cfg["max_gen_len"],
                length_penalty=decoding_cfg["length_penalty"],
                no_repeat_ngram_size=decoding_cfg["no_repeat_ngram_size"],
            )
            if isinstance(batch_predictions, str):
                batch_predictions = [batch_predictions]

            decoded_targets = vocab.batch_decode(tgt.tolist())
            if dataset is not None and hasattr(dataset, "source_text"):
                decoded_sources = [dataset.source_text(row_offset + idx) for idx in range(batch_size)]
            else:
                decoded_sources = vocab.batch_decode(src.cpu().tolist())

            predictions.extend(batch_predictions)
            references.extend(decoded_targets)
            fk_before.extend(flesch_kincaid_grade(text) for text in decoded_sources)
            fk_after.extend(flesch_kincaid_grade(text) for text in batch_predictions)
            row_offset += batch_size

    rouge_1 = np.mean([rouge_n(pred, ref, n=1)["f1"] for pred, ref in zip(predictions, references)]) if predictions else 0.0
    rouge_2 = np.mean([rouge_n(pred, ref, n=2)["f1"] for pred, ref in zip(predictions, references)]) if predictions else 0.0
    rouge_l_score = np.mean([rouge_l(pred, ref)["f1"] for pred, ref in zip(predictions, references)]) if predictions else 0.0
    semantic_scores = evaluate_semantic(predictions, references) if predictions and compute_semantic else {}

    return {
        "ROUGE-1": round(float(rouge_1), 4),
        "ROUGE-2": round(float(rouge_2), 4),
        "ROUGE-L": round(float(rouge_l_score), 4),
        "FK Grade (before)": round(float(np.mean(fk_before)), 2) if fk_before else 0.0,
        "FK Grade (after)": round(float(np.mean(fk_after)), 2) if fk_after else 0.0,
        "FK Reduction": round(float(np.mean(fk_before) - np.mean(fk_after)), 2) if fk_before else 0.0,
        **semantic_scores,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained simplification model.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--config", default=None, help="Optional JSON config override.")
    parser.add_argument("--split", default="data/processed/test.pt", help="Serialized split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu or cuda.")
    parser.add_argument("--output", default="results/evaluation_results.json", help="Where to save metrics.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap for faster evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    if args.config is None and isinstance(checkpoint.get("config"), dict):
        config = checkpoint["config"]

    vocab_path = Path(config["paths"]["processed_dir"]) / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocabulary file: {vocab_path}")

    vocab = Vocabulary.load(str(vocab_path))
    embedding_wrapper = build_embedding_wrapper(config, vocab=vocab)
    test_rows = load_serialized_split(args.split)
    dataset = DischargeSummaryDataset(
        test_rows,
        vocab=vocab,
        max_src=config["data"]["max_src_len"],
        max_tgt=config["data"]["max_tgt_len"],
        source_encoder=embedding_wrapper,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size or config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_transformer_model(vocab_size=len(vocab), config=config, embedding_wrapper=embedding_wrapper)
    load_model_state(model, checkpoint["model_state"])
    model.to(device)

    metrics = evaluate_model(model=model, dataloader=dataloader, vocab=vocab, device=device, config=config, max_batches=args.max_batches)
    save_json(metrics, args.output)
    print(metrics)


if __name__ == "__main__":
    main()
