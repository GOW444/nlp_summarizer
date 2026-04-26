"""CLI for generation and note simplification."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import load_config
from src.embeddings import build_embedding_wrapper
from src.factory import build_transformer_model
from src.models.tfidf_extractor import TFIDFExtractor
from src.pipeline import SimplificationPipeline
from src.preprocessing import normalize_text
from src.utils import get_device, load_checkpoint, load_model_state
from src.vocab import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simplified discharge text.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained model checkpoint.")
    parser.add_argument("--text", required=True, help="Input note or sentence.")
    parser.add_argument("--config", default=None, help="Optional JSON config override.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu or cuda.")
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

    model = build_transformer_model(vocab_size=len(vocab), config=config, embedding_wrapper=embedding_wrapper)
    load_model_state(model, checkpoint["model_state"])
    model.to(device)
    model.eval()

    extractor = TFIDFExtractor()
    extractor.fit([normalize_text(args.text)])
    pipeline = SimplificationPipeline(
            model=model,
            vocab=vocab,
            config=config,
            device=device,
            extractor=extractor,
            complexity_classifier=None,
            source_encoder=embedding_wrapper,
            )
    prediction = pipeline.simplify(args.text)
    print(prediction)


if __name__ == "__main__":
    main()
