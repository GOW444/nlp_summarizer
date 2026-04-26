"""Standalone inference entrypoint for trained simplification checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.config import load_config
from src.embeddings import build_embedding_wrapper
from src.factory import build_transformer_model
from src.preprocessing import normalize_text
from src.utils import get_device, load_checkpoint, load_model_state
from src.vocab import Vocabulary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for single-string inference."""

    parser = argparse.ArgumentParser(description="Run inference with the best saved simplifier checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Path to a saved checkpoint.")
    parser.add_argument("--text", required=True, help="Raw input string to simplify.")
    parser.add_argument("--config", default=None, help="Optional config override.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu or cuda.")
    return parser.parse_args()


def load_model(checkpoint_path: str, config_path: str | None, device: torch.device):
    """Load vocabulary, embedding strategy, model, and checkpoint weights."""

    config = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    if config_path is None and isinstance(checkpoint.get("config"), dict):
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
    return model, vocab, embedding_wrapper, config


@torch.no_grad()
def generate_with_confidence(
    model,
    src: torch.Tensor,
    vocab: Vocabulary,
    max_len: int,
) -> tuple[str, float]:
    """Greedily decode text and estimate confidence from selected token probabilities."""

    memory, src_mask = model.encode(src)
    sequence = [vocab.word2idx["<SOS>"]]
    selected_probs: list[float] = []

    for _ in range(max_len):
        tgt = torch.tensor(sequence, dtype=torch.long, device=src.device).unsqueeze(0)
        logits = model.decode(tgt=tgt, memory=memory, src_mask=src_mask)
        probabilities = torch.softmax(logits[:, -1, :], dim=-1).squeeze(0)
        token_id = int(torch.argmax(probabilities).item())
        selected_probs.append(float(probabilities[token_id].item()))
        sequence.append(token_id)
        if token_id == vocab.word2idx["<EOS>"]:
            break

    confidence = sum(selected_probs) / max(1, len(selected_probs))
    return vocab.decode(sequence), confidence


def predict(text: str, checkpoint_path: str, config_path: str | None = None, device_name: str | None = None) -> dict[str, Any]:
    """Predict a simplified output and confidence score for a raw string."""

    device = get_device(device_name)
    model, vocab, embedding_wrapper, config = load_model(checkpoint_path, config_path, device)
    normalized_text = normalize_text(text)
    encoded = embedding_wrapper.encode_source(
        normalized_text,
        max_len=config["data"]["max_src_len"],
        vocab=vocab,
    )
    src = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    prediction, confidence = generate_with_confidence(
        model=model,
        src=src,
        vocab=vocab,
        max_len=config["decoding"]["max_gen_len"],
    )
    return {
        "predicted_class": "simplified_text",
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "embedding_type": embedding_wrapper.embedding_type,
        "approach_name": embedding_wrapper.approach_name,
    }


def main() -> None:
    """Run CLI inference and print a JSON payload."""

    args = parse_args()
    result = predict(
        text=args.text,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
