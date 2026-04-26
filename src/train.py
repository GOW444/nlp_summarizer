"""Training loop for the from-scratch Transformer simplifier."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.dataset import DischargeSummaryDataset, collate_fn, load_serialized_split
from src.embeddings import build_embedding_wrapper
from src.evaluate import evaluate_model
from src.factory import build_transformer_model
from src.utils import AverageMeter, count_parameters, ensure_dir, get_device, save_checkpoint, set_seed
from src.vocab import Vocabulary


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            smooth = torch.full_like(pred, self.smoothing / max(1, self.vocab_size - 2))
            smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            smooth[:, self.pad_idx] = 0
            smooth[target == self.pad_idx] = 0
        return torch.nn.functional.kl_div(torch.log_softmax(pred, dim=-1), smooth, reduction="batchmean")


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    clip: float,
    device: torch.device,
    scheduler=None,
    epoch: int | None = None,
) -> float:
    model.train()
    loss_meter = AverageMeter()

    desc = f"train e{epoch}" if epoch is not None else "train"
    for src, tgt, _, _ in tqdm(dataloader, desc=desc, leave=False):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), n=src.size(0))

    return loss_meter.avg


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device: torch.device, epoch: int | None = None) -> float:
    model.eval()
    loss_meter = AverageMeter()

    desc = f"val-loss e{epoch}" if epoch is not None else "val-loss"
    for src, tgt, _, _ in tqdm(dataloader, desc=desc, leave=False):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        loss_meter.update(loss.item(), n=src.size(0))

    return loss_meter.avg


def _build_loader(rows, vocab, config, shuffle: bool, source_encoder=None) -> DataLoader:
    dataset = DischargeSummaryDataset(
        rows,
        vocab=vocab,
        max_src=config["data"]["max_src_len"],
        max_tgt=config["data"]["max_tgt_len"],
        source_encoder=source_encoder,
    )
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
    )


def _require_training_files(processed_dir: Path) -> None:
    required = [processed_dir / "train.pt", processed_dir / "val.pt", processed_dir / "vocab.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing training artifacts. Run `python -m src.prepare_data` once the dataset is available.\n"
            + "\n".join(missing)
        )


def _safe_metric_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value).strip("-").lower()


class MetricsCSVLogger:
    """Append per-epoch metrics in a comparison-friendly CSV file."""

    FIELDNAMES = [
        "epoch",
        "embedding_type",
        "approach_name",
        "lr",
        "train_loss",
        "val_loss",
        "score",
        "epoch_time_sec",
        "trainable_params",
        "is_best",
        "ROUGE-1",
        "ROUGE-2",
        "ROUGE-L",
        "FK Grade (before)",
        "FK Grade (after)",
        "FK Reduction",
    ]

    def __init__(self, results_dir: Path, approach_name: str, header: dict[str, object]) -> None:
        self.path = results_dir / f"metrics_{_safe_metric_name(approach_name)}.csv"
        self.header = header
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8", newline="") as handle:
                header_text = " ".join(f"{key}={value}" for key, value in header.items())
                handle.write(f"# {header_text}\n")
                writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES, extrasaction="ignore")
                writer.writeheader()

    def append(self, metrics: dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES, extrasaction="ignore")
            writer.writerow(metrics)


def _format_epoch_table(metrics: dict[str, object]) -> str:
    display_keys = [
        "epoch",
        "train_loss",
        "val_loss",
        "score",
        "ROUGE-L",
        "FK Reduction",
        "epoch_time_sec",
        "is_best",
    ]
    rows = []
    for key in display_keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            value = f"{value:.4f}"
        rows.append((key, value))

    try:
        from tabulate import tabulate

        return tabulate(rows, headers=["metric", "value"], tablefmt="github")
    except ImportError:
        width = max(len(key) for key, _ in rows) if rows else 0
        return "\n".join(f"{key:<{width}} : {value}" for key, value in rows)


def _apply_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    if args.embedding_type:
        config.setdefault("embeddings", {})["type"] = args.embedding_type
    if args.approach_name:
        config.setdefault("embeddings", {})["approach_name"] = args.approach_name
    if args.bert_model_name:
        config.setdefault("embeddings", {})["bert_model_name"] = args.bert_model_name
    if args.gensim_path:
        config.setdefault("embeddings", {})["gensim_path"] = args.gensim_path
    if args.glove_path:
        config.setdefault("embeddings", {})["glove_path"] = args.glove_path
        config.setdefault("paths", {})["glove_path"] = args.glove_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the discharge summary simplifier.")
    parser.add_argument("--config", default=None, help="Optional JSON config override.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu or cuda.")
    parser.add_argument("--embedding-type", choices=["random", "glove", "gensim", "word2vec", "fasttext", "bert"], default=None)
    parser.add_argument("--approach-name", default=None, help="Override the metric/checkpoint approach label.")
    parser.add_argument("--bert-model-name", default=None, help="Hugging Face model name for BERT embeddings.")
    parser.add_argument("--gensim-path", default=None, help="Path to Gensim/Word2Vec/FastText vectors.")
    parser.add_argument("--glove-path", default=None, help="Path to a GloVe text file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _apply_cli_overrides(config, args)
    processed_dir = Path(config["paths"]["processed_dir"])
    checkpoint_dir = ensure_dir(config["paths"]["checkpoint_dir"])
    results_dir = ensure_dir(config["paths"]["results_dir"])
    _require_training_files(processed_dir)

    set_seed(config["training"]["seed"])
    device = get_device(args.device)

    vocab = Vocabulary.load(str(processed_dir / "vocab.json"))
    train_rows = load_serialized_split(str(processed_dir / "train.pt"))
    val_rows = load_serialized_split(str(processed_dir / "val.pt"))

    embedding_wrapper = build_embedding_wrapper(config, vocab=vocab)
    approach_name = embedding_wrapper.approach_name

    train_loader = _build_loader(train_rows, vocab, config, shuffle=True, source_encoder=embedding_wrapper)
    val_loader = _build_loader(val_rows, vocab, config, shuffle=False, source_encoder=embedding_wrapper)

    model = build_transformer_model(
        vocab_size=len(vocab),
        config=config,
        embedding_wrapper=embedding_wrapper,
    ).to(device)
    trainable_params = count_parameters(model)
    print(f"Embedding approach: {approach_name} ({embedding_wrapper.embedding_dim}d source features)")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["training"]["lr"],
        steps_per_epoch=max(1, len(train_loader)),
        epochs=config["training"]["epochs"],
        pct_start=config["training"]["warmup_pct"],
    )
    criterion = LabelSmoothingLoss(
        vocab_size=len(vocab),
        pad_idx=0,
        smoothing=config["training"]["label_smoothing"],
    )

    best_score = float("-inf")
    selection_metric = config["training"]["selection_metric"].lower()
    metrics_logger = MetricsCSVLogger(
        results_dir=results_dir,
        approach_name=approach_name,
        header={
            "EMBEDDING_TYPE": embedding_wrapper.embedding_type.upper(),
            "APPROACH": approach_name,
            "LR": config["training"]["lr"],
            "MODEL_DIM": config["model"]["model_dim"],
        },
    )

    for epoch in range(1, config["training"]["epochs"] + 1):
        epoch_start = time.perf_counter()
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            clip=config["training"]["clip"],
            device=device,
            scheduler=scheduler,
            epoch=epoch,
        )
        val_loss = evaluate_epoch(model=model, dataloader=val_loader, criterion=criterion, device=device, epoch=epoch)

        metrics = {"val_loss": val_loss}
        if selection_metric == "rouge_l":
            generation_metrics = evaluate_model(
                model=model,
                dataloader=val_loader,
                vocab=vocab,
                device=device,
                config=config,
                max_batches=config["training"]["max_val_generation_batches"],
                compute_semantic=False,
            )
            metrics.update(generation_metrics)
            score = metrics["ROUGE-L"]
        else:
            score = -val_loss

        is_best = score > best_score
        epoch_time = time.perf_counter() - epoch_start
        log_row = {
            "epoch": epoch,
            "embedding_type": embedding_wrapper.embedding_type,
            "approach_name": approach_name,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6),
            "score": round(float(score), 6),
            "epoch_time_sec": round(epoch_time, 3),
            "trainable_params": trainable_params,
            "is_best": is_best,
            **metrics,
        }
        metrics_logger.append(log_row)

        print("\n" + "=" * 80)
        print(f"Epoch {epoch}/{config['training']['epochs']} | {approach_name}")
        print("-" * 80)
        print(_format_epoch_table(log_row))
        print(f"Metrics appended to {metrics_logger.path}")

        if is_best:
            best_score = score
            checkpoint_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
                "metrics": log_row,
                "approach_name": approach_name,
            }
            approach_checkpoint = checkpoint_dir / f"best_model_{_safe_metric_name(approach_name)}.pt"
            save_checkpoint(checkpoint_state, approach_checkpoint)
            save_checkpoint(checkpoint_state, checkpoint_dir / "best_model.pt")
            print(f"Saved new best checkpoint to {approach_checkpoint}")


if __name__ == "__main__":
    main()
