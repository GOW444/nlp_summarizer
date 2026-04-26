"""
Audit FK grade direction in train/val/test splits and optionally filter bad pairs.

Usage:
    # Audit only (no changes)
    python audit_fk.py

    # Audit + filter and overwrite splits
    python audit_fk.py --filter

    # Audit + filter with custom threshold (default 0.8)
    python audit_fk.py --filter --threshold 0.75
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import textstat


def fk_grade(text: str) -> float:
    return textstat.flesch_kincaid_grade(text)


def audit_split(rows: list[dict], split_name: str) -> tuple[list[dict], dict]:
    total = len(rows)
    good = []   # FK(target) < FK(source)
    bad = []    # FK(target) >= FK(source)
    deltas = []

    for row in rows:
        src = row.get("note", "")
        tgt = row.get("answer", "")
        fk_src = fk_grade(src)
        fk_tgt = fk_grade(tgt)
        delta = fk_src - fk_tgt  # positive = target is simpler (good)
        deltas.append(delta)
        if fk_tgt < fk_src:
            good.append(row)
        else:
            bad.append(row)

    good_pct = len(good) / total * 100 if total > 0 else 0
    avg_delta = sum(deltas) / len(deltas) if deltas else 0

    stats = {
        "split": split_name,
        "total": total,
        "good (tgt simpler)": len(good),
        "bad (tgt harder/equal)": len(bad),
        "good_pct": round(good_pct, 2),
        "avg_fk_delta": round(avg_delta, 3),
    }
    return good, stats


def print_stats(stats: dict) -> None:
    print(f"\n{'='*50}")
    print(f"Split: {stats['split']}")
    print(f"  Total pairs       : {stats['total']}")
    print(f"  Target simpler    : {stats['good (tgt simpler)']}  ({stats['good_pct']}%)")
    print(f"  Target harder/eq  : {stats['bad (tgt harder/equal)']}")
    print(f"  Avg FK delta      : {stats['avg_fk_delta']}  (positive = simpler on avg)")
    if stats["good_pct"] < 80:
        print(f"  ⚠️  WARNING: only {stats['good_pct']}% pairs have simpler targets (want >80%)")
    else:
        print(f"  ✅ Good: {stats['good_pct']}% pairs have simpler targets")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed", help="Directory with train/val/test.pt")
    parser.add_argument("--filter", action="store_true", help="Filter bad pairs and overwrite splits")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum fraction of good pairs required before filtering is triggered (default 0.8)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    splits = ["train", "val", "test"]

    all_stats = []
    filtered = {}

    for split in splits:
        path = data_dir / f"{split}.pt"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue

        print(f"Loading {path} ...")
        rows = torch.load(str(path))
        good_rows, stats = audit_split(rows, split)
        print_stats(stats)
        all_stats.append(stats)
        filtered[split] = good_rows

    # Summary
    total_all = sum(s["total"] for s in all_stats)
    good_all = sum(s["good (tgt simpler)"] for s in all_stats)
    overall_pct = good_all / total_all * 100 if total_all else 0
    print(f"\n{'='*50}")
    print(f"OVERALL: {good_all}/{total_all} pairs have simpler targets ({overall_pct:.1f}%)")

    if not args.filter:
        print("\nRun with --filter to remove bad pairs and overwrite splits.")
        return

    # Filter and overwrite
    print(f"\nFiltering splits (keeping only pairs where FK(target) < FK(source)) ...")
    for split in splits:
        if split not in filtered:
            continue
        path = data_dir / f"{split}.pt"
        original = torch.load(str(path))
        good_rows = filtered[split]
        removed = len(original) - len(good_rows)
        torch.save(good_rows, str(path))
        print(f"  {split}: {len(original)} → {len(good_rows)} rows ({removed} removed)")

    print("\nDone. Re-run src.train to train on the filtered splits.")
    print("NOTE: vocab.json was NOT rebuilt. If you removed many rows, consider re-running src.prepare_data.")


if __name__ == "__main__":
    main()
