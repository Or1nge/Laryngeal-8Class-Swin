#!/usr/bin/env python3
"""Evaluate one glottis binary checkpoint on the frozen split."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from common import (
    DEFAULT_SPLIT_PATH,
    binary_metrics,
    collect_outputs,
    load_checkpoint_model,
    load_split_dataframe,
    preload_split_cache,
    save_confusion_matrix_png,
    save_predictions,
    save_roc_pr_png,
    setup_device,
    threshold_table,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--cache-device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = setup_device()
    model, ckpt = load_checkpoint_model(args.checkpoint, device)
    cfg = ckpt["cfg"]
    threshold = args.threshold
    if threshold is None:
        threshold = ckpt.get("recommended_threshold")
    if threshold is None:
        threshold = 0.5
    output_dir = args.output_dir or (
        args.checkpoint.parent / f"reeval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_split_dataframe(args.split)
    caches = preload_split_cache(df, cfg, device=device, cache_device=args.cache_device)
    rows = []
    for split_name in ("train", "val", "test"):
        outputs = collect_outputs(model, caches[split_name], int(cfg["eval_batch_size"]), device)
        metrics = binary_metrics(outputs["y_true"], outputs["probs_glottis"], threshold=threshold)
        metrics.update({"split": split_name})
        rows.append(metrics)
        save_predictions(split_name, outputs, caches[split_name], threshold, output_dir)
        threshold_table(outputs["y_true"], outputs["probs_glottis"]).to_csv(
            output_dir / f"threshold_metrics_{split_name}.csv",
            index=False,
        )
        if split_name == "test":
            save_confusion_matrix_png(
                outputs["y_true"],
                outputs["probs_glottis"],
                threshold,
                output_dir / "confusion_matrix_test.png",
            )
            save_roc_pr_png(outputs["y_true"], outputs["probs_glottis"], output_dir / "roc_pr_test.png")
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    write_json(
        output_dir / "evaluation_metadata.json",
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "split": str(args.split.resolve()),
            "threshold": threshold,
            "model_key": ckpt.get("model_key"),
            "model_name": ckpt.get("model_name"),
            "pretrained_source": ckpt.get("pretrained_source"),
        },
    )
    print(f"Evaluation written to: {output_dir}")


if __name__ == "__main__":
    main()
