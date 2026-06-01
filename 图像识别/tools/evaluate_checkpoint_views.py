"""Evaluate a checkpoint on original split images and optional ROI-view manifests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

IMAGE_TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMAGE_TASK_DIR))

from shared import (  # noqa: E402
    DISPLAY_NAMES,
    LABEL_DICT,
    HierarchicalImageClassifier,
    LaryngealDataset,
    build_transforms,
    discover_images,
    evaluate,
    evaluate_hierarchical,
    init_label_mapping,
    is_voc_label,
    load_config,
    load_dataset_split,
    seed_everything,
    setup_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--roi-view", action="append", default=[], help="Format split=/path/to/roi_crop_manifest.csv")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    return parser.parse_args()


def parse_roi_views(values: list[str]) -> dict[str, Path]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--roi-view must use split=path, got {value!r}")
        split_name, path = value.split("=", 1)
        parsed[split_name] = Path(path)
    return parsed


def dataframe_from_roi_manifest(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            output_path = row.get("output_path") or row.get("crop_path")
            class_name = row.get("class_name")
            if not output_path or not class_name:
                continue
            label = LABEL_DICT[class_name]
            rows.append(
                {
                    "image_path": output_path,
                    "patient_name": Path(row.get("relative_path") or output_path).stem,
                    "source_folder": class_name,
                    "label": label,
                    "label_name": DISPLAY_NAMES[label],
                    "is_voc": is_voc_label(label),
                    "roi_action": row.get("action", ""),
                    "cropped": row.get("cropped", ""),
                }
            )
    return pd.DataFrame(rows)


def evaluate_df(model, df, cfg, criterion, device, num_classes, out_prefix: Path) -> dict:
    _, eval_tf = build_transforms(cfg)
    dataset = LaryngealDataset(df, eval_tf, cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("eval_batch_size", 512)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    metrics = evaluate(model, loader, criterion, device, num_classes, return_preds=True)
    hier = evaluate_hierarchical(model, loader, device, num_classes)
    labels = list(DISPLAY_NAMES.keys())
    target_names = [DISPLAY_NAMES[idx] for idx in labels]
    report = classification_report(
        metrics["y_true"],
        metrics["y_pred"],
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=labels)
    pd.DataFrame(report).transpose().to_csv(out_prefix.with_suffix(".classification_report.csv"))
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(out_prefix.with_suffix(".confusion_matrix.csv"))
    return {
        "n": int(len(df)),
        "loss": metrics["loss"],
        "accuracy": metrics["acc"],
        "macro_f1": metrics["f1"],
        "auc": metrics["auc"],
        "hierarchical_voc_acc": hier["hier_acc"],
        "hierarchical_voc_f1": hier["voc_f1"],
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(args.config))
    cfg["eval_batch_size"] = min(int(cfg.get("eval_batch_size", 512)), 512)
    init_label_mapping(cfg)
    seed_everything(int(cfg.get("seed", 42)))
    device = setup_device()
    num_classes = len(LABEL_DICT)

    df = discover_images()
    train_df, val_df, test_df = load_dataset_split(df)
    original_dfs = {"train": train_df, "val": val_df, "test": test_df}
    roi_views = parse_roi_views(args.roi_view)

    model = HierarchicalImageClassifier(num_classes=num_classes, cfg=cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))

    rows = []
    summary = {"checkpoint": str(args.checkpoint), "views": {}}
    for split_name in args.splits:
        original_metrics = evaluate_df(
            model,
            original_dfs[split_name],
            cfg,
            criterion,
            device,
            num_classes,
            args.out_dir / f"{split_name}_original",
        )
        summary["views"][f"{split_name}_original"] = original_metrics
        rows.append({"view": f"{split_name}_original", **original_metrics})

        if split_name in roi_views:
            roi_df = dataframe_from_roi_manifest(roi_views[split_name])
            roi_metrics = evaluate_df(
                model,
                roi_df,
                cfg,
                criterion,
                device,
                num_classes,
                args.out_dir / f"{split_name}_roi",
            )
            summary["views"][f"{split_name}_roi"] = roi_metrics
            rows.append({"view": f"{split_name}_roi", **roi_metrics})

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(args.out_dir / "view_metrics.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
