#!/usr/bin/env python3
"""Train classic baseline models and build model-comparison tables/figures."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import (  # noqa: E402
    BEST_MODEL_PATH,
    DISPLAY_NAMES,
    GPUAugment,
    LABEL_DICT,
    RESULTS_DIR,
    LaryngealDataset,
    WarmupCosineScheduler,
    build_balanced_sampler,
    build_transforms,
    create_classification_metrics,
    create_loaders,
    discover_images,
    gpu_normalise,
    init_label_mapping,
    load_config,
    load_dataset_split,
    maybe_prefetch_loader,
    preload_image_cache,
    seed_everything,
    setup_device,
    train_one_epoch,
    evaluate,
)
import shared  # noqa: E402


DEFAULT_MODELS = {
    "mobilenetv2": "mobilenetv2_100.ra_in1k",
    "densenet121": "densenet121.ra_in1k",
    "resnet50": "resnet50.a1_in1k",
}


LITERATURE_ROWS = [
    {
        "model": "MobileNetV2 + Xception",
        "comparison_group": "literature_reference",
        "source": "Emre et al. 2025, J Clin Med",
        "task": "vocal-fold healthy/nodule/polyp classification",
        "dataset_relation": "external_laryngoscopy_different_dataset",
        "accuracy": 0.980,
        "balanced_accuracy": np.nan,
        "macro_precision": np.nan,
        "macro_recall_sensitivity": np.nan,
        "macro_f1": 0.977,
        "macro_auroc_ovr": np.nan,
        "macro_auprc_ovr": np.nan,
        "notes": "Fine-tuned hybrid model; class F1 values 1.00, 0.96, 0.97.",
    },
    {
        "model": "Swin Transformer V1-tiny",
        "comparison_group": "literature_reference",
        "source": "Kim et al. 2026, Scientific Reports",
        "task": "high-quality laryngoscopy image detection",
        "dataset_relation": "external_laryngoscopy_quality_task",
        "accuracy": 0.951,
        "balanced_accuracy": np.nan,
        "macro_precision": 0.849,
        "macro_recall_sensitivity": 0.913,
        "macro_f1": 0.879,
        "macro_auroc_ovr": 0.979,
        "macro_auprc_ovr": 0.927,
        "notes": "Reported binary high-vs-other quality-classification result.",
    },
    {
        "model": "Prototypical ConvNeXt-Tiny",
        "comparison_group": "literature_reference",
        "source": "Merabet et al. 2025/2026, Int J Med Inform",
        "task": "colon histopathology few-shot classification",
        "dataset_relation": "external_other_body_site",
        "accuracy": 0.985,
        "balanced_accuracy": np.nan,
        "macro_precision": np.nan,
        "macro_recall_sensitivity": np.nan,
        "macro_f1": np.nan,
        "macro_auroc_ovr": np.nan,
        "macro_auprc_ovr": np.nan,
        "notes": "In-domain few-shot accuracy; paper also reports 0.900 external EBHI accuracy.",
    },
]


class TimmClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool, drop_rate: float):
        super().__init__()
        self.model_name = model_name
        self.net = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

    def forward(self, x):
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config_phase2.json")
    parser.add_argument("--output-dir", type=Path, default=Path(RESULTS_DIR) / "model_comparison")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS.keys()),
        help="Model keys or timm model names. Defaults: mobilenetv2 densenet121 resnet50.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--drop-rate", type=float, default=0.2)
    parser.add_argument("--sampler-balance-alpha", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses config batch_size.")
    parser.add_argument("--eval-batch-size", type=int, default=0, help="0 uses config eval_batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Defaults to config num_workers.")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-image-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def model_key(model_arg: str) -> tuple[str, str]:
    if model_arg in DEFAULT_MODELS:
        return model_arg, DEFAULT_MODELS[model_arg]
    key = (
        model_arg.replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )
    return key, model_arg


def active_label_names() -> list[str]:
    return [DISPLAY_NAMES[idx] for idx in range(len(DISPLAY_NAMES))]


def safe_metric(fn, default=np.nan):
    try:
        return fn()
    except ValueError:
        return default


def ovr_auc_summary(y_true: np.ndarray, probs: np.ndarray, num_classes: int, average: str) -> float:
    labels = list(range(num_classes))
    y_bin = label_binarize(y_true, classes=labels)
    scores = []
    weights = []
    for idx in labels:
        if len(np.unique(y_bin[:, idx])) < 2:
            continue
        scores.append(roc_auc_score(y_bin[:, idx], probs[:, idx]))
        weights.append(y_bin[:, idx].sum())
    if not scores:
        return np.nan
    if average == "weighted":
        return float(np.average(scores, weights=weights))
    return float(np.mean(scores))


def summary_metrics(split_name: str, outputs: dict, num_classes: int) -> dict:
    y_true = outputs["y_true"]
    y_pred = outputs["y_pred"]
    probs = outputs["probs"]
    labels = list(range(num_classes))
    y_bin = label_binarize(y_true, classes=labels)
    return {
        "split": split_name,
        "support": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall_sensitivity": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall_sensitivity": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_auroc_ovr": ovr_auc_summary(y_true, probs, num_classes, average="macro"),
        "weighted_auroc_ovr": ovr_auc_summary(y_true, probs, num_classes, average="weighted"),
        "macro_auprc_ovr": safe_metric(lambda: average_precision_score(y_bin, probs, average="macro")),
        "weighted_auprc_ovr": safe_metric(lambda: average_precision_score(y_bin, probs, average="weighted")),
    }


def per_class_metrics(split_name: str, outputs: dict, num_classes: int) -> pd.DataFrame:
    y_true = outputs["y_true"]
    y_pred = outputs["y_pred"]
    probs = outputs["probs"]
    labels = list(range(num_classes))
    label_names = active_label_names()
    y_bin = label_binarize(y_true, classes=labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    rows = []
    for idx, name in enumerate(label_names):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        rows.append(
            {
                "split": split_name,
                "label": name,
                "precision": precision[idx],
                "recall_sensitivity": recall[idx],
                "specificity": tn / max(tn + fp, 1),
                "f1": f1[idx],
                "support": support[idx],
                "one_vs_rest_auroc": safe_metric(lambda i=idx: roc_auc_score(y_bin[:, i], probs[:, i])),
                "one_vs_rest_auprc": safe_metric(lambda i=idx: average_precision_score(y_bin[:, i], probs[:, i])),
            }
        )
    return pd.DataFrame(rows)


def voc_binary_metrics(outputs: dict) -> dict:
    non_voc_label = shared.NON_VOC_LABEL
    y_true = (outputs["y_true"] != non_voc_label).astype(int)
    voc_indices = [idx for idx in range(len(DISPLAY_NAMES)) if idx != non_voc_label]
    voc_score = outputs["probs"][:, voc_indices].sum(axis=1)
    non_voc_score = outputs["probs"][:, non_voc_label]
    y_pred = (voc_score > non_voc_score).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "voc_accuracy": accuracy_score(y_true, y_pred),
        "voc_precision_ppv": precision_score(y_true, y_pred, zero_division=0),
        "voc_recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "voc_specificity": tn / max(tn + fp, 1),
        "voc_f1": f1_score(y_true, y_pred, zero_division=0),
        "voc_auroc": safe_metric(lambda: roc_auc_score(y_true, voc_score)),
        "voc_auprc": safe_metric(lambda: average_precision_score(y_true, voc_score)),
        "voc_tn": tn,
        "voc_fp": fp,
        "voc_fn": fn,
        "voc_tp": tp,
    }


@torch.inference_mode()
def collect_outputs(model, loader, frame: pd.DataFrame, device) -> dict:
    model.eval()
    labels_all = []
    probs_all = []
    preds_all = []
    amp_context = torch.amp.autocast(device_type=device.type) if device.type == "cuda" else nullcontext()
    for images, labels, _is_voc in maybe_prefetch_loader(loader, device):
        images = gpu_normalise(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)
        with amp_context:
            logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        labels_all.append(labels.detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())
        preds_all.append(preds.detach().cpu().numpy())
    return {
        "y_true": np.concatenate(labels_all),
        "y_pred": np.concatenate(preds_all),
        "probs": np.concatenate(probs_all),
        "paths": frame["image_path"].to_numpy(),
        "patients": frame["patient_name"].to_numpy(),
        "source_folders": frame["source_folder"].to_numpy(),
    }


def save_predictions(split_name: str, outputs: dict, output_dir: Path) -> None:
    label_names = active_label_names()
    df = pd.DataFrame(
        {
            "image_path": outputs["paths"],
            "patient_name": outputs["patients"],
            "source_folder": outputs["source_folders"],
            "true_label": [label_names[i] for i in outputs["y_true"]],
            "pred_label": [label_names[i] for i in outputs["y_pred"]],
            "confidence": outputs["probs"].max(axis=1),
            "correct": outputs["y_true"] == outputs["y_pred"],
        }
    )
    for idx, name in enumerate(label_names):
        df[f"prob_{name}"] = outputs["probs"][:, idx]
    df.to_csv(output_dir / f"predictions_{split_name}.csv", index=False)


def train_model(
    model_key_name: str,
    timm_name: str,
    cfg: dict,
    loaders: dict,
    split_frames: dict,
    args: argparse.Namespace,
    device,
    output_dir: Path,
) -> pd.DataFrame:
    run_dir = output_dir / "internal_baselines" / model_key_name
    if run_dir.exists():
        if args.force:
            shutil.rmtree(run_dir)
        elif (run_dir / "summary_metrics.csv").exists():
            print(f"Skipping {model_key_name}: existing summary_metrics.csv")
            return pd.read_csv(run_dir / "summary_metrics.csv")
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_key": model_key_name,
        "timm_model": timm_name,
        "pretrained": not args.no_pretrained,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "drop_rate": args.drop_rate,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\n=== Training baseline: {model_key_name} ({timm_name}) ===")
    model = TimmClassifier(
        timm_name,
        num_classes=len(LABEL_DICT),
        pretrained=not args.no_pretrained,
        drop_rate=args.drop_rate,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters trainable/total: {trainable_params:,}/{total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=torch.cuda.is_available(),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=min(3, max(args.epochs // 5, 1)),
        total_epochs=args.epochs,
        warmup_lr=1e-6,
        min_lr=args.learning_rate * 0.1,
    )
    scaler = torch.amp.GradScaler(device.type)
    gpu_aug = GPUAugment(cfg).to(device)
    cls_metrics = create_classification_metrics(len(LABEL_DICT), device)

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    checkpoint_path = run_dir / "best_model.pth"
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_f1, train_acc, train_auc = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            scaler,
            device,
            grad_accum=1,
            num_classes=len(LABEL_DICT),
            cfg=cfg,
            cls_metrics=cls_metrics,
            gpu_aug=gpu_aug,
        )
        val_metrics = evaluate(model, loaders["val"], criterion, device, len(LABEL_DICT), cls_metrics=cls_metrics)
        scheduler.step()
        val_score = 0.7 * val_metrics["f1"] + 0.3 * val_metrics["auc"]
        improved = val_score > best_score + args.min_delta
        if improved:
            best_score = val_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "train_f1": train_f1,
                "train_acc": train_acc,
                "train_auc": train_auc,
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["f1"],
                "val_acc": val_metrics["acc"],
                "val_auc": val_metrics["auc"],
                "val_score": val_score,
                "best_val_score": best_score,
                "improved": improved,
            }
        )
        star = "*" if improved else " "
        print(
            f" {star} epoch {epoch:02d}/{args.epochs} "
            f"train_f1={train_f1:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_auc={val_metrics['auc']:.4f}"
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}.")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
    model.eval()

    outputs_by_split = {}
    for split_name in ("train", "val", "test"):
        outputs_by_split[split_name] = collect_outputs(model, loaders[split_name], split_frames[split_name], device)
        save_predictions(split_name, outputs_by_split[split_name], run_dir)

    summary_rows = []
    per_class_frames = []
    for split_name, outputs in outputs_by_split.items():
        row = summary_metrics(split_name, outputs, len(LABEL_DICT))
        if split_name == "test":
            row.update(voc_binary_metrics(outputs))
        row.update(
            {
                "model_key": model_key_name,
                "timm_model": timm_name,
                "comparison_group": "internal_baseline",
                "best_epoch": best_epoch,
                "train_seconds": time.time() - started,
            }
        )
        summary_rows.append(row)
        per_class = per_class_metrics(split_name, outputs, len(LABEL_DICT))
        per_class.insert(0, "model_key", model_key_name)
        per_class_frames.append(per_class)

    summary_df = pd.DataFrame(summary_rows)
    per_class_df = pd.concat(per_class_frames, ignore_index=True)
    summary_df.to_csv(run_dir / "summary_metrics.csv", index=False)
    per_class_df.to_csv(run_dir / "per_class_metrics.csv", index=False)
    print(f"Saved baseline run: {run_dir}")
    return summary_df


def current_model_row() -> pd.DataFrame:
    results_dir = Path(RESULTS_DIR)
    current_eval_dir = results_dir / "model_comparison" / "current_checkpoint_v69_eval"
    if (current_eval_dir / "summary_metrics.csv").exists():
        summary_path = current_eval_dir / "summary_metrics.csv"
        voc_path = current_eval_dir / "voc_binary_metrics.csv"
        source = "current checkpoint evaluated on current split"
        dataset_relation = "same_current_frozen_test_split"
    else:
        summary_path = results_dir / "literature_aligned_metrics" / "summary_metrics.csv"
        voc_path = results_dir / "literature_aligned_metrics" / "voc_binary_metrics.csv"
        source = "current project"
        dataset_relation = "project_reported_split"
    if not summary_path.exists():
        return pd.DataFrame()
    summary = pd.read_csv(summary_path)
    test = summary[summary["split"] == "test"].copy()
    if test.empty:
        return pd.DataFrame()
    test = test.iloc[[0]].copy()
    if voc_path.exists():
        voc = pd.read_csv(voc_path)
        voc_test = voc[voc["split"] == "test"]
        if not voc_test.empty:
            for col in ["accuracy", "recall_sensitivity", "specificity", "f1", "auroc", "auprc"]:
                test[f"voc_{col}"] = float(voc_test.iloc[0][col])
    test["model"] = "Ours: KG-SupCon + Swin-B"
    test["model_key"] = "ours_kg_supcon_swin_b"
    test["comparison_group"] = "ours_same_split"
    test["source"] = source
    test["task"] = "8-class laryngeal multiclass classification"
    test["dataset_relation"] = dataset_relation
    note = "Two-stage Knowledge-Guided SupCon pretraining followed by CE fine-tuning."
    checkpoint_path = Path(BEST_MODEL_PATH)
    split_path = PROJECT_ROOT / "dataset_split.json"
    if checkpoint_path.exists() and split_path.exists() and checkpoint_path.stat().st_mtime < split_path.stat().st_mtime:
        note += " Checkpoint file predates the current v6.9 split; retrain Phase 1/2 before treating this as final strict same-split manuscript evidence."
    test["notes"] = note
    return test


def collect_baseline_test_rows(output_dir: Path) -> pd.DataFrame:
    rows = []
    base_dir = output_dir / "internal_baselines"
    if not base_dir.exists():
        return pd.DataFrame()
    for summary_path in sorted(base_dir.glob("*/summary_metrics.csv")):
        df = pd.read_csv(summary_path)
        test = df[df["split"] == "test"].copy()
        if test.empty:
            continue
        test = test.iloc[[0]].copy()
        model_key_name = str(test.iloc[0]["model_key"])
        timm_model = str(test.iloc[0].get("timm_model", model_key_name))
        test["model"] = f"Baseline: {model_key_name}"
        test["source"] = timm_model
        test["task"] = "8-class laryngeal multiclass classification"
        test["dataset_relation"] = "same_patient_level_split"
        test["notes"] = "ImageNet-pretrained timm baseline, single-stage CE fine-tuning."
        rows.append(test)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_comparison_report(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    internal_rows = []
    ours = current_model_row()
    if not ours.empty:
        internal_rows.append(ours)
    baselines = collect_baseline_test_rows(output_dir)
    if not baselines.empty:
        internal_rows.append(baselines)
    internal_df = pd.concat(internal_rows, ignore_index=True) if internal_rows else pd.DataFrame()
    literature_df = pd.DataFrame(LITERATURE_ROWS)

    if not internal_df.empty:
        for col in literature_df.columns:
            if col not in internal_df:
                internal_df[col] = np.nan
        internal_df = internal_df[literature_df.columns.tolist() + [c for c in internal_df.columns if c not in literature_df.columns]]

    combined = pd.concat([internal_df, literature_df], ignore_index=True, sort=False)
    combined.to_csv(output_dir / "model_comparison_summary.csv", index=False)
    literature_df.to_csv(output_dir / "literature_sota_reference.csv", index=False)
    if not internal_df.empty:
        internal_df.to_csv(output_dir / "internal_same_split_comparison.csv", index=False)
        plot_internal_comparison(internal_df, output_dir / "internal_same_split_metrics.png")
        plot_voc_comparison(internal_df, output_dir / "internal_voc_binary_metrics.png")
        per_class_df = collect_internal_per_class_rows(output_dir)
        if not per_class_df.empty:
            per_class_df.to_csv(output_dir / "internal_per_class_f1.csv", index=False)
            plot_per_class_f1_heatmap(per_class_df, output_dir / "internal_per_class_f1_heatmap.png")
    plot_combined_overview(combined, output_dir / "model_comparison_overview.png")
    write_readme(output_dir, internal_df, literature_df)


def _model_order(df: pd.DataFrame) -> pd.DataFrame:
    order = {"ours_same_split": 0, "internal_baseline": 1, "literature_reference": 2}
    return df.assign(_order=df["comparison_group"].map(order).fillna(9)).sort_values(["_order", "model"]).drop(columns="_order")


def _apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.6,
        }
    )


def _short_model_name(name: str) -> str:
    return (
        str(name)
        .replace("Ours: ", "Ours\n")
        .replace("Baseline: ", "Baseline\n")
        .replace("Swin Transformer V1-tiny", "Swin Transformer\nV1-tiny")
        .replace("MobileNetV2 + Xception", "MobileNetV2\n+ Xception")
        .replace("Prototypical ConvNeXt-Tiny", "Prototypical\nConvNeXt-Tiny")
    )


def _group_label(value: str) -> str:
    labels = {
        "ours_same_split": "Same frozen test split",
        "internal_baseline": "Same frozen test split",
        "literature_reference": "Literature reference",
    }
    return labels.get(value, str(value).replace("_", " ").title())


def _save_figure(fig, output_path: Path) -> None:
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


def collect_internal_per_class_rows(output_dir: Path) -> pd.DataFrame:
    rows = []
    results_dir = Path(RESULTS_DIR)
    current_eval_dir = results_dir / "model_comparison" / "current_checkpoint_v69_eval"
    ours_path = (
        current_eval_dir / "per_class_metrics.csv"
        if (current_eval_dir / "per_class_metrics.csv").exists()
        else results_dir / "literature_aligned_metrics" / "per_class_metrics.csv"
    )
    if ours_path.exists():
        ours = pd.read_csv(ours_path)
        ours = ours[ours["split"] == "test"].copy()
        if not ours.empty:
            ours["model"] = "Ours: KG-SupCon + Swin-B"
            ours["model_key"] = "ours_kg_supcon_swin_b"
            ours["comparison_group"] = "ours_same_split"
            rows.append(ours)

    base_dir = output_dir / "internal_baselines"
    if base_dir.exists():
        for per_class_path in sorted(base_dir.glob("*/per_class_metrics.csv")):
            model_key_name = per_class_path.parent.name
            df = pd.read_csv(per_class_path)
            df = df[df["split"] == "test"].copy()
            if df.empty:
                continue
            df["model"] = f"Baseline: {model_key_name}"
            df["model_key"] = model_key_name
            df["comparison_group"] = "internal_baseline"
            rows.append(df)

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True, sort=False)
    result["_class_order"] = result["label"].map({name: idx for idx, name in enumerate(active_label_names())})
    result["_model_order"] = result["comparison_group"].map({"ours_same_split": 0, "internal_baseline": 1}).fillna(9)
    result = result.sort_values(["_model_order", "model", "_class_order"], kind="stable")
    return result.drop(columns=[col for col in ["_model_order", "_class_order"] if col in result])


def plot_internal_comparison(df: pd.DataFrame, output_path: Path) -> None:
    _apply_publication_style()
    df = _model_order(df.copy())
    metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Acc."),
        ("macro_f1", "Macro F1"),
        ("macro_auroc_ovr", "Macro AUROC"),
        ("macro_auprc_ovr", "Macro AUPRC"),
    ]
    metrics = [(metric, name) for metric, name in metrics if metric in df]
    labels = [_short_model_name(name) for name in df["model"]]
    y = np.arange(len(df))
    colors = df["comparison_group"].map(
        {
            "ours_same_split": "#1F4E79",
            "internal_baseline": "#6B7280",
        }
    ).fillna("#6B7280")
    markers = df["comparison_group"].map(
        {
            "ours_same_split": "D",
            "internal_baseline": "o",
        }
    ).fillna("o")

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(2.35 * len(metrics) + 2.8, max(3.2, 0.45 * len(df) + 2.0)),
        sharey=True,
        facecolor="white",
    )
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric, title) in zip(axes, metrics):
        values = pd.to_numeric(df[metric], errors="coerce")
        ax.hlines(y, 0, values, color="#D1D5DB", linewidth=1.1, zorder=1)
        for idx, value in enumerate(values):
            if pd.isna(value):
                continue
            ax.scatter(
                value,
                y[idx],
                s=58 if df.iloc[idx]["comparison_group"] == "ours_same_split" else 46,
                color=colors.iloc[idx],
                marker=markers.iloc[idx],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.text(min(float(value) + 0.018, 1.015), y[idx], f"{float(value):.3f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.04)
        ax.set_title(title, pad=8, fontweight="bold")
        ax.grid(axis="x", linestyle="-", alpha=0.75)
        ax.set_xlabel("Score")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    fig.suptitle("Frozen Test Split: Internal Model Comparison", x=0.02, ha="left", fontweight="bold")
    fig.text(
        0.02,
        0.925,
        "All points are evaluated on the current frozen test split; baseline rows were reimplemented on the current train/val/test split.",
        ha="left",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.0)
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_voc_comparison(df: pd.DataFrame, output_path: Path) -> None:
    _apply_publication_style()
    df = _model_order(df.copy())
    metrics = [
        ("voc_f1", "VOC F1"),
        ("voc_recall_sensitivity", "Sensitivity"),
        ("voc_specificity", "Specificity"),
        ("voc_auroc", "AUROC"),
        ("voc_auprc", "AUPRC"),
    ]
    metrics = [(metric, name) for metric, name in metrics if metric in df]
    labels = [_short_model_name(name) for name in df["model"]]
    y = np.arange(len(df))
    colors = df["comparison_group"].map(
        {
            "ours_same_split": "#1F4E79",
            "internal_baseline": "#6B7280",
        }
    ).fillna("#6B7280")
    markers = df["comparison_group"].map(
        {
            "ours_same_split": "D",
            "internal_baseline": "o",
        }
    ).fillna("o")

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(2.25 * len(metrics) + 2.8, max(3.2, 0.45 * len(df) + 2.0)),
        sharey=True,
        facecolor="white",
    )
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric, title) in zip(axes, metrics):
        values = pd.to_numeric(df[metric], errors="coerce")
        ax.hlines(y, 0, values, color="#D1D5DB", linewidth=1.1, zorder=1)
        for idx, value in enumerate(values):
            if pd.isna(value):
                continue
            ax.scatter(
                value,
                y[idx],
                s=58 if df.iloc[idx]["comparison_group"] == "ours_same_split" else 46,
                color=colors.iloc[idx],
                marker=markers.iloc[idx],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.text(min(float(value) + 0.006, 1.015), y[idx], f"{float(value):.3f}", va="center", fontsize=8)
        ax.set_xlim(0.92, 1.005)
        ax.set_title(title, pad=8, fontweight="bold")
        ax.grid(axis="x", linestyle="-", alpha=0.75)
        ax.set_xlabel("Score")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    fig.suptitle("Frozen Test Split: VOC vs Non-VOC Performance", x=0.02, ha="left", fontweight="bold")
    fig.text(
        0.02,
        0.925,
        "Binary hierarchy check derived from the same multiclass probabilities.",
        ha="left",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.0)
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_per_class_f1_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    _apply_publication_style()
    ordered = _model_order(df.copy())
    class_names = active_label_names()
    matrix = ordered.pivot_table(index="model", columns="label", values="f1", aggfunc="first")
    model_order = ordered[["model", "comparison_group"]].drop_duplicates()["model"].tolist()
    matrix = matrix.reindex(index=model_order, columns=class_names)

    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12.6, max(3.4, 0.45 * len(matrix) + 2.4)), facecolor="white")
    im = ax.imshow(values, cmap="YlGnBu", vmin=0.45, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([_short_model_name(name) for name in matrix.index])
    ax.set_title("Per-Class F1 on the Frozen Test Split", loc="left", fontweight="bold", pad=16)
    ax.set_xlabel("Class")
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                continue
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.62 or value > 0.86 else "#111827",
                fontsize=8,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("F1 score")
    fig.text(
        0.01,
        0.91,
        "Darker cells indicate stronger class-level discrimination; values are not prevalence-weighted.",
        ha="left",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_combined_overview(df: pd.DataFrame, output_path: Path) -> None:
    _apply_publication_style()
    df = _model_order(df.copy())
    metric = "accuracy"
    plot_df = df.dropna(subset=[metric]).copy()
    plot_df["comparison_scope"] = plot_df["comparison_group"].map(_group_label)
    colors = plot_df["comparison_group"].map(
        {
            "ours_same_split": "#1F4E79",
            "internal_baseline": "#6B7280",
            "literature_reference": "#A16207",
        }
    ).fillna("#6B7280")
    markers = plot_df["comparison_group"].map(
        {
            "ours_same_split": "D",
            "internal_baseline": "o",
            "literature_reference": "s",
        }
    ).fillna("o")

    fig, ax = plt.subplots(figsize=(10.8, max(4.2, 0.52 * len(plot_df) + 2.2)), facecolor="white")
    y = np.arange(len(plot_df))
    values = plot_df[metric].astype(float)
    lower_bound = max(0.0, float(values.min()) - 0.08)
    ax.hlines(y, lower_bound, values, color="#CBD5E1", linewidth=1.4, zorder=1)
    for idx, value in enumerate(values):
        ax.scatter(
            value,
            y[idx],
            s=68 if plot_df.iloc[idx]["comparison_group"] == "ours_same_split" else 52,
            color=colors.iloc[idx],
            marker=markers.iloc[idx],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(min(float(value) + 0.012, 1.015), y[idx], f"{float(value):.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([_short_model_name(name) for name in plot_df["model"]])
    ax.set_xlim(lower_bound, 1.02)
    ax.set_xlabel("Reported accuracy")
    ax.set_title("Accuracy Overview by Evidence Scope", loc="left", fontweight="bold", pad=18)
    ax.grid(axis="x", linestyle="-", alpha=0.75)
    ax.tick_params(axis="y", length=0)

    groups = plot_df["comparison_scope"].tolist()
    start = 0
    for idx in range(1, len(groups) + 1):
        if idx == len(groups) or groups[idx] != groups[start]:
            end = idx - 1
            ax.text(
                lower_bound + 0.005,
                start + 0.18,
                groups[start],
                ha="left",
                va="top",
                fontsize=8,
                fontweight="bold",
                color="#374151",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.92},
            )
            if idx < len(groups):
                ax.axhline(idx - 0.5, color="#9CA3AF", linewidth=0.8, linestyle=(0, (2, 2)))
            start = idx

    ax.text(
        0.0,
        1.02,
        "Upper group: current frozen test split. Lower group: external papers with different tasks/datasets.",
        transform=ax.transAxes,
        ha="left",
        fontsize=9,
        color="#374151",
    )
    ax.invert_yaxis()
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str], floatfmt: str = ".4f") -> str:
    table = df[columns].copy()
    headers = columns
    rows = []
    for _, row in table.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append("" if pd.isna(value) else format(float(value), floatfmt))
            else:
                values.append("" if pd.isna(value) else str(value))
        rows.append(values)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(values) + " |" for values in rows)
    return "\n".join(lines)


def write_readme(output_dir: Path, internal_df: pd.DataFrame, literature_df: pd.DataFrame) -> None:
    lines = [
        "# Model Comparison",
        "",
        "This folder separates direct same-split model comparisons from external literature reference rows.",
        "",
        "## Interpretation",
        "",
        "- **Same frozen test split** rows are evaluated on the current project test split; baseline rows were trained on the current train/val/test split.",
        "- The current `best_model.pth` predates the v6.9 split file. Keep its row for visual audit/context, but rerun Phase 1/2 before treating it as final strict same-split manuscript evidence.",
        "- **Literature reference** rows are context only: datasets, labels, disease scope, and tasks differ, so they should not be ranked against the project model.",
        "",
        "![Frozen test split internal comparison](internal_same_split_metrics.png)",
        "",
        "![Accuracy overview by evidence scope](model_comparison_overview.png)",
        "",
        "![Per-class F1 heatmap](internal_per_class_f1_heatmap.png)",
        "",
        "![VOC binary performance](internal_voc_binary_metrics.png)",
        "",
        "## Internal Same-Split Models",
        "",
    ]
    if internal_df.empty:
        lines.append("No internal baseline runs found yet.")
    else:
        cols = ["model", "accuracy", "balanced_accuracy", "macro_f1", "macro_auroc_ovr", "macro_auprc_ovr"]
        lines.append(markdown_table(internal_df, cols))
    lines.extend(
        [
            "",
            "## Literature Reference Rows",
            "",
            "These rows are not direct head-to-head comparisons because the datasets, labels, and tasks differ.",
            "",
            markdown_table(
                literature_df,
                ["model", "source", "task", "accuracy", "macro_f1", "macro_auroc_ovr", "macro_auprc_ovr", "notes"],
            ),
            "",
            "## Files",
            "",
            "- `internal_same_split_comparison.csv`: current checkpoint plus trained baselines evaluated on the current frozen test split.",
            "- `literature_sota_reference.csv`: selected SOTA/reference rows from the three archived papers.",
            "- `model_comparison_summary.csv`: combined table for manuscript drafting.",
            "- `internal_per_class_f1.csv`: test-set per-class F1 values for internal comparison rows.",
            "- `internal_same_split_metrics.png`: small-multiple point-line panels for current frozen-test metrics.",
            "- `internal_per_class_f1_heatmap.png`: per-class F1 heatmap across internal models.",
            "- `internal_voc_binary_metrics.png`: VOC vs Non-VOC hierarchy metrics across internal models.",
            "- `model_comparison_overview.png`: grouped horizontal point-line accuracy overview separating internal and literature evidence.",
            "- `internal_baselines/<model>/`: one folder per trained baseline with history, metrics, predictions, and checkpoint.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(args.config))
    init_label_mapping(cfg)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    seed_everything(seed)

    cfg = dict(cfg)
    cfg["batch_size"] = args.batch_size if args.batch_size > 0 else cfg.get("batch_size", 256)
    cfg["eval_batch_size"] = args.eval_batch_size if args.eval_batch_size > 0 else cfg.get("eval_batch_size", 512)
    cfg["num_workers"] = cfg.get("num_workers", 0) if args.num_workers is None else args.num_workers
    cfg["learning_rate"] = args.learning_rate
    cfg["weight_decay"] = args.weight_decay
    cfg["label_smoothing"] = args.label_smoothing
    cfg["sampler_balance_alpha"] = args.sampler_balance_alpha

    if not args.report_only:
        device = setup_device()
        df = discover_images()
        train_df, val_df, test_df = load_dataset_split(df)
        image_cache = None if args.no_image_cache else preload_image_cache(train_df, val_df, test_df, cfg=cfg)
        sampler = build_balanced_sampler(train_df, hierarchical=False, balance_alpha=args.sampler_balance_alpha)
        loaders = create_loaders(train_df, val_df, test_df, cfg, image_cache=image_cache, train_sampler=sampler)
        split_frames = {"train": train_df, "val": val_df, "test": test_df}

        for model_arg in args.models:
            key, timm_name = model_key(model_arg)
            train_model(key, timm_name, cfg, loaders, split_frames, args, device, output_dir)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    build_comparison_report(output_dir)
    print(f"Saved model comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
