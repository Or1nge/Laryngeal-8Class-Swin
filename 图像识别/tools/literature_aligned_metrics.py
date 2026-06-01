#!/usr/bin/env python3
"""Generate paper-style evaluation tables and figures for the laryngeal model."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shared  # noqa: E402
from shared import (  # noqa: E402
    ATTENTION_MAP_PATH,
    BEST_MODEL_PATH,
    DISPLAY_NAMES,
    HISTORY_CSV_PATH,
    LABEL_DICT,
    RESULTS_DIR,
    LaryngealDataset,
    HierarchicalImageClassifier,
    build_transforms,
    discover_images,
    gpu_normalise,
    init_label_mapping,
    load_config,
    load_dataset_split,
    maybe_prefetch_loader,
    preload_image_cache,
    setup_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config_phase2.json")
    parser.add_argument("--model", type=Path, default=Path(BEST_MODEL_PATH))
    parser.add_argument("--output-dir", type=Path, default=Path(RESULTS_DIR) / "literature_aligned_metrics")
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses config eval_batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Defaults to config num_workers.")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Defaults to config prefetch_factor.")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--no-image-cache", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tsne", action="store_true")
    return parser.parse_args()


def force_timm_no_pretrained():
    original_create_model = shared.timm.create_model

    def create_model_no_pretrained(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    shared.timm.create_model = create_model_no_pretrained
    return original_create_model


def restore_timm_create_model(original_create_model):
    shared.timm.create_model = original_create_model


def active_label_names() -> list[str]:
    return [DISPLAY_NAMES[idx] for idx in range(len(DISPLAY_NAMES))]


def make_loader(
    df: pd.DataFrame,
    cfg: dict,
    eval_tf,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    persistent_workers: bool,
    image_cache: dict | None,
) -> DataLoader:
    dataset = LaryngealDataset(df, eval_tf, cfg, image_cache=image_cache)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


@torch.inference_mode()
def collect_outputs(model, loader, df: pd.DataFrame, device, collect_features: bool = False) -> dict:
    model.eval()
    labels_all = []
    probs_all = []
    preds_all = []
    is_voc_all = []
    features_all = []

    amp_context = (
        torch.amp.autocast(device_type=device.type)
        if device.type == "cuda"
        else nullcontext()
    )

    for images, labels, is_voc in maybe_prefetch_loader(loader, device):
        images = gpu_normalise(images.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)
        with amp_context:
            if collect_features:
                features = model.backbone(images)
                logits = model.classifier(features)
                features_for_plot = features
                if features_for_plot.ndim > 2:
                    features_for_plot = torch.flatten(features_for_plot, start_dim=1)
            else:
                logits = model(images)
                features_for_plot = None
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        labels_all.append(labels.detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())
        preds_all.append(preds.detach().cpu().numpy())
        if isinstance(is_voc, torch.Tensor):
            is_voc_all.append(is_voc.detach().cpu().numpy())
        else:
            is_voc_all.append(np.asarray(is_voc))
        if collect_features:
            features_all.append(features_for_plot.detach().float().cpu().numpy())

    result = {
        "y_true": np.concatenate(labels_all),
        "y_pred": np.concatenate(preds_all),
        "probs": np.concatenate(probs_all),
        "is_voc": np.concatenate(is_voc_all).astype(bool),
        "paths": df["image_path"].to_numpy(),
        "patients": df["patient_name"].to_numpy(),
        "source_folders": df["source_folder"].to_numpy(),
    }
    if collect_features:
        result["features"] = np.concatenate(features_all)
    return result


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
        specificity = tn / max(tn + fp, 1)
        auroc = safe_metric(lambda i=idx: roc_auc_score(y_bin[:, i], probs[:, i]))
        auprc = safe_metric(lambda i=idx: average_precision_score(y_bin[:, i], probs[:, i]))
        rows.append(
            {
                "split": split_name,
                "label": name,
                "precision": precision[idx],
                "recall_sensitivity": recall[idx],
                "specificity": specificity,
                "f1": f1[idx],
                "support": support[idx],
                "one_vs_rest_auroc": auroc,
                "one_vs_rest_auprc": auprc,
            }
        )
    return pd.DataFrame(rows)


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


def voc_binary_metrics(split_name: str, outputs: dict) -> dict:
    non_voc_label = shared.NON_VOC_LABEL
    y_true = (outputs["y_true"] != non_voc_label).astype(int)
    voc_indices = [idx for idx in range(len(DISPLAY_NAMES)) if idx != non_voc_label]
    voc_score = outputs["probs"][:, voc_indices].sum(axis=1)
    non_voc_score = outputs["probs"][:, non_voc_label]
    y_pred = (voc_score > non_voc_score).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "split": split_name,
        "negative_class": DISPLAY_NAMES[non_voc_label],
        "positive_class": "VOC",
        "support": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_ppv": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": safe_metric(lambda: roc_auc_score(y_true, voc_score)),
        "auprc": safe_metric(lambda: average_precision_score(y_true, voc_score)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def bootstrap_ci(outputs: dict, num_classes: int, iterations: int, seed: int) -> pd.DataFrame:
    if iterations <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    y_true = outputs["y_true"]
    class_indices = [np.where(y_true == idx)[0] for idx in range(num_classes)]
    values = []
    for _ in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices if len(indices)]
        )
        sampled_outputs = {
            "y_true": outputs["y_true"][sampled],
            "y_pred": outputs["y_pred"][sampled],
            "probs": outputs["probs"][sampled],
        }
        row = summary_metrics("bootstrap", sampled_outputs, num_classes)
        row.update({f"voc_{k}": v for k, v in voc_binary_metrics("bootstrap", sampled_outputs).items() if isinstance(v, (int, float, np.integer, np.floating))})
        values.append(row)
    df = pd.DataFrame(values)
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall_sensitivity",
        "macro_f1",
        "macro_auroc_ovr",
        "macro_auprc_ovr",
        "voc_accuracy",
        "voc_recall_sensitivity",
        "voc_specificity",
        "voc_f1",
        "voc_auroc",
        "voc_auprc",
    ]
    rows = []
    for metric in metrics:
        if metric not in df:
            continue
        series = df[metric].dropna()
        if series.empty:
            continue
        rows.append(
            {
                "split": "test",
                "metric": metric,
                "mean": series.mean(),
                "ci_low_2.5": np.percentile(series, 2.5),
                "ci_high_97.5": np.percentile(series, 97.5),
                "bootstrap_iterations": iterations,
            }
        )
    return pd.DataFrame(rows)


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


def plot_confusion(outputs: dict, output_path: Path, normalized: bool) -> None:
    labels = list(range(len(DISPLAY_NAMES)))
    label_names = active_label_names()
    cm = confusion_matrix(outputs["y_true"], outputs["y_pred"], labels=labels)
    if normalized:
        cm_plot = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fmt = ".1%"
        title = "Test Confusion Matrix (Row-Normalized)"
        cmap = "YlGnBu"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Test Confusion Matrix (Counts)"
        cmap = "Blues"

    fig, ax = plt.subplots(figsize=(11, 9), facecolor="white")
    im = ax.imshow(cm_plot, cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=35, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontweight="bold")
    threshold = np.nanmax(cm_plot) * 0.55 if cm_plot.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = format(cm_plot[i, j], fmt)
            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color="white" if cm_plot[i, j] > threshold else "#111827",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_roc(outputs: dict, output_path: Path) -> None:
    y_true = outputs["y_true"]
    probs = outputs["probs"]
    labels = list(range(len(DISPLAY_NAMES)))
    label_names = active_label_names()
    y_bin = label_binarize(y_true, classes=labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(9, 8), facecolor="white")
    fpr_grid = np.linspace(0, 1, 300)
    interp_tprs = []
    for idx, color in zip(labels, colors):
        if len(np.unique(y_bin[:, idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, idx], probs[:, idx])
        auc_value = roc_auc_score(y_bin[:, idx], probs[:, idx])
        interp = np.interp(fpr_grid, fpr, tpr)
        interp[0] = 0.0
        interp_tprs.append(interp)
        ax.plot(fpr, tpr, color=color, lw=1.2, alpha=0.75, label=f"{label_names[idx]} ({auc_value:.3f})")
    if interp_tprs:
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[-1] = 1.0
        ax.plot(
            fpr_grid,
            mean_tpr,
            color="#111827",
            lw=2.6,
            label=f"Macro ({np.trapezoid(mean_tpr, fpr_grid):.3f})",
        )
    if y_bin.shape == probs.shape:
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
        ax.plot(fpr_micro, tpr_micro, color="#6B7280", lw=2.0, ls="--", label=f"Micro ({roc_auc_score(y_bin, probs, average='micro'):.3f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1, ls=":")
    ax.set_title("Test One-vs-Rest ROC Curves", fontweight="bold")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pr(outputs: dict, output_path: Path) -> None:
    y_true = outputs["y_true"]
    probs = outputs["probs"]
    labels = list(range(len(DISPLAY_NAMES)))
    label_names = active_label_names()
    y_bin = label_binarize(y_true, classes=labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(9, 8), facecolor="white")
    for idx, color in zip(labels, colors):
        if len(np.unique(y_bin[:, idx])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, idx], probs[:, idx])
        ap = average_precision_score(y_bin[:, idx], probs[:, idx])
        ax.plot(recall, precision, color=color, lw=1.2, alpha=0.75, label=f"{label_names[idx]} ({ap:.3f})")
    ap_macro = safe_metric(lambda: average_precision_score(y_bin, probs, average="macro"))
    ap_micro = safe_metric(lambda: average_precision_score(y_bin, probs, average="micro"))
    ax.text(
        0.03,
        0.05,
        f"Macro AUPRC = {ap_macro:.3f}\nMicro AUPRC = {ap_micro:.3f}",
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#D1D5DB"},
        fontsize=10,
    )
    ax.set_title("Test One-vs-Rest Precision-Recall Curves", fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_bars(per_class_df: pd.DataFrame, output_path: Path) -> None:
    df = per_class_df[per_class_df["split"] == "test"].copy()
    df = df.sort_values("f1")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    height = 0.23
    ax.barh(y - height, df["precision"], height=height, label="Precision", color="#2563EB")
    ax.barh(y, df["recall_sensitivity"], height=height, label="Recall", color="#F59E0B")
    ax.barh(y + height, df["f1"], height=height, label="F1", color="#10B981")
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Score")
    ax.set_title("Test Per-Class Precision, Recall, and F1", fontweight="bold")
    ax.grid(axis="x", alpha=0.22, linestyle="--")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_support(outputs: dict, output_path: Path) -> None:
    label_names = active_label_names()
    counts = pd.Series(outputs["y_true"]).value_counts().reindex(range(len(label_names)), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    ax.bar(label_names, counts.values, color="#4F46E5")
    ax.set_title("Test Class Support", fontweight="bold")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + max(counts.values) * 0.01, str(value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_voc_binary(outputs: dict, output_path: Path) -> None:
    non_voc_label = shared.NON_VOC_LABEL
    y_true = (outputs["y_true"] != non_voc_label).astype(int)
    voc_indices = [idx for idx in range(len(DISPLAY_NAMES)) if idx != non_voc_label]
    voc_score = outputs["probs"][:, voc_indices].sum(axis=1)
    non_voc_score = outputs["probs"][:, non_voc_label]
    y_pred = (voc_score > non_voc_score).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor="white")
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-VOC", "VOC"])
    ax.set_yticklabels(["Non-VOC", "VOC"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("VOC Binary Confusion Matrix", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#111827")

    fpr, tpr, _ = roc_curve(y_true, voc_score)
    axes[1].plot(fpr, tpr, color="#2563EB", lw=2.4, label=f"AUROC {roc_auc_score(y_true, voc_score):.3f}")
    axes[1].plot([0, 1], [0, 1], color="#9CA3AF", lw=1, ls=":")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("VOC-vs-NonVOC ROC", fontweight="bold")
    axes[1].grid(alpha=0.22, linestyle="--")
    axes[1].legend(frameon=False)

    precision, recall, _ = precision_recall_curve(y_true, voc_score)
    axes[2].plot(recall, precision, color="#F59E0B", lw=2.4, label=f"AUPRC {average_precision_score(y_true, voc_score):.3f}")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_title("VOC-vs-NonVOC PR", fontweight="bold")
    axes[2].grid(alpha=0.22, linestyle="--")
    axes[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_training_curves(output_path: Path) -> None:
    history_path = Path(HISTORY_CSV_PATH)
    if not history_path.exists():
        return
    history = pd.read_csv(history_path)
    cfg = load_config(str(PROJECT_ROOT / "config_phase2.json"))
    f1_weight = cfg.get("selection_f1_weight", 0.7)
    auc_weight = cfg.get("selection_auc_weight", 0.3)
    weight_sum = f1_weight + auc_weight
    f1_weight = f1_weight / weight_sum
    auc_weight = auc_weight / weight_sum
    score = f1_weight * history["val_f1"] + auc_weight * history["val_auc"]
    best_idx = score.idxmax()
    best_epoch = int(history.loc[best_idx, "epoch"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="white")
    axes = axes.ravel()
    plots = [
        ("train_loss", "val_loss", "Loss", "upper right"),
        ("train_f1", "val_f1", "Macro F1", "lower right"),
        ("train_acc", "val_acc", "Accuracy", "lower right"),
        ("train_auc", "val_auc", "AUROC", "lower right"),
    ]
    for ax, (train_col, val_col, title, legend_loc) in zip(axes, plots):
        if train_col in history:
            ax.plot(history["epoch"], history[train_col], color="#2563EB", lw=2, label="Train")
        if val_col in history:
            ax.plot(history["epoch"], history[val_col], color="#F59E0B", lw=2, label="Val")
            ax.scatter([best_epoch], [history.loc[best_idx, val_col]], color="#DC2626", zorder=5)
            ax.axvline(best_epoch, color="#DC2626", ls="--", lw=1, alpha=0.55)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.22, linestyle="--")
        ax.legend(loc=legend_loc, frameon=False)
        if title != "Loss":
            ax.set_ylim(0, 1.02)
    fig.suptitle(f"Training Curves (Best validation composite epoch {best_epoch})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_tsne(outputs: dict, output_path: Path, seed: int) -> None:
    if "features" not in outputs:
        return
    features = outputs["features"]
    y_true = outputs["y_true"]
    n = len(y_true)
    perplexity = max(5, min(30, (n - 1) // 3))
    embedded = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)
    label_names = active_label_names()
    colors = plt.cm.tab10(np.linspace(0, 1, len(label_names)))
    fig, ax = plt.subplots(figsize=(9, 8), facecolor="white")
    for idx, color in enumerate(colors):
        mask = y_true == idx
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            s=16,
            alpha=0.72,
            color=color,
            label=label_names[idx],
            edgecolors="none",
        )
    ax.set_title("Test Feature t-SNE", fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.18, linestyle="--")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(output_dir: Path, summary_df: pd.DataFrame, voc_df: pd.DataFrame, ci_df: pd.DataFrame) -> None:
    test = summary_df[summary_df["split"] == "test"].iloc[0]
    voc = voc_df[voc_df["split"] == "test"].iloc[0]
    lines = [
        "# Literature-Aligned Model Evaluation",
        "",
        "Generated from the current `best_model.pth` with the frozen patient-level split.",
        "",
        "## Test Summary",
        "",
        f"- Accuracy: {test['accuracy']:.4f}",
        f"- Balanced accuracy: {test['balanced_accuracy']:.4f}",
        f"- Macro precision / recall / F1: {test['macro_precision']:.4f} / {test['macro_recall_sensitivity']:.4f} / {test['macro_f1']:.4f}",
        f"- Macro AUROC / AUPRC: {test['macro_auroc_ovr']:.4f} / {test['macro_auprc_ovr']:.4f}",
        f"- VOC-vs-NonVOC accuracy / sensitivity / specificity: {voc['accuracy']:.4f} / {voc['recall_sensitivity']:.4f} / {voc['specificity']:.4f}",
        f"- VOC-vs-NonVOC AUROC / AUPRC: {voc['auroc']:.4f} / {voc['auprc']:.4f}",
        "",
        "## Files",
        "",
        "- `summary_metrics.csv`: split-level multiclass metrics.",
        "- `per_class_metrics.csv`: class-level precision, sensitivity, specificity, F1, AUROC, AUPRC.",
        "- `voc_binary_metrics.csv`: binary VOC vs Non-VOC metrics.",
        "- `bootstrap_ci_test.csv`: stratified bootstrap 95% CIs for primary test metrics.",
        "- `predictions_train.csv`, `predictions_val.csv`, `predictions_test.csv`: image-level predictions and probabilities.",
        "- `confusion_matrix_test_counts.png` and `confusion_matrix_test_normalized.png`.",
        "- `roc_curves_test.png` and `precision_recall_curves_test.png`.",
        "- `voc_binary_roc_pr_confusion_test.png`.",
        "- `per_class_metric_bars_test.png` and `support_distribution_test.png`.",
        "- `training_curves_literature_style.png`.",
        "- `tsne_test_features.png` when t-SNE generation is enabled.",
        "- `gradcam_maps_existing.png`: copied from the existing project Grad-CAM output when available.",
        "- `gradcam_class_balanced_test.png`: optional class-balanced random Grad-CAM panel with original image beside each attention map.",
        "",
    ]
    if not ci_df.empty:
        lines.extend(["## Bootstrap CIs", ""])
        for _, row in ci_df.iterrows():
            lines.append(
                f"- {row['metric']}: {row['mean']:.4f} "
                f"[{row['ci_low_2.5']:.4f}, {row['ci_high_97.5']:.4f}]"
            )
        lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(args.config))
    init_label_mapping(cfg)

    device = setup_device()
    df = discover_images()
    train_df, val_df, test_df = load_dataset_split(df)
    _, eval_tf = build_transforms(cfg)
    batch_size = args.batch_size if args.batch_size > 0 else cfg.get("eval_batch_size", 128)
    num_workers = cfg.get("num_workers", 0) if args.num_workers is None else args.num_workers
    prefetch_factor = cfg.get("prefetch_factor", 4) if args.prefetch_factor is None else args.prefetch_factor
    persistent_workers = cfg.get("persistent_workers", True) and not args.no_persistent_workers
    image_cache = None if args.no_image_cache else preload_image_cache(train_df, val_df, test_df, cfg=cfg)
    split_frames = {"train": train_df, "val": val_df, "test": test_df}
    loaders = {
        name: make_loader(
            frame,
            cfg,
            eval_tf,
            batch_size,
            num_workers,
            prefetch_factor,
            persistent_workers,
            image_cache,
        )
        for name, frame in split_frames.items()
    }

    original_create_model = force_timm_no_pretrained()
    try:
        model = HierarchicalImageClassifier(num_classes=len(LABEL_DICT), cfg=cfg).to(device)
    finally:
        restore_timm_create_model(original_create_model)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    outputs_by_split = {}
    for split_name, loader in loaders.items():
        outputs_by_split[split_name] = collect_outputs(
            model,
            loader,
            split_frames[split_name],
            device,
            collect_features=(split_name == "test" and not args.skip_tsne),
        )
        save_predictions(split_name, outputs_by_split[split_name], output_dir)

    num_classes = len(LABEL_DICT)
    summary_df = pd.DataFrame(
        [summary_metrics(split, outputs, num_classes) for split, outputs in outputs_by_split.items()]
    )
    per_class_df = pd.concat(
        [per_class_metrics(split, outputs, num_classes) for split, outputs in outputs_by_split.items()],
        ignore_index=True,
    )
    voc_df = pd.DataFrame([voc_binary_metrics(split, outputs) for split, outputs in outputs_by_split.items()])
    ci_df = bootstrap_ci(outputs_by_split["test"], num_classes, args.bootstrap, args.seed)

    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)
    voc_df.to_csv(output_dir / "voc_binary_metrics.csv", index=False)
    ci_df.to_csv(output_dir / "bootstrap_ci_test.csv", index=False)

    test_outputs = outputs_by_split["test"]
    plot_confusion(test_outputs, output_dir / "confusion_matrix_test_counts.png", normalized=False)
    plot_confusion(test_outputs, output_dir / "confusion_matrix_test_normalized.png", normalized=True)
    plot_roc(test_outputs, output_dir / "roc_curves_test.png")
    plot_pr(test_outputs, output_dir / "precision_recall_curves_test.png")
    plot_per_class_bars(per_class_df, output_dir / "per_class_metric_bars_test.png")
    plot_support(test_outputs, output_dir / "support_distribution_test.png")
    plot_voc_binary(test_outputs, output_dir / "voc_binary_roc_pr_confusion_test.png")
    plot_training_curves(output_dir / "training_curves_literature_style.png")
    if not args.skip_tsne:
        plot_tsne(test_outputs, output_dir / "tsne_test_features.png", args.seed)

    gradcam_path = Path(ATTENTION_MAP_PATH)
    if gradcam_path.exists():
        shutil.copy2(gradcam_path, output_dir / "gradcam_maps_existing.png")

    write_report(output_dir, summary_df, voc_df, ci_df)
    print(f"Saved literature-aligned evaluation to {output_dir}")


if __name__ == "__main__":
    main()
