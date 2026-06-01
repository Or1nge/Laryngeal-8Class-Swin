#!/usr/bin/env python3
"""Generate class-balanced Grad-CAM panels with color-coded correctness."""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shared  # noqa: E402
from shared import (  # noqa: E402
    BEST_MODEL_PATH,
    DISPLAY_NAMES,
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
    setup_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config_phase2.json")
    parser.add_argument("--model", type=Path, default=Path(BEST_MODEL_PATH))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(RESULTS_DIR) / "literature_aligned_metrics" / "gradcam_class_balanced_test.png",
    )
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses config eval_batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Defaults to config num_workers.")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Defaults to config prefetch_factor.")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--cam-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
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


def get_target_layers(model):
    last_stage = model.backbone.layers[-1]
    return [last_stage.blocks[-1]]


def reshape_transform(tensor):
    if tensor.dim() == 3:
        batch, hw, channels = tensor.shape
        h = w = int(hw ** 0.5)
        return tensor.reshape(batch, h, w, channels).permute(0, 3, 1, 2)
    if tensor.dim() == 4 and tensor.shape[-1] != tensor.shape[-2]:
        return tensor.permute(0, 3, 1, 2)
    return tensor


def make_loader(
    df: pd.DataFrame,
    cfg: dict,
    eval_tf,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    persistent_workers: bool,
    return_visual: bool = False,
) -> DataLoader:
    dataset = LaryngealDataset(df, eval_tf, cfg, return_visual=return_visual)
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
def collect_predictions(model, loader, dataframe: pd.DataFrame, device) -> pd.DataFrame:
    model.eval()
    rows = []
    paths = dataframe["image_path"].tolist()
    cursor = 0
    amp_context = torch.amp.autocast(device_type=device.type) if device.type == "cuda" else nullcontext()
    for images, labels, _is_voc in loader:
        batch_paths = paths[cursor : cursor + len(labels)]
        cursor += len(labels)
        images = gpu_normalise(images.to(device, non_blocking=True))
        with amp_context:
            logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        confs = probs.max(dim=1).values.cpu().numpy()
        for path, true_id, pred_id, confidence in zip(batch_paths, labels.numpy(), preds, confs):
            rows.append(
                {
                    "image_path": path,
                    "true_id": int(true_id),
                    "pred_id": int(pred_id),
                    "confidence": float(confidence),
                    "correct": int(true_id) == int(pred_id),
                }
            )
    return pd.DataFrame(rows)


def sample_examples(test_df: pd.DataFrame, samples_per_class: int, seed: int) -> pd.DataFrame:
    sampled_parts = []
    for label_id in range(len(DISPLAY_NAMES)):
        class_df = test_df[test_df["label"] == label_id].copy()
        if class_df.empty:
            continue
        n = min(samples_per_class, len(class_df))
        sampled = class_df.sample(n=n, random_state=seed + label_id).reset_index(drop=True)
        sampled["sample_order"] = np.arange(n)
        sampled_parts.append(sampled)
    if not sampled_parts:
        return pd.DataFrame()
    selected = pd.concat(sampled_parts, ignore_index=True)
    return selected.sort_values(["label", "sample_order"]).reset_index(drop=True)


def attach_predictions(selected_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    selected = selected_df.reset_index(drop=True).copy()
    pred_df = pred_df.reset_index(drop=True)
    selected["true_id"] = pred_df["true_id"].astype(int)
    selected["pred_id"] = pred_df["pred_id"].astype(int)
    selected["confidence"] = pred_df["confidence"].astype(float)
    selected["correct"] = pred_df["correct"].astype(bool)
    selected["true_name"] = selected["true_id"].map(DISPLAY_NAMES)
    selected["pred_name"] = selected["pred_id"].map(DISPLAY_NAMES)
    selected["selected_pred_id"] = selected["pred_id"]
    selected["selected_confidence"] = selected["confidence"]
    selected["selected_correct"] = selected["correct"]
    return selected


def build_overlay(original: np.ndarray, cam_map: np.ndarray, image_size: int) -> np.ndarray:
    rgb_img = cv2.resize(original, (image_size, image_size)).astype(np.float32) / 255.0
    cam_resized = cv2.resize(cam_map, (image_size, image_size))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = 0.48 * heatmap + 0.52 * rgb_img
    return np.uint8(255 * np.clip(overlay, 0, 1))


def style_axis(ax, edge_color: str | None = None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(edge_color is not None)
        if edge_color is not None:
            spine.set_linewidth(2.0)
            spine.set_edgecolor(edge_color)


def generate_gradcam_panel(
    model,
    selected_df: pd.DataFrame,
    eval_tf,
    cfg: dict,
    device,
    output: Path,
    cam_batch_size: int,
) -> None:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if selected_df.empty:
        raise RuntimeError("No selected examples for Grad-CAM.")

    dataset = LaryngealDataset(selected_df, eval_tf, cfg, return_visual=True)
    loader = DataLoader(dataset, batch_size=max(1, min(cam_batch_size, len(dataset))), shuffle=False, num_workers=0)
    cam = GradCAM(model=model, target_layers=get_target_layers(model), reshape_transform=reshape_transform)
    grayscale_parts = []
    original_parts = []
    label_parts = []
    cursor = 0
    for images, labels, _is_voc, originals, _paths in loader:
        batch_size = len(labels)
        images = gpu_normalise(images.to(device))
        pred_ids = selected_df.iloc[cursor : cursor + batch_size]["selected_pred_id"].astype(int).tolist()
        cursor += batch_size
        cam_input = images.detach().clone().requires_grad_(True)
        targets = [ClassifierOutputTarget(pred_id) for pred_id in pred_ids]
        grayscale_parts.append(cam(input_tensor=cam_input, targets=targets))
        original_parts.extend([original.numpy().astype(np.uint8) for original in originals])
        label_parts.extend(labels.cpu().tolist())
    cam.activations_and_grads.release()
    grayscale_cams = np.concatenate(grayscale_parts, axis=0)

    max_samples = max(1, int(selected_df.groupby("label").size().max()))
    n_cols = max_samples * 2
    n_rows = len(DISPLAY_NAMES)
    fig_w = 3.2 * n_cols
    fig_h = 3.6 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), facecolor="white")
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    label_ids = list(range(len(DISPLAY_NAMES)))
    row_map = {label_id: row_idx for row_idx, label_id in enumerate(label_ids)}
    used_samples = {label_id: 0 for label_id in label_ids}
    image_size = cfg["image_size"]

    for idx, item in selected_df.iterrows():
        label_id = int(item["label"])
        row_idx = row_map[label_id]
        sample_idx = used_samples[label_id]
        used_samples[label_id] += 1
        original_col = sample_idx * 2
        cam_col = original_col + 1

        original = original_parts[idx]
        original_show = cv2.resize(original, (image_size, image_size))
        overlay = build_overlay(original, grayscale_cams[idx], image_size)
        correct = bool(item["selected_correct"])
        title_color = "#16A34A" if correct else "#DC2626"
        title_prefix = "OK" if correct else "ERR"
        true_name = DISPLAY_NAMES[int(label_parts[idx])]
        pred_name = DISPLAY_NAMES[int(item["selected_pred_id"])]
        confidence = float(item["selected_confidence"])

        axes[row_idx, original_col].imshow(original_show)
        axes[row_idx, original_col].set_title("Original", color="#374151", fontsize=9, fontweight="bold")
        style_axis(axes[row_idx, original_col], "#D1D5DB")

        ax = axes[row_idx, cam_col]
        ax.imshow(overlay)
        ax.set_title(
            f"{title_prefix}  T:{true_name}\nP:{pred_name}  conf={confidence:.3f}",
            color=title_color,
            fontsize=10,
            fontweight="bold",
        )
        style_axis(ax, title_color)

    for label_id, row_idx in row_map.items():
        axes[row_idx, 0].set_ylabel(DISPLAY_NAMES[label_id], fontsize=11, fontweight="bold")
        for sample_idx in range(used_samples[label_id], max_samples):
            axes[row_idx, sample_idx * 2].axis("off")
            axes[row_idx, sample_idx * 2 + 1].axis("off")

    fig.suptitle(
        "Class-Balanced Random Test Grad-CAM (green = correct, red = wrong)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(str(args.config))
    init_label_mapping(cfg)
    device = setup_device()

    df = discover_images()
    _train_df, _val_df, test_df = load_dataset_split(df)
    _, eval_tf = build_transforms(cfg)
    batch_size = args.batch_size if args.batch_size > 0 else cfg.get("eval_batch_size", 128)
    num_workers = cfg.get("num_workers", 0) if args.num_workers is None else args.num_workers
    prefetch_factor = cfg.get("prefetch_factor", 4) if args.prefetch_factor is None else args.prefetch_factor
    persistent_workers = cfg.get("persistent_workers", True) and not args.no_persistent_workers

    original_create_model = force_timm_no_pretrained()
    try:
        model = HierarchicalImageClassifier(num_classes=len(LABEL_DICT), cfg=cfg).to(device)
    finally:
        restore_timm_create_model(original_create_model)
    model.load_state_dict(torch.load(args.model, map_location=device), strict=True)
    model.eval()

    selected_samples = sample_examples(test_df, args.samples_per_class, args.seed)
    prediction_loader = make_loader(
        selected_samples,
        cfg,
        eval_tf,
        batch_size,
        num_workers,
        prefetch_factor,
        persistent_workers,
    )
    pred_df = collect_predictions(model, prediction_loader, selected_samples, device)
    selected = attach_predictions(selected_samples, pred_df)
    selected.to_csv(args.output.with_suffix(".csv"), index=False)
    generate_gradcam_panel(model, selected, eval_tf, cfg, device, args.output, args.cam_batch_size)
    print(f"Saved Grad-CAM panel: {args.output}")
    print(f"Saved selection manifest: {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
