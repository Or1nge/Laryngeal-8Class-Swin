#!/usr/bin/env python3
"""Evaluate ROI localizer or reflection gate checkpoints on a BAGLS manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import (
    ROI_RESULTS_DIR,
    BaglsSegmentationDataset,
    ReflectionDataset,
    binary_metrics,
    dice_bce_loss,
    dice_score_from_logits,
    dice_scores_from_logits,
    ensure_dir,
    iter_batches,
    localizer_loss,
    load_roi_localizer_model,
    load_roi_reflection_model,
    normalise_localizer_batch,
    normalise_reflection_batch,
    parse_bbox,
    resolve_device,
    threshold_table,
    write_json,
)


def _mask_bbox_str(mask: np.ndarray) -> str:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return ""
    return f"{int(xs.min())},{int(ys.min())},{int(xs.max() + 1)},{int(ys.max() + 1)}"


def _overlay_image(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    image_f = image.astype(np.float32)
    overlay = image_f.copy()
    gt_mask = gt > 0
    pred_mask = pred > 0
    fp = pred_mask & ~gt_mask
    fn = gt_mask & ~pred_mask
    tp = pred_mask & gt_mask
    overlay[tp] = 0.55 * overlay[tp] + 0.45 * np.array([0, 220, 0], dtype=np.float32)
    overlay[fp] = 0.55 * overlay[fp] + 0.45 * np.array([255, 0, 0], dtype=np.float32)
    overlay[fn] = 0.55 * overlay[fn] + 0.45 * np.array([0, 80, 255], dtype=np.float32)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


@torch.inference_mode()
def eval_localizer(
    checkpoint: Path,
    manifest: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
    cache_device: torch.device | None = None,
    save_worst_overlays: int = 100,
) -> None:
    model, cfg, _ckpt = load_roi_localizer_model(checkpoint, device)
    channels_last = bool(cfg.get("channels_last", False)) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    rows = []
    for split in ("train", "val", "test"):
        ds = BaglsSegmentationDataset(manifest, cfg, split=split, cache_device=cache_device)
        if len(ds) == 0:
            continue
        losses, dices, pred_rows = [], [], []
        worst: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray]] = []
        offset = 0
        for images, masks in iter_batches(
            ds,
            int(cfg.get("eval_batch_size", 64)),
            shuffle=False,
            device=device,
            augment=False,
            channels_last=channels_last,
        ):
            raw_images = images.detach().cpu()
            images = normalise_localizer_batch(images)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            logits = model(images)
            losses.append(float(localizer_loss(logits, masks, cfg).cpu()))
            dices.append(float(dice_score_from_logits(logits, masks).cpu()))
            probs = torch.sigmoid(logits)
            pred = probs >= float(cfg.get("prediction_threshold", 0.5))
            dice_values = dice_scores_from_logits(logits, masks).detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy().astype(np.uint8)
            mask_np = (masks.detach().cpu().numpy() > 0.5).astype(np.uint8)
            probs_np = probs.detach().cpu().numpy()
            batch_size = int(mask_np.shape[0])
            for j in range(batch_size):
                row = ds.frame.iloc[offset + j]
                gt = mask_np[j, 0]
                pr = pred_np[j, 0]
                gt_area = float(gt.mean())
                pred_area = float(pr.mean())
                fp = int(((pr == 1) & (gt == 0)).sum())
                fn = int(((pr == 0) & (gt == 1)).sum())
                pred_rows.append(
                    {
                        "image_path": row.get("image_path", ""),
                        "mask_path": row.get("mask_path", ""),
                        "width": row.get("width", np.nan),
                        "height": row.get("height", np.nan),
                        "glottis_area_ratio": row.get("glottis_area_ratio", np.nan),
                        "reflection_label": row.get("reflection_label", np.nan),
                        "per_sample_dice": float(dice_values[j]),
                        "pred_area_ratio": pred_area,
                        "gt_area_ratio": gt_area,
                        "prob_area_mean": float(probs_np[j, 0].mean()),
                        "fp": fp,
                        "fn": fn,
                        "empty_gt": bool(gt_area <= float(cfg.get("empty_area_threshold", 1e-8))),
                        "empty_false_positive": bool(gt_area <= float(cfg.get("empty_area_threshold", 1e-8)) and pred_area > float(cfg.get("empty_pred_area_threshold", 0.001))),
                        "pred_bbox": _mask_bbox_str(pr),
                        "gt_bbox": _mask_bbox_str(gt),
                        "manifest_gt_bbox": row.get("roi_bbox_xyxy", ""),
                    }
                )
                if split == "test" and save_worst_overlays > 0:
                    image_np = np.clip(raw_images[j].permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
                    worst.append((float(dice_values[j]), offset + j, image_np, gt.copy(), pr.copy()))
            offset += batch_size
        pred_df = pd.DataFrame(pred_rows)
        pred_df.to_csv(output_dir / f"localizer_predictions_{split}.csv", index=False)
        empty = pred_df[pred_df["empty_gt"] == True]
        rows.append(
            {
                "split": split,
                "loss": float(np.mean(losses)),
                "dice": float(np.mean(dices)),
                "samples": len(ds),
                "mean_per_sample_dice": float(pred_df["per_sample_dice"].mean()),
                "nonempty_mean_dice": float(pred_df.loc[pred_df["empty_gt"] == False, "per_sample_dice"].mean()),
                "empty_count": int(len(empty)),
                "empty_false_positive_count": int(empty["empty_false_positive"].sum()) if len(empty) else 0,
                "empty_false_positive_rate": float(empty["empty_false_positive"].mean()) if len(empty) else 0.0,
                "pred_area_ratio_mean": float(pred_df["pred_area_ratio"].mean()),
                "gt_area_ratio_mean": float(pred_df["gt_area_ratio"].mean()),
            }
        )
        if split == "test" and save_worst_overlays > 0 and worst:
            overlay_dir = output_dir / "worst_test_overlays"
            overlay_dir.mkdir(parents=True, exist_ok=True)
            for rank, (dice, idx, image_np, gt, pr) in enumerate(sorted(worst, key=lambda item: item[0])[:save_worst_overlays], start=1):
                _overlay_image(image_np, gt, pr).save(overlay_dir / f"{rank:03d}_idx{idx}_dice{dice:.4f}.png")
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    write_json(output_dir / "metrics.json", {"rows": rows})


@torch.inference_mode()
def eval_reflection(checkpoint: Path, manifest: pd.DataFrame, output_dir: Path, device: torch.device) -> None:
    model, cfg, _ckpt = load_roi_reflection_model(checkpoint, device)
    metric_rows = []
    for split in ("train", "val", "test"):
        ds = ReflectionDataset(manifest, cfg, split=split, cache_device=None)
        if len(ds) == 0:
            continue
        y_valid, y_reflect, p_valid, p_reflect = [], [], [], []
        for images, valid, reflect in iter_batches(ds, int(cfg.get("eval_batch_size", 128)), shuffle=False):
            out = model(normalise_reflection_batch(images.to(device)))
            y_valid.extend(valid.numpy())
            y_reflect.extend(reflect.numpy())
            p_valid.extend(out["roi_valid_prob"].cpu().numpy())
            p_reflect.extend(out["reflect_prob"].cpu().numpy())
        pred = pd.DataFrame({"y_valid": y_valid, "y_reflect": y_reflect, "p_valid": p_valid, "p_reflect": p_reflect})
        pred.to_csv(output_dir / f"predictions_{split}.csv", index=False)
        threshold_table(np.asarray(y_reflect), np.asarray(p_reflect), prefix="reflect_").to_csv(
            output_dir / f"threshold_metrics_reflect_{split}.csv",
            index=False,
        )
        metric_rows.append({"split": split, **binary_metrics(np.asarray(y_valid), np.asarray(p_valid), threshold=float(cfg.get("valid_threshold", 0.55)), prefix="valid_")})
        metric_rows.append({"split": split, **binary_metrics(np.asarray(y_reflect), np.asarray(p_reflect), threshold=float(cfg.get("reflect_threshold", 0.65)), prefix="reflect_")})
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROI_RESULTS_DIR / "eval")
    parser.add_argument("--task", choices=["auto", "localizer", "reflection"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--save-worst-overlays", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split row cap for smoke checks only.")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    manifest = pd.read_csv(args.manifest)
    if args.limit:
        manifest = manifest.groupby("split", group_keys=False).head(int(args.limit)).copy()
    device = resolve_device(args.device)
    cache_device = resolve_device(args.cache_device)
    if cache_device.type == "cpu":
        cache_device = None
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    task = args.task
    if task == "auto":
        task = str(ckpt.get("role", "reflection"))
    if task == "localizer":
        eval_localizer(args.checkpoint, manifest, output_dir, device, cache_device=cache_device, save_worst_overlays=int(args.save_worst_overlays))
    else:
        eval_reflection(args.checkpoint, manifest, output_dir, device)
    print(f"Evaluation written to {output_dir}")


if __name__ == "__main__":
    main()
