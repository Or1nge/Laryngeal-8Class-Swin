#!/usr/bin/env python3
"""Export offline ROI sidecar scores for the local 8-class image project."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    ROI_RESULTS_DIR,
    bbox_to_str,
    crop_bbox,
    expand_bbox,
    load_rgb,
    load_roi_localizer_model,
    load_roi_reflection_model,
    mask_to_bbox,
    normalise_localizer_batch,
    normalise_reflection_batch,
    parse_bbox,
    preprocess_localizer_frame,
    preprocess_reflection_roi,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SKIP_TOKENS = ("_seg", "_mask", "segmentation", "mask")


def discover_project_images(image_root: Path, limit: int | None = None) -> list[Path]:
    if not image_root.exists():
        raise FileNotFoundError(f"image_root does not exist: {image_root}")
    paths = [
        p for p in image_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not any(token in p.stem.lower() or token in p.parent.name.lower() for token in SKIP_TOKENS)
    ]
    paths = sorted(paths)
    paths = paths[:limit] if limit else paths
    if not paths:
        raise RuntimeError(f"No project images found under {image_root}")
    return paths


@torch.inference_mode()
def localize_batch(model, cfg, rgbs: list[np.ndarray], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    tensors = [preprocess_localizer_frame(rgb, cfg) for rgb in rgbs]
    batch = normalise_localizer_batch(torch.stack(tensors).to(device))
    logits = model(batch)
    probs = torch.sigmoid(logits[:, 0]).cpu().numpy()
    bboxes = []
    valid_probs = []
    for prob, rgb in zip(probs, rgbs):
        box = mask_to_bbox(prob > 0.5)
        if box is None:
            bboxes.append([np.nan, np.nan, np.nan, np.nan])
            valid_probs.append(float(np.nanmax(prob)))
            continue
        h, w = rgb.shape[:2]
        scale_x = w / prob.shape[1]
        scale_y = h / prob.shape[0]
        scaled = [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y]
        bboxes.append(expand_bbox(scaled, w, h, scale_w=float(cfg.get("roi_expand_w", 1.8)), scale_h=float(cfg.get("roi_expand_h", 1.6))) or [np.nan, np.nan, np.nan, np.nan])
        valid_probs.append(float(np.nanmax(prob)))
    return np.asarray(bboxes, dtype=np.float32), np.asarray(valid_probs, dtype=np.float32)


@torch.inference_mode()
def reflect_batch(model, cfg, crops: list[np.ndarray], device: torch.device) -> np.ndarray:
    if not crops:
        return np.asarray([], dtype=np.float32)
    tensors = [preprocess_reflection_roi(crop, cfg) for crop in crops]
    batch = normalise_reflection_batch(torch.stack(tensors).to(device))
    out = model(batch)
    return out["reflect_prob"].detach().cpu().numpy().astype(np.float32)


def bucket(valid_prob: float, reflect_prob: float, valid_thr: float, reflect_thr: float, severe_thr: float) -> tuple[str, str, float]:
    if not np.isfinite(valid_prob) or valid_prob < valid_thr:
        return "invalid_roi", "low_valid", 0.0
    if not np.isfinite(reflect_prob):
        return "missing", "unknown", 1.0
    if reflect_prob >= severe_thr:
        return "severe", "severe_reflection", 0.35
    if reflect_prob >= reflect_thr:
        return "mild", "reflection", max(0.35, 1.0 - reflect_prob / max(severe_thr, 1e-6))
    return "none", "clean", 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path(os.environ.get("LARYNX_IMAGE_DIR", "/home/or1ngelinux/CVProjects/Larynx/Laryngeal_Dataset_Processed")))
    parser.add_argument("--localizer-checkpoint", type=Path, required=True)
    parser.add_argument("--reflection-checkpoint", type=Path, required=True)
    parser.add_argument("--scores-csv", type=Path, default=ROI_RESULTS_DIR / "roi_scores.csv")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--valid-threshold", type=float, default=0.55)
    parser.add_argument("--reflect-threshold", type=float, default=0.65)
    parser.add_argument("--severe-reflect-threshold", type=float, default=0.85)
    args = parser.parse_args()

    device = torch.device("cuda" if args.cache_device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu")
    localizer, localizer_cfg, _ = load_roi_localizer_model(args.localizer_checkpoint, device)
    reflector, reflector_cfg, _ = load_roi_reflection_model(args.reflection_checkpoint, device)
    paths = discover_project_images(args.image_root.expanduser(), limit=args.limit)
    rows = []
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        rgbs = [load_rgb(path) for path in batch_paths]
        bboxes, valid_probs = localize_batch(localizer, localizer_cfg, rgbs, device)
        crops, crop_map = [], []
        for idx, (rgb, bbox) in enumerate(zip(rgbs, bboxes)):
            crop = crop_bbox(rgb, bbox)
            if crop is not None and parse_bbox(bbox) is not None:
                crops.append(crop)
                crop_map.append(idx)
        reflect_probs = np.full(len(batch_paths), np.nan, dtype=np.float32)
        if crops:
            values = reflect_batch(reflector, reflector_cfg, crops, device)
            reflect_probs[crop_map] = values
        for path, bbox, valid_prob, reflect_prob in zip(batch_paths, bboxes, valid_probs, reflect_probs):
            severity, roi_bucket, weight = bucket(
                float(valid_prob),
                float(reflect_prob),
                args.valid_threshold,
                args.reflect_threshold,
                args.severe_reflect_threshold,
            )
            rows.append(
                {
                    "image_path": str(path),
                    "roi_bbox_xyxy": bbox_to_str(bbox if np.isfinite(bbox).all() else None),
                    "roi_valid_prob": float(valid_prob),
                    "reflect_prob": float(reflect_prob) if np.isfinite(reflect_prob) else np.nan,
                    "reflect_severity": severity,
                    "roi_bucket": roi_bucket,
                    "roi_weight": float(weight),
                    "score_source": "bagls_roi_reflection_v1",
                    "localizer_checkpoint": str(args.localizer_checkpoint),
                    "reflection_checkpoint": str(args.reflection_checkpoint),
                }
            )
        print(f"scored {min(start + len(batch_paths), len(paths))}/{len(paths)}")
    args.scores_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.scores_csv, index=False)
    print(f"Wrote {args.scores_csv}")


if __name__ == "__main__":
    main()
