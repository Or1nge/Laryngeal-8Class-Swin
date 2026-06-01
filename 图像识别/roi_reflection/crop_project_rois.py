#!/usr/bin/env python3
"""Crop local 8-class images with one or more ROI localizer checkpoints."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from common import (
    ROI_RESULTS_DIR,
    bbox_to_str,
    crop_bbox,
    ensure_dir,
    expand_bbox,
    load_rgb,
    load_roi_localizer_model,
    mask_to_bbox,
    normalise_localizer_batch,
    preprocess_localizer_frame,
    resolve_device,
    write_json,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SKIP_TOKENS = ("_seg", "_mask", "segmentation", "mask")


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected name=/path form, got {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Checkpoint name is empty in {value!r}")
    return name, Path(path).expanduser()


def parse_combo(value: str) -> tuple[str, list[tuple[str, float]]]:
    if "=" not in value:
        raise ValueError(f"Expected combo=name1,name2 form, got {value!r}")
    name, labels = value.split("=", 1)
    parsed: list[tuple[str, float]] = []
    for item in labels.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            label, weight = item.split(":", 1)
            parsed.append((label.strip(), float(weight)))
        else:
            parsed.append((item, 1.0))
    if not parsed:
        raise ValueError(f"Combo {name!r} has no checkpoint labels.")
    if sum(weight for _label, weight in parsed) <= 0:
        raise ValueError(f"Combo {name!r} must have positive total weight.")
    return name.strip(), parsed


def read_include_list(include_list: Path, image_root: Path) -> list[Path]:
    include_list = include_list.expanduser().resolve()
    if not include_list.exists():
        raise FileNotFoundError(f"include list does not exist: {include_list}")

    entries: list[str] = []
    if include_list.suffix.lower() == ".csv":
        with include_list.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            path_field = next(
                (
                    name
                    for name in ("image_path", "input_path", "relative_path", "path")
                    if name in fieldnames
                ),
                None,
            )
            if path_field is None:
                raise ValueError(
                    f"{include_list} must contain one of image_path/input_path/relative_path/path columns."
                )
            entries = [str(row.get(path_field, "")).strip() for row in reader]
    else:
        entries = [
            line.strip()
            for line in include_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    paths: list[Path] = []
    missing: list[str] = []
    for entry in entries:
        path = Path(entry).expanduser()
        if not path.is_absolute():
            path = image_root / path
        path = path.resolve()
        if not path.exists():
            missing.append(entry)
            continue
        try:
            path.relative_to(image_root)
        except ValueError as exc:
            raise ValueError(f"Included path is outside --image-root: {path}") from exc
        paths.append(path)

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} included images were not found; first missing: {preview}")

    unique_paths = list(dict.fromkeys(paths))
    if not unique_paths:
        raise RuntimeError(f"No usable paths found in include list: {include_list}")
    return unique_paths


def discover_project_images(
    image_root: Path,
    limit: int | None = None,
    include_list: Path | None = None,
) -> list[Path]:
    if not image_root.exists():
        raise FileNotFoundError(f"image root does not exist: {image_root}")
    if include_list is not None:
        paths = [
            path
            for path in read_include_list(include_list, image_root)
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and not any(token in path.stem.lower() or token in path.parent.name.lower() for token in SKIP_TOKENS)
        ]
    else:
        paths = [
            path
            for path in image_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and not any(token in path.stem.lower() or token in path.parent.name.lower() for token in SKIP_TOKENS)
        ]
        paths = sorted(paths)
    if limit is not None:
        paths = paths[: int(limit)]
    if not paths:
        raise RuntimeError(f"No project images found under {image_root}")
    return paths


def black_border_crop(rgb: np.ndarray, threshold: int) -> np.ndarray:
    bbox = black_border_bbox(rgb, threshold)
    if bbox is None:
        return rgb
    crop = crop_bbox(rgb, bbox)
    return rgb if crop is None else crop


def black_border_bbox(rgb: np.ndarray, threshold: int) -> list[float] | None:
    gray = np.asarray(rgb).max(axis=2)
    coords = np.argwhere(gray > int(threshold))
    if coords.size == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return [float(x0), float(y0), float(x1), float(y1)]


def bbox_area(bbox: list[float] | None) -> float:
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_size(bbox: list[float] | None) -> tuple[float, float]:
    if bbox is None:
        return 0.0, 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def clamp_bbox_to_region(bbox: list[float] | None, region: list[float]) -> list[float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    rx1, ry1, rx2, ry2 = [float(v) for v in region]
    nx1 = max(rx1, min(rx2, x1))
    ny1 = max(ry1, min(ry2, y1))
    nx2 = max(rx1, min(rx2, x2))
    ny2 = max(ry1, min(ry2, y2))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return [nx1, ny1, nx2, ny2]


def centered_bbox_in_region(cx: float, cy: float, width: float, height: float, region: list[float]) -> list[float] | None:
    rx1, ry1, rx2, ry2 = [float(v) for v in region]
    region_w = max(0.0, rx2 - rx1)
    region_h = max(0.0, ry2 - ry1)
    if region_w <= 0 or region_h <= 0:
        return None
    width = min(max(1.0, float(width)), region_w)
    height = min(max(1.0, float(height)), region_h)
    cx = max(rx1, min(rx2, float(cx)))
    cy = max(ry1, min(ry2, float(cy)))
    x1 = cx - width / 2.0
    y1 = cy - height / 2.0
    x2 = x1 + width
    y2 = y1 + height
    if x1 < rx1:
        x2 += rx1 - x1
        x1 = rx1
    if y1 < ry1:
        y2 += ry1 - y1
        y1 = ry1
    if x2 > rx2:
        x1 -= x2 - rx2
        x2 = rx2
    if y2 > ry2:
        y1 -= y2 - ry2
        y2 = ry2
    return [max(rx1, x1), max(ry1, y1), min(rx2, x2), min(ry2, y2)]


def expand_to_min_safe_context(
    raw_bbox: list[float],
    expanded_bbox: list[float],
    effective_bbox: list[float],
    min_width_ratio: float,
    min_height_ratio: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> list[float] | None:
    effective_w, effective_h = bbox_size(effective_bbox)
    effective_area = bbox_area(effective_bbox)
    if effective_w <= 0 or effective_h <= 0 or effective_area <= 0:
        return None

    clamped_expanded = clamp_bbox_to_region(expanded_bbox, effective_bbox)
    if clamped_expanded is None:
        clamped_expanded = effective_bbox

    current_w, current_h = bbox_size(clamped_expanded)
    min_w = effective_w * max(0.0, min(1.0, float(min_width_ratio)))
    min_h = effective_h * max(0.0, min(1.0, float(min_height_ratio)))
    min_area = effective_area * max(0.0, min(1.0, float(min_area_ratio)))
    max_area = effective_area * max(0.0, min(1.0, float(max_area_ratio)))
    if max_area < min_area:
        max_area = min_area

    target_w = min(effective_w, max(current_w, min_w, 1.0))
    target_h = min(effective_h, max(current_h, min_h, 1.0))
    if target_w * target_h < min_area:
        target_w = min(effective_w, max(target_w, min_area / max(target_h, 1.0)))
    if target_w * target_h < min_area:
        target_h = min(effective_h, max(target_h, min_area / max(target_w, 1.0)))
    if target_w * target_h > max_area and max_area >= min_area:
        scale = (max_area / max(target_w * target_h, 1.0)) ** 0.5
        scaled_w = max(min_w, target_w * scale)
        scaled_h = max(min_h, target_h * scale)
        if scaled_w * scaled_h >= min_area:
            target_w = min(effective_w, scaled_w)
            target_h = min(effective_h, scaled_h)

    x1, y1, x2, y2 = [float(v) for v in raw_bbox]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return centered_bbox_in_region(cx, cy, target_w, target_h, effective_bbox)


def crop_safety_metrics(bbox: list[float] | None, effective_bbox: list[float]) -> dict[str, float]:
    crop_w, crop_h = bbox_size(bbox)
    effective_w, effective_h = bbox_size(effective_bbox)
    effective_area = bbox_area(effective_bbox)
    return {
        "crop_width_ratio": float(crop_w / effective_w) if effective_w > 0 else 0.0,
        "crop_height_ratio": float(crop_h / effective_h) if effective_h > 0 else 0.0,
        "crop_area_ratio": float(bbox_area(bbox) / effective_area) if effective_area > 0 else 0.0,
    }


def is_safe_crop(
    bbox: list[float] | None,
    effective_bbox: list[float],
    min_width_ratio: float,
    min_height_ratio: float,
    min_area_ratio: float,
) -> bool:
    metrics = crop_safety_metrics(bbox, effective_bbox)
    return (
        metrics["crop_width_ratio"] + 1e-9 >= float(min_width_ratio)
        and metrics["crop_height_ratio"] + 1e-9 >= float(min_height_ratio)
        and metrics["crop_area_ratio"] + 1e-9 >= float(min_area_ratio)
    )


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(path, quality=95, subsampling=0)
    else:
        image.save(path)


def prob_to_original_tensor(prob: torch.Tensor, cfg: dict[str, Any], width: int, height: int) -> torch.Tensor:
    prob = prob.detach().float()
    if prob.dim() == 3:
        prob = prob[0]
    proc_h, proc_w = prob.shape[-2:]
    resize_mode = str(cfg.get("resize_mode", "resize")).lower()
    if resize_mode in {"letterbox", "pad", "resize_pad"}:
        scale = min(float(proc_w) / max(width, 1), float(proc_h) / max(height, 1))
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        x0 = max(0, (proc_w - new_w) // 2)
        y0 = max(0, (proc_h - new_h) // 2)
        content = prob[y0 : y0 + new_h, x0 : x0 + new_w]
        if content.numel() == 0:
            return torch.zeros((height, width), dtype=torch.float32, device=prob.device)
        restored = F.interpolate(
            content.view(1, 1, content.shape[0], content.shape[1]),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    else:
        restored = F.interpolate(
            prob.view(1, 1, proc_h, proc_w),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    return restored[0, 0]


def prob_to_original(prob: torch.Tensor, cfg: dict[str, Any], width: int, height: int) -> np.ndarray:
    restored = prob_to_original_tensor(prob.detach().float().cpu(), cfg, width, height)
    return restored.numpy().astype(np.float32)


def tensor_mask_to_bbox(mask: torch.Tensor) -> list[float] | None:
    coords = mask.nonzero(as_tuple=False)
    if coords.numel() == 0:
        return None
    y_min = coords[:, 0].min()
    x_min = coords[:, 1].min()
    y_max = coords[:, 0].max() + 1
    x_max = coords[:, 1].max() + 1
    return [float(x_min.item()), float(y_min.item()), float(x_max.item()), float(y_max.item())]


@torch.inference_mode()
def predict_original_probs(
    models: dict[str, torch.nn.Module],
    cfgs: dict[str, dict[str, Any]],
    labels: list[str],
    rgbs: list[np.ndarray],
    device: torch.device,
) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    for label in labels:
        cfg = cfgs[label]
        batch = torch.stack([preprocess_localizer_frame(rgb, cfg) for rgb in rgbs]).to(device, non_blocking=True)
        batch = normalise_localizer_batch(batch)
        if bool(cfg.get("channels_last", False)) and device.type == "cuda":
            batch = batch.contiguous(memory_format=torch.channels_last)
        logits = models[label](batch)
        probs = torch.sigmoid(logits)
        if probs.dim() == 3:
            probs = probs[:, None]
        out[label] = [
            prob_to_original(probs[i, 0], cfg, width=rgb.shape[1], height=rgb.shape[0])
            for i, rgb in enumerate(rgbs)
        ]
    return out


@torch.inference_mode()
def predict_combo_stats_gpu(
    models: dict[str, torch.nn.Module],
    cfgs: dict[str, dict[str, Any]],
    labels: list[str],
    combo: list[tuple[str, float]],
    rgbs: list[np.ndarray],
    device: torch.device,
    threshold: float,
) -> list[dict[str, Any]]:
    total = float(sum(weight for _label, weight in combo))
    weights = {label: float(weight / total) for label, weight in combo}
    combo_probs: list[torch.Tensor | None] = [None] * len(rgbs)
    for label in labels:
        cfg = cfgs[label]
        batch = torch.stack([preprocess_localizer_frame(rgb, cfg) for rgb in rgbs]).to(device, non_blocking=True)
        batch = normalise_localizer_batch(batch)
        if bool(cfg.get("channels_last", False)) and device.type == "cuda":
            batch = batch.contiguous(memory_format=torch.channels_last)
        logits = models[label](batch)
        probs = torch.sigmoid(logits)
        if probs.dim() == 3:
            probs = probs[:, None]
        weight = weights[label]
        for i, rgb in enumerate(rgbs):
            restored = prob_to_original_tensor(probs[i, 0], cfg, width=rgb.shape[1], height=rgb.shape[0])
            weighted = restored * weight
            combo_probs[i] = weighted if combo_probs[i] is None else combo_probs[i] + weighted

    rows: list[dict[str, Any]] = []
    for prob in combo_probs:
        if prob is None or prob.numel() == 0:
            rows.append({"valid_prob": 0.0, "roi_area_ratio": 0.0, "raw_bbox": None})
            continue
        finite = torch.isfinite(prob)
        valid_prob = float(prob[finite].max().item()) if bool(finite.any()) else 0.0
        mask = torch.nan_to_num(prob, nan=float("-inf")) >= float(threshold)
        roi_area_ratio = float(mask.float().mean().item()) if mask.numel() else 0.0
        rows.append(
            {
                "valid_prob": valid_prob,
                "roi_area_ratio": roi_area_ratio,
                "raw_bbox": tensor_mask_to_bbox(mask) if bool(mask.any()) else None,
            }
        )
    return rows


def combo_probability(
    probs_by_label: dict[str, list[np.ndarray]],
    combo: list[tuple[str, float]],
    index: int,
) -> np.ndarray:
    total = float(sum(weight for _label, weight in combo))
    weighted = [probs_by_label[label][index] * float(weight / total) for label, weight in combo]
    return np.stack(weighted, axis=0).sum(axis=0).astype(np.float32)


def resolve_expand_value(value: float | None, cfgs: dict[str, dict[str, Any]], combo: list[tuple[str, float]], key: str, default: float) -> float:
    if value is not None:
        return float(value)
    total = float(sum(weight for _label, weight in combo))
    if total <= 0:
        return float(default)
    return float(sum(float(cfgs[label].get(key, default)) * float(weight / total) for label, weight in combo))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path(os.environ.get("LARYNX_IMAGE_DIR", "/home/or1ngelinux/CVProjects/Larynx/Laryngeal_Dataset_Processed")))
    parser.add_argument("--output-root", type=Path, default=ROI_RESULTS_DIR / "project_roi_crops")
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--checkpoint", action="append", required=True, help="name=/path/to/roi_localizer_best.pth")
    parser.add_argument("--combo", action="append", default=None, help="combo_name=checkpoint_name[:weight],checkpoint_name[:weight]")
    parser.add_argument("--combo-name", default=None, help="Which combo to use when more than one --combo is supplied.")
    parser.add_argument("--include-list", type=Path, default=None, help="Optional CSV/TXT of image_path or relative_path entries to crop.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--postprocess-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Run ROI probability restoration, thresholding, and bbox extraction on CUDA when available; cpu keeps the legacy path.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-roi-area-ratio", type=float, default=0.00025, help="Minimum thresholded ROI mask area ratio before the ROI is considered usable.")
    parser.add_argument("--min-area-ratio", type=float, dest="min_roi_area_ratio", default=argparse.SUPPRESS, help="Deprecated alias for --min-roi-area-ratio.")
    parser.add_argument("--expand-w", type=float, default=None)
    parser.add_argument("--expand-h", type=float, default=None)
    parser.add_argument("--min-crop-width-ratio", type=float, default=0.60, help="Minimum output crop width as a fraction of the black-border valid image width.")
    parser.add_argument("--min-crop-height-ratio", type=float, default=0.60, help="Minimum output crop height as a fraction of the black-border valid image height.")
    parser.add_argument("--min-crop-area-ratio", type=float, default=0.50, help="Minimum output crop area as a fraction of the black-border valid image area.")
    parser.add_argument("--max-crop-area-ratio", type=float, default=1.00, help="Soft maximum output crop area as a fraction of the black-border valid image area.")
    parser.add_argument("--fallback", choices=["original", "black-border-crop"], default="black-border-crop")
    parser.add_argument("--black-border-threshold", type=int, default=15)
    args = parser.parse_args()

    image_root = args.image_root.expanduser().resolve()
    output_root = ensure_dir(args.output_root.expanduser().resolve())
    manifest_csv = args.manifest_csv or (output_root / "manifest.csv")
    checkpoints = dict(parse_named_path(value) for value in args.checkpoint)
    combos = dict(parse_combo(value) for value in args.combo) if args.combo else {}
    if not combos:
        default_name = next(iter(checkpoints)) if len(checkpoints) == 1 else "ensemble"
        combos[default_name] = [(label, 1.0) for label in checkpoints]
    combo_name = args.combo_name or next(iter(combos))
    if combo_name not in combos:
        raise ValueError(f"Unknown combo {combo_name!r}; available combos: {sorted(combos)}")
    combo = combos[combo_name]
    missing = sorted({label for label, _weight in combo if label not in checkpoints})
    if missing:
        raise ValueError(f"Combo {combo_name!r} references unknown checkpoints: {missing}")

    device = resolve_device(args.device)
    if args.postprocess_device == "cuda" and device.type != "cuda":
        raise RuntimeError("--postprocess-device cuda requires --device cuda/auto with CUDA available.")
    use_gpu_postprocess = device.type == "cuda" and args.postprocess_device in {"auto", "cuda"}
    postprocess_device = "cuda" if use_gpu_postprocess else "cpu"
    print(f"Using {postprocess_device} ROI postprocess")
    models: dict[str, torch.nn.Module] = {}
    cfgs: dict[str, dict[str, Any]] = {}
    for label, checkpoint_path in checkpoints.items():
        model, cfg, _ckpt = load_roi_localizer_model(checkpoint_path, device)
        if bool(cfg.get("channels_last", False)) and device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        models[label] = model.eval()
        cfgs[label] = cfg

    expand_w = resolve_expand_value(args.expand_w, cfgs, combo, "roi_expand_w", 1.8)
    expand_h = resolve_expand_value(args.expand_h, cfgs, combo, "roi_expand_h", 1.6)
    paths = discover_project_images(image_root, limit=args.limit, include_list=args.include_list)
    rows: list[dict[str, Any]] = []
    used_labels = sorted({label for label, _weight in combo})

    for start in range(0, len(paths), max(1, int(args.batch_size))):
        batch_paths = paths[start : start + max(1, int(args.batch_size))]
        rgbs = [load_rgb(path) for path in batch_paths]
        if use_gpu_postprocess:
            combo_stats = predict_combo_stats_gpu(models, cfgs, used_labels, combo, rgbs, device, threshold=float(args.threshold))
            probs_by_label = None
        else:
            probs_by_label = predict_original_probs(models, cfgs, used_labels, rgbs, device)
            combo_stats = None
        for idx, (path, rgb) in enumerate(zip(batch_paths, rgbs)):
            rel = path.relative_to(image_root)
            out_path = output_root / rel
            image_h, image_w = rgb.shape[:2]
            effective_bbox = black_border_bbox(rgb, args.black_border_threshold) or [0.0, 0.0, float(image_w), float(image_h)]
            effective_crop = crop_bbox(rgb, effective_bbox)
            if effective_crop is None:
                effective_bbox = [0.0, 0.0, float(image_w), float(image_h)]
                effective_crop = rgb
            if combo_stats is not None:
                stats = combo_stats[idx]
                valid_prob = float(stats["valid_prob"])
                roi_area_ratio = float(stats["roi_area_ratio"])
                raw_bbox = stats["raw_bbox"]
            else:
                assert probs_by_label is not None
                avg_prob = combo_probability(probs_by_label, combo, idx)
                valid_prob = float(np.nanmax(avg_prob)) if avg_prob.size else 0.0
                mask = avg_prob >= float(args.threshold)
                roi_area_ratio = float(mask.mean()) if mask.size else 0.0
                raw_bbox = mask_to_bbox(mask) if mask.size and bool(mask.any()) else None
            expanded_bbox = None
            safe_bbox = None
            safe_crop_status = "fallback_no_roi"
            if raw_bbox is None:
                safe_crop_status = "fallback_no_roi"
            elif roi_area_ratio < float(args.min_roi_area_ratio):
                safe_crop_status = "fallback_small_roi"
            else:
                expanded_bbox = expand_bbox(raw_bbox, image_w, image_h, scale_w=expand_w, scale_h=expand_h)
                if expanded_bbox is not None:
                    safe_bbox = expand_to_min_safe_context(
                        raw_bbox=raw_bbox,
                        expanded_bbox=expanded_bbox,
                        effective_bbox=effective_bbox,
                        min_width_ratio=args.min_crop_width_ratio,
                        min_height_ratio=args.min_crop_height_ratio,
                        min_area_ratio=args.min_crop_area_ratio,
                        max_area_ratio=args.max_crop_area_ratio,
                    )
                if is_safe_crop(
                    safe_bbox,
                    effective_bbox,
                    min_width_ratio=args.min_crop_width_ratio,
                    min_height_ratio=args.min_crop_height_ratio,
                    min_area_ratio=args.min_crop_area_ratio,
                ):
                    safe_crop_status = "safe_roi_crop"
                else:
                    safe_bbox = None
                    safe_crop_status = "fallback_unsafe_crop"

            if safe_bbox is not None:
                output_bbox = safe_bbox
                output_crop = crop_bbox(rgb, output_bbox)
                crop_status = "roi_crop"
            else:
                if args.fallback == "black-border-crop":
                    output_bbox = effective_bbox
                    output_crop = effective_crop
                else:
                    output_bbox = [0.0, 0.0, float(image_w), float(image_h)]
                    output_crop = rgb
                crop_status = f"fallback_{args.fallback.replace('-', '_')}"
            if output_crop is None:
                output_bbox = [0.0, 0.0, float(image_w), float(image_h)]
                output_crop = rgb
                crop_status = "fallback_original"
                safe_crop_status = "fallback_unsafe_crop"
            save_rgb(output_crop, out_path)
            crop_metrics = crop_safety_metrics(output_bbox, effective_bbox)
            output_h, output_w = output_crop.shape[:2]
            rows.append(
                {
                    "input_path": str(path),
                    "output_path": str(out_path),
                    "relative_path": rel.as_posix(),
                    "class_folder": rel.parts[0] if rel.parts else "",
                    "image_width": int(image_w),
                    "image_height": int(image_h),
                    "effective_bbox": bbox_to_str(effective_bbox),
                    "effective_width": int(round(bbox_size(effective_bbox)[0])),
                    "effective_height": int(round(bbox_size(effective_bbox)[1])),
                    "raw_roi_bbox": bbox_to_str(raw_bbox),
                    "expanded_roi_bbox": bbox_to_str(expanded_bbox),
                    "safe_roi_bbox": bbox_to_str(safe_bbox),
                    "bbox": bbox_to_str(output_bbox),
                    "output_width": int(output_w),
                    "output_height": int(output_h),
                    "roi_area_ratio": roi_area_ratio,
                    "crop_width_ratio": crop_metrics["crop_width_ratio"],
                    "crop_height_ratio": crop_metrics["crop_height_ratio"],
                    "crop_area_ratio": crop_metrics["crop_area_ratio"],
                    "valid_prob": valid_prob,
                    "crop_status": crop_status,
                    "safe_crop_status": safe_crop_status,
                    "combo_name": combo_name,
                    "combo": ",".join(f"{label}:{weight:g}" for label, weight in combo),
                    "threshold": float(args.threshold),
                    "min_roi_area_ratio": float(args.min_roi_area_ratio),
                    "min_crop_width_ratio": float(args.min_crop_width_ratio),
                    "min_crop_height_ratio": float(args.min_crop_height_ratio),
                    "min_crop_area_ratio": float(args.min_crop_area_ratio),
                    "max_crop_area_ratio": float(args.max_crop_area_ratio),
                    "fallback": args.fallback,
                }
            )
        print(f"cropped {min(start + len(batch_paths), len(paths))}/{len(paths)}")

    ensure_dir(manifest_csv.parent)
    frame = pd.DataFrame(rows)
    frame.to_csv(manifest_csv, index=False)
    summary = {
        "image_root": str(image_root),
        "output_root": str(output_root),
        "manifest_csv": str(manifest_csv),
        "samples": int(len(frame)),
        "combo_name": combo_name,
        "combo": [{"name": label, "weight": weight, "checkpoint": str(checkpoints[label])} for label, weight in combo],
        "threshold": float(args.threshold),
        "min_roi_area_ratio": float(args.min_roi_area_ratio),
        "min_crop_width_ratio": float(args.min_crop_width_ratio),
        "min_crop_height_ratio": float(args.min_crop_height_ratio),
        "min_crop_area_ratio": float(args.min_crop_area_ratio),
        "max_crop_area_ratio": float(args.max_crop_area_ratio),
        "expand_w": float(expand_w),
        "expand_h": float(expand_h),
        "fallback": args.fallback,
        "postprocess_device": postprocess_device,
        "crop_status_counts": frame["crop_status"].value_counts().to_dict(),
        "safe_crop_status_counts": frame["safe_crop_status"].value_counts().to_dict(),
        "min_observed_crop_width_ratio": float(frame["crop_width_ratio"].min()) if len(frame) else None,
        "min_observed_crop_height_ratio": float(frame["crop_height_ratio"].min()) if len(frame) else None,
        "min_observed_crop_area_ratio": float(frame["crop_area_ratio"].min()) if len(frame) else None,
    }
    write_json(output_root / "summary.json", summary)
    print(f"Wrote crops under {output_root}")
    print(f"Wrote manifest {manifest_csv}")


if __name__ == "__main__":
    main()
