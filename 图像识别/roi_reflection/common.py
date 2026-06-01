"""Shared utilities for BAGLS-driven ROI validity and reflection gating."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    import cv2
except Exception:  # pragma: no cover - optional acceleration
    cv2 = None

try:
    import timm
except Exception:  # pragma: no cover - fallback model still works
    timm = None


MODULE_DIR = Path(__file__).resolve().parent
IMAGE_PROJECT_DIR = MODULE_DIR.parent
WORKTREE_DIR = IMAGE_PROJECT_DIR.parent


def detect_workspace_dir(worktree_dir: Path) -> Path:
    if worktree_dir.parent.name == "worktrees":
        return worktree_dir.parent.parent
    return worktree_dir


WORKSPACE_DIR = Path(os.environ.get("LARYNX_WORKSPACE_DIR", detect_workspace_dir(WORKTREE_DIR))).resolve()
WORKTREE_NAME = os.environ.get("LARYNX_WORKTREE_NAME", WORKTREE_DIR.name)
RESULTS_ROOT = Path(os.environ.get("LARYNX_RESULTS_ROOT", WORKSPACE_DIR / "Results")).resolve()
RESULTS_DIR = Path(os.environ.get("LARYNX_RESULTS_DIR", RESULTS_ROOT / WORKTREE_NAME)).resolve()
ROI_RESULTS_DIR = RESULTS_DIR / "roi_reflection"
DEFAULT_BAGLS_ROOT = Path(os.environ.get("BAGLS_ROOT", "/mnt/data/LarynxData/BAGLS")).expanduser()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_TOKENS = ("mask", "seg", "segmentation", "label")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def default_localizer_config() -> dict[str, Any]:
    return {
        "seed": 42,
        "image_size": 256,
        "resize_mode": "resize",
        "letterbox_pad_value": 0,
        "random_scale_pad_prob": 0.0,
        "random_scale_pad_range": [0.85, 1.15],
        "batch_size": 32,
        "eval_batch_size": 64,
        "epochs": 30,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "dice_weight": 0.5,
        "bce_weight": 0.5,
        "tversky_weight": 0.0,
        "tversky_alpha": 0.35,
        "tversky_beta": 0.65,
        "focal_bce_gamma": 0.0,
        "empty_area_threshold": 1e-8,
        "empty_pred_area_threshold": 0.001,
        "area_aware_sampling": False,
        "area_bin_edges": [0.0, 1e-8, 0.001, 0.005, 0.02, 0.08, 1.0],
        "empty_sample_weight": 2.0,
        "large_area_sample_weight": 2.0,
        "sampling_replacement": True,
        "selection_metric": "val_dice",
        "selection_empty_fp_weight": 0.5,
        "selection_area_mae_weight": 0.0,
        "early_stopping_patience": 8,
        "early_stopping_min_delta": 0.001,
        "cache_device": "auto",
        "num_workers": 0,
        "model_arch": "timm_unet",
        "backbone": "swin_tiny_patch4_window7_224",
        "pretrained": True,
        "model_kwargs": {},
        "channels_last": True,
        "compile_model": False,
        "batched_gpu_augment": True,
        "base_channels": 32,
        "decoder_channels": [256, 128, 64, 48],
        "dropout": 0.05,
        "augment_train": True,
        "horizontal_flip_prob": 0.5,
        "affine_prob": 0.65,
        "affine_degrees": 8.0,
        "affine_translate": [0.05, 0.05],
        "affine_scale": [0.92, 1.08],
        "gamma_prob": 0.35,
        "gamma_range": [0.85, 1.15],
        "brightness_prob": 0.35,
        "brightness_range": [0.9, 1.1],
        "contrast_prob": 0.35,
        "contrast_range": [0.9, 1.1],
        "blur_prob": 0.12,
        "blur_kernel_size": 3,
        "blur_sigma": [0.1, 0.8],
        "noise_prob": 0.20,
        "noise_std": 0.015,
        "roi_expand_w": 1.8,
        "roi_expand_h": 1.6,
    }


def default_reflection_config() -> dict[str, Any]:
    return {
        "seed": 42,
        "image_size": 224,
        "batch_size": 256,
        "eval_batch_size": 512,
        "epochs": 40,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "early_stopping_patience": 8,
        "cache_device": "auto",
        "backbone": "mobilenetv3_small_100",
        "pretrained": True,
        "focal_gamma": 2.0,
        "valid_loss_weight": 0.5,
        "reflect_loss_weight": 1.0,
        "valid_threshold": 0.55,
        "reflect_threshold": 0.65,
        "severe_reflect_threshold": 0.85,
        "soft_weight_min": 0.35,
    }


def load_json_config(path: Path | str | None, defaults: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(defaults)
    if path:
        with Path(path).open("r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(value: str | None = "auto") -> torch.device:
    value = str(value or "auto").lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device(value)


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamped_run_dir(root: Path | str, suffix: str) -> Path:
    base = ensure_dir(root) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
    if not base.exists():
        base.mkdir(parents=True)
        return base
    for idx in range(2, 1000):
        candidate = base.with_name(f"{base.name}_{idx}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise FileExistsError(f"Could not allocate run dir under {root}")


def write_json(path: Path | str, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def git_provenance() -> dict[str, Any]:
    def run(args: list[str], default: str = "unknown") -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=WORKTREE_DIR,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            return result.stdout.strip() or default
        except Exception:
            return default

    dirty = run(["status", "--porcelain"], default="")
    return {
        "branch": run(["branch", "--show-current"], default="detached"),
        "commit": run(["rev-parse", "--short", "HEAD"]),
        "full_commit": run(["rev-parse", "HEAD"]),
        "dirty": bool(dirty),
    }


def save_run_provenance(run_dir: Path, cfg: dict[str, Any], command: list[str] | None = None) -> None:
    write_json(
        run_dir / "provenance.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "worktree_dir": WORKTREE_DIR,
            "results_dir": RESULTS_DIR,
            "git": git_provenance(),
            "command": command or sys.argv,
            "cfg": cfg,
        },
    )


def copy_notice(run_dir: Path) -> None:
    notice = MODULE_DIR / "NOTICE"
    if notice.exists():
        shutil.copy2(notice, run_dir / "NOTICE")


def load_rgb(path: Path | str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(path: Path | str) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return (mask > 0).astype(np.float32)


def resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.BICUBIC), dtype=np.uint8)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray((mask > 0).astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 0


def letterbox_rgb(rgb: np.ndarray, size: int, pad_value: int = 0) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    width, height = image.size
    scale = min(float(size) / max(width, 1), float(size) / max(height, 1))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), color=(int(pad_value), int(pad_value), int(pad_value)))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def letterbox_mask(mask: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
    width, height = image.size
    scale = min(float(size) / max(width, 1), float(size) / max(height, 1))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), Image.NEAREST)
    canvas = Image.new("L", (size, size), color=0)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return np.asarray(canvas, dtype=np.uint8) > 0


def preprocess_localizer_rgb(rgb: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    size = int(cfg.get("image_size", 256))
    if str(cfg.get("resize_mode", "resize")).lower() in {"letterbox", "pad", "resize_pad"}:
        return letterbox_rgb(rgb, size, pad_value=int(cfg.get("letterbox_pad_value", 0)))
    return resize_rgb(rgb, size)


def preprocess_localizer_mask(mask: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    size = int(cfg.get("image_size", 256))
    if str(cfg.get("resize_mode", "resize")).lower() in {"letterbox", "pad", "resize_pad"}:
        return letterbox_mask(mask, size)
    return resize_mask(mask, size)


def chw_float_tensor(rgb: np.ndarray) -> torch.Tensor:
    arr = np.asarray(rgb, dtype=np.uint8)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255.0)


def normalise_batch(batch: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=batch.device, dtype=batch.dtype)
    std = IMAGENET_STD.to(device=batch.device, dtype=batch.dtype)
    return (batch - mean) / std


def preprocess_localizer_frame(rgb: np.ndarray | Image.Image, cfg: dict[str, Any] | None = None) -> torch.Tensor:
    cfg = cfg or default_localizer_config()
    arr = np.asarray(rgb.convert("RGB") if isinstance(rgb, Image.Image) else rgb, dtype=np.uint8)
    return chw_float_tensor(preprocess_localizer_rgb(arr, cfg))


def preprocess_reflection_roi(rgb: np.ndarray | Image.Image, cfg: dict[str, Any] | None = None) -> torch.Tensor:
    cfg = cfg or default_reflection_config()
    arr = np.asarray(rgb.convert("RGB") if isinstance(rgb, Image.Image) else rgb, dtype=np.uint8)
    return chw_float_tensor(resize_rgb(arr, int(cfg.get("image_size", 224))))


def normalise_localizer_batch(batch: torch.Tensor) -> torch.Tensor:
    return normalise_batch(batch)


def normalise_reflection_batch(batch: torch.Tensor) -> torch.Tensor:
    return normalise_batch(batch)


def mask_to_bbox(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def expand_bbox(
    bbox: Iterable[float] | None,
    width: int,
    height: int,
    scale_w: float = 1.8,
    scale_h: float = 1.6,
) -> list[float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, x2 - x1) * float(scale_w)
    bh = max(1.0, y2 - y1) * float(scale_h)
    nx1 = max(0.0, cx - bw / 2.0)
    ny1 = max(0.0, cy - bh / 2.0)
    nx2 = min(float(width), cx + bw / 2.0)
    ny2 = min(float(height), cy + bh / 2.0)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return [nx1, ny1, nx2, ny2]


def bbox_to_str(bbox: Iterable[float] | None) -> str:
    if bbox is None:
        return ""
    return ",".join(f"{float(v):.2f}" for v in bbox)


def parse_bbox(value: str | Iterable[float] | float | None) -> list[float] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    else:
        parts = list(value)
    if len(parts) != 4:
        return None
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    if not np.isfinite(vals).all() or vals[2] <= vals[0] or vals[3] <= vals[1]:
        return None
    return vals


def crop_bbox(rgb: np.ndarray, bbox: Iterable[float] | None) -> np.ndarray | None:
    bbox = parse_bbox(bbox)
    if bbox is None:
        return None
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    left = max(0, min(w - 1, int(math.floor(x1))))
    top = max(0, min(h - 1, int(math.floor(y1))))
    right = max(left + 1, min(w, int(math.ceil(x2))))
    bottom = max(top + 1, min(h, int(math.ceil(y2))))
    return rgb[top:bottom, left:right]


def _largest_component_ratio(mask: np.ndarray) -> float:
    binary = np.asarray(mask, dtype=np.uint8)
    total = int(binary.sum())
    if total == 0:
        return 0.0
    if cv2 is not None:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            return 0.0
        largest = int(stats[1:, cv2.CC_STAT_AREA].max())
        return largest / max(total, 1)

    visited = np.zeros(binary.shape, dtype=bool)
    best = 0
    h, w = binary.shape
    for y, x in zip(*np.where(binary > 0)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        area = 0
        while stack:
            cy, cx = stack.pop()
            area += 1
            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        best = max(best, area)
    return best / max(total, 1)


def specular_seed_metrics(
    roi_rgb: np.ndarray,
    value_thr: int = 245,
    sat_thr: int = 30,
    min_cc_area: int = 24,
) -> dict[str, Any]:
    if roi_rgb is None or roi_rgb.size == 0:
        return {
            "specular_area_ratio": 0.0,
            "largest_specular_cc_ratio": 0.0,
            "white_ratio": 0.0,
            "sharpness": 0.0,
            "seed_pixels": 0,
        }
    if cv2 is not None:
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        seed = ((hsv[:, :, 2] > value_thr) & (hsv[:, :, 1] < sat_thr)).astype(np.uint8)
        if seed.any():
            num, labels, stats, _ = cv2.connectedComponentsWithStats(seed, connectivity=8)
            clean = np.zeros_like(seed)
            for idx in range(1, num):
                if int(stats[idx, cv2.CC_STAT_AREA]) >= int(min_cc_area):
                    clean[labels == idx] = 1
            seed = clean
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        white_ratio = float((gray > value_thr).mean())
    else:
        arr = roi_rgb.astype(np.float32)
        maxc = arr.max(axis=2)
        minc = arr.min(axis=2)
        sat = np.where(maxc > 0, (maxc - minc) / maxc * 255.0, 0.0)
        gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
        seed = ((maxc > value_thr) & (sat < sat_thr)).astype(np.uint8)
        sharpness = float(np.var(np.gradient(gray)[0]) + np.var(np.gradient(gray)[1]))
        white_ratio = float((gray > value_thr).mean())
    area = float(seed.mean())
    return {
        "specular_area_ratio": area,
        "largest_specular_cc_ratio": float(_largest_component_ratio(seed) * area),
        "white_ratio": white_ratio,
        "sharpness": sharpness,
        "seed_pixels": int(seed.sum()),
    }


def reflection_label_from_metrics(metrics: dict[str, Any]) -> tuple[int, str, str]:
    area = float(metrics.get("specular_area_ratio", 0.0))
    largest = float(metrics.get("largest_specular_cc_ratio", 0.0))
    white = float(metrics.get("white_ratio", 0.0))
    if area >= 0.08 or largest >= 0.05 or white >= 0.18:
        return 1, "severe", "heuristic"
    if area >= 0.025 or largest >= 0.015 or white >= 0.08:
        return 1, "mild", "heuristic"
    return 0, "none", "heuristic"


def infer_recording_id(path: Path, meta: dict[str, Any] | None = None) -> str:
    meta = meta or {}
    for key in ("recording_id", "video_id", "sequence_id", "hsv_id", "case_id"):
        if meta.get(key):
            return str(meta[key])
    match = re.match(r"([A-Za-z]*\d+)", path.stem)
    if match:
        return match.group(1)
    return path.parent.name


def read_meta(path: Path) -> dict[str, Any]:
    candidates = [
        path.with_suffix(".json"),
        path.with_suffix(".meta"),
        path.with_name(path.stem + ".json"),
        path.with_name(path.stem + ".meta"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _is_mask_like(path: Path) -> bool:
    stem = path.stem.lower()
    parent = path.parent.name.lower()
    return any(token in stem or token in parent for token in MASK_TOKENS)


def _mask_candidates(image_path: Path, root: Path, all_masks: dict[str, Path]) -> list[Path]:
    stem = image_path.stem
    names = [
        stem,
        stem + "_seg",
        stem + "_mask",
        stem + "_segmentation",
        stem.replace("_image", "_seg"),
        stem.replace("image", "seg"),
    ]
    parent = image_path.parent.relative_to(root).as_posix().lower()
    keys = [f"{parent}/{name.lower()}" for name in names]
    return [all_masks[key] for key in keys if key in all_masks]


def discover_bagls_pairs(bagls_root: Path | str, limit: int | None = None) -> list[dict[str, Any]]:
    root = Path(bagls_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"BAGLS root not found: {root}")

    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    mask_files = [p for p in files if _is_mask_like(p)]
    mask_map = {p.relative_to(root).with_suffix("").as_posix().lower(): p for p in mask_files}
    image_files = [p for p in files if not _is_mask_like(p)]
    rows: list[dict[str, Any]] = []
    for image_path in sorted(image_files):
        candidates = _mask_candidates(image_path, root, mask_map)
        if not candidates:
            continue
        mask_path = candidates[0]
        rel = image_path.relative_to(root).as_posix().lower()
        official = "official_test" if rel.startswith("test/") or "/test/" in rel else "official_train"
        meta = read_meta(image_path)
        rows.append(
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "bagls_split": official,
                "recording_id": infer_recording_id(image_path, meta),
                "institution": meta.get("institution", meta.get("clinic", "")),
                "modality": meta.get("modality", ""),
            }
        )
        if limit is not None and len(rows) >= int(limit):
            break
    if not rows:
        raise RuntimeError(
            f"No BAGLS image/mask pairs found under {root}. "
            "Expected extracted training/test folders containing images plus *_seg or *_mask files."
        )
    return rows


def assign_recording_splits(rows: list[dict[str, Any]], seed: int = 42, val_fraction: float = 0.1) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    train_records = sorted({r["recording_id"] for r in rows if r["bagls_split"] == "official_train"})
    rng.shuffle(train_records)
    val_count = max(1, int(round(len(train_records) * float(val_fraction)))) if len(train_records) > 1 else 0
    val_records = set(train_records[:val_count])
    out = []
    for row in rows:
        item = dict(row)
        if row["bagls_split"] == "official_test":
            item["split"] = "test"
        elif row["recording_id"] in val_records:
            item["split"] = "val"
        else:
            item["split"] = "train"
        out.append(item)
    return out


def build_manifest_rows(
    bagls_root: Path | str,
    limit: int | None = None,
    seed: int = 42,
    val_fraction: float = 0.1,
    expand_w: float = 1.8,
    expand_h: float = 1.6,
) -> pd.DataFrame:
    rows = assign_recording_splits(discover_bagls_pairs(bagls_root, limit=limit), seed=seed, val_fraction=val_fraction)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        try:
            rgb = load_rgb(row["image_path"])
            mask = load_mask(row["mask_path"])
        except Exception as exc:
            item = dict(row)
            item.update({"load_error": repr(exc), "roi_valid_label": 0, "reflection_label": 0})
            enriched.append(item)
            continue
        h, w = rgb.shape[:2]
        bbox = mask_to_bbox(mask)
        roi_bbox = expand_bbox(bbox, w, h, scale_w=expand_w, scale_h=expand_h)
        roi = crop_bbox(rgb, roi_bbox)
        metrics = specular_seed_metrics(roi)
        reflection_label, severity, source = reflection_label_from_metrics(metrics)
        glottis_area_ratio = float(mask.mean())
        item = dict(row)
        item.update(metrics)
        item.update(
            {
                "width": int(w),
                "height": int(h),
                "roi_bbox_xyxy": bbox_to_str(roi_bbox),
                "glottis_area_ratio": glottis_area_ratio,
                "roi_valid_label": int(roi_bbox is not None and glottis_area_ratio > 0),
                "reflection_label": int(reflection_label),
                "reflection_severity": severity,
                "label_source": source,
                "review_needed": bool(severity == "mild"),
                "load_error": "",
            }
        )
        enriched.append(item)
    return pd.DataFrame(enriched)


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.enc1 = self._block(in_channels, c)
        self.enc2 = self._block(c, c * 2)
        self.enc3 = self._block(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = self._block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = self._block(c * 2, c)
        self.out = nn.Conv2d(c, 1, 1)

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


class ConvBnAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.0):
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(float(dropout)))
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = ConvBnAct(in_ch, out_ch, stride and 3, dropout=dropout)
        if stride != 1:
            self.conv1[0].stride = (stride, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return F.relu(out + self.skip(x), inplace=True)


class ResidualUNet(nn.Module):
    """Dependency-light fallback with more depth and residual skips than TinyUNet."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, dropout: float = 0.05):
        super().__init__()
        c = int(base_channels)
        self.model_impl = "residual_unet"
        self.stem = ResidualBlock(in_channels, c, dropout=dropout)
        self.enc2 = ResidualBlock(c, c * 2, stride=2, dropout=dropout)
        self.enc3 = ResidualBlock(c * 2, c * 4, stride=2, dropout=dropout)
        self.enc4 = ResidualBlock(c * 4, c * 8, stride=2, dropout=dropout)
        self.bottleneck = ResidualBlock(c * 8, c * 8, dropout=dropout)
        self.dec3 = ResidualBlock(c * 8 + c * 4, c * 4, dropout=dropout)
        self.dec2 = ResidualBlock(c * 4 + c * 2, c * 2, dropout=dropout)
        self.dec1 = ResidualBlock(c * 2 + c, c, dropout=dropout)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.stem(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.bottleneck(self.enc4(e3))
        d3 = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


class TimmEncoderUNet(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet34.a1_in1k",
        pretrained: bool = True,
        decoder_channels: Iterable[int] = (256, 128, 64, 32),
        dropout: float = 0.05,
        image_size: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm is not available")
        self.backbone_name = backbone
        self.pretrained_source = "timm_pretrained" if pretrained else "random_init"
        kwargs = dict(model_kwargs or {})
        if image_size and "swin" in backbone.lower() and "img_size" not in kwargs:
            kwargs["img_size"] = int(image_size)
        out_indices = tuple(int(v) for v in kwargs.pop("out_indices", (0, 1, 2, 3, 4)))
        try:
            self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=out_indices, **kwargs)
        except Exception:
            if out_indices == (0, 1, 2, 3, 4):
                self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3), **kwargs)
            else:
                raise
        channels = list(self.encoder.feature_info.channels())
        decoder_channels = [int(v) for v in decoder_channels]
        if len(decoder_channels) < len(channels) - 1:
            decoder_channels.extend([decoder_channels[-1]] * (len(channels) - 1 - len(decoder_channels)))
        self.center = ResidualBlock(channels[-1], decoder_channels[0], dropout=dropout)
        blocks = []
        in_ch = decoder_channels[0]
        for skip_ch, out_ch in zip(reversed(channels[:-1]), decoder_channels[1:]):
            blocks.append(ResidualBlock(in_ch + skip_ch, int(out_ch), dropout=dropout))
            in_ch = int(out_ch)
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Conv2d(in_ch, 1, 1)
        self.model_impl = "timm_unet"

    @staticmethod
    def _as_nchw(feature: torch.Tensor, channels: int) -> torch.Tensor:
        if feature.dim() != 4:
            raise ValueError(f"Expected 4D feature map, got shape {tuple(feature.shape)}")
        if feature.shape[1] == channels:
            return feature
        if feature.shape[-1] == channels:
            return feature.permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        raw_features = self.encoder(x)
        channels = list(self.encoder.feature_info.channels())
        features = [self._as_nchw(feat, ch) for feat, ch in zip(raw_features, channels)]
        out = self.center(features[-1])
        for block, skip in zip(self.blocks, reversed(features[:-1])):
            out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            out = block(torch.cat([out, skip], dim=1))
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return self.out(out)


class _SmallCnnBackbone(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )
        self.num_features = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualHeadReflectionGate(nn.Module):
    def __init__(self, backbone: str = "mobilenetv3_small_100", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.pretrained_source = "fallback_cnn"
        if timm is not None:
            try:
                self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
                self.feature_dim = int(getattr(self.backbone, "num_features", 1024))
                self.pretrained_source = "timm_pretrained" if pretrained else "random_init"
            except Exception:
                self.backbone = _SmallCnnBackbone()
                self.feature_dim = self.backbone.num_features
        else:
            self.backbone = _SmallCnnBackbone()
            self.feature_dim = self.backbone.num_features
        self.feature_dim = self._infer_feature_dim(self.feature_dim)
        self.valid_head = nn.Linear(self.feature_dim, 1)
        self.reflect_head = nn.Linear(self.feature_dim, 1)

    def _infer_feature_dim(self, fallback: int) -> int:
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                feat = self.backbone(torch.zeros(1, 3, 224, 224))
            if feat.dim() > 2:
                feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            return int(feat.shape[1])
        except Exception:
            return int(fallback)
        finally:
            self.backbone.train(was_training)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        if feat.dim() > 2:
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        valid_logits = self.valid_head(feat).squeeze(1)
        reflect_logits = self.reflect_head(feat).squeeze(1)
        return {
            "valid_logits": valid_logits,
            "reflect_logits": reflect_logits,
            "roi_valid_prob": torch.sigmoid(valid_logits),
            "reflect_prob": torch.sigmoid(reflect_logits),
        }


def make_localizer_model(cfg: dict[str, Any] | None = None) -> nn.Module:
    cfg = cfg or default_localizer_config()
    arch = str(cfg.get("model_impl") or cfg.get("model_arch", "tiny_unet")).lower()
    if arch in {"tiny", "tiny_unet", "tinyunet"}:
        model = TinyUNet(base_channels=int(cfg.get("base_channels", 32)))
        model.model_impl = "tiny_unet"
        return model
    if arch in {"residual", "residual_unet", "resunet", "res_unet", "residual_unet_fallback"}:
        return ResidualUNet(
            base_channels=int(cfg.get("base_channels", 32)),
            dropout=float(cfg.get("dropout", 0.05)),
        )
    if arch in {"timm", "timm_unet", "encoder_unet"}:
        try:
            return TimmEncoderUNet(
                backbone=str(cfg.get("backbone", "resnet34.a1_in1k")),
                pretrained=bool(cfg.get("pretrained", True)),
                decoder_channels=cfg.get("decoder_channels", [256, 128, 64, 32]),
                dropout=float(cfg.get("dropout", 0.05)),
                image_size=int(cfg.get("image_size", 256)),
                model_kwargs=dict(cfg.get("model_kwargs", {})),
            )
        except Exception as exc:
            print(f"WARNING: timm localizer unavailable ({exc!r}); falling back to residual_unet.", file=sys.stderr)
            model = ResidualUNet(
                base_channels=int(cfg.get("base_channels", 32)),
                dropout=float(cfg.get("dropout", 0.05)),
            )
            model.model_impl = "residual_unet_fallback"
            return model
    raise ValueError(f"Unknown localizer model_arch: {cfg.get('model_arch')!r}")


def make_reflection_model(cfg: dict[str, Any] | None = None) -> DualHeadReflectionGate:
    cfg = cfg or default_reflection_config()
    return DualHeadReflectionGate(
        backbone=str(cfg.get("backbone", "mobilenetv3_small_100")),
        pretrained=bool(cfg.get("pretrained", True)),
    )


def checkpoint_payload(model: nn.Module, cfg: dict[str, Any], role: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "role": role,
        "cfg": cfg,
        "model_state_dict": model.state_dict(),
        "git": git_provenance(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    return payload


def _load_state_dict(path: Path | str, map_location: torch.device) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict):
        return ckpt
    return {"model_state_dict": ckpt}


def _checkpoint_state(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    if state is None and all(isinstance(value, torch.Tensor) for value in ckpt.values()):
        state = ckpt
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a model state dict.")
    return state


def _validate_checkpoint_role(ckpt: dict[str, Any], expected_role: str, model_path: Path | str) -> None:
    role = ckpt.get("role")
    if role is not None and str(role) != expected_role:
        raise ValueError(f"{model_path} is a {role!r} checkpoint, expected {expected_role!r}.")


def load_roi_localizer_model(model_path: Path | str, device: torch.device | str | None = None):
    device = resolve_device(str(device) if device is not None else "auto")
    ckpt = _load_state_dict(model_path, device)
    _validate_checkpoint_role(ckpt, "localizer", model_path)
    cfg = default_localizer_config()
    cfg.update(ckpt.get("cfg", {}))
    state = _checkpoint_state(ckpt)
    if "model_arch" not in ckpt.get("cfg", {}) and any(key.startswith("enc1.") for key in state):
        cfg["model_arch"] = "tiny_unet"
        cfg["model_impl"] = "tiny_unet"
    model = make_localizer_model(cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg, ckpt


def load_roi_reflection_model(model_path: Path | str, device: torch.device | str | None = None):
    device = resolve_device(str(device) if device is not None else "auto")
    ckpt = _load_state_dict(model_path, device)
    _validate_checkpoint_role(ckpt, "reflection", model_path)
    cfg = default_reflection_config()
    cfg.update(ckpt.get("cfg", {}))
    cfg["pretrained"] = False
    model = make_reflection_model(cfg).to(device)
    model.load_state_dict(_checkpoint_state(ckpt), strict=True)
    model.eval()
    return model, cfg, ckpt


load_localizer_model = load_roi_localizer_model
load_reflection_model = load_roi_reflection_model
load_localizer_checkpoint = load_roi_localizer_model
load_reflection_checkpoint = load_roi_reflection_model
load_roi_localizer = load_roi_localizer_model
load_roi_reflection_classifier = load_roi_reflection_model
load_checkpoint_localizer = load_roi_localizer_model
load_checkpoint_reflection = load_roi_reflection_model


def decode_reflection_output(output: Any) -> dict[str, torch.Tensor]:
    if isinstance(output, dict):
        return output
    if isinstance(output, (tuple, list)):
        if len(output) >= 2:
            return {"valid_logits": output[0], "reflect_logits": output[1]}
        output = output[0]
    return {"reflect_logits": output}


def dice_score_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.dim()))
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    return ((2.0 * inter + eps) / (denom + eps)).mean()


def dice_scores_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.dim()))
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    return (2.0 * inter + eps) / (denom + eps)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 0.5, bce_weight: float = 0.5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice_loss = 1.0 - dice_score_from_logits(logits, target)
    return float(bce_weight) * bce + float(dice_weight) * dice_loss


def localizer_loss(logits: torch.Tensor, target: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    bce_raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    gamma = float(cfg.get("focal_bce_gamma", 0.0))
    if gamma > 0:
        probs = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, probs, 1.0 - probs).clamp(1e-6, 1.0)
        bce = ((1.0 - pt) ** gamma * bce_raw).mean()
    else:
        bce = bce_raw.mean()
    dice_loss = 1.0 - dice_score_from_logits(logits, target)
    dims = tuple(range(1, logits.dim()))
    probs = torch.sigmoid(logits)
    tp = (probs * target).sum(dim=dims)
    fp = (probs * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - probs) * target).sum(dim=dims)
    alpha = float(cfg.get("tversky_alpha", 0.35))
    beta = float(cfg.get("tversky_beta", 0.65))
    tversky = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
    tversky_loss = 1.0 - tversky.mean()
    return (
        float(cfg.get("bce_weight", 0.5)) * bce
        + float(cfg.get("dice_weight", 0.5)) * dice_loss
        + float(cfg.get("tversky_weight", 0.0)) * tversky_loss
    )


def localizer_batch_metrics(logits: torch.Tensor, target: torch.Tensor, cfg: dict[str, Any]) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= float(cfg.get("prediction_threshold", 0.5))).float()
    dims = tuple(range(1, pred.dim()))
    gt_area = target.mean(dim=dims)
    pred_area = pred.mean(dim=dims)
    empty = gt_area <= float(cfg.get("empty_area_threshold", 1e-8))
    nonempty = ~empty
    dice_values = dice_scores_from_logits(logits, target)
    empty_fp = empty & (pred_area > float(cfg.get("empty_pred_area_threshold", 0.001)))
    fp_ratio = (pred * (1.0 - target)).sum(dim=dims) / torch.clamp((1.0 - target).sum(dim=dims), min=1.0)
    fn_ratio = ((1.0 - pred) * target).sum(dim=dims) / torch.clamp(target.sum(dim=dims), min=1.0)
    return {
        "dice": float(dice_values.mean().detach().cpu()),
        "nonempty_dice": float(dice_values[nonempty].mean().detach().cpu()) if bool(nonempty.any()) else float("nan"),
        "empty_count": int(empty.sum().detach().cpu()),
        "empty_fp_count": int(empty_fp.sum().detach().cpu()),
        "area_mae": float((pred_area - gt_area).abs().mean().detach().cpu()),
        "fp_ratio": float(fp_ratio.mean().detach().cpu()),
        "fn_ratio": float(fn_ratio.mean().detach().cpu()),
    }


def binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5, prefix: str = "") -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=np.float32)
    finite = np.isfinite(probs)
    if not finite.any():
        return {f"{prefix}threshold": threshold, f"{prefix}accuracy": float("nan")}
    y_true = y_true[finite]
    probs = probs[finite]
    pred = (probs >= float(threshold)).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    out = {
        f"{prefix}threshold": float(threshold),
        f"{prefix}accuracy": float((tp + tn) / max(len(y_true), 1)),
        f"{prefix}precision": float(precision),
        f"{prefix}recall": float(recall),
        f"{prefix}specificity": float(specificity),
        f"{prefix}f1": float(f1),
        f"{prefix}tp": tp,
        f"{prefix}tn": tn,
        f"{prefix}fp": fp,
        f"{prefix}fn": fn,
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out[f"{prefix}auroc"] = float(roc_auc_score(y_true, probs)) if len(set(y_true.tolist())) > 1 else float("nan")
        out[f"{prefix}auprc"] = float(average_precision_score(y_true, probs)) if len(set(y_true.tolist())) > 1 else float("nan")
    except Exception:
        out[f"{prefix}auroc"] = float("nan")
        out[f"{prefix}auprc"] = float("nan")
    return out


def threshold_table(y_true: np.ndarray, probs: np.ndarray, prefix: str = "") -> pd.DataFrame:
    return pd.DataFrame(binary_metrics(y_true, probs, threshold=t, prefix=prefix) for t in np.linspace(0.01, 0.99, 99))


def choose_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    min_specificity: float = 0.85,
    min_recall: float = 0.50,
    prefix: str = "",
) -> dict[str, Any]:
    table = threshold_table(y_true, probs, prefix=prefix)
    spec_col = f"{prefix}specificity"
    recall_col = f"{prefix}recall"
    f1_col = f"{prefix}f1"
    thr_col = f"{prefix}threshold"
    candidates = table[(table[spec_col] >= min_specificity) & (table[recall_col] >= min_recall)]
    if candidates.empty:
        candidates = table
    best = candidates.sort_values([f1_col, spec_col, recall_col], ascending=False).iloc[0]
    return {"threshold": float(best[thr_col]), "metrics": best.to_dict()}


@dataclass
class SegmentationBatch:
    image: torch.Tensor
    mask: torch.Tensor


class BaglsSegmentationDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cfg: dict[str, Any], split: str | None = None, cache_device: torch.device | None = None):
        frame = manifest.copy()
        if split is not None:
            frame = frame[frame["split"] == split].copy()
        self.frame = frame.reset_index(drop=True)
        self.cfg = cfg
        self.split = split
        self.augment = bool(cfg.get("augment_train", False)) and split == "train"
        self.image_size = int(cfg.get("image_size", 256))
        self.cache_device = cache_device
        self.sample_weights = self._build_sample_weights()
        self.cached_images = None
        self.cached_masks = None
        if cache_device is not None and len(self.frame):
            images, masks = [], []
            for row in self.frame.itertuples(index=False):
                rgb = preprocess_localizer_rgb(load_rgb(row.image_path), self.cfg)
                images.append(torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(torch.uint8))
                mask = preprocess_localizer_mask(load_mask(row.mask_path), self.cfg).astype(np.uint8)
                masks.append(torch.from_numpy(mask[None, :, :]))
            self.cached_images = torch.stack(images).to(cache_device)
            self.cached_masks = torch.stack(masks).to(cache_device)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._get_item(idx, augment=self.augment)

    def _get_item(self, idx: int, augment: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cached_images is not None and self.cached_masks is not None:
            image = self.cached_images[idx]
            mask = self.cached_masks[idx]
            if image.dtype == torch.uint8:
                image = image.float().div_(255.0)
            else:
                image = image.float()
            mask = mask.float()
            if augment:
                image, mask = self._apply_train_augment(image, mask)
            return image, mask
        row = self.frame.iloc[idx]
        image = preprocess_localizer_frame(load_rgb(row["image_path"]), self.cfg)
        mask = preprocess_localizer_mask(load_mask(row["mask_path"]), self.cfg).astype(np.float32)
        mask_t = torch.from_numpy(mask[None, :, :])
        if augment:
            image, mask_t = self._apply_train_augment(image, mask_t)
        return image, mask_t

    def _build_sample_weights(self) -> np.ndarray | None:
        if not bool(self.cfg.get("area_aware_sampling", False)) or self.split != "train" or len(self.frame) == 0:
            return None
        areas = self.frame.get("glottis_area_ratio", pd.Series(np.zeros(len(self.frame)))).fillna(0.0).astype(float).to_numpy()
        edges = np.asarray(self.cfg.get("area_bin_edges", [0.0, 1e-8, 0.001, 0.005, 0.02, 0.08, 1.0]), dtype=np.float64)
        edges = np.unique(np.sort(edges))
        if len(edges) < 2:
            return None
        bins = np.clip(np.digitize(areas, edges[1:-1], right=False), 0, len(edges) - 2)
        counts = np.bincount(bins, minlength=len(edges) - 1).astype(np.float64)
        weights = 1.0 / np.maximum(counts[bins], 1.0)
        empty_thr = float(self.cfg.get("empty_area_threshold", 1e-8))
        weights[areas <= empty_thr] *= float(self.cfg.get("empty_sample_weight", 1.0))
        if len(edges) >= 2:
            weights[areas >= float(edges[-2])] *= float(self.cfg.get("large_area_sample_weight", 1.0))
        weights = weights / np.maximum(weights.sum(), 1e-12)
        return weights.astype(np.float64)

    def epoch_indices(self, shuffle: bool) -> np.ndarray:
        n = len(self)
        if not shuffle:
            return np.arange(n)
        if self.sample_weights is None:
            indices = np.arange(n)
            np.random.shuffle(indices)
            return indices
        return np.random.choice(np.arange(n), size=n, replace=bool(self.cfg.get("sampling_replacement", True)), p=self.sample_weights)

    @staticmethod
    def _sample_range(value: Any, default: tuple[float, float]) -> float:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            low, high = float(value[0]), float(value[1])
        else:
            low, high = default
        return float(np.random.uniform(low, high))

    def _apply_train_augment(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        if np.random.random() < float(cfg.get("horizontal_flip_prob", 0.0)):
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if np.random.random() < float(cfg.get("affine_prob", 0.0)):
            degrees = float(cfg.get("affine_degrees", 0.0))
            angle = float(np.random.uniform(-degrees, degrees))
            translate_cfg = cfg.get("affine_translate", [0.0, 0.0])
            max_dx = float(translate_cfg[0]) if isinstance(translate_cfg, (list, tuple)) and translate_cfg else 0.0
            max_dy = float(translate_cfg[1]) if isinstance(translate_cfg, (list, tuple)) and len(translate_cfg) > 1 else max_dx
            translate = [
                int(round(np.random.uniform(-max_dx, max_dx) * image.shape[-1])),
                int(round(np.random.uniform(-max_dy, max_dy) * image.shape[-2])),
            ]
            scale = self._sample_range(cfg.get("affine_scale", [1.0, 1.0]), (1.0, 1.0))
            image = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            )

        if np.random.random() < float(cfg.get("gamma_prob", 0.0)):
            image = TF.adjust_gamma(image.clamp(0.0, 1.0), gamma=self._sample_range(cfg.get("gamma_range", [1.0, 1.0]), (1.0, 1.0)))
        if np.random.random() < float(cfg.get("brightness_prob", 0.0)):
            image = TF.adjust_brightness(image, self._sample_range(cfg.get("brightness_range", [1.0, 1.0]), (1.0, 1.0)))
        if np.random.random() < float(cfg.get("contrast_prob", 0.0)):
            image = TF.adjust_contrast(image, self._sample_range(cfg.get("contrast_range", [1.0, 1.0]), (1.0, 1.0)))
        if np.random.random() < float(cfg.get("blur_prob", 0.0)):
            kernel = int(cfg.get("blur_kernel_size", 3))
            kernel = kernel if kernel % 2 == 1 else kernel + 1
            sigma = self._sample_range(cfg.get("blur_sigma", [0.1, 0.8]), (0.1, 0.8))
            image = TF.gaussian_blur(image, kernel_size=[kernel, kernel], sigma=[sigma, sigma])
        if np.random.random() < float(cfg.get("noise_prob", 0.0)):
            image = image + torch.randn_like(image) * float(cfg.get("noise_std", 0.015))
        return image.clamp(0.0, 1.0), (mask > 0.5).float()

    def get_batch(
        self,
        indices: np.ndarray | torch.Tensor,
        *,
        device: torch.device | None = None,
        augment: bool | None = None,
        channels_last: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        augment = self.augment if augment is None else bool(augment)
        if self.cached_images is None or self.cached_masks is None:
            batch = [self._get_item(int(i), augment=augment) for i in np.asarray(indices).tolist()]
            images, masks = (torch.stack(list(col)) for col in zip(*batch))
            if device is not None:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            return images, masks

        cache_device = self.cached_images.device
        if torch.is_tensor(indices):
            idx = indices.to(cache_device, non_blocking=True).long()
        else:
            idx = torch.as_tensor(indices, device=cache_device, dtype=torch.long)
        if idx.numel() == 0:
            raise ValueError("Empty batch indices.")
        is_contiguous = bool(idx.numel() > 1 and torch.equal(idx, torch.arange(int(idx[0]), int(idx[0]) + idx.numel(), device=idx.device)))
        if is_contiguous:
            start = int(idx[0].item())
            stop = start + int(idx.numel())
            images_u8 = self.cached_images[start:stop]
            masks_u8 = self.cached_masks[start:stop]
        else:
            images_u8 = self.cached_images.index_select(0, idx)
            masks_u8 = self.cached_masks.index_select(0, idx)
        target_device = device or cache_device
        images = images_u8.to(device=target_device, dtype=torch.float32, non_blocking=True).div_(255.0)
        masks = masks_u8.to(device=target_device, dtype=torch.float32, non_blocking=True)
        if augment and bool(self.cfg.get("batched_gpu_augment", True)):
            images, masks = batched_localizer_augment(images, masks, self.cfg)
        elif augment:
            augmented = [self._apply_train_augment(img, mask) for img, mask in zip(images, masks)]
            images, masks = (torch.stack(list(col)) for col in zip(*augmented))
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        return images, masks


def _prob_mask(batch_size: int, prob: float, device: torch.device) -> torch.Tensor:
    if prob <= 0:
        return torch.zeros(batch_size, device=device, dtype=torch.bool)
    if prob >= 1:
        return torch.ones(batch_size, device=device, dtype=torch.bool)
    return torch.rand(batch_size, device=device) < float(prob)


def _sample_range_tensor(batch_size: int, value: Any, default: tuple[float, float], device: torch.device) -> torch.Tensor:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
    else:
        low, high = default
    return torch.empty(batch_size, device=device).uniform_(low, high)


def _batched_affine(images: torch.Tensor, masks: torch.Tensor, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _channels, height, width = images.shape
    device = images.device
    selected = _prob_mask(batch_size, float(cfg.get("affine_prob", 0.0)), device)
    if not bool(selected.any()):
        return images, masks
    degrees = float(cfg.get("affine_degrees", 0.0))
    angles = torch.empty(batch_size, device=device).uniform_(-degrees, degrees) * math.pi / 180.0
    scales = _sample_range_tensor(batch_size, cfg.get("affine_scale", [1.0, 1.0]), (1.0, 1.0), device).clamp_min(1e-3)
    translate_cfg = cfg.get("affine_translate", [0.0, 0.0])
    max_dx = float(translate_cfg[0]) if isinstance(translate_cfg, (list, tuple)) and translate_cfg else 0.0
    max_dy = float(translate_cfg[1]) if isinstance(translate_cfg, (list, tuple)) and len(translate_cfg) > 1 else max_dx
    tx = torch.empty(batch_size, device=device).uniform_(-max_dx, max_dx) * 2.0
    ty = torch.empty(batch_size, device=device).uniform_(-max_dy, max_dy) * 2.0
    cos = torch.cos(angles) / scales
    sin = torch.sin(angles) / scales
    theta = torch.zeros(batch_size, 2, 3, device=device, dtype=images.dtype)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=images.dtype).expand(batch_size, -1, -1)
    theta = torch.where(selected.view(batch_size, 1, 1), theta, identity)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    images = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    masks = F.grid_sample(masks, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return images, (masks > 0.5).float()


def _batched_scale_pad(images: torch.Tensor, masks: torch.Tensor, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    prob = float(cfg.get("random_scale_pad_prob", 0.0))
    if prob <= 0 or float(torch.rand((), device=images.device)) >= prob:
        return images, masks
    size = int(images.shape[-1])
    scale = float(_sample_range_tensor(1, cfg.get("random_scale_pad_range", [1.0, 1.0]), (1.0, 1.0), images.device).item())
    scaled = max(16, int(round(size * scale)))
    if scaled == size:
        return images, masks
    images_s = F.interpolate(images, size=(scaled, scaled), mode="bilinear", align_corners=False)
    masks_s = F.interpolate(masks, size=(scaled, scaled), mode="nearest")
    if scaled > size:
        max_y = scaled - size
        max_x = scaled - size
        top = int(torch.randint(0, max_y + 1, (), device=images.device).item())
        left = int(torch.randint(0, max_x + 1, (), device=images.device).item())
        return images_s[:, :, top : top + size, left : left + size], masks_s[:, :, top : top + size, left : left + size]
    pad_total = size - scaled
    top = int(torch.randint(0, pad_total + 1, (), device=images.device).item())
    left = int(torch.randint(0, pad_total + 1, (), device=images.device).item())
    bottom = pad_total - top
    right = pad_total - left
    images_p = F.pad(images_s, (left, right, top, bottom), value=float(cfg.get("letterbox_pad_value", 0)) / 255.0)
    masks_p = F.pad(masks_s, (left, right, top, bottom), value=0.0)
    return images_p, masks_p


def batched_localizer_augment(images: torch.Tensor, masks: torch.Tensor, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(images.shape[0])
    device = images.device
    if batch_size == 0:
        return images, masks
    flip = _prob_mask(batch_size, float(cfg.get("horizontal_flip_prob", 0.0)), device)
    if bool(flip.any()):
        images[flip] = torch.flip(images[flip], dims=(-1,))
        masks[flip] = torch.flip(masks[flip], dims=(-1,))
    images, masks = _batched_scale_pad(images, masks, cfg)
    if bool(cfg.get("batched_affine_enabled", True)):
        images, masks = _batched_affine(images, masks, cfg)

    gamma_selected = _prob_mask(batch_size, float(cfg.get("gamma_prob", 0.0)), device)
    gamma = torch.ones(batch_size, device=device)
    gamma = torch.where(gamma_selected, _sample_range_tensor(batch_size, cfg.get("gamma_range", [1.0, 1.0]), (1.0, 1.0), device), gamma)
    images = images.clamp(1e-6, 1.0).pow(gamma.view(-1, 1, 1, 1))

    brightness_selected = _prob_mask(batch_size, float(cfg.get("brightness_prob", 0.0)), device)
    brightness = torch.ones(batch_size, device=device)
    brightness = torch.where(
        brightness_selected,
        _sample_range_tensor(batch_size, cfg.get("brightness_range", [1.0, 1.0]), (1.0, 1.0), device),
        brightness,
    )
    images = images * brightness.view(-1, 1, 1, 1)

    contrast_selected = _prob_mask(batch_size, float(cfg.get("contrast_prob", 0.0)), device)
    contrast = torch.ones(batch_size, device=device)
    contrast = torch.where(
        contrast_selected,
        _sample_range_tensor(batch_size, cfg.get("contrast_range", [1.0, 1.0]), (1.0, 1.0), device),
        contrast,
    )
    mean = images.mean(dim=(-2, -1), keepdim=True)
    images = (images - mean) * contrast.view(-1, 1, 1, 1) + mean

    noise = _prob_mask(batch_size, float(cfg.get("noise_prob", 0.0)), device)
    if bool(noise.any()):
        images = images + torch.randn_like(images) * float(cfg.get("noise_std", 0.015)) * noise.view(-1, 1, 1, 1)
    return images.clamp(0.0, 1.0), (masks > 0.5).float()


class ReflectionDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cfg: dict[str, Any], split: str | None = None, cache_device: torch.device | None = None):
        frame = manifest.copy()
        if split is not None:
            frame = frame[frame["split"] == split].copy()
        frame = frame[frame["roi_bbox_xyxy"].fillna("").astype(str).str.len() > 0].copy()
        self.frame = frame.reset_index(drop=True)
        self.cfg = cfg
        self.cache_device = cache_device
        self.cached_images = None
        if cache_device is not None and len(self.frame):
            tensors = []
            for row in self.frame.itertuples(index=False):
                rgb = load_rgb(row.image_path)
                roi = crop_bbox(rgb, row.roi_bbox_xyxy)
                if roi is None:
                    roi = rgb
                tensors.append(preprocess_reflection_roi(roi, cfg))
            self.cached_images = torch.stack(tensors).to(cache_device)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[idx]
        if self.cached_images is not None:
            image = self.cached_images[idx]
        else:
            rgb = load_rgb(row["image_path"])
            roi = crop_bbox(rgb, row["roi_bbox_xyxy"])
            if roi is None:
                roi = rgb
            image = preprocess_reflection_roi(roi, self.cfg)
        valid = torch.tensor(float(row.get("roi_valid_label", 1)), dtype=torch.float32)
        reflect = torch.tensor(float(row.get("reflection_label", 0)), dtype=torch.float32)
        return image, valid, reflect


def iter_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    *,
    device: torch.device | None = None,
    augment: bool | None = None,
    channels_last: bool = False,
):
    n = len(dataset)
    if hasattr(dataset, "epoch_indices"):
        indices = dataset.epoch_indices(shuffle)
    else:
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
    for start in range(0, n, int(batch_size)):
        idxs = indices[start : start + int(batch_size)]
        if hasattr(dataset, "get_batch"):
            yield dataset.get_batch(idxs, device=device, augment=augment, channels_last=channels_last)
            continue
        batch = [dataset[int(i)] for i in idxs]
        columns = list(zip(*batch))
        yield tuple(torch.stack(list(col)) for col in columns)


def save_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)
