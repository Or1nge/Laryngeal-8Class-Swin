#!/usr/bin/env python3
"""Run image-checkpoint inference over laryngeal videos.

The script treats a video as a bag of sampled frames. It does not train a
temporal model. Instead, it compares several weak-supervision aggregation
rules so the existing still-image classifier can be evaluated on small video
sets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "图像识别"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shared  # noqa: E402
from shared import (  # noqa: E402
    BEST_MODEL_PATH,
    LABEL_DICT,
    LABEL_NAMES,
    RESULTS_DIR,
    VOC_LABELS,
    HierarchicalImageClassifier,
    _build_base_preprocess,
    gpu_normalise,
    init_label_mapping,
    load_config,
    setup_device,
)

GLOTTIS_BINARY_DIR = PROJECT_ROOT / "glottis_binary"
if str(GLOTTIS_BINARY_DIR) not in sys.path:
    sys.path.insert(0, str(GLOTTIS_BINARY_DIR))

from common import default_train_config, load_checkpoint_model  # noqa: E402

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
DEFAULT_VIDEO_ROOT = Path(
    os.environ.get("LARYNX_VIDEO_ROOT", "/mnt/data/LarynxData/videos/ali_larynx_20260505")
).expanduser()
DEFAULT_VIDEO_LABEL_MAP = os.environ.get("LARYNX_VIDEO_LABEL_MAP")
DEFAULT_GLOTTIS_GATE_MODEL = Path(
    os.environ.get(
        "LARYNX_GLOTTIS_GATE_MODEL",
        Path(RESULTS_DIR)
        / "glottis_binary_benchmarks"
        / "20260505_183333_parallel"
        / "swin_base"
        / "best_model.pth",
    )
)

DEFAULT_FOLDER_LABEL_MAP = {
    "normal/healthy-larynx": "Normal",
    "cancer/laryngeal-cancer": "Cancer",
    "benign/reinke-edema": "Reinke-Edema",
    "benign/vocal-cord-polyp": "Vocal-Cord-Polyp",
    "benign/vocal-cord-leukoplakia": "Vocal-Cord-Leukoplakia",
}

LABEL_DISPLAY_ZH = {
    "Non-Vocal-Cord": "非声带图片",
    "Normal": "正常",
    "Reinke-Edema": "任克氏水肿",
    "Vocal-Cord-Cyst": "声带囊肿",
    "Vocal-Cord-Polyp": "声带息肉",
    "Vocal-Cord-Leukoplakia": "声带白斑",
    "Vocal-Cord-Granuloma": "声带肉芽肿",
    "Cancer": "癌",
}

CHINESE_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


@dataclass(frozen=True)
class VideoItem:
    video_path: Path
    true_label: str
    video_id: str
    start_sec: float | None = None
    end_sec: float | None = None


@dataclass(frozen=True)
class RuleVariant:
    name: str
    gate: str
    threshold: float | None = None


RULE_VARIANTS = [
    RuleVariant("all_frames", "all"),
    RuleVariant("voc_sum_gt_nonvoc", "voc_margin", 0.0),
    RuleVariant("voc_sum_margin_0_10", "voc_margin", 0.10),
    RuleVariant("nonvoc_lt_0_50", "nonvoc_lt", 0.50),
    RuleVariant("nonvoc_lt_0_35", "nonvoc_lt", 0.35),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate still-image larynx classifier predictions over videos."
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=DEFAULT_VIDEO_ROOT,
        help="Root containing video folders. Defaults to LARYNX_VIDEO_ROOT or the shared data disk.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional CSV with video_path,true_label and optional video_id,"
            "valid_start_sec,valid_end_sec. Paths may be absolute or relative to --video-root."
        ),
    )
    parser.add_argument(
        "--folder-label-map",
        type=Path,
        default=Path(DEFAULT_VIDEO_LABEL_MAP) if DEFAULT_VIDEO_LABEL_MAP else None,
        help="Optional JSON mapping relative folder paths to checkpoint labels.",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config_phase2.json")
    parser.add_argument("--model", type=Path, default=Path(BEST_MODEL_PATH))
    parser.add_argument("--output-dir", type=Path, default=Path(RESULTS_DIR) / "video_inference")
    parser.add_argument("--sample-fps", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--decode-mode",
        choices=["sequential", "seek"],
        default="sequential",
        help=(
            "sequential reads forward through the video and samples by frame index; "
            "seek preserves the older per-sample timestamp seeking path."
        ),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision for model inference.",
    )
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument(
        "--frame-threshold",
        type=float,
        default=0.50,
        help="Frame-level probability threshold used only for evidence segment reporting.",
    )
    parser.add_argument(
        "--candidate-labels",
        default="auto",
        help="'auto' uses labels present in the video set, 'all-voc' uses all VOC labels, or comma list.",
    )
    parser.add_argument(
        "--allow-unlabeled",
        action="store_true",
        help="Include videos whose folder does not map to a checkpoint label; useful for tbr review batches.",
    )
    parser.add_argument("--unlabeled-label", default="Unknown")
    parser.add_argument("--min-sharpness", type=float, default=5.0)
    parser.add_argument("--min-brightness", type=float, default=5.0)
    parser.add_argument("--max-brightness", type=float, default=250.0)
    parser.add_argument("--max-black-ratio", type=float, default=0.85)
    parser.add_argument(
        "--max-white-ratio",
        type=float,
        default=0.12,
        help=(
            "Reject frames where more than this fraction of pixels are saturated white. "
            "This suppresses reflection-dominated frames that can mimic leukoplakia."
        ),
    )
    parser.add_argument(
        "--min-evidence-duration-sec",
        type=float,
        default=1.0,
        help="Mark a diagnosis low-confidence if frame-threshold evidence is shorter than this duration.",
    )
    parser.add_argument("--keyframes-per-video", type=int, default=3)
    parser.add_argument(
        "--save-keyframes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save ordinary top-scoring JPG keyframes in addition to diagnosis Grad-CAM outputs.",
    )
    parser.add_argument(
        "--no-keyframes",
        action="store_true",
        help="Deprecated compatibility flag; ordinary keyframes are already disabled by default.",
    )
    parser.add_argument(
        "--glottis-gate-model",
        type=Path,
        default=DEFAULT_GLOTTIS_GATE_MODEL,
        help="Swin binary glottis/non-glottis checkpoint used before 8-class disease inference.",
    )
    parser.add_argument(
        "--glottis-gate-threshold",
        type=float,
        default=0.94,
        help="Keep frames with binary prob_glottis >= this threshold before 8-class inference.",
    )
    parser.add_argument(
        "--glottis-gate-fallback-threshold",
        type=float,
        default=None,
        help="Fallback threshold when too few frames survive the high-specificity gate.",
    )
    parser.add_argument(
        "--min-glottis-gate-frames",
        type=int,
        default=5,
        help="Use fallback threshold if fewer than this many quality frames pass the main glottis gate.",
    )
    parser.add_argument(
        "--max-segment-gap-sec",
        type=float,
        default=1.0,
        help="Merge gate/evidence segments across short gaps such as glare or transient blur.",
    )
    parser.add_argument(
        "--glottis-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the binary glottis gate before the 8-class classifier.",
    )
    parser.add_argument(
        "--diagnosis-variant",
        default="voc_sum_gt_nonvoc",
        choices=[variant.name for variant in RULE_VARIANTS],
        help="Rule variant used for patient-level diagnosis summary and Grad-CAM frame selection.",
    )
    parser.add_argument(
        "--gradcam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write original/model-input/Grad-CAM comparison images for patient-level selected frames.",
    )
    parser.add_argument("--device", default=None, help="Override torch device, e.g. cpu or cuda:0.")
    return parser.parse_args()


def normalize_rel_path(path: Path | str) -> str:
    return Path(path).as_posix().strip("/")


def safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "video"


def safe_filename_part(text: str) -> str:
    text = re.sub(r'[\\/:\*\?"<>\|\s]+', "_", str(text))
    text = re.sub(r"_+", "_", text)
    return text.strip("._") or "item"


def label_zh(label: str) -> str:
    return LABEL_DISPLAY_ZH.get(label, label)


def configure_matplotlib_cjk_font() -> str | None:
    for font_path in CHINESE_FONT_CANDIDATES:
        path = Path(font_path)
        if not path.exists():
            continue
        font_manager.fontManager.addfont(str(path))
        font_name = font_manager.FontProperties(fname=str(path)).get_name()
        plt.rcParams["font.family"] = font_name
        plt.rcParams["axes.unicode_minus"] = False
        return font_name
    return None


def load_folder_label_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return DEFAULT_FOLDER_LABEL_MAP
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {normalize_rel_path(k): str(v) for k, v in data.items()}


def infer_label_from_folder(video_path: Path, video_root: Path, folder_map: dict[str, str]) -> str | None:
    try:
        rel_parent = video_path.parent.relative_to(video_root).as_posix()
    except ValueError:
        rel_parent = video_path.parent.as_posix()

    rel_parent = normalize_rel_path(rel_parent)
    if rel_parent in folder_map:
        return folder_map[rel_parent]
    for folder, label in folder_map.items():
        if rel_parent.endswith(folder):
            return label
    return None


def discover_videos(
    video_root: Path,
    folder_map: dict[str, str],
    allow_unlabeled: bool = False,
    unlabeled_label: str = "Unknown",
) -> list[VideoItem]:
    items: list[VideoItem] = []
    for path in sorted(video_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        label = infer_label_from_folder(path, video_root, folder_map)
        if label is None:
            if not allow_unlabeled:
                continue
            label = unlabeled_label
        rel = path.relative_to(video_root).with_suffix("")
        items.append(VideoItem(path, label, safe_id(rel.as_posix())))
    return items


def load_manifest(manifest_path: Path, video_root: Path) -> list[VideoItem]:
    df = pd.read_csv(manifest_path)
    required = {"video_path", "true_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    items: list[VideoItem] = []
    for row in df.to_dict("records"):
        video_path = Path(str(row["video_path"]))
        if not video_path.is_absolute():
            video_path = video_root / video_path
        video_id = str(row.get("video_id") or video_path.stem)
        start = row.get("valid_start_sec")
        end = row.get("valid_end_sec")
        start_sec = None if pd.isna(start) or start == "" else float(start)
        end_sec = None if pd.isna(end) or end == "" else float(end)
        items.append(VideoItem(video_path, str(row["true_label"]), safe_id(video_id), start_sec, end_sec))
    return items


def frame_quality(rgb: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    black_ratio = float((gray < 8).mean())
    white_ratio = float((gray > 245).mean())
    return {
        "brightness": brightness,
        "sharpness": sharpness,
        "black_ratio": black_ratio,
        "white_ratio": white_ratio,
    }


def quality_keep(metrics: dict[str, float], args: argparse.Namespace) -> bool:
    return (
        metrics["sharpness"] >= args.min_sharpness
        and args.min_brightness <= metrics["brightness"] <= args.max_brightness
        and metrics["black_ratio"] <= args.max_black_ratio
        and metrics["white_ratio"] <= args.max_white_ratio
    )


def append_sampled_frame(rows: list[dict], bgr: np.ndarray, frame_idx: int, native_fps: float, args: argparse.Namespace) -> None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    metrics = frame_quality(rgb)
    rows.append(
        {
            "time_sec": float(frame_idx / native_fps),
            "native_frame_idx": int(frame_idx),
            "rgb": rgb,
            "quality_keep": quality_keep(metrics, args),
            **metrics,
        }
    )


def sample_video_frames_seek(item: VideoItem, sample_fps: float, args: argparse.Namespace) -> list[dict]:
    cap = cv2.VideoCapture(str(item.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {item.video_path}")
    if sample_fps <= 0:
        cap.release()
        raise ValueError(f"--sample-fps must be > 0, got {sample_fps}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if native_fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Cannot determine fps/frame count for {item.video_path}")

    duration = frame_count / native_fps
    start_sec = max(0.0, item.start_sec or 0.0)
    end_sec = min(duration, item.end_sec if item.end_sec is not None else duration)
    if end_sec <= start_sec:
        cap.release()
        raise ValueError(f"Invalid video interval for {item.video_path}: {start_sec}-{end_sec}")

    step = 1.0 / sample_fps
    times = np.arange(start_sec, end_sec, step, dtype=np.float64)
    rows: list[dict] = []
    for time_sec in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)
        ok, bgr = cap.read()
        if not ok:
            continue
        append_sampled_frame(rows, bgr, int(round(time_sec * native_fps)), native_fps, args)

    cap.release()
    return rows


def sample_video_frames_sequential(item: VideoItem, sample_fps: float, args: argparse.Namespace) -> list[dict]:
    cap = cv2.VideoCapture(str(item.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {item.video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if native_fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Cannot determine fps/frame count for {item.video_path}")

    duration = frame_count / native_fps
    start_sec = max(0.0, item.start_sec or 0.0)
    end_sec = min(duration, item.end_sec if item.end_sec is not None else duration)
    if end_sec <= start_sec:
        cap.release()
        raise ValueError(f"Invalid video interval for {item.video_path}: {start_sec}-{end_sec}")

    if sample_fps <= 0:
        cap.release()
        raise ValueError(f"--sample-fps must be > 0, got {sample_fps}")

    times = np.arange(start_sec, end_sec, 1.0 / sample_fps, dtype=np.float64)
    target_indices = np.rint(times * native_fps).astype(np.int64)
    start_frame = max(0, int(math.floor(start_sec * native_fps)))
    end_frame = min(frame_count, int(math.ceil(end_sec * native_fps)))
    target_indices = np.unique(target_indices[(target_indices >= start_frame) & (target_indices < end_frame)])
    if target_indices.size == 0:
        cap.release()
        return []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_idx = start_frame
    rows: list[dict] = []

    for target_idx in target_indices.tolist():
        while current_idx < target_idx:
            if not cap.grab():
                cap.release()
                return rows
            current_idx += 1

        ok, bgr = cap.read()
        if not ok:
            break
        append_sampled_frame(rows, bgr, current_idx, native_fps, args)
        current_idx += 1

    cap.release()
    return rows


def sample_video_frames(item: VideoItem, sample_fps: float, args: argparse.Namespace) -> list[dict]:
    if args.decode_mode == "seek":
        return sample_video_frames_seek(item, sample_fps, args)
    return sample_video_frames_sequential(item, sample_fps, args)


def build_model(cfg: dict, model_path: Path, device: torch.device) -> HierarchicalImageClassifier:
    original_create_model = shared.timm.create_model

    def create_model_no_pretrained(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    shared.timm.create_model = create_model_no_pretrained
    try:
        model = HierarchicalImageClassifier(num_classes=len(LABEL_DICT), cfg=cfg).to(device)
    finally:
        shared.timm.create_model = original_create_model

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def build_glottis_gate_model(model_path: Path, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Glottis gate checkpoint not found: {model_path}. "
            "Pass --no-glottis-gate to disable it or --glottis-gate-model to set a checkpoint."
        )
    model, checkpoint = load_checkpoint_model(model_path, device)
    cfg = default_train_config()
    cfg.update(checkpoint.get("cfg", {}))
    model.eval()
    return model, cfg, checkpoint


def glottis_preprocess_frame(rgb: np.ndarray, cfg: dict) -> torch.Tensor:
    image = Image.fromarray(rgb).convert("RGB")
    image = shared.CropBlackBorders(cfg.get("crop_black_threshold", 15))(image)
    resize_size = int(cfg.get("resize_size", 256))
    image_size = int(cfg.get("image_size", 224))
    image = image.resize((resize_size, resize_size), Image.BICUBIC)
    left = max((resize_size - image_size) // 2, 0)
    top = max((resize_size - image_size) // 2, 0)
    image = image.crop((left, top, left + image_size, top + image_size))
    array = np.asarray(image, dtype=np.uint8)
    chw = np.ascontiguousarray(array.transpose(2, 0, 1))
    return torch.from_numpy(chw).float().div_(255.0)


@torch.inference_mode()
def infer_glottis_gate(
    model,
    frames: list[dict],
    cfg: dict,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
    threshold: float,
    fallback_threshold: float | None,
    min_gate_frames: int,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    quality_indices = [i for i, row in enumerate(frames) if row["quality_keep"]]
    probs = np.full(len(frames), np.nan, dtype=np.float32)
    if not quality_indices:
        return probs, np.zeros(len(frames), dtype=bool), float(threshold), False

    for start in range(0, len(quality_indices), batch_size):
        batch_indices = quality_indices[start : start + batch_size]
        tensors = [glottis_preprocess_frame(frames[idx]["rgb"], cfg) for idx in batch_indices]
        batch = torch.stack(tensors).to(device)
        batch = gpu_normalise(batch)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
            logits = model(batch)
        batch_probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs[batch_indices] = batch_probs

    finite = np.isfinite(probs)
    keep = finite & (probs >= float(threshold))
    fallback_used = False
    used_threshold = float(threshold)
    if (
        int(keep.sum()) < int(min_gate_frames)
        and fallback_threshold is not None
        and np.isfinite(float(fallback_threshold))
        and float(fallback_threshold) < float(threshold)
    ):
        fallback_keep = finite & (probs >= float(fallback_threshold))
        if int(fallback_keep.sum()) > int(keep.sum()):
            keep = fallback_keep
            used_threshold = float(fallback_threshold)
            fallback_used = True
    return probs, keep, used_threshold, fallback_used


def annotate_glottis_gate(
    frames: list[dict],
    probs: np.ndarray | None,
    keep_mask: np.ndarray,
    threshold: float,
    fallback_used: bool,
    enabled: bool,
) -> None:
    for idx, frame in enumerate(frames):
        frame["glottis_gate_enabled"] = bool(enabled)
        frame["glottis_prob"] = float(probs[idx]) if probs is not None and np.isfinite(probs[idx]) else float("nan")
        frame["glottis_gate_keep"] = bool(keep_mask[idx])
        frame["glottis_gate_threshold"] = float(threshold) if enabled else float("nan")
        frame["glottis_gate_fallback_used"] = bool(fallback_used)


@torch.inference_mode()
def infer_frames(
    model: HierarchicalImageClassifier,
    frames: list[dict],
    preprocess,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
    eligible_mask: np.ndarray | None = None,
) -> np.ndarray:
    if eligible_mask is None:
        kept_indices = [i for i, row in enumerate(frames) if row["quality_keep"]]
    else:
        kept_indices = [
            i for i, row in enumerate(frames)
            if row["quality_keep"] and bool(eligible_mask[i])
        ]
    probs = np.full((len(frames), len(LABEL_DICT)), np.nan, dtype=np.float32)
    if not kept_indices:
        return probs

    for start in range(0, len(kept_indices), batch_size):
        batch_indices = kept_indices[start : start + batch_size]
        tensors = []
        for idx in batch_indices:
            img = Image.fromarray(frames[idx]["rgb"])
            tensors.append(preprocess(img))
        batch = torch.stack(tensors).to(device)
        batch = gpu_normalise(batch)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
            logits = model(batch)
        batch_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probs[batch_indices] = batch_probs
    return probs


def top_fraction_mean(values: np.ndarray, fraction: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(finite.size * fraction)))
    top = np.partition(finite, -k)[-k:]
    return float(top.mean())


def build_gate_mask(variant: RuleVariant, probs: np.ndarray) -> np.ndarray:
    finite = np.isfinite(probs).all(axis=1)
    if variant.gate == "all":
        return finite
    non_voc = probs[:, shared.NON_VOC_LABEL]
    voc_indices = sorted(VOC_LABELS)
    voc_sum = probs[:, voc_indices].sum(axis=1)
    if variant.gate == "voc_margin":
        return finite & ((voc_sum - non_voc) > float(variant.threshold or 0.0))
    if variant.gate == "nonvoc_lt":
        return finite & (non_voc < float(variant.threshold))
    raise ValueError(f"Unknown gate variant: {variant}")


def merge_segments(
    times: np.ndarray,
    mask: np.ndarray,
    sample_fps: float,
    max_gap_factor: float = 1.5,
    max_gap_sec: float | None = None,
) -> list[tuple[float, float, int]]:
    selected = times[mask]
    if selected.size == 0:
        return []
    max_gap = max_gap_factor / sample_fps
    if max_gap_sec is not None and max_gap_sec > 0:
        max_gap = max(max_gap, float(max_gap_sec))
    segments: list[tuple[float, float, int]] = []
    start = float(selected[0])
    prev = float(selected[0])
    count = 1
    for t in selected[1:]:
        t = float(t)
        if t - prev <= max_gap:
            prev = t
            count += 1
            continue
        segments.append((start, prev + 1.0 / sample_fps, count))
        start = t
        prev = t
        count = 1
    segments.append((start, prev + 1.0 / sample_fps, count))
    return segments


def format_segments(segments: list[tuple[float, float, int]]) -> str:
    return ";".join(f"{s:.2f}-{e:.2f}s({n})" for s, e, n in segments)


def get_rule_variant(name: str) -> RuleVariant:
    for variant in RULE_VARIANTS:
        if variant.name == name:
            return variant
    raise ValueError(f"Unknown rule variant: {name}")


def patient_info(item: VideoItem, video_root: Path) -> tuple[str, str, str]:
    try:
        parts = item.video_path.relative_to(video_root).parts
        group = parts[0] if len(parts) > 1 else item.video_path.parent.name
    except ValueError:
        group = item.video_path.parent.name
    patient_name = group.split("_", 1)[0] if "_" in group else group
    return safe_id(group), patient_name, group


def segment_for_time(segments: list[tuple[float, float, int]], time_sec: float) -> str:
    for start, end, count in segments:
        if start <= time_sec <= end:
            return f"{start:.2f}-{end:.2f}s({count})"
    return ""


def build_diagnosis_record(
    item: VideoItem,
    frames: list[dict],
    probs: np.ndarray,
    video_rows: list[dict],
    args: argparse.Namespace,
) -> dict:
    variant = get_rule_variant(args.diagnosis_variant)
    row = next((r for r in video_rows if r["variant"] == variant.name), None)
    if row is None:
        raise RuntimeError(f"Missing video row for diagnosis variant: {variant.name}")

    patient_id, patient_name, patient_folder = patient_info(item, args.video_root)
    pred_label = str(row["pred_label"])
    pred_idx = LABEL_DICT[pred_label]
    times = np.array([frame["time_sec"] for frame in frames], dtype=np.float64)
    gate_mask = build_gate_mask(variant, probs)
    finite_pred = gate_mask & np.isfinite(probs[:, pred_idx])

    best_frame_idx = None
    selected_time = float("nan")
    selected_prob = float("nan")
    selected_rgb = None
    if finite_pred.any():
        candidate_indices = np.where(finite_pred)[0]
        best_frame_idx = int(candidate_indices[np.argmax(probs[candidate_indices, pred_idx])])
        selected_time = float(times[best_frame_idx])
        selected_prob = float(probs[best_frame_idx, pred_idx])
        selected_rgb = frames[best_frame_idx]["rgb"]

    evidence_mask = finite_pred & (probs[:, pred_idx] >= args.frame_threshold)
    evidence_segments = merge_segments(times, evidence_mask, args.sample_fps, max_gap_sec=args.max_segment_gap_sec)
    top_frame_segment = ""
    if best_frame_idx is not None:
        top_frame_segment = segment_for_time(evidence_segments, selected_time)
        if not top_frame_segment:
            top_frame_segment = f"{selected_time:.2f}s(top_frame_below_threshold)"
    low_confidence_reasons = []
    if not np.isfinite(selected_prob) or selected_prob < args.frame_threshold:
        low_confidence_reasons.append("selected_prob_below_frame_threshold")
    if args.glottis_gate and int(row.get("glottis_gate_frames", 0)) < int(args.min_glottis_gate_frames):
        low_confidence_reasons.append("too_few_glottis_gate_frames")
    evidence_duration_sec = float(evidence_mask.sum() / args.sample_fps)
    if evidence_duration_sec < float(args.min_evidence_duration_sec):
        low_confidence_reasons.append("too_short_evidence_duration")
    low_confidence = bool(low_confidence_reasons)

    return {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "patient_folder": patient_folder,
        "video_id": item.video_id,
        "video_path": str(item.video_path),
        "true_label": item.true_label,
        "label_known": item.true_label in LABEL_DICT,
        "diagnosis_variant": variant.name,
        "pred_label": pred_label,
        "pred_label_zh": label_zh(pred_label),
        "diagnosis_call": "review_required" if low_confidence else pred_label,
        "diagnosis_call_zh": "待人工复核" if low_confidence else label_zh(pred_label),
        "pred_score": row["pred_score"],
        "lesion_segments_sec": format_segments(evidence_segments),
        "selected_lesion_segment_sec": top_frame_segment,
        "selected_frame_time_sec": selected_time,
        "selected_frame_prob": selected_prob,
        "selected_frame_idx": int(frames[best_frame_idx]["native_frame_idx"]) if best_frame_idx is not None else "",
        "gate_frames": row["gate_frames"],
        "gate_duration_sec": row["gate_duration_sec"],
        "glottis_gate_frames": row.get("glottis_gate_frames", ""),
        "glottis_gate_threshold": row.get("glottis_gate_threshold", ""),
        "glottis_gate_fallback_used": row.get("glottis_gate_fallback_used", ""),
        "evidence_frames_at_threshold": int(evidence_mask.sum()),
        "evidence_duration_sec": evidence_duration_sec,
        "low_confidence": low_confidence,
        "low_confidence_reasons": ";".join(low_confidence_reasons),
        "gradcam_path": "",
        "_selected_rgb": selected_rgb,
        "_pred_idx": pred_idx,
    }


def score_video(
    item: VideoItem,
    frames: list[dict],
    probs: np.ndarray,
    candidate_indices: list[int],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    times = np.array([row["time_sec"] for row in frames], dtype=np.float64)
    non_voc = probs[:, shared.NON_VOC_LABEL]
    voc_sum = probs[:, sorted(VOC_LABELS)].sum(axis=1)
    all_scores_rows: list[dict] = []
    segment_rows: list[dict] = []

    for variant in RULE_VARIANTS:
        gate_mask = build_gate_mask(variant, probs)
        gate_segments = merge_segments(times, gate_mask, args.sample_fps, max_gap_sec=args.max_segment_gap_sec)
        score_by_idx = {
            idx: top_fraction_mean(probs[gate_mask, idx], args.top_fraction)
            for idx in candidate_indices
        }
        pred_idx = max(
            score_by_idx,
            key=lambda idx: -1.0 if math.isnan(score_by_idx[idx]) else score_by_idx[idx],
        )
        pred_label = LABEL_NAMES[pred_idx]
        pred_score = score_by_idx[pred_idx]
        label_known = item.true_label in LABEL_DICT
        evidence_mask = gate_mask & (probs[:, pred_idx] >= args.frame_threshold)
        evidence_segments = merge_segments(times, evidence_mask, args.sample_fps, max_gap_sec=args.max_segment_gap_sec)

        row = {
            "video_id": item.video_id,
            "video_path": str(item.video_path),
            "true_label": item.true_label,
            "label_known": label_known,
            "variant": variant.name,
            "pred_label": pred_label,
            "correct": pred_label == item.true_label if label_known else "",
            "pred_score": pred_score,
            "sampled_frames": len(frames),
            "quality_kept_frames": int(sum(bool(frame["quality_keep"]) for frame in frames)),
            "eight_class_inferred_frames": int(np.isfinite(probs).all(axis=1).sum()),
            "glottis_gate_enabled": bool(args.glottis_gate),
            "glottis_gate_frames": int(sum(bool(frame.get("glottis_gate_keep")) for frame in frames)),
            "glottis_gate_threshold": float(frames[0].get("glottis_gate_threshold", float("nan"))) if frames else float("nan"),
            "glottis_gate_fallback_used": bool(frames[0].get("glottis_gate_fallback_used", False)) if frames else False,
            "gate_frames": int(gate_mask.sum()),
            "gate_duration_sec": float(gate_mask.sum() / args.sample_fps),
            "gate_segments": format_segments(gate_segments),
            "evidence_frames_at_threshold": int(evidence_mask.sum()),
            "evidence_duration_sec": float(evidence_mask.sum() / args.sample_fps),
            "evidence_segments": format_segments(evidence_segments),
            "non_voc_prob_mean": float(np.nanmean(non_voc)),
            "non_voc_prob_p90": float(np.nanpercentile(non_voc, 90)),
            "voc_sum_prob_mean": float(np.nanmean(voc_sum)),
            "voc_sum_prob_p90": float(np.nanpercentile(voc_sum, 90)),
        }
        for idx in candidate_indices:
            row[f"score_{LABEL_NAMES[idx]}"] = score_by_idx[idx]
            row[f"frames_{LABEL_NAMES[idx]}_ge_threshold"] = int(
                (gate_mask & (probs[:, idx] >= args.frame_threshold)).sum()
            )
        all_scores_rows.append(row)

        segment_rows.extend(
            {
                "video_id": item.video_id,
                "true_label": item.true_label,
                "variant": variant.name,
                "segment_type": "gate",
                "start_sec": start,
                "end_sec": end,
                "duration_sec": end - start,
                "frame_count": count,
                "pred_label": pred_label,
            }
            for start, end, count in gate_segments
        )
        segment_rows.extend(
            {
                "video_id": item.video_id,
                "true_label": item.true_label,
                "variant": variant.name,
                "segment_type": "pred_evidence",
                "start_sec": start,
                "end_sec": end,
                "duration_sec": end - start,
                "frame_count": count,
                "pred_label": pred_label,
            }
            for start, end, count in evidence_segments
        )

    return all_scores_rows, segment_rows


def write_frame_rows(
    writer: csv.DictWriter,
    item: VideoItem,
    frames: list[dict],
    probs: np.ndarray,
) -> None:
    labels = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
    for idx, row in enumerate(frames):
        prob = probs[idx]
        finite = np.isfinite(prob).all()
        if finite:
            pred_idx = int(np.argmax(prob))
            non_voc = float(prob[shared.NON_VOC_LABEL])
            voc_sum = float(prob[sorted(VOC_LABELS)].sum())
        else:
            pred_idx = -1
            non_voc = float("nan")
            voc_sum = float("nan")
        out = {
            "video_id": item.video_id,
            "video_path": str(item.video_path),
            "true_label": item.true_label,
            "time_sec": row["time_sec"],
            "native_frame_idx": row["native_frame_idx"],
            "quality_keep": row["quality_keep"],
            "brightness": row["brightness"],
            "sharpness": row["sharpness"],
            "black_ratio": row["black_ratio"],
            "white_ratio": row["white_ratio"],
            "glottis_gate_enabled": row.get("glottis_gate_enabled", False),
            "glottis_prob": row.get("glottis_prob", float("nan")),
            "glottis_gate_keep": row.get("glottis_gate_keep", row["quality_keep"]),
            "glottis_gate_threshold": row.get("glottis_gate_threshold", float("nan")),
            "glottis_gate_fallback_used": row.get("glottis_gate_fallback_used", False),
            "pred_argmax": LABEL_NAMES.get(pred_idx, "quality_filtered"),
            "non_voc_prob": non_voc,
            "voc_sum_prob": voc_sum,
        }
        for label_idx, label_name in enumerate(labels):
            out[f"prob_{label_name}"] = float(prob[label_idx]) if finite else float("nan")
        writer.writerow(out)


def save_keyframes(
    output_dir: Path,
    item: VideoItem,
    frames: list[dict],
    probs: np.ndarray,
    video_rows: list[dict],
    args: argparse.Namespace,
) -> None:
    if args.no_keyframes or not args.save_keyframes or args.keyframes_per_video <= 0:
        return

    for row in video_rows:
        variant = row["variant"]
        pred_label = row["pred_label"]
        pred_idx = LABEL_DICT[pred_label]
        gate_mask = build_gate_mask(
            next(v for v in RULE_VARIANTS if v.name == variant),
            probs,
        )
        candidate = np.where(gate_mask & np.isfinite(probs[:, pred_idx]))[0]
        if candidate.size == 0:
            continue
        ranked = candidate[np.argsort(probs[candidate, pred_idx])][::-1]
        target_dir = output_dir / "keyframes" / variant / f"{item.video_id}__pred_{safe_id(pred_label)}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for rank, frame_idx in enumerate(ranked[: args.keyframes_per_video], start=1):
            time_sec = frames[int(frame_idx)]["time_sec"]
            score = float(probs[int(frame_idx), pred_idx])
            filename = f"rank{rank:02d}_t{time_sec:07.2f}s_p{score:.3f}.jpg"
            Image.fromarray(frames[int(frame_idx)]["rgb"]).save(target_dir / filename, quality=92)


def get_gradcam_target_layers(model: HierarchicalImageClassifier):
    last_stage = model.backbone.layers[-1]
    return [last_stage.blocks[-1]]


def gradcam_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 3:
        batch, hw, channels = tensor.shape
        h = w = int(hw ** 0.5)
        return tensor.reshape(batch, h, w, channels).permute(0, 3, 1, 2)
    if tensor.dim() == 4 and tensor.shape[-1] != tensor.shape[-2]:
        return tensor.permute(0, 3, 1, 2)
    return tensor


def build_cam_overlay(rgb_img: np.ndarray, cam_map: np.ndarray) -> np.ndarray:
    base = rgb_img.astype(np.float32) / 255.0
    cam_resized = cv2.resize(cam_map, (rgb_img.shape[1], rgb_img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.uint8(255 * np.clip(0.48 * heatmap + 0.52 * base, 0, 1))


def save_gradcam_comparisons(
    model: HierarchicalImageClassifier,
    records: list[dict],
    cfg: dict,
    device: torch.device,
    output_dir: Path,
) -> None:
    records_with_frames = [record for record in records if record.get("_selected_rgb") is not None]
    if not records_with_frames:
        return

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    gradcam_dir = output_dir / "diagnosis_gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib_cjk_font()
    tensor_preprocess = _build_base_preprocess(cfg, to_tensor=True)
    visual_preprocess = _build_base_preprocess(cfg, to_tensor=False)
    cam = GradCAM(
        model=model,
        target_layers=get_gradcam_target_layers(model),
        reshape_transform=gradcam_reshape_transform,
    )

    model.eval()
    for record in records_with_frames:
        raw_rgb = record["_selected_rgb"]
        pred_idx = int(record["_pred_idx"])
        raw_pil = Image.fromarray(raw_rgb)
        model_input_rgb = np.array(visual_preprocess(raw_pil), dtype=np.uint8)
        image_tensor = tensor_preprocess(raw_pil).unsqueeze(0).to(device)
        image_tensor = gpu_normalise(image_tensor)
        cam_input = image_tensor.detach().clone().requires_grad_(True)
        target = [ClassifierOutputTarget(pred_idx)]
        with torch.enable_grad():
            grayscale_cam = cam(input_tensor=cam_input, targets=target)[0]
        overlay = build_cam_overlay(model_input_rgb, grayscale_cam)

        video_stem = Path(str(record["video_path"])).stem
        pred_label_name = str(record["pred_label"])
        pred_label_display = label_zh(pred_label_name)
        filename = (
            f"{safe_filename_part(record['patient_name'])}_{safe_filename_part(record['patient_id'])}"
            f"__{safe_filename_part(video_stem)}"
            f"__{safe_filename_part(pred_label_display)}__t{float(record['selected_frame_time_sec']):07.2f}s.png"
        )
        out_path = gradcam_dir / filename

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor="white")
        axes[0].imshow(raw_rgb)
        axes[0].set_title("Raw video frame")
        axes[1].imshow(model_input_rgb)
        axes[1].set_title("Model input")
        axes[2].imshow(overlay)
        axes[2].set_title(f"Grad-CAM: {pred_label_display}")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"{record['patient_name']} ({record['patient_id']}) | {pred_label_display} | "
            f"t={float(record['selected_frame_time_sec']):.2f}s | p={float(record['selected_frame_prob']):.3f}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
        record["gradcam_path"] = str(out_path)

    cam.activations_and_grads.release()


def public_diagnosis_columns() -> list[str]:
    return [
        "patient_id",
        "patient_name",
        "patient_folder",
        "video_count",
        "selected_video_id",
        "selected_video_path",
        "diagnosis_variant",
        "pred_label",
        "pred_label_zh",
        "diagnosis_call",
        "diagnosis_call_zh",
        "pred_score",
        "lesion_segments_sec",
        "selected_lesion_segment_sec",
        "selected_frame_time_sec",
        "selected_frame_prob",
        "gate_duration_sec",
        "glottis_gate_frames",
        "glottis_gate_threshold",
        "glottis_gate_fallback_used",
        "evidence_duration_sec",
        "low_confidence",
        "low_confidence_reasons",
        "per_video_predictions",
        "gradcam_path",
    ]


def write_diagnosis_outputs(
    records: list[dict],
    model: HierarchicalImageClassifier,
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    selected_records: list[dict] = []
    summary_rows: list[dict] = []
    records_by_patient: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        records_by_patient[str(record["patient_id"])].append(record)

    for patient_id in sorted(records_by_patient):
        patient_records = records_by_patient[patient_id]
        selected_record = sorted(
            patient_records,
            key=lambda record: (
                -1.0 if not np.isfinite(record["selected_frame_prob"]) else float(record["selected_frame_prob"]),
                -1.0 if pd.isna(record["pred_score"]) else float(record["pred_score"]),
            ),
            reverse=True,
        )[0]
        selected_records.append(selected_record)

    if args.gradcam:
        save_gradcam_comparisons(model, selected_records, cfg, device, output_dir)

    public_records = [{k: v for k, v in record.items() if not k.startswith("_")} for record in records]
    evidence_df = pd.DataFrame(public_records)
    evidence_df.to_csv(output_dir / "diagnosis_evidence.csv", index=False)

    if selected_records:
        for selected_record in selected_records:
            patient_records = records_by_patient[str(selected_record["patient_id"])]
            per_video = "; ".join(
                f"{record['video_id']}:{record['pred_label']}({float(record['selected_frame_prob']):.3f})"
                for record in patient_records
                if np.isfinite(record["selected_frame_prob"])
            )
            summary_rows.append(
                {
                    "patient_id": selected_record["patient_id"],
                    "patient_name": selected_record["patient_name"],
                    "patient_folder": selected_record["patient_folder"],
                    "video_count": int(len(patient_records)),
                    "selected_video_id": selected_record["video_id"],
                    "selected_video_path": selected_record["video_path"],
                    "diagnosis_variant": selected_record["diagnosis_variant"],
                    "pred_label": selected_record["pred_label"],
                    "pred_label_zh": selected_record["pred_label_zh"],
                    "diagnosis_call": selected_record["diagnosis_call"],
                    "diagnosis_call_zh": selected_record["diagnosis_call_zh"],
                    "pred_score": selected_record["pred_score"],
                    "lesion_segments_sec": selected_record["lesion_segments_sec"],
                    "selected_lesion_segment_sec": selected_record["selected_lesion_segment_sec"],
                    "selected_frame_time_sec": selected_record["selected_frame_time_sec"],
                    "selected_frame_prob": selected_record["selected_frame_prob"],
                    "gate_duration_sec": selected_record["gate_duration_sec"],
                    "glottis_gate_frames": selected_record["glottis_gate_frames"],
                    "glottis_gate_threshold": selected_record["glottis_gate_threshold"],
                    "glottis_gate_fallback_used": selected_record["glottis_gate_fallback_used"],
                    "evidence_duration_sec": selected_record["evidence_duration_sec"],
                    "low_confidence": selected_record["low_confidence"],
                    "low_confidence_reasons": selected_record["low_confidence_reasons"],
                    "per_video_predictions": per_video,
                    "gradcam_path": selected_record["gradcam_path"],
                }
            )

    summary_df = pd.DataFrame(summary_rows, columns=public_diagnosis_columns())
    summary_df.to_csv(output_dir / "diagnosis_summary.csv", index=False)


def resolve_candidate_indices(value: str, items: list[VideoItem]) -> list[int]:
    if value == "auto":
        names = sorted({item.true_label for item in items if item.true_label in LABEL_DICT}, key=LABEL_DICT.get)
        if not names:
            names = [LABEL_NAMES[idx] for idx in sorted(VOC_LABELS)]
    elif value == "all-voc":
        names = [LABEL_NAMES[idx] for idx in sorted(VOC_LABELS)]
    else:
        names = [part.strip() for part in value.split(",") if part.strip()]
    missing = [name for name in names if name not in LABEL_DICT]
    if missing:
        raise ValueError(f"Candidate labels are absent from this checkpoint/config: {missing}")
    return [LABEL_DICT[name] for name in names]


def summarize(video_rows: list[dict], candidate_indices: list[int], args: argparse.Namespace) -> dict:
    by_variant: dict[str, dict] = {}
    rows_by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in video_rows:
        rows_by_variant[row["variant"]].append(row)

    for variant, rows in rows_by_variant.items():
        labeled_rows = [row for row in rows if row.get("label_known")]
        correct = sum(1 for row in labeled_rows if row["correct"] is True)
        total = len(rows)
        labeled_total = len(labeled_rows)
        by_variant[variant] = {
            "videos": total,
            "labeled_videos": labeled_total,
            "correct": correct,
            "accuracy": correct / labeled_total if labeled_total else float("nan"),
            "mean_gate_frames": float(np.mean([row["gate_frames"] for row in rows])) if rows else 0.0,
            "mean_gate_duration_sec": float(np.mean([row["gate_duration_sec"] for row in rows])) if rows else 0.0,
            "pred_counts": {
                str(label): int(count)
                for label, count in pd.Series([row["pred_label"] for row in rows]).value_counts().items()
            } if rows else {},
        }
    return {
        "candidate_labels": [LABEL_NAMES[idx] for idx in candidate_indices],
        "sample_fps": args.sample_fps,
        "decode_mode": args.decode_mode,
        "batch_size": args.batch_size,
        "amp": bool(args.amp),
        "glottis_gate": {
            "enabled": bool(args.glottis_gate),
            "model": str(args.glottis_gate_model) if args.glottis_gate else "",
            "threshold": float(args.glottis_gate_threshold),
            "fallback_threshold": (
                None
                if args.glottis_gate_fallback_threshold is None
                else float(args.glottis_gate_fallback_threshold)
            ),
            "min_gate_frames": int(args.min_glottis_gate_frames),
            "max_segment_gap_sec": float(args.max_segment_gap_sec),
        },
        "top_fraction": args.top_fraction,
        "frame_threshold": args.frame_threshold,
        "save_keyframes": bool(args.save_keyframes and not args.no_keyframes),
        "quality_filter": {
            "min_sharpness": args.min_sharpness,
            "min_brightness": args.min_brightness,
            "max_brightness": args.max_brightness,
            "max_black_ratio": args.max_black_ratio,
            "max_white_ratio": args.max_white_ratio,
        },
        "min_evidence_duration_sec": args.min_evidence_duration_sec,
        "variants": by_variant,
    }


def main() -> None:
    args = parse_args()
    args.video_root = args.video_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(args.config))
    init_label_mapping(cfg)
    device = torch.device(args.device) if args.device else setup_device()

    folder_map = load_folder_label_map(args.folder_label_map)
    items = (
        load_manifest(args.manifest, args.video_root)
        if args.manifest
        else discover_videos(args.video_root, folder_map, args.allow_unlabeled, args.unlabeled_label)
    )
    if not items:
        raise RuntimeError(f"No usable videos found under {args.video_root}")

    candidate_indices = resolve_candidate_indices(args.candidate_labels, items)
    missing_labels = sorted(
        {
            item.true_label
            for item in items
            if item.true_label not in LABEL_DICT and item.true_label != args.unlabeled_label
        }
    )
    if missing_labels:
        raise ValueError(f"Video labels not present in checkpoint/config: {missing_labels}")

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Videos: {len(items)} from {args.video_root}")
    print(f"Checkpoint labels: {list(LABEL_DICT.keys())}")
    print(f"Video candidate labels: {[LABEL_NAMES[idx] for idx in candidate_indices]}")
    print(f"Decode mode: {args.decode_mode}, batch size: {args.batch_size}, AMP: {args.amp and device.type == 'cuda'}")

    preprocess = _build_base_preprocess(cfg, to_tensor=True)
    model = build_model(cfg, args.model, device)
    glottis_model = None
    glottis_cfg = None
    if args.glottis_gate:
        glottis_model, glottis_cfg, glottis_checkpoint = build_glottis_gate_model(
            args.glottis_gate_model.resolve(), device
        )
        checkpoint_threshold = glottis_checkpoint.get("recommended_threshold")
        if checkpoint_threshold is not None and args.glottis_gate_threshold is None:
            args.glottis_gate_threshold = float(checkpoint_threshold)
        print(
            f"Glottis gate: {args.glottis_gate_model} "
            f"threshold={args.glottis_gate_threshold:.2f}, "
            f"fallback={args.glottis_gate_fallback_threshold if args.glottis_gate_fallback_threshold is not None else 'disabled'}"
        )

    labels = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
    frame_fieldnames = [
        "video_id",
        "video_path",
        "true_label",
        "time_sec",
        "native_frame_idx",
        "quality_keep",
        "brightness",
        "sharpness",
        "black_ratio",
        "white_ratio",
        "glottis_gate_enabled",
        "glottis_prob",
        "glottis_gate_keep",
        "glottis_gate_threshold",
        "glottis_gate_fallback_used",
        "pred_argmax",
        "non_voc_prob",
        "voc_sum_prob",
        *[f"prob_{label}" for label in labels],
    ]

    all_video_rows: list[dict] = []
    all_segment_rows: list[dict] = []
    diagnosis_records: list[dict] = []
    frame_csv_path = args.output_dir / "frame_predictions.csv"
    with frame_csv_path.open("w", newline="", encoding="utf-8") as f:
        frame_writer = csv.DictWriter(f, fieldnames=frame_fieldnames)
        frame_writer.writeheader()
        for item in items:
            print(f"Processing {item.video_id}: {item.video_path}")
            frames = sample_video_frames(item, args.sample_fps, args)
            if args.glottis_gate:
                glottis_probs, glottis_keep, used_threshold, fallback_used = infer_glottis_gate(
                    glottis_model,
                    frames,
                    glottis_cfg,
                    device,
                    args.batch_size,
                    args.amp,
                    args.glottis_gate_threshold,
                    args.glottis_gate_fallback_threshold,
                    args.min_glottis_gate_frames,
                )
            else:
                glottis_probs = None
                glottis_keep = np.array([bool(frame["quality_keep"]) for frame in frames], dtype=bool)
                used_threshold = float("nan")
                fallback_used = False
            annotate_glottis_gate(
                frames,
                glottis_probs,
                glottis_keep,
                used_threshold,
                fallback_used,
                args.glottis_gate,
            )
            probs = infer_frames(
                model,
                frames,
                preprocess,
                device,
                args.batch_size,
                args.amp,
                eligible_mask=glottis_keep if args.glottis_gate else None,
            )
            write_frame_rows(frame_writer, item, frames, probs)
            video_rows, segment_rows = score_video(item, frames, probs, candidate_indices, args)
            all_video_rows.extend(video_rows)
            all_segment_rows.extend(segment_rows)
            diagnosis_records.append(build_diagnosis_record(item, frames, probs, video_rows, args))
            save_keyframes(args.output_dir, item, frames, probs, video_rows, args)

    video_df = pd.DataFrame(all_video_rows)
    segment_df = pd.DataFrame(all_segment_rows)
    video_df.to_csv(args.output_dir / "video_predictions.csv", index=False)
    segment_df.to_csv(args.output_dir / "video_segments.csv", index=False)
    summary = summarize(all_video_rows, candidate_indices, args)
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_diagnosis_outputs(diagnosis_records, model, cfg, device, args.output_dir, args)

    print("\nSummary by variant:")
    for variant, stats in summary["variants"].items():
        if stats["labeled_videos"]:
            print(
                f"  {variant}: {stats['correct']}/{stats['labeled_videos']} labeled "
                f"acc={stats['accuracy']:.3f}, mean_gate_frames={stats['mean_gate_frames']:.1f}"
            )
        else:
            print(
                f"  {variant}: videos={stats['videos']}, labeled=0, "
                f"acc=n/a, mean_gate_frames={stats['mean_gate_frames']:.1f}"
            )
    print(f"\nWrote: {args.output_dir}")
    print(f"Wrote diagnosis summary: {args.output_dir / 'diagnosis_summary.csv'}")


if __name__ == "__main__":
    main()
