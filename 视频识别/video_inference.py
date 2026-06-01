#!/usr/bin/env python3
"""Run image-checkpoint inference over laryngeal videos.

The script treats a video as a bag of sampled frames. It does not train a
temporal model. The standard pipeline applies quality filtering, binary glottis
gating, and YOLO-Pose+DINOv3 ROI acceptance before aggregating 8-class disease
probabilities over accepted ROI crops. If strict ROI accepts no frame in a
video, the pipeline falls back to full-frame Swin inference for frames that
passed the upstream quality and glottis gates.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "图像识别"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shared  # noqa: E402
from shared import (  # noqa: E402
    BEST_MODEL_PATH,
    LABEL_DICT,
    LABEL_NAMES,
    RESULTS_DIR,
    RESULTS_ROOT,
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
    os.environ.get("LARYNX_VIDEO_ROOT", "/mnt/data/LarynxData/videos/classified_videos")
).expanduser()
DEFAULT_VIDEO_LABEL_MAP = os.environ.get("LARYNX_VIDEO_LABEL_MAP")
DEFAULT_RENDER_VIDEO_DIR = Path(
    os.environ.get("LARYNX_OUTPUT_VIDEO_DIR", str(Path(shared.WORKSPACE_DIR) / "Output_Video"))
).expanduser()
def _first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


DEFAULT_EIGHT_CLASS_MODEL = Path(
    os.environ.get(
        "LARYNX_EIGHT_CLASS_MODEL",
        str(
            _first_existing_path(
                Path(RESULTS_DIR) / "roi_reflection" / "eight_class_roi_soft" / "best_model.pth",
                Path(RESULTS_ROOT) / "bagls_roi_reflection" / "best_model.pth",
                Path(BEST_MODEL_PATH),
                Path(RESULTS_ROOT) / "main" / "best_model.pth",
            )
        ),
    )
)
DEFAULT_GLOTTIS_GATE_MODEL = Path(
    os.environ.get(
        "LARYNX_GLOTTIS_GATE_MODEL",
        str(
            _first_existing_path(
                Path(RESULTS_DIR)
                / "glottis_binary_benchmarks"
                / "20260505_183333_parallel"
                / "swin_base"
                / "best_model.pth",
                Path(RESULTS_ROOT)
                / "main"
                / "glottis_binary_benchmarks"
                / "20260505_183333_parallel"
                / "swin_base"
                / "best_model.pth",
            )
        ),
    )
)
DEFAULT_YOLO_POSE_ROI_MODEL = Path(
    os.environ.get(
        "LARYNX_ROI_LOCALIZER_MODEL",
        "/home/or1ngelinux/CVProjects/Larynx/YOLOPoseVocalFold/Results/"
        "dinov3_aux_full_pipeline_v12_20260524/stage1_pose/"
        "yolo11m_stage1_manual_mixedneg60_blackpad_containment_l0p05_v12/"
        "weights/best.pt",
    )
)
DEFAULT_DINO_AUX_ROI_MODEL = Path(
    os.environ.get(
        "LARYNX_ROI_DINO_AUX_MODEL",
        "/home/or1ngelinux/CVProjects/Larynx/YOLOPoseVocalFold/Results/"
        "dinov3_keypoint_aux/"
        "dinov3_vits16_oriented_point_region_hardneg_448_ldp200_v12_20260524/"
        "weights/best_aux_head.pt",
    )
)
DEFAULT_DINO_AUX_CODE_ROOT = Path(
    os.environ.get(
        "LARYNX_ROI_DINO_AUX_CODE_ROOT",
        "/home/or1ngelinux/CVProjects/Larynx/YOLOPoseVocalFold",
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
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


@dataclass(frozen=True)
class VideoItem:
    video_path: Path
    true_label: str
    video_id: str
    start_sec: float | None = None
    end_sec: float | None = None


@dataclass
class DinoAuxBundle:
    checkpoint_path: Path
    code_root: Path
    device: torch.device
    imgsz: int
    extractor: Any
    head: Any
    aux_cfg: Any
    prediction_input_cls: Any
    attach_aux_score: Any


@dataclass
class RoiGateBundle:
    model: Any
    weights_path: Path
    device_arg: str | None
    aux: DinoAuxBundle | None = None


STANDARD_VARIANT = "standard_roi_glottis_top_fraction"


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
    parser.add_argument("--model", type=Path, default=DEFAULT_EIGHT_CLASS_MODEL)
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
    parser.add_argument(
        "--leukoplakia-min-evidence-duration-sec",
        type=float,
        default=2.0,
        help="Additional low-confidence floor for leukoplakia calls because short clean-ROI bursts can mimic plaques.",
    )
    parser.add_argument(
        "--normal-conflict-prob-threshold",
        type=float,
        default=0.85,
        help="Mark low-confidence when Normal has a high peak but too little sustained Normal evidence.",
    )
    parser.add_argument(
        "--normal-conflict-max-evidence-duration-sec",
        type=float,
        default=1.0,
        help="Normal evidence shorter than this duration is treated as a conflict, not an automatic Normal rescue.",
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
        help="Enable the binary glottis gate before ROI and 8-class inference.",
    )
    parser.add_argument(
        "--roi-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable YOLO-Pose+DINOv3 ROI acceptance and square crop preprocessing.",
    )
    parser.add_argument(
        "--roi-localizer-model",
        type=Path,
        default=DEFAULT_YOLO_POSE_ROI_MODEL,
        help="YOLO-Pose glottic ROI checkpoint used to crop every frame before classification.",
    )
    parser.add_argument(
        "--roi-valid-threshold",
        type=float,
        default=0.25,
        help="Minimum YOLO ROI bbox confidence used before DINOv3 auxiliary scoring.",
    )
    parser.add_argument(
        "--roi-accept-threshold",
        type=float,
        default=0.25,
        help="Minimum combined YOLO/DINO ROI confidence required to keep a frame for Swin inference.",
    )
    parser.add_argument(
        "--roi-dino-aux-model",
        type=Path,
        default=DEFAULT_DINO_AUX_ROI_MODEL,
        help="DINOv3 auxiliary point-region checkpoint used together with the YOLO-Pose ROI localizer.",
    )
    parser.add_argument(
        "--roi-dino-aux-code-root",
        type=Path,
        default=DEFAULT_DINO_AUX_CODE_ROOT,
        help="Checkout containing the DINOv3 auxiliary scoring code.",
    )
    parser.add_argument(
        "--roi-dino-aux",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the DINOv3 auxiliary ROI scorer after YOLO-Pose produces candidate keypoints.",
    )
    parser.add_argument(
        "--roi-dino-aux-accept-threshold",
        type=float,
        default=0.30,
        help="Minimum DINOv3 point-region score required to accept a YOLO ROI crop.",
    )
    parser.add_argument(
        "--roi-dino-aux-imgsz",
        type=int,
        default=None,
        help="Override DINOv3 auxiliary input size; defaults to the checkpoint config.",
    )
    parser.add_argument(
        "--roi-dino-aux-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for DINOv3 auxiliary ROI scoring. auto reuses the main inference device.",
    )
    parser.add_argument(
        "--roi-black-border-luma-floor",
        type=float,
        default=8.0,
        help="Brightness floor for removing existing black borders before YOLO-Pose/DINO ROI scoring.",
    )
    parser.add_argument(
        "--roi-imgsz",
        type=int,
        default=960,
        help="YOLO-Pose inference image size.",
    )
    parser.add_argument(
        "--roi-blackpad-fraction",
        type=float,
        default=0.30,
        help="Black border fraction added to each side before YOLO-Pose ROI localization.",
    )
    parser.add_argument(
        "--roi-blackpad-min-padding",
        type=int,
        default=80,
        help="Minimum black border pixels added before YOLO-Pose ROI localization.",
    )
    parser.add_argument(
        "--roi-cache-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for YOLO-Pose ROI inference. auto reuses the main inference device.",
    )
    parser.add_argument(
        "--roi-allow-missing-model",
        action="store_true",
        help="Use full-frame square fallback if the YOLO-Pose ROI model is missing; otherwise missing files are errors.",
    )
    parser.add_argument(
        "--roi-video-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If strict ROI accepts no frame in a video, infer on upstream quality+glottis full frames.",
    )
    parser.add_argument(
        "--gradcam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write original/model-input/Grad-CAM comparison images for patient-level selected frames.",
    )
    parser.add_argument(
        "--render-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render an annotated 8-fps diagnostic overlay video with ROI soft mask, Grad-CAM, and vote trend.",
    )
    parser.add_argument(
        "--render-video-dir",
        type=Path,
        default=DEFAULT_RENDER_VIDEO_DIR,
        help="Directory for annotated overlay videos. Defaults to the workspace Output_Video folder.",
    )
    parser.add_argument(
        "--render-chart-window-sec",
        type=float,
        default=5.0,
        help="Rolling time window shown in the disease-vote trend chart.",
    )
    parser.add_argument(
        "--render-roi-min-brightness",
        type=float,
        default=0.35,
        help="Minimum brightness multiplier outside low-probability ROI regions in rendered video.",
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


def resolve_roi_device(value: str, main_device: torch.device) -> torch.device:
    if value == "auto":
        return main_device
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--roi-cache-device cuda was requested but CUDA is not available.")
        return main_device if main_device.type == "cuda" else torch.device("cuda")
    raise ValueError(f"Unknown ROI cache device: {value}")


def resolve_checkpoint_path(path: Path, role: str) -> Path:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"ROI {role} checkpoint not found: {path}. "
            "Pass --no-roi-gate to disable ROI cropping or set --roi-localizer-model."
        )
    return path.resolve()


def resolve_aux_code_root(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    module_path = path / "tools" / "score_predictions_with_dinov3_aux.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"DINOv3 auxiliary ROI code not found: {module_path}. "
            "Set --roi-dino-aux-code-root or pass --no-roi-dino-aux."
        )
    return path


def yolo_device_arg(device: torch.device) -> str:
    if device.type == "cpu":
        return "cpu"
    index = device.index if device.index is not None else torch.cuda.current_device()
    return str(index)


def import_dino_aux_module(code_root: Path) -> Any:
    module_name = "_larynx_roi_dinov3_aux_score"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    module_path = code_root / "tools" / "score_predictions_with_dinov3_aux.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load DINOv3 auxiliary scorer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_dino_aux_bundle(args: argparse.Namespace, main_device: torch.device) -> DinoAuxBundle:
    aux_path = resolve_checkpoint_path(args.roi_dino_aux_model, "DINOv3 auxiliary")
    code_root = resolve_aux_code_root(args.roi_dino_aux_code_root)
    aux_device = resolve_roi_device(args.roi_dino_aux_device, main_device)
    score_module = import_dino_aux_module(code_root)
    checkpoint = torch.load(aux_path, map_location="cpu")
    aux_cfg = score_module.load_aux_config_from_checkpoint(checkpoint)
    imgsz = int(args.roi_dino_aux_imgsz or checkpoint.get("config", {}).get("imgsz", 448))
    args.roi_dino_aux_imgsz = imgsz
    extractor = score_module.load_dinov3_extractor(aux_cfg, aux_device)
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    head = score_module.DinoV3KeypointAuxHead(
        int(checkpoint["feature_dim"]),
        patch_output_size=aux_cfg.oriented_patch_output_size,
        point_hidden_dim=aux_cfg.point_hidden_dim,
        include_coordinates=aux_cfg.include_point_coordinates,
        include_valid_mask=aux_cfg.include_valid_mask,
    ).to(aux_device)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    return DinoAuxBundle(
        checkpoint_path=aux_path,
        code_root=code_root,
        device=aux_device,
        imgsz=imgsz,
        extractor=extractor,
        head=head,
        aux_cfg=aux_cfg,
        prediction_input_cls=score_module.DinoV3PredictionInput,
        attach_aux_score=score_module.attach_aux_score,
    )


def build_roi_gate_bundle(args: argparse.Namespace, main_device: torch.device) -> RoiGateBundle | None:
    try:
        roi_device = resolve_roi_device(args.roi_cache_device, main_device)
        weights_path = resolve_checkpoint_path(args.roi_localizer_model, "YOLO-Pose localizer")
    except FileNotFoundError as exc:
        if args.roi_allow_missing_model:
            print(f"WARNING: ROI localizer unavailable: {exc}. Using full-frame square fallback.")
            return None
        raise

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO-Pose ROI localization. "
            "Install it or pass --no-roi-gate to bypass ROI cropping."
        ) from exc
    try:
        aux_bundle = build_dino_aux_bundle(args, main_device) if args.roi_dino_aux else None
    except FileNotFoundError as exc:
        if args.roi_allow_missing_model:
            print(f"WARNING: ROI auxiliary scorer unavailable: {exc}. Using full-frame square fallback.")
            return None
        raise
    return RoiGateBundle(
        model=YOLO(str(weights_path)),
        weights_path=weights_path,
        device_arg=yolo_device_arg(roi_device),
        aux=aux_bundle,
    )


def _scale_bbox_xyxy(box: np.ndarray, width: int, height: int) -> np.ndarray:
    box = box.astype(np.float32, copy=True)
    if not np.isfinite(box).all():
        return np.full(4, np.nan, dtype=np.float32)
    if float(np.nanmax(np.abs(box))) <= 1.5:
        box[[0, 2]] *= float(width)
        box[[1, 3]] *= float(height)
    x1, y1, x2, y2 = box.tolist()
    x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    if x2 <= x1 or y2 <= y1:
        return np.full(4, np.nan, dtype=np.float32)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _crop_roi(rgb: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    if not np.isfinite(bbox).all():
        return None
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    left = max(0, min(w - 1, int(math.floor(float(x1)))))
    top = max(0, min(h - 1, int(math.floor(float(y1)))))
    right = max(left + 1, min(w, int(math.ceil(float(x2)))))
    bottom = max(top + 1, min(h, int(math.ceil(float(y2)))))
    return rgb[top:bottom, left:right]


def format_bbox_xyxy(bbox: Iterable[float] | None) -> str:
    if bbox is None:
        return ""
    values = list(bbox)
    if len(values) != 4 or not np.isfinite(np.asarray(values, dtype=np.float32)).all():
        return ""
    return ",".join(f"{float(value):.1f}" for value in values)


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
    eligible_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    if eligible_mask is None:
        eligible_mask = np.array([bool(row["quality_keep"]) for row in frames], dtype=bool)
    quality_indices = [
        i for i, row in enumerate(frames)
        if row["quality_keep"] and bool(eligible_mask[i])
    ]
    probs = np.full(len(frames), np.nan, dtype=np.float32)
    if not quality_indices:
        return probs, np.zeros(len(frames), dtype=bool), float(threshold), False

    for start in range(0, len(quality_indices), batch_size):
        batch_indices = quality_indices[start : start + batch_size]
        tensors = [glottis_preprocess_frame(frame_model_rgb(frames[idx]), cfg) for idx in batch_indices]
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


def blackpad_padding(width: int, height: int, fraction: float, min_padding: int) -> int:
    return max(int(min_padding), int(round(max(width, height) * float(fraction))))


def add_black_border(rgb: np.ndarray, fraction: float, min_padding: int) -> tuple[np.ndarray, int]:
    h, w = rgb.shape[:2]
    padding = blackpad_padding(w, h, fraction=fraction, min_padding=min_padding)
    padded = cv2.copyMakeBorder(
        rgb,
        padding,
        padding,
        padding,
        padding,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return padded, padding


def crop_black_borders_rgb(rgb: np.ndarray, threshold: float = 15) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = content_bbox_from_black_border(rgb, threshold=threshold)
    if x1 <= x0 or y1 <= y0:
        h, w = rgb.shape[:2]
        return rgb, np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32)
    return rgb[y0:y1, x0:x1], np.array([x0, y0, x1, y1], dtype=np.float32)


def square_resize_rgb(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    side = max(1, int(max(h, w)))
    if h == side and w == side:
        return rgb
    return cv2.resize(rgb, (side, side), interpolation=cv2.INTER_LINEAR)


def frame_model_rgb(frame: dict) -> np.ndarray:
    value = frame.get("roi_model_rgb")
    return value if isinstance(value, np.ndarray) else frame["rgb"]


def _rect_soft_mask(shape: tuple[int, int], bbox: np.ndarray | None) -> np.ndarray | None:
    if bbox is None or not np.isfinite(bbox).all():
        return None
    h, w = shape
    x1, y1, x2, y2 = bbox
    left = max(0, min(w - 1, int(math.floor(float(x1)))))
    top = max(0, min(h - 1, int(math.floor(float(y1)))))
    right = max(left + 1, min(w, int(math.ceil(float(x2)))))
    bottom = max(top + 1, min(h, int(math.ceil(float(y2)))))
    mask = np.zeros((h, w), dtype=np.float32)
    mask[top:bottom, left:right] = 1.0
    return mask


def _fallback_square_roi(rgb: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    cropped, bbox = crop_black_borders_rgb(rgb, threshold=float(args.roi_black_border_luma_floor))
    return square_resize_rgb(cropped), bbox


def _extract_best_yolo_candidate(
    result: Any,
    padding: int,
    crop_bbox: np.ndarray,
    cropped_width: int,
    cropped_height: int,
) -> dict[str, Any] | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    best = int(np.nanargmax(confs))
    conf = float(confs[best])
    cropped_box = xyxy[best].astype(np.float32, copy=True)
    cropped_box[[0, 2]] -= float(padding)
    cropped_box[[1, 3]] -= float(padding)
    cropped_box = _scale_bbox_xyxy(cropped_box, cropped_width, cropped_height)
    if not np.isfinite(cropped_box).all():
        return None
    original_box = cropped_box.copy()
    original_box[[0, 2]] += float(crop_bbox[0])
    original_box[[1, 3]] += float(crop_bbox[1])

    cropped_keypoints: list[list[float]] | None = None
    keypoints = getattr(result, "keypoints", None)
    keypoint_data = getattr(keypoints, "data", None)
    if keypoint_data is not None:
        rows = keypoint_data.detach().cpu().numpy()[best]
        cropped_keypoints = []
        for row in rows[:3]:
            cropped_keypoints.append(
                [
                    float(row[0]) - float(padding),
                    float(row[1]) - float(padding),
                    float(row[2]) if len(row) > 2 else conf,
                ]
            )
    return {
        "bbox": original_box,
        "cropped_bbox": cropped_box,
        "confidence": conf,
        "cropped_keypoints": cropped_keypoints,
    }


def score_dino_aux_candidate(
    bundle: DinoAuxBundle | None,
    cropped_rgb: np.ndarray,
    cropped_keypoints: list[list[float]] | None,
    *,
    padding_px: int,
    frame_idx: int,
) -> dict[str, Any] | None:
    if bundle is None or cropped_keypoints is None or len(cropped_keypoints) < 3:
        return None
    image = Image.fromarray(cropped_rgb).convert("RGB")
    dinov3_input = bundle.prediction_input_cls(
        image=image,
        keypoints=cropped_keypoints[:3],
        image_source=f"video_frame_{frame_idx}#black_border_removed",
        image_source_field="in_memory_cropped_frame",
        padding_px=float(padding_px),
        warnings=[],
    )
    record: dict[str, Any] = {"flags": []}
    record = bundle.attach_aux_score(
        record,
        dinov3_input=dinov3_input,
        extractor=bundle.extractor,
        head=bundle.head,
        aux_cfg=bundle.aux_cfg,
        device=bundle.device,
        imgsz=bundle.imgsz,
    )
    return record.get("dinov3_aux")


def empty_roi_gate_result(n: int) -> dict[str, np.ndarray]:
    return {
        "bboxes": np.full((n, 4), np.nan, dtype=np.float32),
        "valid_probs": np.full(n, np.nan, dtype=np.float32),
        "yolo_probs": np.full(n, np.nan, dtype=np.float32),
        "dino_aux_scores": np.full(n, np.nan, dtype=np.float32),
        "dino_aux_factors": np.full(n, np.nan, dtype=np.float32),
        "dino_aux_direct_accept": np.zeros(n, dtype=bool),
        "reflect_probs": np.full(n, np.nan, dtype=np.float32),
        "soft_masks": np.full(n, None, dtype=object),
        "model_rgbs": np.full(n, None, dtype=object),
        "severity": np.full(n, "not_evaluated", dtype=object),
        "keep": np.zeros(n, dtype=bool),
        "reason": np.full(n, "not_eligible_upstream", dtype=object),
    }


@torch.inference_mode()
def infer_roi_gate(
    frames: list[dict],
    bundle: RoiGateBundle | None,
    args: argparse.Namespace,
    eligible_mask: np.ndarray,
    batch_size: int,
    use_amp: bool,
) -> dict[str, np.ndarray]:
    result = empty_roi_gate_result(len(frames))
    eligible_indices = [
        i for i, row in enumerate(frames)
        if row["quality_keep"] and bool(eligible_mask[i])
    ]
    if not eligible_indices:
        return result

    if bundle is None:
        for idx in eligible_indices:
            result["keep"][idx] = False
            result["severity"][idx] = "invalid"
            result["reason"][idx] = "roi_model_missing_invalid_frame"
        return result

    yolo_batch_size = max(1, int(batch_size))
    for start in range(0, len(eligible_indices), yolo_batch_size):
        batch_indices = eligible_indices[start : start + yolo_batch_size]
        cropped_rgbs: list[np.ndarray] = []
        crop_bboxes: list[np.ndarray] = []
        padded_rgbs: list[np.ndarray] = []
        paddings: list[int] = []
        for idx in batch_indices:
            cropped_rgb, crop_bbox = crop_black_borders_rgb(
                frames[idx]["rgb"],
                threshold=float(args.roi_black_border_luma_floor),
            )
            padded, padding = add_black_border(
                cropped_rgb,
                fraction=float(args.roi_blackpad_fraction),
                min_padding=int(args.roi_blackpad_min_padding),
            )
            cropped_rgbs.append(cropped_rgb)
            crop_bboxes.append(crop_bbox)
            padded_rgbs.append(padded)
            paddings.append(padding)

        predict_kwargs: dict[str, Any] = {
            "source": [Image.fromarray(rgb) for rgb in padded_rgbs],
            "conf": float(args.roi_valid_threshold),
            "imgsz": int(args.roi_imgsz),
            "stream": False,
            "verbose": False,
        }
        if bundle.device_arg is not None:
            predict_kwargs["device"] = bundle.device_arg
        results = bundle.model.predict(**predict_kwargs)

        for frame_idx, yolo_result, padding, cropped_rgb, crop_bbox in zip(
            batch_indices,
            results,
            paddings,
            cropped_rgbs,
            crop_bboxes,
        ):
            rgb = frames[frame_idx]["rgb"]
            h, w = rgb.shape[:2]
            cropped_h, cropped_w = cropped_rgb.shape[:2]
            candidate = _extract_best_yolo_candidate(
                yolo_result,
                padding,
                crop_bbox,
                cropped_w,
                cropped_h,
            )
            if candidate is None:
                result["valid_probs"][frame_idx] = float("nan")
                result["severity"][frame_idx] = "invalid"
                result["reason"][frame_idx] = "roi_reject_no_box"
                result["keep"][frame_idx] = False
                continue

            bbox = candidate["bbox"]
            conf = float(candidate["confidence"])
            dino_aux = score_dino_aux_candidate(
                bundle.aux,
                cropped_rgb,
                candidate.get("cropped_keypoints"),
                padding_px=padding,
                frame_idx=frame_idx,
            )
            confidence_factor = float(dino_aux.get("confidence_factor", 1.0)) if dino_aux else 1.0
            direct_accept = bool(dino_aux.get("direct_accept", False)) if dino_aux else False
            combined_conf = max(0.0, min(1.0, conf * confidence_factor))
            if direct_accept:
                combined_conf = max(combined_conf, float(args.roi_accept_threshold))
            dino_score = float(dino_aux.get("point_region_score", np.nan)) if dino_aux else float("nan")
            dino_ok = (
                not args.roi_dino_aux
                or direct_accept
                or (np.isfinite(dino_score) and dino_score >= float(args.roi_dino_aux_accept_threshold))
            )
            roi_accepted = bool(combined_conf >= float(args.roi_accept_threshold) and dino_ok)
            if not roi_accepted:
                result["bboxes"][frame_idx] = bbox
                result["valid_probs"][frame_idx] = combined_conf
                result["yolo_probs"][frame_idx] = conf
                if dino_aux:
                    result["dino_aux_scores"][frame_idx] = dino_score
                    result["dino_aux_factors"][frame_idx] = confidence_factor
                    result["dino_aux_direct_accept"][frame_idx] = direct_accept
                result["soft_masks"][frame_idx] = _rect_soft_mask((h, w), bbox)
                result["severity"][frame_idx] = "invalid"
                result["reason"][frame_idx] = "roi_reject_low_yolo_dino_score"
                result["keep"][frame_idx] = False
                continue

            crop = _crop_roi(rgb, bbox)
            if crop is None:
                result["bboxes"][frame_idx] = bbox
                result["yolo_probs"][frame_idx] = conf
                result["valid_probs"][frame_idx] = combined_conf
                if dino_aux:
                    result["dino_aux_scores"][frame_idx] = dino_score
                    result["dino_aux_factors"][frame_idx] = confidence_factor
                    result["dino_aux_direct_accept"][frame_idx] = direct_accept
                result["soft_masks"][frame_idx] = _rect_soft_mask((h, w), bbox)
                result["severity"][frame_idx] = "invalid"
                result["reason"][frame_idx] = "roi_reject_empty_crop"
                result["keep"][frame_idx] = False
                continue

            result["bboxes"][frame_idx] = bbox
            result["valid_probs"][frame_idx] = combined_conf
            result["yolo_probs"][frame_idx] = conf
            if dino_aux:
                result["dino_aux_scores"][frame_idx] = dino_score
                result["dino_aux_factors"][frame_idx] = confidence_factor
                result["dino_aux_direct_accept"][frame_idx] = direct_accept
            result["model_rgbs"][frame_idx] = square_resize_rgb(crop)
            result["soft_masks"][frame_idx] = _rect_soft_mask((h, w), bbox)
            result["severity"][frame_idx] = "detected_dinov3_aux" if dino_aux else "detected"
            result["reason"][frame_idx] = "roi_detected_dinov3_aux" if dino_aux else "roi_detected"
            result["keep"][frame_idx] = True

    return result


def annotate_roi_gate(
    frames: list[dict],
    roi_result: dict[str, np.ndarray] | None,
    args: argparse.Namespace,
) -> None:
    for idx, frame in enumerate(frames):
        frame["roi_gate_enabled"] = bool(args.roi_gate)
        if not args.roi_gate or roi_result is None:
            frame["roi_bbox_xyxy"] = ""
            frame["roi_valid_prob"] = float("nan")
            frame["roi_yolo_prob"] = float("nan")
            frame["roi_dino_aux_score"] = float("nan")
            frame["roi_dino_aux_factor"] = float("nan")
            frame["roi_dino_aux_direct_accept"] = False
            frame["roi_reflect_prob"] = float("nan")
            frame["roi_reflect_severity"] = "not_run"
            frame["roi_gate_keep"] = True
            frame["roi_filter_reason"] = "roi_gate_disabled"
            frame["_roi_soft_mask"] = None
            frame["_roi_crop_bbox"] = None
            frame["roi_model_rgb"] = square_resize_rgb(frame["rgb"])
            continue
        frame["roi_bbox_xyxy"] = format_bbox_xyxy(roi_result["bboxes"][idx])
        valid_prob = roi_result["valid_probs"][idx]
        yolo_prob = roi_result["yolo_probs"][idx]
        dino_score = roi_result["dino_aux_scores"][idx]
        dino_factor = roi_result["dino_aux_factors"][idx]
        reflect_prob = roi_result["reflect_probs"][idx]
        frame["roi_valid_prob"] = float(valid_prob) if np.isfinite(valid_prob) else float("nan")
        frame["roi_yolo_prob"] = float(yolo_prob) if np.isfinite(yolo_prob) else float("nan")
        frame["roi_dino_aux_score"] = float(dino_score) if np.isfinite(dino_score) else float("nan")
        frame["roi_dino_aux_factor"] = float(dino_factor) if np.isfinite(dino_factor) else float("nan")
        frame["roi_dino_aux_direct_accept"] = bool(roi_result["dino_aux_direct_accept"][idx])
        frame["roi_reflect_prob"] = float(reflect_prob) if np.isfinite(reflect_prob) else float("nan")
        frame["roi_reflect_severity"] = str(roi_result["severity"][idx])
        frame["roi_gate_keep"] = bool(roi_result["keep"][idx])
        frame["roi_filter_reason"] = str(roi_result["reason"][idx])
        frame["_roi_soft_mask"] = roi_result["soft_masks"][idx]
        frame["_roi_crop_bbox"] = roi_result["bboxes"][idx]
        model_rgb = roi_result["model_rgbs"][idx]
        frame["roi_model_rgb"] = model_rgb if isinstance(model_rgb, np.ndarray) else square_resize_rgb(frame["rgb"])


def roi_gate_keep_mask(frames: list[dict], args: argparse.Namespace) -> np.ndarray:
    if not args.roi_gate:
        return np.ones(len(frames), dtype=bool)
    return np.array([bool(frame.get("roi_gate_keep", False)) for frame in frames], dtype=bool)


def apply_video_level_roi_fallback(frames: list[dict], upstream_mask: np.ndarray) -> np.ndarray:
    fallback_keep = np.array([bool(value) for value in upstream_mask], dtype=bool)
    for idx, keep in enumerate(fallback_keep):
        if not keep:
            continue
        frame = frames[idx]
        frame["roi_gate_keep"] = True
        frame["roi_filter_reason"] = "roi_video_fallback_full_frame"
        frame["roi_reflect_severity"] = "video_fallback_full_frame"
        frame["_roi_soft_mask"] = None
        frame["_roi_crop_bbox"] = None
        frame["roi_model_rgb"] = frame["rgb"]
    return fallback_keep


def roi_reason_counts(frames: list[dict], enabled: bool) -> Counter:
    counts: Counter = Counter()
    if not enabled:
        return counts
    for frame in frames:
        reason = str(frame.get("roi_filter_reason", ""))
        if reason in {"", "roi_pass", "roi_detected", "roi_detected_dinov3_aux", "roi_gate_disabled", "not_eligible_upstream"}:
            continue
        counts[reason] += 1
    return counts


def format_reason_counts(counts: Counter) -> str:
    return ";".join(f"{reason}:{count}" for reason, count in sorted(counts.items()))


def parse_reason_counts(value: str) -> Counter:
    counts: Counter = Counter()
    if not value:
        return counts
    for part in str(value).split(";"):
        if not part or ":" not in part:
            continue
        key, raw_count = part.rsplit(":", 1)
        try:
            counts[key] += int(raw_count)
        except ValueError:
            continue
    return counts


def roi_video_stats(frames: list[dict], args: argparse.Namespace) -> dict[str, Any]:
    if not args.roi_gate:
        return {
            "roi_gate_enabled": False,
            "roi_gate_keep_frames": "",
            "roi_gate_filtered_frames": "",
            "roi_invalid_frames": "",
            "roi_reflection_mild_frames": "",
            "roi_reflection_severe_frames": "",
            "roi_filter_reasons": "",
        }
    reasons = roi_reason_counts(frames, enabled=True)
    invalid_reasons = {
        "roi_model_missing_invalid_frame",
        "roi_reject_no_box",
        "roi_reject_low_yolo_dino_score",
        "roi_reject_empty_crop",
    }
    return {
        "roi_gate_enabled": True,
        "roi_gate_keep_frames": int(sum(bool(frame.get("roi_gate_keep")) for frame in frames)),
        "roi_gate_filtered_frames": int(
            sum(str(frame.get("roi_filter_reason")) in invalid_reasons for frame in frames)
        ),
        "roi_invalid_frames": int(sum(str(frame.get("roi_filter_reason")) in invalid_reasons for frame in frames)),
        "roi_reflection_mild_frames": 0,
        "roi_reflection_severe_frames": 0,
        "roi_filter_reasons": format_reason_counts(reasons),
    }


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
            img = Image.fromarray(frame_model_rgb(frames[idx]))
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


def top_fraction_vote_k(candidate_count: int, fraction: float) -> int:
    if candidate_count <= 0:
        return 0
    return max(1, min(candidate_count, int(math.floor(candidate_count * float(fraction)))))


def vote_labels_for_frame(prob: np.ndarray, candidate_indices: list[int], top_k: int) -> list[int]:
    if top_k <= 0:
        return []
    values = prob[candidate_indices]
    if not np.isfinite(values).all():
        return []
    order = np.argsort(values)[::-1]
    return [candidate_indices[int(pos)] for pos in order[:top_k]]


def build_standard_gate_mask(
    frames: list[dict],
    probs: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    finite = np.isfinite(probs).all(axis=1)
    quality_keep = np.array([bool(frame["quality_keep"]) for frame in frames], dtype=bool)
    gate_mask = finite & quality_keep
    if args.glottis_gate:
        gate_mask &= np.array([bool(frame.get("glottis_gate_keep", False)) for frame in frames], dtype=bool)
    if args.roi_gate:
        gate_mask &= roi_gate_keep_mask(frames, args)
    return gate_mask


def vote_summary(
    frames: list[dict],
    probs: np.ndarray,
    candidate_indices: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    gate_mask = build_standard_gate_mask(frames, probs, args)
    top_k = top_fraction_vote_k(len(candidate_indices), args.top_fraction)
    counts: Counter[int] = Counter()
    prob_sums: Counter[int] = Counter()
    for frame_idx, keep in enumerate(gate_mask):
        if not keep:
            continue
        voted = vote_labels_for_frame(probs[frame_idx], candidate_indices, top_k)
        for label_idx in voted:
            counts[label_idx] += 1
            prob_sums[label_idx] += float(probs[frame_idx, label_idx])
    total_votes = int(sum(counts.values()))
    if total_votes:
        pred_idx = max(
            candidate_indices,
            key=lambda idx: (
                counts[idx],
                prob_sums[idx] / counts[idx] if counts[idx] else -1.0,
                -idx,
            ),
        )
        pred_score = float(counts[pred_idx] / total_votes)
    else:
        pred_idx = candidate_indices[0]
        pred_score = float("nan")
    leader_votes = max(counts.values()) if counts else 0
    return {
        "gate_mask": gate_mask,
        "top_k": top_k,
        "counts": counts,
        "prob_sums": prob_sums,
        "total_votes": total_votes,
        "leader_votes": int(leader_votes),
        "pred_idx": int(pred_idx),
        "pred_score": pred_score,
    }


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
    row = next((r for r in video_rows if r["variant"] == STANDARD_VARIANT), None)
    if row is None:
        raise RuntimeError(f"Missing video row for standard variant: {STANDARD_VARIANT}")

    patient_id, patient_name, patient_folder = patient_info(item, args.video_root)
    pred_label = str(row["pred_label"])
    pred_idx = LABEL_DICT[pred_label]
    times = np.array([frame["time_sec"] for frame in frames], dtype=np.float64)
    gate_mask = build_standard_gate_mask(frames, probs, args)
    finite_pred = gate_mask & np.isfinite(probs[:, pred_idx])

    best_frame_idx = None
    selected_time = float("nan")
    selected_prob = float("nan")
    selected_rgb = None
    selected_model_rgb = None
    if finite_pred.any():
        candidate_indices = np.where(finite_pred)[0]
        best_frame_idx = int(candidate_indices[np.argmax(probs[candidate_indices, pred_idx])])
        selected_time = float(times[best_frame_idx])
        selected_prob = float(probs[best_frame_idx, pred_idx])
        selected_rgb = frames[best_frame_idx]["rgb"]
        selected_model_rgb = frame_model_rgb(frames[best_frame_idx])

    evidence_mask = finite_pred & (probs[:, pred_idx] >= args.frame_threshold)
    evidence_segments = merge_segments(times, evidence_mask, args.sample_fps, max_gap_sec=args.max_segment_gap_sec)
    normal_peak_prob = float("nan")
    normal_evidence_duration_sec = float("nan")
    if "Normal" in LABEL_DICT:
        normal_idx = LABEL_DICT["Normal"]
        finite_normal = gate_mask & np.isfinite(probs[:, normal_idx])
        if finite_normal.any():
            normal_peak_prob = float(np.nanmax(probs[finite_normal, normal_idx]))
            normal_evidence_mask = finite_normal & (probs[:, normal_idx] >= args.frame_threshold)
            normal_evidence_duration_sec = float(normal_evidence_mask.sum() / args.sample_fps)
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
    if args.roi_gate and int(row.get("roi_gate_keep_frames", 0) or 0) == 0:
        low_confidence_reasons.append("no_roi_gate_frames")
    if best_frame_idx is not None and frames[best_frame_idx].get("roi_reflect_severity") == "mild":
        low_confidence_reasons.append("selected_roi_mild_reflection")
    evidence_duration_sec = float(evidence_mask.sum() / args.sample_fps)
    if evidence_duration_sec < float(args.min_evidence_duration_sec):
        low_confidence_reasons.append("too_short_evidence_duration")
    if (
        pred_label == "Vocal-Cord-Leukoplakia"
        and evidence_duration_sec < float(args.leukoplakia_min_evidence_duration_sec)
    ):
        low_confidence_reasons.append("leukoplakia_short_evidence_duration")
    if (
        np.isfinite(normal_peak_prob)
        and normal_peak_prob >= float(args.normal_conflict_prob_threshold)
        and np.isfinite(normal_evidence_duration_sec)
        and normal_evidence_duration_sec < float(args.normal_conflict_max_evidence_duration_sec)
    ):
        low_confidence_reasons.append("normal_conflict_short_evidence")
    low_confidence = bool(low_confidence_reasons)
    selected_frame = frames[best_frame_idx] if best_frame_idx is not None else {}

    return {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "patient_folder": patient_folder,
        "video_id": item.video_id,
        "video_path": str(item.video_path),
        "true_label": item.true_label,
        "label_known": item.true_label in LABEL_DICT,
        "diagnosis_variant": STANDARD_VARIANT,
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
        "roi_gate_enabled": row.get("roi_gate_enabled", False),
        "roi_valid_threshold": row.get("roi_valid_threshold", ""),
        "roi_reflect_threshold": row.get("roi_reflect_threshold", ""),
        "roi_severe_reflect_threshold": row.get("roi_severe_reflect_threshold", ""),
        "roi_gate_keep_frames": row.get("roi_gate_keep_frames", ""),
        "roi_gate_filtered_frames": row.get("roi_gate_filtered_frames", ""),
        "roi_filter_reasons": row.get("roi_filter_reasons", ""),
        "selected_roi_bbox_xyxy": selected_frame.get("roi_bbox_xyxy", ""),
        "selected_roi_valid_prob": selected_frame.get("roi_valid_prob", float("nan")),
        "selected_roi_yolo_prob": selected_frame.get("roi_yolo_prob", float("nan")),
        "selected_roi_dino_aux_score": selected_frame.get("roi_dino_aux_score", float("nan")),
        "selected_roi_dino_aux_factor": selected_frame.get("roi_dino_aux_factor", float("nan")),
        "selected_roi_dino_aux_direct_accept": selected_frame.get("roi_dino_aux_direct_accept", False),
        "selected_roi_reflect_prob": selected_frame.get("roi_reflect_prob", float("nan")),
        "selected_roi_reflect_severity": selected_frame.get("roi_reflect_severity", ""),
        "selected_roi_filter_reason": selected_frame.get("roi_filter_reason", ""),
        "evidence_frames_at_threshold": int(evidence_mask.sum()),
        "evidence_duration_sec": evidence_duration_sec,
        "normal_peak_prob": normal_peak_prob,
        "normal_evidence_duration_sec": normal_evidence_duration_sec,
        "low_confidence": low_confidence,
        "low_confidence_reasons": ";".join(low_confidence_reasons),
        "gradcam_path": "",
        "_selected_rgb": selected_rgb,
        "_selected_model_rgb": selected_model_rgb if best_frame_idx is not None else None,
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
    roi_stats = roi_video_stats(frames, args)
    votes = vote_summary(frames, probs, candidate_indices, args)
    gate_mask = votes["gate_mask"]
    gate_segments = merge_segments(times, gate_mask, args.sample_fps, max_gap_sec=args.max_segment_gap_sec)
    vote_counts: Counter[int] = votes["counts"]
    total_votes = int(votes["total_votes"])
    leader_votes = int(votes["leader_votes"])
    score_by_idx = {
        idx: float(vote_counts[idx] / total_votes) if total_votes else float("nan")
        for idx in candidate_indices
    }
    pred_idx = int(votes["pred_idx"])
    pred_label = LABEL_NAMES[pred_idx]
    pred_score = float(votes["pred_score"])
    label_known = item.true_label in LABEL_DICT
    evidence_mask = gate_mask & (probs[:, pred_idx] >= args.frame_threshold)
    evidence_segments = merge_segments(times, evidence_mask, args.sample_fps, max_gap_sec=args.max_segment_gap_sec)

    row = {
        "video_id": item.video_id,
        "video_path": str(item.video_path),
        "true_label": item.true_label,
        "label_known": label_known,
        "variant": STANDARD_VARIANT,
        "pred_label": pred_label,
        "correct": pred_label == item.true_label if label_known else "",
        "pred_score": pred_score,
        "aggregation_method": "per_frame_top_fraction_vote",
        "top_fraction_vote_k": int(votes["top_k"]),
        "total_top_fraction_votes": total_votes,
        "leader_votes": leader_votes,
        "sampled_frames": len(frames),
        "quality_kept_frames": int(sum(bool(frame["quality_keep"]) for frame in frames)),
        "eight_class_inferred_frames": int(np.isfinite(probs).all(axis=1).sum()),
        "glottis_gate_enabled": bool(args.glottis_gate),
        "glottis_gate_frames": int(sum(bool(frame.get("glottis_gate_keep")) for frame in frames)),
        "glottis_gate_threshold": float(frames[0].get("glottis_gate_threshold", float("nan"))) if frames else float("nan"),
        "glottis_gate_fallback_used": bool(frames[0].get("glottis_gate_fallback_used", False)) if frames else False,
        **roi_stats,
        "roi_valid_threshold": float(args.roi_valid_threshold) if args.roi_gate else float("nan"),
        "roi_reflect_threshold": float("nan"),
        "roi_severe_reflect_threshold": float("nan"),
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
        row[f"votes_{LABEL_NAMES[idx]}"] = int(vote_counts[idx])
        row[f"leader_relative_{LABEL_NAMES[idx]}"] = (
            float(vote_counts[idx] / leader_votes) if leader_votes else float("nan")
        )
        row[f"frames_{LABEL_NAMES[idx]}_ge_threshold"] = int(
            (gate_mask & (probs[:, idx] >= args.frame_threshold)).sum()
        )
    all_scores_rows.append(row)

    segment_rows.extend(
        {
            "video_id": item.video_id,
            "true_label": item.true_label,
            "variant": STANDARD_VARIANT,
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
            "variant": STANDARD_VARIANT,
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
            "roi_gate_enabled": row.get("roi_gate_enabled", False),
            "roi_bbox_xyxy": row.get("roi_bbox_xyxy", ""),
            "roi_valid_prob": row.get("roi_valid_prob", float("nan")),
            "roi_yolo_prob": row.get("roi_yolo_prob", float("nan")),
            "roi_dino_aux_score": row.get("roi_dino_aux_score", float("nan")),
            "roi_dino_aux_factor": row.get("roi_dino_aux_factor", float("nan")),
            "roi_dino_aux_direct_accept": row.get("roi_dino_aux_direct_accept", False),
            "roi_reflect_prob": row.get("roi_reflect_prob", float("nan")),
            "roi_reflect_severity": row.get("roi_reflect_severity", "not_run"),
            "roi_gate_keep": row.get("roi_gate_keep", True),
            "roi_filter_reason": row.get("roi_filter_reason", "roi_gate_disabled"),
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
        gate_mask = build_standard_gate_mask(frames, probs, args)
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
        source_rgb = record.get("_selected_model_rgb")
        if not isinstance(source_rgb, np.ndarray):
            source_rgb = raw_rgb
        pred_idx = int(record["_pred_idx"])
        model_source_pil = Image.fromarray(source_rgb)
        model_input_rgb = np.array(visual_preprocess(model_source_pil), dtype=np.uint8)
        image_tensor = tensor_preprocess(model_source_pil).unsqueeze(0).to(device)
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


RENDER_LABEL_COLORS = {
    "Normal": (82, 190, 128),
    "Reinke-Edema": (86, 180, 233),
    "Vocal-Cord-Cyst": (238, 130, 238),
    "Vocal-Cord-Polyp": (245, 166, 35),
    "Vocal-Cord-Leukoplakia": (248, 231, 28),
    "Vocal-Cord-Granuloma": (231, 76, 60),
    "Cancer": (155, 89, 182),
    "Non-Vocal-Cord": (180, 180, 180),
}


def render_label_color(label: str) -> tuple[int, int, int]:
    return RENDER_LABEL_COLORS.get(label, (230, 230, 230))


def load_pil_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in CHINESE_FONT_CANDIDATES:
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_shadow_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int] = (245, 245, 245),
) -> None:
    x, y = xy
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)


def content_bbox_from_black_border(rgb: np.ndarray, threshold: float = 15) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    coords = np.argwhere(gray > float(threshold))
    h, w = gray.shape[:2]
    if coords.size == 0:
        return 0, 0, w, h
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return int(x0), int(y0), int(x1), int(y1)


def detect_left_panel_width(frames: list[dict]) -> int:
    if not frames:
        return 420
    widths: list[int] = []
    stride = max(1, len(frames) // 12)
    for frame in frames[::stride]:
        gray = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2GRAY)
        dark_ratio = (gray < 12).mean(axis=0)
        left = 0
        for value in dark_ratio:
            if value >= 0.80:
                left += 1
            else:
                break
        widths.append(left)
    frame_w = int(frames[0]["rgb"].shape[1])
    detected = int(np.median(widths)) if widths else 0
    return int(min(max(360, detected - 16), frame_w * 0.45))


def apply_roi_soft_filter(rgb: np.ndarray, mask: np.ndarray | None, min_brightness: float) -> np.ndarray:
    if mask is None:
        return rgb
    h, w = rgb.shape[:2]
    soft = np.asarray(mask, dtype=np.float32)
    if soft.shape[:2] != (h, w):
        soft = cv2.resize(soft, (w, h), interpolation=cv2.INTER_LINEAR)
    soft = cv2.GaussianBlur(np.clip(soft, 0.0, 1.0), (0, 0), sigmaX=5.0)
    floor = float(np.clip(min_brightness, 0.05, 1.0))
    weight = floor + (1.0 - floor) * soft
    filtered = rgb.astype(np.float32) * weight[:, :, None]
    return np.uint8(np.clip(filtered, 0, 255))


def cam_to_frame_map(raw_rgb: np.ndarray, cam_map: np.ndarray, cfg: dict) -> np.ndarray:
    h, w = raw_rgb.shape[:2]
    full = np.zeros((h, w), dtype=np.float32)
    x0, y0, x1, y1 = content_bbox_from_black_border(
        raw_rgb,
        threshold=int(cfg.get("crop_black_threshold", 15)),
    )
    if x1 <= x0 or y1 <= y0:
        return cv2.resize(cam_map, (w, h), interpolation=cv2.INTER_LINEAR)
    resized = cv2.resize(cam_map.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    full[y0:y1, x0:x1] = np.clip(resized, 0.0, 1.0)
    return full


def cam_to_frame_map_for_frame(frame: dict, cam_map: np.ndarray, cfg: dict) -> np.ndarray:
    raw_rgb = frame["rgb"]
    h, w = raw_rgb.shape[:2]
    full = np.zeros((h, w), dtype=np.float32)
    bbox = frame.get("_roi_crop_bbox")
    if isinstance(bbox, np.ndarray) and np.isfinite(bbox).all():
        x1, y1, x2, y2 = bbox
        left = max(0, min(w - 1, int(math.floor(float(x1)))))
        top = max(0, min(h - 1, int(math.floor(float(y1)))))
        right = max(left + 1, min(w, int(math.ceil(float(x2)))))
        bottom = max(top + 1, min(h, int(math.ceil(float(y2)))))
        resized = cv2.resize(cam_map.astype(np.float32), (right - left, bottom - top), interpolation=cv2.INTER_LINEAR)
        full[top:bottom, left:right] = np.clip(resized, 0.0, 1.0)
        return full
    return cam_to_frame_map(raw_rgb, cam_map, cfg)


def overlay_cam_on_frame(rgb: np.ndarray, cam_map: np.ndarray | None) -> np.ndarray:
    if cam_map is None:
        return rgb
    heatmap = cv2.applyColorMap(np.uint8(255 * np.clip(cam_map, 0.0, 1.0)), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32)
    alpha = (0.12 + 0.34 * np.clip(cam_map, 0.0, 1.0))[:, :, None]
    mixed = rgb.astype(np.float32) * (1.0 - alpha) + heatmap * alpha
    return np.uint8(np.clip(mixed, 0, 255))


def compute_render_cam_maps(
    model: HierarchicalImageClassifier,
    frames: list[dict],
    probs: np.ndarray,
    pred_idx: int,
    cfg: dict,
    device: torch.device,
    args: argparse.Namespace,
    gate_mask: np.ndarray,
) -> list[np.ndarray | None]:
    maps: list[np.ndarray | None] = [None] * len(frames)
    valid_indices = [
        idx for idx, keep in enumerate(gate_mask)
        if keep and np.isfinite(probs[idx, pred_idx])
    ]
    if not valid_indices:
        return maps

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    tensor_preprocess = _build_base_preprocess(cfg, to_tensor=True)
    cam = GradCAM(
        model=model,
        target_layers=get_gradcam_target_layers(model),
        reshape_transform=gradcam_reshape_transform,
    )
    model.eval()
    for idx in valid_indices:
        raw_pil = Image.fromarray(frame_model_rgb(frames[idx]))
        image_tensor = tensor_preprocess(raw_pil).unsqueeze(0).to(device)
        image_tensor = gpu_normalise(image_tensor)
        cam_input = image_tensor.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            grayscale_cam = cam(input_tensor=cam_input, targets=[ClassifierOutputTarget(pred_idx)])[0]
        maps[idx] = cam_to_frame_map_for_frame(frames[idx], grayscale_cam, cfg)
    cam.activations_and_grads.release()
    return maps


def build_render_timeline(
    frames: list[dict],
    probs: np.ndarray,
    candidate_indices: list[int],
    args: argparse.Namespace,
    gate_mask: np.ndarray,
) -> list[dict[str, Any]]:
    top_k = top_fraction_vote_k(len(candidate_indices), args.top_fraction)
    vote_events: list[tuple[float, int]] = []
    last_values = {idx: 0.0 for idx in candidate_indices}
    current_evidence = 0.0
    longest_evidence = 0.0
    timeline: list[dict[str, Any]] = []
    step_sec = 1.0 / float(args.sample_fps)
    window_sec = max(step_sec, float(args.render_chart_window_sec))

    for frame_idx, frame in enumerate(frames):
        time_sec = float(frame["time_sec"])
        valid = bool(gate_mask[frame_idx]) and np.isfinite(probs[frame_idx, candidate_indices]).all()
        voted: list[int] = []
        if valid:
            current_evidence += step_sec
            longest_evidence = max(longest_evidence, current_evidence)
            voted = vote_labels_for_frame(probs[frame_idx], candidate_indices, top_k)
            for label_idx in voted:
                vote_events.append((time_sec, label_idx))
            cutoff = time_sec - window_sec
            vote_events = [(t, label_idx) for t, label_idx in vote_events if t >= cutoff]
            counts: Counter[int] = Counter(label_idx for _t, label_idx in vote_events)
            leader = max(counts.values()) if counts else 0
            last_values = {
                idx: float(counts[idx] / leader * 100.0) if leader else 0.0
                for idx in candidate_indices
            }
        else:
            current_evidence = 0.0
        timeline.append(
            {
                "valid": valid,
                "voted": voted,
                "values": dict(last_values),
                "current_evidence_sec": float(current_evidence if valid else 0.0),
                "longest_evidence_sec": float(longest_evidence),
            }
        )
    return timeline


def draw_vote_chart(
    draw: ImageDraw.ImageDraw,
    frames: list[dict],
    timeline: list[dict[str, Any]],
    frame_idx: int,
    candidate_indices: list[int],
    box: tuple[int, int, int, int],
    args: argparse.Namespace,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, fill=(18, 22, 28), outline=(85, 92, 105), width=1)
    for frac in (0.25, 0.50, 0.75):
        y = int(y1 - frac * (y1 - y0))
        draw.line((x0 + 8, y, x1 - 8, y), fill=(50, 56, 66), width=1)

    current_time = float(frames[frame_idx]["time_sec"])
    start_time = current_time - float(args.render_chart_window_sec)
    visible = [
        idx for idx in range(frame_idx + 1)
        if float(frames[idx]["time_sec"]) >= start_time
    ]
    if len(visible) < 2:
        visible = list(range(max(0, frame_idx - 1), frame_idx + 1))

    def x_for_time(t: float) -> int:
        denom = max(float(args.render_chart_window_sec), 1e-6)
        return int(x0 + 10 + np.clip((t - start_time) / denom, 0.0, 1.0) * max(1, x1 - x0 - 20))

    def y_for_value(v: float) -> int:
        return int(y1 - 10 - np.clip(v, 0.0, 100.0) / 100.0 * max(1, y1 - y0 - 20))

    for label_idx in candidate_indices:
        label = LABEL_NAMES[label_idx]
        color = render_label_color(label)
        points = [
            (x_for_time(float(frames[idx]["time_sec"])), y_for_value(float(timeline[idx]["values"][label_idx])))
            for idx in visible
        ]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        elif points:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

    draw_shadow_text(draw, (x0 + 10, y0 + 8), f"近{float(args.render_chart_window_sec):g}秒投票趋势", font, (245, 245, 245))
    draw_shadow_text(draw, (x0 + 10, y1 - 26), "领先病种=100%", font, (200, 205, 215))


def draw_render_panel(
    image: Image.Image,
    item: VideoItem,
    frames: list[dict],
    probs: np.ndarray,
    frame_idx: int,
    candidate_indices: list[int],
    pred_idx: int,
    timeline: list[dict[str, Any]],
    vote_counts: Counter[int],
    panel_w: int,
    final_longest_evidence: float,
    args: argparse.Namespace,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, panel_w, image.height), fill=(4, 7, 12, 218))
    font_title = load_pil_font(30)
    font_main = load_pil_font(24)
    font_small = load_pil_font(18)
    font_tiny = load_pil_font(15)

    frame = frames[frame_idx]
    state = timeline[frame_idx]
    prob = probs[frame_idx]
    final_label = LABEL_NAMES[pred_idx]
    final_call = label_zh(final_label) if final_longest_evidence >= float(args.min_evidence_duration_sec) else "待人工复核"
    valid_text = "有效ROI帧" if state["valid"] else "无效/未入池"
    valid_color = (92, 220, 150) if state["valid"] else (240, 120, 90)
    if np.isfinite(prob[candidate_indices]).all():
        current_idx = candidate_indices[int(np.argmax(prob[candidate_indices]))]
        current_text = f"{label_zh(LABEL_NAMES[current_idx])}  p={float(prob[current_idx]):.2f}"
    else:
        current_text = "-"
    glottis_prob = frame.get("glottis_prob", float("nan"))
    glottis_enabled = bool(frame.get("glottis_gate_enabled", False))
    glottis_value = 1 if bool(frame.get("glottis_gate_keep", False)) else 0
    roi_valid = frame.get("roi_valid_prob", float("nan"))
    roi_dino = frame.get("roi_dino_aux_score", float("nan"))
    roi_reflect = frame.get("roi_reflect_prob", float("nan"))
    roi_severity = str(frame.get("roi_reflect_severity", ""))

    x = 24
    y = 24
    draw_shadow_text(draw, (x, y), "视频诊断叠加", font_title, (250, 250, 250))
    y += 48
    draw_shadow_text(draw, (x, y), f"最终: {final_call}", font_main, render_label_color(final_label))
    y += 38
    draw_shadow_text(draw, (x, y), f"状态: {valid_text}", font_main, valid_color)
    y += 38
    if glottis_enabled:
        glottis_prob_text = f"{float(glottis_prob):.2f}" if np.isfinite(glottis_prob) else "-"
        glottis_text = f"声门区域(0/1): {glottis_value}  p={glottis_prob_text}"
    else:
        glottis_text = "声门0/1 gate: 未启用"
    draw_shadow_text(draw, (x, y), glottis_text, font_small)
    y += 30
    draw_shadow_text(draw, (x, y), f"当前Top1: {current_text}", font_small)
    y += 30
    roi_valid_text = f"{float(roi_valid):.2f}" if np.isfinite(roi_valid) else "-"
    roi_dino_text = f"{float(roi_dino):.2f}" if np.isfinite(roi_dino) else "-"
    roi_reflect_text = f"{float(roi_reflect):.2f}" if np.isfinite(roi_reflect) else "-"
    draw_shadow_text(draw, (x, y), f"ROI valid={roi_valid_text}  DINO={roi_dino_text}  reflect={roi_reflect_text}", font_small)
    y += 28
    draw_shadow_text(draw, (x, y), f"ROI状态: {roi_severity}", font_small)
    y += 34
    draw_shadow_text(draw, (x, y), f"当前证据: {float(state['current_evidence_sec']):.2f}s", font_main)
    y += 34
    draw_shadow_text(draw, (x, y), f"最长证据: {float(state['longest_evidence_sec']):.2f}s", font_main)
    y += 34
    draw_shadow_text(draw, (x, y), f"证据阈值: >= {float(args.min_evidence_duration_sec):.2f}s", font_small)
    y += 28
    draw_shadow_text(draw, (x, y), f"时间: {float(frame['time_sec']):.2f}s", font_small)

    chart_top = min(image.height - 440, max(y + 28, 410))
    chart_box = (20, chart_top, panel_w - 20, chart_top + 250)
    draw_vote_chart(draw, frames, timeline, frame_idx, candidate_indices, chart_box, args, font_tiny)

    legend_y = chart_box[3] + 18
    leader = max(vote_counts.values()) if vote_counts else 0
    for rank, label_idx in enumerate(candidate_indices):
        label = LABEL_NAMES[label_idx]
        col = rank % 2
        row = rank // 2
        lx = 24 + col * max(170, (panel_w - 48) // 2)
        ly = legend_y + row * 28
        color = render_label_color(label)
        draw.rectangle((lx, ly + 5, lx + 16, ly + 21), fill=color)
        current_value = float(state["values"][label_idx])
        count = int(vote_counts[label_idx])
        suffix = f"{current_value:3.0f}%/{count}"
        draw_shadow_text(draw, (lx + 22, ly), f"{label_zh(label)} {suffix}", font_tiny, (235, 238, 242))
    if leader:
        draw_shadow_text(draw, (24, image.height - 34), f"总投票: {sum(vote_counts.values())}  领先票: {leader}", font_tiny)


def render_diagnostic_video(
    item: VideoItem,
    frames: list[dict],
    probs: np.ndarray,
    candidate_indices: list[int],
    model: HierarchicalImageClassifier,
    cfg: dict,
    device: torch.device,
    args: argparse.Namespace,
) -> Path | None:
    if not frames:
        return None
    render_votes = vote_summary(frames, probs, candidate_indices, args)
    pred_idx = int(render_votes["pred_idx"])
    gate_mask = render_votes["gate_mask"]
    timeline = build_render_timeline(frames, probs, candidate_indices, args, gate_mask)
    final_longest = max((float(row["longest_evidence_sec"]) for row in timeline), default=0.0)
    cam_maps = compute_render_cam_maps(model, frames, probs, pred_idx, cfg, device, args, gate_mask)

    args.render_video_dir.mkdir(parents=True, exist_ok=True)
    pred_label = LABEL_NAMES[pred_idx]
    out_path = args.render_video_dir / (
        f"{safe_filename_part(item.video_id)}__{safe_filename_part(label_zh(pred_label))}"
        "__diagnostic_overlay.mp4"
    )
    temp_path = out_path.with_name(f"{out_path.stem}__tmp_mp4v{out_path.suffix}")
    height, width = frames[0]["rgb"].shape[:2]
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.sample_fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create temporary output video: {temp_path}")

    panel_w = detect_left_panel_width(frames)
    for frame_idx, frame in enumerate(frames):
        rgb = frame["rgb"].copy()
        if bool(gate_mask[frame_idx]):
            rgb = apply_roi_soft_filter(
                rgb,
                frame.get("_roi_soft_mask"),
                min_brightness=float(args.render_roi_min_brightness),
            )
            rgb = overlay_cam_on_frame(rgb, cam_maps[frame_idx])
        pil = Image.fromarray(rgb)
        draw_render_panel(
            pil,
            item,
            frames,
            probs,
            frame_idx,
            candidate_indices,
            pred_idx,
            timeline,
            render_votes["counts"],
            panel_w,
            final_longest,
            args,
        )
        writer.write(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))
    writer.release()
    encode_h264_video(temp_path, out_path)

    metadata = {
        "video_path": str(item.video_path),
        "output_video": str(out_path),
        "sample_fps": float(args.sample_fps),
        "aggregation_method": "per_frame_top_fraction_vote",
        "top_fraction": float(args.top_fraction),
        "top_fraction_vote_k": int(render_votes["top_k"]),
        "candidate_labels": [LABEL_NAMES[idx] for idx in candidate_indices],
        "pred_label": pred_label,
        "pred_label_zh": label_zh(pred_label),
        "diagnosis_call_zh": label_zh(pred_label)
        if final_longest >= float(args.min_evidence_duration_sec)
        else "待人工复核",
        "vote_counts": {LABEL_NAMES[idx]: int(render_votes["counts"][idx]) for idx in candidate_indices},
        "total_votes": int(render_votes["total_votes"]),
        "longest_evidence_duration_sec": float(final_longest),
        "min_evidence_duration_sec": float(args.min_evidence_duration_sec),
        "render_chart_window_sec": float(args.render_chart_window_sec),
    }
    with out_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return out_path


def encode_h264_video(temp_path: Path, out_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(temp_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to write H.264 diagnostic videos.") from exc
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


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
        "roi_gate_enabled",
        "roi_valid_threshold",
        "roi_reflect_threshold",
        "roi_severe_reflect_threshold",
        "roi_gate_keep_frames",
        "roi_gate_filtered_frames",
        "roi_filter_reasons",
        "selected_roi_bbox_xyxy",
        "selected_roi_valid_prob",
        "selected_roi_yolo_prob",
        "selected_roi_dino_aux_score",
        "selected_roi_dino_aux_factor",
        "selected_roi_dino_aux_direct_accept",
        "selected_roi_reflect_prob",
        "selected_roi_reflect_severity",
        "selected_roi_filter_reason",
        "evidence_duration_sec",
        "normal_peak_prob",
        "normal_evidence_duration_sec",
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
                    "roi_gate_enabled": selected_record["roi_gate_enabled"],
                    "roi_valid_threshold": selected_record["roi_valid_threshold"],
                    "roi_reflect_threshold": selected_record["roi_reflect_threshold"],
                    "roi_severe_reflect_threshold": selected_record["roi_severe_reflect_threshold"],
                    "roi_gate_keep_frames": selected_record["roi_gate_keep_frames"],
                    "roi_gate_filtered_frames": selected_record["roi_gate_filtered_frames"],
                    "roi_filter_reasons": selected_record["roi_filter_reasons"],
                    "selected_roi_bbox_xyxy": selected_record["selected_roi_bbox_xyxy"],
                    "selected_roi_valid_prob": selected_record["selected_roi_valid_prob"],
                    "selected_roi_yolo_prob": selected_record["selected_roi_yolo_prob"],
                    "selected_roi_dino_aux_score": selected_record["selected_roi_dino_aux_score"],
                    "selected_roi_dino_aux_factor": selected_record["selected_roi_dino_aux_factor"],
                    "selected_roi_dino_aux_direct_accept": selected_record["selected_roi_dino_aux_direct_accept"],
                    "selected_roi_reflect_prob": selected_record["selected_roi_reflect_prob"],
                    "selected_roi_reflect_severity": selected_record["selected_roi_reflect_severity"],
                    "selected_roi_filter_reason": selected_record["selected_roi_filter_reason"],
                    "evidence_duration_sec": selected_record["evidence_duration_sec"],
                    "normal_peak_prob": selected_record["normal_peak_prob"],
                    "normal_evidence_duration_sec": selected_record["normal_evidence_duration_sec"],
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
        variant_roi_reasons: Counter = Counter()
        for row in rows:
            variant_roi_reasons.update(parse_reason_counts(str(row.get("roi_filter_reasons", ""))))
        by_variant[variant] = {
            "videos": total,
            "labeled_videos": labeled_total,
            "correct": correct,
            "accuracy": correct / labeled_total if labeled_total else float("nan"),
            "mean_gate_frames": float(np.mean([row["gate_frames"] for row in rows])) if rows else 0.0,
            "mean_gate_duration_sec": float(np.mean([row["gate_duration_sec"] for row in rows])) if rows else 0.0,
            "mean_roi_gate_keep_frames": (
                float(np.mean([int(row.get("roi_gate_keep_frames") or 0) for row in rows]))
                if args.roi_gate and rows else 0.0
            ),
            "roi_filter_reason_counts": dict(sorted(variant_roi_reasons.items())),
            "pred_counts": {
                str(label): int(count)
                for label, count in pd.Series([row["pred_label"] for row in rows]).value_counts().items()
            } if rows else {},
        }
    unique_video_rows: dict[str, dict] = {}
    for row in video_rows:
        unique_video_rows.setdefault(str(row["video_id"]), row)
    roi_reasons: Counter = Counter()
    for row in unique_video_rows.values():
        roi_reasons.update(parse_reason_counts(str(row.get("roi_filter_reasons", ""))))
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
        "roi_gate": {
            "enabled": bool(args.roi_gate),
            "localizer_model": str(args.roi_localizer_model) if args.roi_gate else "",
            "localizer_type": "yolo_pose_dinov3_strict_roi_crop",
            "dino_aux_enabled": bool(args.roi_gate and args.roi_dino_aux),
            "dino_aux_model": str(args.roi_dino_aux_model) if args.roi_gate and args.roi_dino_aux else "",
            "dino_aux_code_root": str(args.roi_dino_aux_code_root) if args.roi_gate and args.roi_dino_aux else "",
            "dino_aux_imgsz": int(args.roi_dino_aux_imgsz or 0) if args.roi_gate and args.roi_dino_aux else "",
            "bbox_conf_threshold": float(args.roi_valid_threshold),
            "accept_threshold": float(args.roi_accept_threshold),
            "dino_aux_accept_threshold": float(args.roi_dino_aux_accept_threshold),
            "imgsz": int(args.roi_imgsz),
            "blackpad_fraction": float(args.roi_blackpad_fraction),
            "blackpad_min_padding": int(args.roi_blackpad_min_padding),
            "black_border_luma_floor": float(args.roi_black_border_luma_floor),
            "cache_device": str(args.roi_cache_device),
            "dino_aux_device": str(args.roi_dino_aux_device),
            "video_level_full_frame_fallback": bool(args.roi_video_fallback),
            "allow_missing_model": bool(args.roi_allow_missing_model),
            "keep_frames": (
                int(sum(int(row.get("roi_gate_keep_frames") or 0) for row in unique_video_rows.values()))
                if args.roi_gate else 0
            ),
            "filtered_frames": (
                int(sum(int(row.get("roi_gate_filtered_frames") or 0) for row in unique_video_rows.values()))
                if args.roi_gate else 0
            ),
            "filter_reason_counts": dict(sorted(roi_reasons.items())),
        },
        "aggregation_method": "per_frame_top_fraction_vote",
        "top_fraction": args.top_fraction,
        "top_fraction_vote_k": top_fraction_vote_k(len(candidate_indices), args.top_fraction),
        "frame_threshold": args.frame_threshold,
        "render_video": {
            "enabled": bool(args.render_video),
            "output_dir": str(args.render_video_dir) if args.render_video else "",
            "chart_window_sec": float(args.render_chart_window_sec),
            "roi_min_brightness": float(args.render_roi_min_brightness),
        },
        "leukoplakia_min_evidence_duration_sec": args.leukoplakia_min_evidence_duration_sec,
        "normal_conflict_prob_threshold": args.normal_conflict_prob_threshold,
        "normal_conflict_max_evidence_duration_sec": args.normal_conflict_max_evidence_duration_sec,
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
    args.render_video_dir = args.render_video_dir.resolve()
    if args.render_video and args.candidate_labels == "auto":
        args.candidate_labels = "all-voc"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(args.config))
    init_label_mapping(cfg)
    device = torch.device(args.device) if args.device else setup_device()
    if args.roi_gate:
        if not (0.0 <= float(args.roi_valid_threshold) <= 1.0):
            raise ValueError("--roi-valid-threshold must be between 0 and 1.")
        if not (0.0 <= float(args.roi_accept_threshold) <= 1.0):
            raise ValueError("--roi-accept-threshold must be between 0 and 1.")
        if not (0.0 <= float(args.roi_dino_aux_accept_threshold) <= 1.0):
            raise ValueError("--roi-dino-aux-accept-threshold must be between 0 and 1.")
        if int(args.roi_imgsz) <= 0:
            raise ValueError("--roi-imgsz must be positive.")
        if args.roi_dino_aux and args.roi_dino_aux_imgsz is not None and int(args.roi_dino_aux_imgsz) <= 0:
            raise ValueError("--roi-dino-aux-imgsz must be positive.")
        if float(args.roi_blackpad_fraction) < 0.0:
            raise ValueError("--roi-blackpad-fraction must be non-negative.")
        if int(args.roi_blackpad_min_padding) < 0:
            raise ValueError("--roi-blackpad-min-padding must be non-negative.")
        if float(args.roi_black_border_luma_floor) < 0.0:
            raise ValueError("--roi-black-border-luma-floor must be non-negative.")

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

    roi_bundle = None
    if args.roi_gate:
        roi_bundle = build_roi_gate_bundle(args, device)
        print(
            "YOLO-Pose + DINOv3 ROI: "
            f"weights={args.roi_localizer_model}, detect_conf>={args.roi_valid_threshold:.2f}, "
            f"accept>={args.roi_accept_threshold:.2f}, dino>={args.roi_dino_aux_accept_threshold:.2f}, "
            f"imgsz={args.roi_imgsz}, blackpad={args.roi_blackpad_fraction:.2f}/"
            f"{args.roi_blackpad_min_padding}px, "
            f"dino_aux={args.roi_dino_aux_model if args.roi_dino_aux else 'disabled'}"
        )

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
        "roi_gate_enabled",
        "roi_bbox_xyxy",
        "roi_valid_prob",
        "roi_yolo_prob",
        "roi_dino_aux_score",
        "roi_dino_aux_factor",
        "roi_dino_aux_direct_accept",
        "roi_reflect_prob",
        "roi_reflect_severity",
        "roi_gate_keep",
        "roi_filter_reason",
        "pred_argmax",
        "non_voc_prob",
        "voc_sum_prob",
        *[f"prob_{label}" for label in labels],
    ]

    all_video_rows: list[dict] = []
    all_segment_rows: list[dict] = []
    diagnosis_records: list[dict] = []
    rendered_video_paths: list[str] = []
    frame_csv_path = args.output_dir / "frame_predictions.csv"
    with frame_csv_path.open("w", newline="", encoding="utf-8") as f:
        frame_writer = csv.DictWriter(f, fieldnames=frame_fieldnames)
        frame_writer.writeheader()
        for item in items:
            print(f"Processing {item.video_id}: {item.video_path}")
            frames = sample_video_frames(item, args.sample_fps, args)
            quality_keep = np.array([bool(frame["quality_keep"]) for frame in frames], dtype=bool)
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
                    eligible_mask=quality_keep,
                )
            else:
                glottis_probs = None
                glottis_keep = np.ones(len(frames), dtype=bool)
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
            roi_eligible = quality_keep & glottis_keep
            if args.roi_gate:
                roi_result = infer_roi_gate(
                    frames,
                    roi_bundle,
                    args,
                    eligible_mask=roi_eligible,
                    batch_size=args.batch_size,
                    use_amp=args.amp,
                )
                annotate_roi_gate(frames, roi_result, args)
                roi_keep = np.array([bool(value) for value in roi_result["keep"]], dtype=bool)
                if args.roi_video_fallback and not bool(roi_keep.any()) and bool(roi_eligible.any()):
                    roi_keep = apply_video_level_roi_fallback(frames, roi_eligible)
            else:
                annotate_roi_gate(frames, None, args)
                roi_keep = np.ones(len(frames), dtype=bool)
            inference_keep = quality_keep & roi_keep & glottis_keep
            probs = infer_frames(
                model,
                frames,
                preprocess,
                device,
                args.batch_size,
                args.amp,
                eligible_mask=inference_keep,
            )
            write_frame_rows(frame_writer, item, frames, probs)
            video_rows, segment_rows = score_video(item, frames, probs, candidate_indices, args)
            all_video_rows.extend(video_rows)
            all_segment_rows.extend(segment_rows)
            diagnosis_records.append(build_diagnosis_record(item, frames, probs, video_rows, args))
            save_keyframes(args.output_dir, item, frames, probs, video_rows, args)
            if args.render_video:
                rendered_path = render_diagnostic_video(
                    item,
                    frames,
                    probs,
                    candidate_indices,
                    model,
                    cfg,
                    device,
                    args,
                )
                if rendered_path is not None:
                    rendered_video_paths.append(str(rendered_path))
                    print(f"Rendered diagnostic video: {rendered_path}")

    video_df = pd.DataFrame(all_video_rows)
    segment_df = pd.DataFrame(all_segment_rows)
    video_df.to_csv(args.output_dir / "video_predictions.csv", index=False)
    segment_df.to_csv(args.output_dir / "video_segments.csv", index=False)
    summary = summarize(all_video_rows, candidate_indices, args)
    summary["rendered_videos"] = rendered_video_paths
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
