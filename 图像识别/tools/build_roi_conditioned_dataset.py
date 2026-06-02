#!/usr/bin/env python3
"""Build split-aware square classifier inputs from ROI prediction JSONL files.

Train policy: keep only successful ROI crops.
Val/test policy: use ROI crops when available; otherwise use the no-black
cropped source image recorded by the ROI predictor and resize it to a square.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DICOM_PATIENT_KEY_PATTERN = re.compile(r"(?:^|_)13\.(\d{13})\.")
DEFAULT_CLASSES = (
    "混杂图片",
    "正常",
    "声带任克水肿",
    "声带囊肿",
    "声带息肉",
    "声带白斑",
    "声带肉芽肿",
    "喉癌",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="Create train/val/test manifests from dataset_split.json.")
    manifest.add_argument("--dataset-split", type=Path, required=True)
    manifest.add_argument("--out-dir", type=Path, required=True)
    manifest.add_argument("--classes", nargs="*", default=list(DEFAULT_CLASSES))

    dataset = subparsers.add_parser("dataset", help="Create ROI-conditioned square image tree.")
    dataset.add_argument("--train-predictions", type=Path, required=True)
    dataset.add_argument("--val-predictions", type=Path, required=True)
    dataset.add_argument("--test-predictions", type=Path, required=True)
    dataset.add_argument("--out-dir", type=Path, required=True)
    dataset.add_argument("--output-size", type=int, default=224)
    dataset.add_argument("--crop-actions", nargs="*", default=["auto_accept", "manual_review"])
    dataset.add_argument("--classes", nargs="*", default=list(DEFAULT_CLASSES))
    dataset.add_argument("--jpeg-quality", type=int, default=95)
    dataset.add_argument("--train-fallback-class", default="")
    dataset.add_argument("--train-fallback-count", type=int, default=0)
    dataset.add_argument("--train-fallback-seed", type=int, default=20260525)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_patient_name(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        prefix = stem.split("_", 1)[0].strip()
        if prefix:
            return prefix
    if len(stem) >= 8 and stem[:8].isdigit():
        return stem[:8]
    match = DICOM_PATIENT_KEY_PATTERN.search(stem)
    if match:
        return f"{int(match.group(1)):08d}"[-8:]
    stripped = stem.strip()
    return stripped if stripped else path.name


def image_paths(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def command_manifest(args: argparse.Namespace) -> None:
    split = json.loads(args.dataset_split.read_text(encoding="utf-8"))
    dataset_root = Path(split["dataset_root"])
    patient_split = split.get("patients", split)
    patient_to_split: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for patient in patient_split.get(split_name, []):
            patient_to_split[str(patient)] = split_name

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split_name: (out_dir / f"{split_name}.jsonl").open("w", encoding="utf-8")
        for split_name in ("train", "val", "test")
    }
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    missing_patients = 0
    try:
        for class_name in args.classes:
            folder = dataset_root / class_name
            if not folder.is_dir():
                continue
            for path in image_paths(folder):
                patient = get_patient_name(path)
                split_name = patient_to_split.get(patient)
                if split_name is None:
                    missing_patients += 1
                    continue
                record = {
                    "source": str(path.resolve()),
                    "original_source": str(path.resolve()),
                    "source_key": str(path.resolve()),
                    "class_name": class_name,
                    "patient_name": patient,
                    "split": split_name,
                }
                handles[split_name].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split_name][class_name] += 1
    finally:
        for handle in handles.values():
            handle.close()

    summary = {
        "dataset_split": str(args.dataset_split),
        "dataset_root": str(dataset_root),
        "out_dir": str(out_dir),
        "counts": {split_name: dict(counts[split_name]) for split_name in ("train", "val", "test")},
        "totals": {split_name: sum(counts[split_name].values()) for split_name in ("train", "val", "test")},
        "missing_patient_images": missing_patients,
    }
    (out_dir / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def clamp_bbox(bbox: Sequence[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    if len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    left = max(0, min(width, int(math.floor(min(x1, x2)))))
    top = max(0, min(height, int(math.floor(min(y1, y2)))))
    right = max(0, min(width, int(math.ceil(max(x1, x2)))))
    bottom = max(0, min(height, int(math.ceil(max(y1, y2)))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def polygon_size(points: Sequence[Sequence[float]]) -> tuple[int, int]:
    top_width = math.dist(points[0], points[1])
    bottom_width = math.dist(points[3], points[2])
    right_height = math.dist(points[1], points[2])
    left_height = math.dist(points[0], points[3])
    return max(1, int(round(max(top_width, bottom_width)))), max(1, int(round(max(left_height, right_height))))


def crop_polygon(image: Image.Image, polygon: Sequence[Sequence[float]]) -> Image.Image | None:
    if len(polygon) != 4:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    width, height = polygon_size(polygon)
    source = np.asarray([[float(x), float(y)] for x, y in polygon], dtype=np.float32)
    target = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(np.asarray(image.convert("RGB")), matrix, (width, height))
    return Image.fromarray(warped)


def crop_prediction(record: dict[str, Any]) -> Image.Image | None:
    source = Path(str(record.get("source", "")))
    if not source.exists():
        return None
    image = Image.open(source).convert("RGB")
    polygon = record.get("usable_box_polygon") or record.get("final_box_polygon")
    if polygon:
        crop = crop_polygon(image, polygon)
        if crop is not None:
            return crop
    bbox = record.get("usable_bbox") or record.get("final_bbox")
    if not bbox:
        return None
    clamped = clamp_bbox(bbox, image.width, image.height)
    if clamped is None:
        return None
    return image.crop(clamped)


def fallback_no_black_image(record: dict[str, Any]) -> Image.Image | None:
    for field in ("cropped_source", "dinov3_source", "original_source"):
        value = record.get(field)
        if not value:
            continue
        path = Path(str(value))
        if path.exists():
            return Image.open(path).convert("RGB")
    return None


def record_key(record: dict[str, Any]) -> str:
    for field in ("source_key", "original_source", "source"):
        value = record.get(field)
        if value:
            return str(value)
    return ""


def square_resize(image: Image.Image, output_size: int) -> Image.Image:
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    return image.convert("RGB").resize((output_size, output_size), resample=resample)


def save_image(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=max(1, min(int(quality), 100)), subsampling=0)
    else:
        image.save(path)


def output_relative_path(record: dict[str, Any], allowed_classes: set[str]) -> Path | None:
    class_name = str(record.get("class_name") or "")
    if class_name not in allowed_classes:
        return None
    original = Path(str(record.get("original_source") or record.get("source") or ""))
    if not original.name:
        return None
    return Path(class_name) / original.name


def process_prediction_file(
    *,
    split_name: str,
    predictions: Path,
    out_dir: Path,
    output_size: int,
    crop_actions: set[str],
    allowed_classes: set[str],
    quality: int,
    records: Iterable[dict[str, Any]] | None = None,
    train_fallback_class: str = "",
    train_fallback_keys: set[str] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, Counter[str]]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    iterator = records if records is not None else read_jsonl(predictions)
    selected_train_fallback_keys = train_fallback_keys or set()
    for record in iterator:
        rel_path = output_relative_path(record, allowed_classes)
        if rel_path is None:
            counts["outside_class_set"] += 1
            continue
        class_name = rel_path.parts[0]
        action = str(record.get("action", ""))
        output_path = out_dir / rel_path
        saved_kind = ""
        error = ""
        image: Image.Image | None = None
        is_train_fallback_class = split_name == "train" and class_name == train_fallback_class
        if is_train_fallback_class:
            if record_key(record) in selected_train_fallback_keys:
                image = fallback_no_black_image(record)
                if image is not None:
                    saved_kind = "train_no_black_fallback"
                else:
                    error = "missing_train_fallback_source"
                    counts["missing_train_fallback_source"] += 1
            else:
                error = "train_fallback_not_selected"
                counts["train_fallback_not_selected"] += 1
        elif action in crop_actions:
            image = crop_prediction(record)
            if image is not None:
                saved_kind = "roi_crop"
            else:
                error = "empty_crop"
                counts["empty_crop"] += 1
        if image is None and split_name in {"val", "test"}:
            image = fallback_no_black_image(record)
            if image is not None:
                saved_kind = "no_black_fallback"
            else:
                error = "missing_fallback_source"
                counts["missing_fallback_source"] += 1
        if image is None:
            saved = False
            counts["skipped"] += 1
            class_counts[class_name]["skipped"] += 1
        else:
            save_image(square_resize(image, output_size), output_path, quality)
            saved = True
            counts[saved_kind] += 1
            class_counts[class_name][saved_kind] += 1
        counts["records"] += 1
        class_counts[class_name]["records"] += 1
        class_counts[class_name][action or "unknown"] += 1
        rows.append(
            {
                "split": split_name,
                "class_name": class_name,
                "original_source": record.get("original_source", ""),
                "source": record.get("source", ""),
                "cropped_source": record.get("cropped_source", ""),
                "output_path": str(output_path) if saved else "",
                "action": action,
                "saved_kind": saved_kind,
                "saved": saved,
                "final_confidence": record.get("final_confidence", ""),
                "dinov3_point_region_score": (record.get("dinov3_aux") or {}).get("point_region_score", ""),
                "dinov3_confidence_factor": (record.get("dinov3_aux") or {}).get("confidence_factor", ""),
                "flags": ";".join(record.get("flags", [])),
                "error": error,
            }
        )
    return rows, counts, class_counts


def command_dataset(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for class_name in args.classes:
        (out_dir / class_name).mkdir(parents=True, exist_ok=True)
    crop_actions = set(args.crop_actions)
    allowed_classes = set(args.classes)
    train_records = list(read_jsonl(args.train_predictions))
    train_fallback_keys: set[str] = set()
    train_fallback_class = str(args.train_fallback_class or "")
    if train_fallback_class and args.train_fallback_count > 0:
        candidates = [record for record in train_records if str(record.get("class_name") or "") == train_fallback_class]
        rng = random.Random(args.train_fallback_seed)
        selected = rng.sample(candidates, min(args.train_fallback_count, len(candidates)))
        train_fallback_keys = {record_key(record) for record in selected if record_key(record)}
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "out_dir": str(out_dir),
        "output_size": args.output_size,
        "crop_actions": sorted(crop_actions),
        "train_fallback_sample": {
            "class_name": train_fallback_class,
            "requested_count": args.train_fallback_count,
            "selected_count": len(train_fallback_keys),
            "seed": args.train_fallback_seed,
            "policy": "replace train ROI-only handling for this class with sampled no-black fallback squares",
        },
        "split_policy": {
            "train": "save only successful ROI crops",
            "val": "save ROI crop when possible, otherwise no-black fallback square",
            "test": "save ROI crop when possible, otherwise no-black fallback square",
        },
        "splits": {},
    }
    for split_name, predictions in (
        ("train", args.train_predictions),
        ("val", args.val_predictions),
        ("test", args.test_predictions),
    ):
        rows, counts, class_counts = process_prediction_file(
            split_name=split_name,
            predictions=predictions,
            out_dir=out_dir,
            output_size=args.output_size,
            crop_actions=crop_actions,
            allowed_classes=allowed_classes,
            quality=args.jpeg_quality,
            records=train_records if split_name == "train" else None,
            train_fallback_class=train_fallback_class,
            train_fallback_keys=train_fallback_keys,
        )
        all_rows.extend(rows)
        summary["splits"][split_name] = {
            "predictions": str(predictions),
            "counts": dict(counts),
            "counts_by_class": {class_name: dict(class_counts[class_name]) for class_name in args.classes},
        }
    manifest_path = out_dir / "roi_conditioned_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "split",
            "class_name",
            "original_source",
            "source",
            "cropped_source",
            "output_path",
            "action",
            "saved_kind",
            "saved",
            "final_confidence",
            "dinov3_point_region_score",
            "dinov3_confidence_factor",
            "flags",
            "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    summary["manifest"] = str(manifest_path)
    (out_dir / "roi_conditioned_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.command == "manifest":
        command_manifest(args)
    elif args.command == "dataset":
        command_dataset(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
