#!/usr/bin/env python3
"""Build a balanced glottis gate split with ROI-cropped positive samples."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import DEFAULT_SPLIT_PATH, LABEL_NAMES, RESULTS_ROOT, WORKTREE_NAME, write_json


VOC7_SOURCE_FOLDERS = {
    "Normal",
    "正常",
    "Cancer",
    "喉癌",
    "Reinke-Edema",
    "声带任克水肿",
    "Vocal-Cord-Cyst",
    "声带囊肿",
    "Vocal-Cord-Polyp",
    "声带息肉",
    "Vocal-Cord-Leukoplakia",
    "声带白斑",
    "Vocal-Cord-Granuloma",
    "声带肉芽肿",
}


def default_experiment_root() -> Path:
    return RESULTS_ROOT / WORKTREE_NAME / "glottis_binary_roi_crops"


def parse_args() -> argparse.Namespace:
    root = default_experiment_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-split", type=Path, default=root / "roi_cropped_glottis_split.json")
    parser.add_argument("--output-manifest", type=Path, default=root / "roi_cropped_glottis_manifest.csv")
    parser.add_argument("--crop-root", type=Path, default=root / "images")
    parser.add_argument("--crop-list", type=Path, default=root / "roi_crop_input_list.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-fraction", type=float, default=0.5)
    parser.add_argument(
        "--glottis-source-folder",
        action="append",
        default=None,
        help="Allowed positive source folder. Defaults to the seven VOC disease/normal classes.",
    )
    return parser.parse_args()


def load_source_split(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "records" not in payload:
        raise ValueError(f"Split file has no records: {path}")
    return payload


def clone_record(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(record, ensure_ascii=False))


def patient_overlap(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups[str(record["split"])].add(str(record.get("patient_group", "")))
    return {
        "train_val": sorted(groups["train"] & groups["val"]),
        "train_test": sorted(groups["train"] & groups["test"]),
        "val_test": sorted(groups["val"] & groups["test"]),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(int(record["label_id"]) for record in records)
    source_counts = Counter(str(record.get("source_folder", "")) for record in records)
    variant_counts = Counter(str(record.get("sample_variant", "")) for record in records)
    crop_counts = Counter(str(record.get("roi_crop_status", "")) for record in records)
    patient_counts: dict[int, set[str]] = defaultdict(set)
    for record in records:
        patient_counts[int(record["label_id"])].add(str(record.get("patient_group", "")))
    return {
        "num_images": len(records),
        "num_patient_groups": len({str(record.get("patient_group", "")) for record in records}),
        "images_per_label": {
            LABEL_NAMES[label]: int(label_counts[label])
            for label in sorted(LABEL_NAMES)
        },
        "patient_groups_per_label": {
            LABEL_NAMES[label]: len(patient_counts[label])
            for label in sorted(LABEL_NAMES)
        },
        "images_per_source_folder": dict(sorted(source_counts.items())),
        "images_per_sample_variant": dict(sorted(variant_counts.items())),
        "images_per_roi_crop_status": dict(sorted(crop_counts.items())),
    }


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "label",
        "label_id",
        "sample_variant",
        "roi_crop_status",
        "source_folder",
        "patient_group",
        "patient_name",
        "relative_path",
        "image_path",
        "original_image_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def write_crop_list(path: Path, crop_records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "source_folder", "relative_path", "image_path", "crop_output_path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in crop_records:
            writer.writerow(
                {
                    "split": record["split"],
                    "source_folder": record.get("source_folder", ""),
                    "relative_path": record["relative_path"],
                    "image_path": record["original_image_path"],
                    "crop_output_path": record["image_path"],
                }
            )


def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.crop_fraction) <= 1.0:
        raise ValueError("--crop-fraction must be between 0 and 1.")

    source_split_path = args.source_split.expanduser().resolve()
    output_split = args.output_split.expanduser().resolve()
    output_manifest = args.output_manifest.expanduser().resolve()
    crop_root = args.crop_root.expanduser().resolve()
    crop_list = args.crop_list.expanduser().resolve()
    allowed_glottis_folders = set(args.glottis_source_folder or VOC7_SOURCE_FOLDERS)
    source = load_source_split(source_split_path)
    rng = random.Random(int(args.seed))

    source_records = [clone_record(record) for record in source["records"]]
    selected_records: list[dict[str, Any]] = []
    crop_records: list[dict[str, Any]] = []
    dropped_counts: dict[str, dict[str, int]] = {}

    for split_name in ("train", "val", "test"):
        split_rows = [record for record in source_records if record.get("split") == split_name]
        glottis_rows = sorted(
            (
                record
                for record in split_rows
                if int(record["label_id"]) == 1 and str(record.get("source_folder")) in allowed_glottis_folders
            ),
            key=lambda row: str(row.get("relative_path", row.get("image_path", ""))),
        )
        non_glottis_rows = sorted(
            (record for record in split_rows if int(record["label_id"]) == 0),
            key=lambda row: str(row.get("relative_path", row.get("image_path", ""))),
        )
        if not glottis_rows or not non_glottis_rows:
            raise RuntimeError(f"Split {split_name} needs both glottis and non_glottis samples.")

        target_count = min(len(glottis_rows), len(non_glottis_rows))
        glottis_selected = rng.sample(glottis_rows, target_count)
        non_glottis_selected = rng.sample(non_glottis_rows, target_count)
        crop_count = int(round(target_count * float(args.crop_fraction)))
        crop_count = min(target_count, max(0, crop_count))
        crop_source_ids = {
            str(Path(record["image_path"]).expanduser().resolve())
            for record in rng.sample(glottis_selected, crop_count)
        }

        dropped_counts[split_name] = {
            "source_glottis_voc7": len(glottis_rows),
            "source_non_glottis": len(non_glottis_rows),
            "selected_glottis": target_count,
            "selected_non_glottis": target_count,
            "roi_cropped_glottis": crop_count,
            "original_glottis": target_count - crop_count,
            "dropped_glottis_for_balance": len(glottis_rows) - target_count,
            "dropped_non_glottis_for_balance": len(non_glottis_rows) - target_count,
        }

        for record in sorted(glottis_selected, key=lambda row: str(row.get("relative_path", ""))):
            out = clone_record(record)
            original_path = str(Path(out["image_path"]).expanduser().resolve())
            out["original_image_path"] = original_path
            if original_path in crop_source_ids:
                crop_path = crop_root / out["relative_path"]
                out["image_path"] = str(crop_path)
                out["sample_variant"] = "roi_cropped_glottis"
                out["roi_crop_status"] = "roi_cropped_copy"
                crop_records.append(out)
            else:
                out["image_path"] = original_path
                out["sample_variant"] = "original_glottis"
                out["roi_crop_status"] = "not_cropped"
            selected_records.append(out)

        for record in sorted(non_glottis_selected, key=lambda row: str(row.get("relative_path", ""))):
            out = clone_record(record)
            original_path = str(Path(out["image_path"]).expanduser().resolve())
            out["image_path"] = original_path
            out["original_image_path"] = original_path
            out["sample_variant"] = "non_glottis"
            out["roi_crop_status"] = "not_applicable"
            selected_records.append(out)

    records_by_split = {
        split_name: [record for record in selected_records if record["split"] == split_name]
        for split_name in ("train", "val", "test")
    }
    overlaps = patient_overlap(selected_records)
    if any(overlaps.values()):
        raise RuntimeError(f"Patient overlap detected after filtering: {overlaps}")

    payload = {
        "task": "glottis_binary_gate_roi_cropped_voc7",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_split": str(source_split_path),
        "source_dataset_root": source.get("dataset_root"),
        "crop_root": str(crop_root),
        "crop_fraction": float(args.crop_fraction),
        "seed": int(args.seed),
        "label_names": LABEL_NAMES,
        "source_glottis_policy": (
            "positives are limited to the seven in-distribution VOC classes from the 8-class task; "
            "half of selected positives point to ROI-cropped copies, half stay original"
        ),
        "non_glottis_policy": "sample the same number of non_glottis records per split without replacement",
        "allowed_glottis_source_folders": sorted(allowed_glottis_folders),
        "balance_summary": dropped_counts,
        "stats": {
            split_name: summarize_records(split_records)
            for split_name, split_records in records_by_split.items()
        },
        "audit": {
            "patient_group_overlap": overlaps,
            "num_crop_records": len(crop_records),
            "crop_list": str(crop_list),
            "output_manifest": str(output_manifest),
        },
        "records": selected_records,
    }

    output_split.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_split, payload)
    write_manifest(output_manifest, selected_records)
    write_crop_list(crop_list, crop_records)
    print(f"Wrote ROI-cropped split: {output_split}")
    print(f"Wrote training manifest: {output_manifest}")
    print(f"Wrote crop input list: {crop_list}")
    for split_name in ("train", "val", "test"):
        stats = payload["stats"][split_name]
        print(f"{split_name}: {stats['images_per_label']}, variants={stats['images_per_sample_variant']}")


if __name__ == "__main__":
    main()
