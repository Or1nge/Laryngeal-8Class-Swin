#!/usr/bin/env python3
"""Shared utilities for glottis/non-glottis binary training."""

from __future__ import annotations

import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

sys.dont_write_bytecode = True

MODULE_DIR = Path(__file__).resolve().parent
IMAGE_RECOGNITION_DIR = MODULE_DIR.parent
PROJECT_ROOT = IMAGE_RECOGNITION_DIR.parent

if str(IMAGE_RECOGNITION_DIR) not in sys.path:
    sys.path.insert(0, str(IMAGE_RECOGNITION_DIR))

from shared import (  # noqa: E402
    CropBlackBorders,
    GPUAugment,
    LOCAL_WEIGHT_CANDIDATES,
    WarmupCosineScheduler,
    gpu_normalise,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DICOM_PATIENT_KEY_PATTERN = re.compile(r"(?:^|_)13\.(\d{13})\.")
LABEL_NAMES = {0: "non_glottis", 1: "glottis"}
DISPLAY_NAMES = {0: "Non-glottis / non-effective vocal fold", 1: "Glottis / effective vocal fold"}

NON_GLOTTIS_SOURCE_FOLDERS = {
    "Non-Vocal-Cord",
    "混杂图片",
    "室带膨隆",
    "不用管—质量图片",
}
GLOTTIS_EXACT_SOURCE_FOLDERS = {
    "Normal",
    "正常",
    "Cancer",
    "喉癌",
}

DEFAULT_SPLIT_PATH = MODULE_DIR / "glottis_binary_split.json"
DEFAULT_MANIFEST_PATH = MODULE_DIR / "glottis_binary_manifest.csv"


def detect_workspace_dir(project_root: Path = PROJECT_ROOT) -> Path:
    if project_root.parent.name == "worktrees":
        return project_root.parent.parent
    return project_root


def detect_worktree_name(project_root: Path = PROJECT_ROOT) -> str:
    if project_root.parent.name == "worktrees":
        return project_root.name
    return project_root.name


WORKSPACE_DIR = Path(os.environ.get("LARYNX_WORKSPACE_DIR", detect_workspace_dir()))
WORKTREE_NAME = os.environ.get("LARYNX_WORKTREE_NAME", detect_worktree_name())
WORKSPACE_PARENT_DIR = WORKSPACE_DIR.parent
DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "LARYNX_IMAGE_DIR",
        next(
            (
                str(path)
                for path in (
                    WORKSPACE_DIR / "Laryngeal_Dataset_Processed",
                    WORKSPACE_PARENT_DIR / "Laryngeal_Dataset_Processed",
                    PROJECT_ROOT.parent / "Laryngeal_Dataset_Processed",
                )
                if path.is_dir()
            ),
            str(WORKSPACE_PARENT_DIR / "Laryngeal_Dataset_Processed"),
        ),
    )
)
RESULTS_ROOT = Path(os.environ.get("LARYNX_RESULTS_ROOT", WORKSPACE_DIR / "Results"))
DEFAULT_BENCHMARK_ROOT = Path(
    os.environ.get(
        "LARYNX_GLOTTIS_BINARY_RESULTS_DIR",
        RESULTS_ROOT / WORKTREE_NAME / "glottis_binary_benchmarks",
    )
)


MODEL_REGISTRY = {
    "resnet50": {
        "timm_name": "resnet50.a1_in1k",
        "method": "ce",
        "description": "ResNet-50 ImageNet CE baseline",
    },
    "vit_base": {
        "timm_name": "vit_base_patch16_224.augreg_in21k_ft_in1k",
        "method": "ce",
        "description": "ViT-B/16 ImageNet-21k CE baseline",
    },
    "swin_base": {
        "timm_name": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        "method": "ce",
        "description": "Swin-B ImageNet-22k CE baseline",
    },
    "supcon_swin_base": {
        "timm_name": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        "method": "supcon_ce",
        "description": "Supervised contrastive pretraining + Swin-B CE fine-tuning",
    },
}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def groups(self) -> dict[str, set[str]]:
        grouped: dict[str, set[str]] = defaultdict(set)
        for item in self.parent:
            grouped[self.find(item)].add(item)
        return grouped


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    return device


def infer_source_folder_label(folder_name: str) -> tuple[int | None, str]:
    if folder_name in NON_GLOTTIS_SOURCE_FOLDERS:
        return 0, "explicit_non_glottis_source_folder"
    if folder_name in GLOTTIS_EXACT_SOURCE_FOLDERS:
        return 1, "explicit_glottis_source_folder"
    if "声带" in folder_name or "Vocal-Cord" in folder_name:
        return 1, "vocal_cord_disease_folder"
    return None, "unmapped_folder_semantics"


def iter_image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_patient_name(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        prefix = stem.split("_", 1)[0].strip()
        if prefix:
            return prefix
    if len(stem) >= 8 and stem[:8].isdigit():
        return stem[:8]
    dicom_patient_key = extract_dicom_patient_key(stem)
    if dicom_patient_key:
        return dicom_patient_key
    stripped_stem = stem.strip()
    return stripped_stem if stripped_stem else path.name


def extract_dicom_patient_key(stem: str) -> str | None:
    match = DICOM_PATIENT_KEY_PATTERN.search(stem)
    if not match:
        return None
    return f"{int(match.group(1)):08d}"[-8:]


def get_patient_aliases(path: Path) -> set[str]:
    stem = path.stem
    aliases = {get_patient_name(path)}

    named_ten_digit = re.match(r"^.+?_(\d{10})(?:$|_)", stem)
    if named_ten_digit:
        aliases.add(named_ten_digit.group(1)[:8])

    dicom_patient_key = extract_dicom_patient_key(stem)
    if dicom_patient_key:
        aliases.add(dicom_patient_key)

    return {alias for alias in aliases if alias}


def discover_binary_rows(dataset_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    source_folder_rows: list[dict[str, Any]] = []
    for folder in sorted((p for p in dataset_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        if folder.name.startswith("__"):
            continue
        image_files = iter_image_files(folder)
        label, reason = infer_source_folder_label(folder.name)
        included = bool(image_files) and label is not None
        source_folder_rows.append(
            {
                "source_folder": folder.name,
                "num_images": len(image_files),
                "label": LABEL_NAMES[label] if label is not None else None,
                "label_id": label,
                "included": included,
                "mapping_reason": reason,
            }
        )
        if not included:
            continue
        for path in image_files:
            rows.append(
                {
                    "image_path": str(path),
                    "relative_path": str(path.relative_to(dataset_root)),
                    "source_folder": folder.name,
                    "patient_name": get_patient_name(path),
                    "patient_aliases": sorted(get_patient_aliases(path)),
                    "label": LABEL_NAMES[int(label)],
                    "label_id": int(label),
                }
            )
    if not rows:
        raise RuntimeError(f"No binary-task images found under {dataset_root}")
    return rows, source_folder_rows


def build_patient_alias_groups(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    union_find = UnionFind()
    for row in rows:
        aliases = list(row["patient_aliases"])
        for alias in aliases:
            union_find.add(alias)
        for alias in aliases[1:]:
            union_find.union(aliases[0], alias)
    return {sorted(aliases)[0]: set(aliases) for aliases in union_find.groups().values()}


def assign_patient_groups(rows: list[dict[str, Any]], alias_groups: dict[str, set[str]]) -> None:
    alias_to_group = {
        alias: group_name
        for group_name, aliases in alias_groups.items()
        for alias in aliases
    }
    for row in rows:
        row["patient_group"] = alias_to_group[row["patient_name"]]


def majority_label_by_patient(rows: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        grouped[row["patient_group"]][int(row["label_id"])] += 1
    return {
        patient: min(counts, key=lambda label: (-counts[label], label))
        for patient, counts in grouped.items()
    }


def split_patients(
    patient_labels: dict[str, int],
    seed: int,
    test_size: float,
    val_size_of_remaining: float,
) -> dict[str, set[str]]:
    patients = sorted(patient_labels)
    labels = [patient_labels[patient] for patient in patients]
    label_counts = Counter(labels)
    if min(label_counts.values()) < 3:
        raise ValueError(f"Need at least three patient groups per binary label: {dict(label_counts)}")

    train_val_patients, test_patients, train_val_labels, _ = train_test_split(
        patients,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=val_size_of_remaining,
        stratify=train_val_labels,
        random_state=seed,
    )
    return {
        "train": set(train_patients),
        "val": set(val_patients),
        "test": set(test_patients),
    }


def patient_overlap(patient_splits: dict[str, set[str]]) -> dict[str, list[str]]:
    return {
        "train_val": sorted(patient_splits["train"] & patient_splits["val"]),
        "train_test": sorted(patient_splits["train"] & patient_splits["test"]),
        "val_test": sorted(patient_splits["val"] & patient_splits["test"]),
    }


def output_patient_aliases(
    patient_splits: dict[str, set[str]],
    alias_groups: dict[str, set[str]],
    observed_patient_names: set[str],
) -> dict[str, list[str]]:
    return {
        split_name: sorted(
            alias
            for group_name in group_names
            for alias in alias_groups[group_name]
            if alias in observed_patient_names
        )
        for split_name, group_names in patient_splits.items()
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    image_counts = Counter(int(row["label_id"]) for row in records)
    patient_counts: dict[int, set[str]] = defaultdict(set)
    source_counts: Counter[str] = Counter()
    for row in records:
        label_id = int(row["label_id"])
        patient_counts[label_id].add(row["patient_group"])
        source_counts[row["source_folder"]] += 1
    return {
        "num_images": len(records),
        "num_patient_groups": len({row["patient_group"] for row in records}),
        "images_per_label": {
            LABEL_NAMES[label]: int(image_counts[label])
            for label in sorted(LABEL_NAMES)
        },
        "patient_groups_per_label": {
            LABEL_NAMES[label]: len(patient_counts[label])
            for label in sorted(LABEL_NAMES)
        },
        "images_per_source_folder": dict(sorted(source_counts.items())),
    }


def build_binary_split(
    dataset_root: Path,
    output_path: Path = DEFAULT_SPLIT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    seed: int = 42,
    test_size: float = 0.1,
    val_size_of_remaining: float = 0.1111,
    force: bool = False,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output_path = output_path.resolve()
    manifest_path = manifest_path.resolve()
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --force.")
    if manifest_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}; pass --force.")

    rows, source_folder_rows = discover_binary_rows(dataset_root)
    alias_groups = build_patient_alias_groups(rows)
    assign_patient_groups(rows, alias_groups)

    patient_labels = majority_label_by_patient(rows)
    patient_splits = split_patients(
        patient_labels,
        seed=seed,
        test_size=test_size,
        val_size_of_remaining=val_size_of_remaining,
    )
    overlaps = patient_overlap(patient_splits)
    if any(overlaps.values()):
        raise RuntimeError(f"Patient group overlap detected: {overlaps}")

    observed_patient_names = {row["patient_name"] for row in rows}
    patient_alias_splits = output_patient_aliases(patient_splits, alias_groups, observed_patient_names)
    alias_overlaps = patient_overlap(
        {split_name: set(aliases) for split_name, aliases in patient_alias_splits.items()}
    )
    if any(alias_overlaps.values()):
        raise RuntimeError(f"Patient alias overlap detected: {alias_overlaps}")

    for row in rows:
        row["split"] = next(
            split_name
            for split_name, patient_group_set in patient_splits.items()
            if row["patient_group"] in patient_group_set
        )

    records_by_split = {
        split_name: [row for row in rows if row["split"] == split_name]
        for split_name in ("train", "val", "test")
    }
    patient_classes: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        patient_classes[row["patient_group"]].add(int(row["label_id"]))
    multi_label_patients = {
        patient: sorted(LABEL_NAMES[label] for label in labels)
        for patient, labels in patient_classes.items()
        if len(labels) > 1
    }
    alias_group_examples = {
        group_name: sorted(aliases)
        for group_name, aliases in sorted(alias_groups.items())
        if len(aliases) > 1
    }

    payload = {
        "task": "glottis_binary_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "seed": int(seed),
        "split_method": "stratified_by_patient_majority_binary_label_with_alias_union",
        "test_size": float(test_size),
        "val_size_of_remaining": float(val_size_of_remaining),
        "label_names": LABEL_NAMES,
        "display_names": DISPLAY_NAMES,
        "patient_id_rules": [
            "use prefix before first underscore for named files",
            "use first eight digits for numeric filenames such as 0005481703.jpg",
            "use the embedded 13.x DICOM-like key for bare DICOM-style filenames",
            "merge filename aliases when a named file exposes a matching numeric patient key",
            "otherwise use the full filename stem",
        ],
        "folder_mapping_policy": [
            "混杂图片 / Non-Vocal-Cord => non_glottis",
            "室带膨隆 and 不用管—质量图片 => non_glottis for the effective-vocal-fold gate; review risk is documented",
            "正常 / Normal, 喉癌 / Cancer, and folders containing 声带 or Vocal-Cord => glottis",
            "empty or unmapped folders are recorded but excluded",
        ],
        "source_folders": source_folder_rows,
        "patients": patient_alias_splits,
        "stats": {
            split_name: summarize_records(split_records)
            for split_name, split_records in records_by_split.items()
        },
        "audit": {
            "num_images": len(rows),
            "num_patient_groups": len(patient_labels),
            "num_patient_aliases": len({row["patient_name"] for row in rows}),
            "patient_group_overlap": overlaps,
            "patient_alias_overlap": alias_overlaps,
            "patient_majority_label_counts": {
                LABEL_NAMES[label]: int(count)
                for label, count in sorted(Counter(patient_labels.values()).items())
            },
            "num_alias_groups": sum(1 for aliases in alias_groups.values() if len(aliases) > 1),
            "patient_alias_group_examples": dict(list(alias_group_examples.items())[:20]),
            "num_multi_label_patients": len(multi_label_patients),
            "multi_label_patient_examples": dict(list(sorted(multi_label_patients.items()))[:20]),
            "risk_notes": [
                "Patient grouping is filename-derived; alias union reduces but cannot fully prove identity equivalence.",
                "室带膨隆 is treated as non_glottis because it is not effective vocal-fold/glottic mucosa; this should be checked clinically if it becomes a major error source.",
                "不用管—质量图片 has only a few images and is included as non_glottis/unusable evidence rather than discarded.",
            ],
        },
        "records": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "label",
                "label_id",
                "source_folder",
                "patient_group",
                "patient_name",
                "relative_path",
                "image_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": row["split"],
                    "label": row["label"],
                    "label_id": row["label_id"],
                    "source_folder": row["source_folder"],
                    "patient_group": row["patient_group"],
                    "patient_name": row["patient_name"],
                    "relative_path": row["relative_path"],
                    "image_path": row["image_path"],
                }
            )
    return payload


def load_split_dataframe(split_path: Path = DEFAULT_SPLIT_PATH) -> pd.DataFrame:
    with Path(split_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    df = pd.DataFrame(payload["records"])
    if df.empty:
        raise RuntimeError(f"No records in split file: {split_path}")
    df["label_id"] = df["label_id"].astype(int)
    return df


def default_train_config() -> dict[str, Any]:
    return {
        "seed": 42,
        "image_size": 224,
        "resize_size": 256,
        "crop_black_threshold": 15,
        "batch_size": 384,
        "eval_batch_size": 768,
        "epochs": 18,
        "patience": 5,
        "early_stopping_min_delta": 0.0005,
        "learning_rate": 8e-5,
        "weight_decay": 0.05,
        "label_smoothing": 0.03,
        "drop_rate": 0.15,
        "drop_path_rate": 0.1,
        "sampler_balance_alpha": 0.75,
        "grad_clip_norm": 1.0,
        "supcon_epochs": 8,
        "supcon_patience": 3,
        "supcon_learning_rate": 5e-5,
        "supcon_temperature": 0.08,
        "supcon_projection_dim": 128,
        "random_resized_crop_scale_min": 0.85,
        "random_resized_crop_scale_max": 1.0,
        "random_resized_crop_ratio_min": 0.9,
        "random_resized_crop_ratio_max": 1.1,
        "random_affine_degrees": 8,
        "random_affine_translate": [0.06, 0.06],
        "random_affine_scale": [0.92, 1.08],
        "random_horizontal_flip_prob": 0.5,
        "random_vertical_flip_prob": 0.0,
        "color_jitter_brightness": 0.15,
        "color_jitter_contrast": 0.15,
        "color_jitter_saturation": 0.08,
        "color_jitter_hue": 0.01,
        "gaussian_blur_prob": 0.15,
        "gaussian_blur_sigma_max": 1.5,
        "random_adjust_sharpness_prob": 0.15,
        "random_adjust_sharpness_factor": 1.4,
        "recommended_min_non_glottis_specificity": 0.99,
        "recommended_min_glottis_recall": 0.85,
    }


def merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def load_image_uint8(path: str | Path, cfg: dict[str, Any]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = CropBlackBorders(cfg.get("crop_black_threshold", 15))(image)
    resize_size = int(cfg.get("resize_size", 256))
    image_size = int(cfg.get("image_size", 224))
    image = image.resize((resize_size, resize_size), Image.BICUBIC)
    left = max((resize_size - image_size) // 2, 0)
    top = max((resize_size - image_size) // 2, 0)
    image = image.crop((left, top, left + image_size, top + image_size))
    array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


@dataclass
class SplitCache:
    images: torch.Tensor
    labels: torch.Tensor
    frame: pd.DataFrame


def preload_split_cache(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    device: torch.device,
    cache_device: str = "cuda",
) -> dict[str, SplitCache]:
    target_device = device if cache_device == "cuda" and device.type == "cuda" else torch.device("cpu")
    caches: dict[str, SplitCache] = {}
    for split_name in ("train", "val", "test"):
        frame = df[df["split"] == split_name].reset_index(drop=True)
        tensors = []
        failures = []
        for idx, path in enumerate(frame["image_path"].tolist(), start=1):
            try:
                tensors.append(load_image_uint8(path, cfg))
            except Exception as exc:  # noqa: BLE001
                failures.append({"image_path": path, "error": repr(exc)})
                tensors.append(torch.zeros(3, cfg["image_size"], cfg["image_size"], dtype=torch.uint8))
            if idx % 1000 == 0:
                print(f"  cached {split_name}: {idx}/{len(frame)} images")
        images = torch.stack(tensors).to(target_device, non_blocking=False)
        labels = torch.tensor(frame["label_id"].to_numpy(), dtype=torch.long, device=target_device)
        caches[split_name] = SplitCache(images=images, labels=labels, frame=frame)
        if failures:
            print(f"Warning: {len(failures)} images failed to load in {split_name}; black placeholders used.")
    return caches


def class_sample_weights(labels: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    labels_cpu = labels.detach().cpu()
    counts = torch.bincount(labels_cpu, minlength=2).float().clamp_min(1.0)
    weights_by_class = (counts.sum() / (len(counts) * counts)).pow(float(alpha))
    return weights_by_class[labels_cpu].to(labels.device, dtype=torch.float32)


def iter_train_batches(cache: SplitCache, batch_size: int, balance_alpha: float):
    weights = class_sample_weights(cache.labels, balance_alpha)
    num_samples = int(cache.labels.numel())
    sampled_indices = torch.multinomial(weights, num_samples=num_samples, replacement=True)
    for start in range(0, num_samples, batch_size):
        idx = sampled_indices[start:start + batch_size]
        yield cache.images[idx], cache.labels[idx]


def iter_eval_batches(cache: SplitCache, batch_size: int):
    num_samples = int(cache.labels.numel())
    for start in range(0, num_samples, batch_size):
        idx = slice(start, min(start + batch_size, num_samples))
        yield cache.images[idx], cache.labels[idx]


def create_timm_backbone(
    model_name: str,
    pretrained: bool,
    drop_rate: float,
    drop_path_rate: float,
) -> tuple[nn.Module, str]:
    kwargs = {
        "num_classes": 0,
        "global_pool": "avg",
        "drop_rate": drop_rate,
        "drop_path_rate": drop_path_rate,
    }

    def _create(pretrained_flag: bool) -> nn.Module:
        try:
            return timm.create_model(model_name, pretrained=pretrained_flag, **kwargs)
        except TypeError:
            slim_kwargs = {k: v for k, v in kwargs.items() if k not in {"drop_path_rate", "drop_rate"}}
            return timm.create_model(model_name, pretrained=pretrained_flag, **slim_kwargs)

    if pretrained and model_name == "swin_base_patch4_window7_224.ms_in22k_ft_in1k":
        model = _create(pretrained_flag=False)
        for weight_path in LOCAL_WEIGHT_CANDIDATES:
            weight_path = Path(weight_path)
            if not weight_path.exists():
                continue
            try:
                from safetensors.torch import load_file

                state_dict = load_file(str(weight_path))
                model.load_state_dict(state_dict, strict=False)
                return model, str(weight_path)
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to load local Swin weights from {weight_path}: {exc}")

    if pretrained:
        try:
            return _create(pretrained_flag=True), "timm_pretrained"
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: pretrained load failed for {model_name}: {exc}")
            print("Retrying with random initialization.")
    return _create(pretrained_flag=False), "random_init"


class FeatureBinaryClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        drop_rate: float = 0.15,
        drop_path_rate: float = 0.1,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.backbone, self.pretrained_source = create_timm_backbone(
            model_name=model_name,
            pretrained=pretrained,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        feature_dim = int(getattr(self.backbone, "num_features"))
        self.feature_dim = feature_dim
        self.projector = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim, projection_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim, 2),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode(x)
        return F.normalize(self.projector(features), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode(x)
        return self.classifier(features)


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.08) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        logits = torch.div(features @ features.T, self.temperature)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=features.device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        positive_count = mask.sum(dim=1).clamp_min(1.0)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positive_count
        return -mean_log_prob_pos.mean()


def safe_metric(fn, default: float = float("nan")) -> float:
    try:
        return float(fn())
    except ValueError:
        return default


def binary_metrics(y_true: np.ndarray, probs_glottis: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (probs_glottis >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "support": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_glottis": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_glottis_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity_non_glottis": float(tn / max(tn + fp, 1)),
        "f1_glottis": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_pass_non_glottis_rate": float(fp / max(tn + fp, 1)),
        "false_block_glottis_rate": float(fn / max(fn + tp, 1)),
        "tn_non_glottis": int(tn),
        "fp_non_glottis_as_glottis": int(fp),
        "fn_glottis_as_non_glottis": int(fn),
        "tp_glottis": int(tp),
        "auroc": safe_metric(lambda: roc_auc_score(y_true, probs_glottis)),
        "auprc": safe_metric(lambda: average_precision_score(y_true, probs_glottis)),
    }


def threshold_table(y_true: np.ndarray, probs_glottis: np.ndarray) -> pd.DataFrame:
    thresholds = np.round(np.linspace(0.01, 0.99, 99), 4)
    return pd.DataFrame(binary_metrics(y_true, probs_glottis, threshold=t) for t in thresholds)


def choose_gate_threshold(
    val_thresholds: pd.DataFrame,
    min_specificity: float,
    min_glottis_recall: float,
) -> dict[str, Any]:
    selected_floor = None
    candidates = pd.DataFrame()
    for floor in [min_specificity, 0.985, 0.98, 0.975, 0.97, 0.95]:
        filtered = val_thresholds[
            (val_thresholds["specificity_non_glottis"] >= floor)
            & (val_thresholds["recall_glottis_sensitivity"] >= min_glottis_recall)
        ].copy()
        if not filtered.empty:
            selected_floor = floor
            candidates = filtered
            break
    if candidates.empty:
        candidates = val_thresholds.copy()
        selected_floor = float(candidates["specificity_non_glottis"].max())
    candidates["gate_score"] = (
        0.65 * candidates["specificity_non_glottis"]
        + 0.25 * candidates["recall_glottis_sensitivity"]
        + 0.10 * candidates["balanced_accuracy"]
    )
    best = candidates.sort_values(
        ["gate_score", "specificity_non_glottis", "recall_glottis_sensitivity", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]
    return {
        "threshold": float(best["threshold"]),
        "selection_specificity_floor": float(selected_floor),
        "val_metrics_at_threshold": {
            key: (float(value) if isinstance(value, (np.floating, float)) else int(value))
            for key, value in best.to_dict().items()
        },
        "selection_rule": (
            "maximize 0.65*specificity_non_glottis + 0.25*glottis_recall "
            "+ 0.10*balanced_accuracy under the strongest available specificity floor"
        ),
    }


@torch.inference_mode()
def collect_outputs(
    model: nn.Module,
    cache: SplitCache,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    y_true = []
    probs = []
    amp_context = torch.amp.autocast(device_type=device.type) if device.type == "cuda" else torch.no_grad()
    for images, labels in iter_eval_batches(cache, batch_size):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images = gpu_normalise(images)
        with amp_context:
            logits = model(images)
        y_true.append(labels.detach().cpu().numpy())
        probs.append(F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    return {
        "y_true": np.concatenate(y_true),
        "probs_glottis": np.concatenate(probs),
    }


def save_predictions(
    split_name: str,
    outputs: dict[str, np.ndarray],
    cache: SplitCache,
    threshold: float,
    output_dir: Path,
) -> None:
    y_true = outputs["y_true"]
    probs = outputs["probs_glottis"]
    y_pred = (probs >= threshold).astype(int)
    frame = cache.frame.copy()
    frame["true_label"] = [LABEL_NAMES[int(label)] for label in y_true]
    frame["prob_glottis"] = probs
    frame["prob_non_glottis"] = 1.0 - probs
    frame["threshold"] = threshold
    frame["pred_label"] = [LABEL_NAMES[int(label)] for label in y_pred]
    frame["correct"] = y_true == y_pred
    frame["error_type"] = np.where(
        frame["correct"],
        "",
        np.where(
            y_true == 0,
            "false_pass_non_glottis_into_gate",
            "false_block_glottis_from_gate",
        ),
    )
    frame.to_csv(output_dir / f"predictions_{split_name}.csv", index=False)
    if split_name == "test":
        errors = frame[~frame["correct"]].copy()
        errors.to_csv(output_dir / "error_samples_test.csv", index=False)


def save_confusion_matrix_png(y_true: np.ndarray, probs_glottis: np.ndarray, threshold: float, path: Path) -> None:
    y_pred = (probs_glottis >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], ["Pred non-glottis", "Pred glottis"])
    ax.set_yticks([0, 1], ["True non-glottis", "True glottis"])
    ax.set_title(f"Test confusion matrix @ threshold {threshold:.2f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_roc_pr_png(y_true: np.ndarray, probs_glottis: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    try:
        fpr, tpr, _ = roc_curve(y_true, probs_glottis)
        axes[0].plot(fpr, tpr, color="#0B6E4F", linewidth=2)
        axes[0].plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
        axes[0].set_title(f"ROC AUC {roc_auc_score(y_true, probs_glottis):.4f}")
    except ValueError:
        axes[0].text(0.5, 0.5, "ROC unavailable", ha="center", va="center")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].grid(alpha=0.25)

    try:
        precision, recall, _ = precision_recall_curve(y_true, probs_glottis)
        axes[1].plot(recall, precision, color="#8F2D56", linewidth=2)
        axes[1].set_title(f"PR AUC {average_precision_score(y_true, probs_glottis):.4f}")
    except ValueError:
        axes[1].text(0.5, 0.5, "PR unavailable", ha="center", va="center")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_provenance() -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:  # noqa: BLE001
            return None

    return {
        "project_root": str(PROJECT_ROOT),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty_status": run_git(["status", "--short"]),
    }


def archive_source_files(run_root: Path) -> None:
    source_dir = run_root / "source_snapshot"
    source_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "common.py",
        "build_manifest_split.py",
        "train_benchmarks.py",
        "evaluate_checkpoint.py",
        "README.md",
    ]:
        src = MODULE_DIR / filename
        if src.exists():
            shutil.copy2(src, source_dir / filename)


def checkpoint_payload(
    model: FeatureBinaryClassifier,
    model_key: str,
    model_info: dict[str, Any],
    cfg: dict[str, Any],
    best_epoch: int,
    threshold: float | None = None,
) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "model_key": model_key,
        "model_info": model_info,
        "model_name": model.model_name,
        "pretrained_source": model.pretrained_source,
        "cfg": cfg,
        "label_names": LABEL_NAMES,
        "display_names": DISPLAY_NAMES,
        "best_epoch": int(best_epoch),
        "recommended_threshold": threshold,
    }


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[FeatureBinaryClassifier, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = default_train_config()
    cfg.update(ckpt.get("cfg", {}))
    model = FeatureBinaryClassifier(
        model_name=ckpt["model_name"],
        pretrained=False,
        drop_rate=float(cfg.get("drop_rate", 0.15)),
        drop_path_rate=float(cfg.get("drop_path_rate", 0.1)),
        projection_dim=int(cfg.get("supcon_projection_dim", 128)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.pretrained_source = ckpt.get("pretrained_source", "checkpoint")
    return model, ckpt
