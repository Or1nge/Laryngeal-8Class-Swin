#!/usr/bin/env python3
"""Move misclassified images from a folder dataset into a review directory."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shared  # noqa: E402
from shared import (  # noqa: E402
    BEST_MODEL_PATH,
    DISPLAY_NAMES,
    FOLDER_TO_LABEL,
    HierarchicalImageClassifier,
    IMAGE_EXTENSIONS,
    LABEL_DICT,
    build_transforms,
    get_voc_label_indices,
    gpu_normalise,
    init_label_mapping,
    load_config,
    setup_device,
)


class FolderImageDataset(Dataset):
    def __init__(self, source_root: Path, transform):
        self.source_root = source_root
        self.transform = transform
        self.rows = []
        for folder_name in sorted(FOLDER_TO_LABEL):
            folder = source_root / folder_name
            if not folder.is_dir():
                continue
            label = FOLDER_TO_LABEL[folder_name]
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.rows.append((path, label, folder_name))
        if not self.rows:
            raise RuntimeError(f"No active-class images found under {source_root}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label, folder_name = self.rows[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label, str(path), folder_name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config_phase2.json")
    parser.add_argument("--model", type=Path, default=Path(BEST_MODEL_PATH))
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses config eval_batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Defaults to config num_workers.")
    parser.add_argument(
        "--mode",
        choices=["hierarchical", "argmax"],
        default="hierarchical",
        help="Prediction rule. hierarchical matches the project inference logic.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing output directory before writing new results.",
    )
    return parser.parse_args()


def hierarchical_predictions_from_logits(logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    voc_indices = torch.tensor(get_voc_label_indices(), dtype=torch.long, device=logits.device)
    non_voc_label = int(shared.NON_VOC_LABEL)
    non_voc_prob = probs[:, non_voc_label]
    voc_prob = probs.index_select(1, voc_indices).sum(dim=1)
    is_vocal_cord = voc_prob > non_voc_prob

    preds = torch.full((logits.size(0),), non_voc_label, dtype=torch.long, device=logits.device)
    if is_vocal_cord.any():
        voc_logits = logits.index_select(1, voc_indices)
        preds[is_vocal_cord] = voc_indices[torch.argmax(voc_logits[is_vocal_cord], dim=1)]
    return preds


def force_timm_no_pretrained():
    original_create_model = shared.timm.create_model

    def create_model_no_pretrained(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    shared.timm.create_model = create_model_no_pretrained
    return original_create_model


def restore_timm_create_model(original_create_model):
    shared.timm.create_model = original_create_model


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}__{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def main():
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root not found: {source_root}")
    if output_root.exists():
        if not args.force:
            raise SystemExit(f"Output directory already exists: {output_root}. Pass --force.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    cfg = load_config(str(args.config))
    init_label_mapping(cfg)
    _, eval_tf = build_transforms(cfg)
    dataset = FolderImageDataset(source_root, eval_tf)
    batch_size = args.batch_size if args.batch_size > 0 else cfg.get("eval_batch_size", 128)
    num_workers = cfg.get("num_workers", 0) if args.num_workers is None else args.num_workers
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = cfg.get("persistent_workers", True)
        prefetch_factor = cfg.get("prefetch_factor", 4)
        if prefetch_factor:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )

    device = setup_device()
    original_create_model = force_timm_no_pretrained()
    try:
        model = HierarchicalImageClassifier(num_classes=len(LABEL_DICT), cfg=cfg).to(device)
    finally:
        restore_timm_create_model(original_create_model)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    rows = []
    total = 0
    correct = 0
    moved = 0
    amp_context = torch.amp.autocast(device_type=device.type) if device.type == "cuda" else nullcontext()

    with torch.inference_mode():
        for images, labels, paths, folder_names in loader:
            labels = labels.to(device, non_blocking=True)
            images = gpu_normalise(images.to(device, non_blocking=True))
            with amp_context:
                logits = model(images)
            probs = F.softmax(logits, dim=1)
            if args.mode == "hierarchical":
                preds = hierarchical_predictions_from_logits(logits, probs)
            else:
                preds = torch.argmax(logits, dim=1)

            total += labels.numel()
            correct_mask = preds.eq(labels)
            correct += int(correct_mask.sum().item())

            for label, pred, path_str, folder_name, prob_row in zip(
                labels.cpu().tolist(),
                preds.cpu().tolist(),
                paths,
                folder_names,
                probs.cpu(),
            ):
                true_name = DISPLAY_NAMES[label]
                pred_name = DISPLAY_NAMES[pred]
                pred_prob = float(prob_row[pred].item())
                src = Path(path_str)
                if label == pred:
                    continue
                target_dir = output_root / f"{folder_name}__pred_{pred_name}"
                target_dir.mkdir(parents=True, exist_ok=True)
                dst = unique_destination(target_dir / src.name)
                shutil.move(str(src), str(dst))
                moved += 1
                rows.append(
                    {
                        "original_path": str(src),
                        "moved_path": str(dst),
                        "true_label": true_name,
                        "pred_label": pred_name,
                        "pred_prob": f"{pred_prob:.6f}",
                    }
                )

    manifest_path = output_root / "mispredictions.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["original_path", "moved_path", "true_label", "pred_label", "pred_prob"],
        )
        writer.writeheader()
        writer.writerows(rows)

    accuracy = correct / max(total, 1)
    print(f"source_root={source_root}")
    print(f"output_root={output_root}")
    print(f"mode={args.mode}")
    print(f"total={total}")
    print(f"correct={correct}")
    print(f"wrong_moved={moved}")
    print(f"accuracy={accuracy:.6f}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
