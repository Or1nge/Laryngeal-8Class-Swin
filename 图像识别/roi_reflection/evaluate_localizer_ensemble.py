#!/usr/bin/env python3
"""Evaluate localizer checkpoint probability ensembles with val-selected thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from common import (
    BaglsSegmentationDataset,
    ROI_RESULTS_DIR,
    ensure_dir,
    load_roi_localizer_model,
    normalise_localizer_batch,
    resolve_device,
    write_json,
)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected name=/path form, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser()


def parse_combo(value: str) -> tuple[str, list[tuple[str, float]]]:
    if "=" not in value:
        raise ValueError(f"Expected combo=name1,name2 form, got {value!r}")
    name, labels = value.split("=", 1)
    parsed = []
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


def mask_bbox_str(mask: np.ndarray) -> str:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return ""
    return f"{int(xs.min())},{int(ys.min())},{int(xs.max() + 1)},{int(ys.max() + 1)}"


def overlay_image(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    out = image.astype(np.float32).copy()
    gt_mask = gt > 0
    pred_mask = pred > 0
    tp = gt_mask & pred_mask
    fp = pred_mask & ~gt_mask
    fn = gt_mask & ~pred_mask
    out[tp] = 0.55 * out[tp] + 0.45 * np.array([0, 220, 0], dtype=np.float32)
    out[fp] = 0.55 * out[fp] + 0.45 * np.array([255, 0, 0], dtype=np.float32)
    out[fn] = 0.55 * out[fn] + 0.45 * np.array([0, 80, 255], dtype=np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def dice_values(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(1, pred.dim()))
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    return (2.0 * inter + eps) / (denom + eps)


def apply_threshold(probs: torch.Tensor, threshold: float, min_pred_area: float) -> torch.Tensor:
    pred = (probs >= float(threshold)).float()
    if min_pred_area > 0:
        pred_area = pred.mean(dim=tuple(range(1, pred.dim())))
        keep = pred_area >= float(min_pred_area)
        pred = pred * keep.view(-1, 1, 1, 1).float()
    return pred


def area_bin(values: pd.Series) -> pd.Series:
    bins = [-1e-12, 1e-8, 0.001, 0.005, 0.02, 0.08, 1.0]
    labels = ["empty", "tiny", "small", "medium", "large", "tail"]
    return pd.cut(values.astype(float), bins=bins, labels=labels, include_lowest=True)


def size_bin(frame: pd.DataFrame) -> pd.Series:
    width = frame["width"].astype(str)
    height = frame["height"].astype(str)
    return height + "x" + width


class EnsembleEvaluator:
    def __init__(
        self,
        manifest: pd.DataFrame,
        checkpoints: dict[str, Path],
        combos: dict[str, list[tuple[str, float]]],
        output_dir: Path,
        device: torch.device,
        cache_device: torch.device | None,
        batch_size: int,
    ):
        self.manifest = manifest
        self.checkpoints = checkpoints
        self.combos = combos
        self.output_dir = output_dir
        self.device = device
        self.cache_device = cache_device
        self.batch_size = int(batch_size)
        self.models = {}
        self.cfgs = {}
        self.channels_last = {}
        for label, checkpoint in checkpoints.items():
            model, cfg, _ckpt = load_roi_localizer_model(checkpoint, device)
            use_channels_last = bool(cfg.get("channels_last", False)) and device.type == "cuda"
            if use_channels_last:
                model = model.to(memory_format=torch.channels_last)
            self.models[label] = model.eval()
            self.cfgs[label] = cfg
            self.channels_last[label] = use_channels_last

    def make_datasets(self, split: str) -> dict[str, BaglsSegmentationDataset]:
        return {
            label: BaglsSegmentationDataset(self.manifest, cfg, split=split, cache_device=self.cache_device)
            for label, cfg in self.cfgs.items()
        }

    def iter_indices(self, n: int):
        for start in range(0, n, self.batch_size):
            yield np.arange(start, min(start + self.batch_size, n))

    @torch.inference_mode()
    def batch_probs(self, datasets: dict[str, BaglsSegmentationDataset], indices: np.ndarray) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        probs = {}
        target_size = None
        target_masks = None
        raw_images = None
        for pos, (label, dataset) in enumerate(datasets.items()):
            images, masks = dataset.get_batch(indices, device=self.device, augment=False, channels_last=self.channels_last[label])
            if pos == 0:
                target_size = masks.shape[-2:]
                target_masks = masks
                raw_images = images.detach().cpu()
            images = normalise_localizer_batch(images)
            if self.channels_last[label]:
                images = images.contiguous(memory_format=torch.channels_last)
            prob = torch.sigmoid(self.models[label](images))
            if target_size is not None and prob.shape[-2:] != target_size:
                prob = F.interpolate(prob, size=target_size, mode="bilinear", align_corners=False)
            probs[label] = prob
        assert target_masks is not None and raw_images is not None
        return probs, target_masks, raw_images

    def combo_prob(self, probs: dict[str, torch.Tensor], labels: list[tuple[str, float]]) -> torch.Tensor:
        total = float(sum(weight for _label, weight in labels))
        weighted = [probs[label] * float(weight / total) for label, weight in labels]
        return torch.stack(weighted, dim=0).sum(dim=0)

    @torch.inference_mode()
    def scan_thresholds(self, thresholds: list[float], min_areas: list[float]) -> dict[str, dict[str, float]]:
        datasets = self.make_datasets("val")
        n = len(next(iter(datasets.values())))
        stats = {
            combo: {
                (threshold, min_area): {"dice_sum": 0.0, "count": 0, "empty": 0, "empty_fp": 0}
                for threshold in thresholds
                for min_area in min_areas
            }
            for combo in self.combos
        }
        for indices in self.iter_indices(n):
            probs, masks, _raw = self.batch_probs(datasets, indices)
            gt_area = masks.mean(dim=tuple(range(1, masks.dim())))
            empty = gt_area <= 1e-8
            for combo_name, labels in self.combos.items():
                combo_probs = self.combo_prob(probs, labels)
                for threshold in thresholds:
                    for min_area in min_areas:
                        pred = apply_threshold(combo_probs, threshold, min_area)
                        values = dice_values(pred, masks)
                        pred_area = pred.mean(dim=tuple(range(1, pred.dim())))
                        empty_fp = empty & (pred_area > 0.001)
                        item = stats[combo_name][(threshold, min_area)]
                        item["dice_sum"] += float(values.sum().detach().cpu())
                        item["count"] += int(values.numel())
                        item["empty"] += int(empty.sum().detach().cpu())
                        item["empty_fp"] += int(empty_fp.sum().detach().cpu())
        selected = {}
        for combo_name, combo_stats in stats.items():
            rows = []
            for (threshold, min_area), item in combo_stats.items():
                row = {
                    "threshold": threshold,
                    "min_pred_area": min_area,
                    "val_mean_dice": item["dice_sum"] / max(item["count"], 1),
                    "val_empty_false_positive_rate": item["empty_fp"] / max(item["empty"], 1),
                    "val_empty_false_positive_count": item["empty_fp"],
                    "val_empty_count": item["empty"],
                }
                rows.append(row)
            scan = pd.DataFrame(rows).sort_values(
                ["val_mean_dice", "val_empty_false_positive_rate"],
                ascending=[False, True],
            )
            scan.insert(0, "combo", combo_name)
            scan.to_csv(self.output_dir / f"threshold_scan_val_{combo_name}.csv", index=False)
            selected[combo_name] = scan.iloc[0].to_dict()
        pd.concat(
            [pd.read_csv(self.output_dir / f"threshold_scan_val_{combo_name}.csv") for combo_name in self.combos],
            ignore_index=True,
        ).to_csv(self.output_dir / "threshold_scan_val.csv", index=False)
        write_json(self.output_dir / "selected_thresholds.json", selected)
        return selected

    @torch.inference_mode()
    def evaluate_selected(self, selected: dict[str, dict[str, float]], save_worst: int, splits: tuple[str, ...]) -> None:
        metric_rows = []
        worst = {combo: [] for combo in self.combos}
        for split in splits:
            datasets = self.make_datasets(split)
            first_dataset = next(iter(datasets.values()))
            n = len(first_dataset)
            stats = {
                combo: {
                    "dice_sum": 0.0,
                    "nonempty_dice_sum": 0.0,
                    "count": 0,
                    "nonempty_count": 0,
                    "empty": 0,
                    "empty_fp": 0,
                    "pred_area_sum": 0.0,
                    "gt_area_sum": 0.0,
                }
                for combo in self.combos
            }
            test_rows_for_summary = {combo: [] for combo in self.combos} if split == "test" else {}
            wrote_header = {combo: False for combo in self.combos}
            for indices in self.iter_indices(n):
                probs, masks, raw_images = self.batch_probs(datasets, indices)
                mask_np = (masks.detach().cpu().numpy() > 0.5).astype(np.uint8)
                chunk_rows = {combo: [] for combo in self.combos}
                for combo_name, labels in self.combos.items():
                    threshold = float(selected[combo_name]["threshold"])
                    min_area = float(selected[combo_name]["min_pred_area"])
                    combo_probs = self.combo_prob(probs, labels)
                    pred = apply_threshold(combo_probs, threshold, min_area)
                    dice_np = dice_values(pred, masks).detach().cpu().numpy()
                    pred_np = pred.detach().cpu().numpy().astype(np.uint8)
                    prob_np = combo_probs.detach().cpu().numpy()
                    for j, idx in enumerate(indices.tolist()):
                        row = first_dataset.frame.iloc[idx]
                        gt = mask_np[j, 0]
                        pr = pred_np[j, 0]
                        gt_area = float(gt.mean())
                        pred_area = float(pr.mean())
                        fp = int(((pr == 1) & (gt == 0)).sum())
                        fn = int(((pr == 0) & (gt == 1)).sum())
                        pred_row = (
                            {
                                "image_path": row.get("image_path", ""),
                                "mask_path": row.get("mask_path", ""),
                                "width": row.get("width", np.nan),
                                "height": row.get("height", np.nan),
                                "glottis_area_ratio": row.get("glottis_area_ratio", np.nan),
                                "reflection_label": row.get("reflection_label", np.nan),
                                "per_sample_dice": float(dice_np[j]),
                                "pred_area_ratio": pred_area,
                                "gt_area_ratio": gt_area,
                                "prob_area_mean": float(prob_np[j, 0].mean()),
                                "fp": fp,
                                "fn": fn,
                                "empty_gt": bool(gt_area <= 1e-8),
                                "empty_false_positive": bool(gt_area <= 1e-8 and pred_area > 0.001),
                                "threshold": threshold,
                                "min_pred_area": min_area,
                                "pred_bbox": mask_bbox_str(pr),
                                "gt_bbox": mask_bbox_str(gt),
                            }
                        )
                        chunk_rows[combo_name].append(pred_row)
                        item = stats[combo_name]
                        item["dice_sum"] += float(dice_np[j])
                        item["count"] += 1
                        if gt_area > 1e-8:
                            item["nonempty_dice_sum"] += float(dice_np[j])
                            item["nonempty_count"] += 1
                        else:
                            item["empty"] += 1
                            if pred_area > 0.001:
                                item["empty_fp"] += 1
                        item["pred_area_sum"] += pred_area
                        item["gt_area_sum"] += gt_area
                        if split == "test":
                            test_rows_for_summary[combo_name].append(pred_row)
                        if split == "test" and save_worst > 0:
                            image_np = np.clip(raw_images[j].permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
                            worst[combo_name].append((float(dice_np[j]), idx, image_np, gt.copy(), pr.copy()))
                for combo_name, rows in chunk_rows.items():
                    if not rows:
                        continue
                    path = self.output_dir / f"localizer_predictions_{combo_name}_{split}.csv"
                    pd.DataFrame(rows).to_csv(path, mode="a", header=not wrote_header[combo_name], index=False)
                    wrote_header[combo_name] = True
            for combo_name, item in stats.items():
                metric_rows.append(
                    {
                        "combo": combo_name,
                        "split": split,
                        "dice": item["dice_sum"] / max(item["count"], 1),
                        "samples": int(item["count"]),
                        "nonempty_mean_dice": item["nonempty_dice_sum"] / max(item["nonempty_count"], 1),
                        "empty_count": int(item["empty"]),
                        "empty_false_positive_count": int(item["empty_fp"]),
                        "empty_false_positive_rate": item["empty_fp"] / max(item["empty"], 1),
                        "pred_area_ratio_mean": item["pred_area_sum"] / max(item["count"], 1),
                        "gt_area_ratio_mean": item["gt_area_sum"] / max(item["count"], 1),
                        "threshold": float(selected[combo_name]["threshold"]),
                        "min_pred_area": float(selected[combo_name]["min_pred_area"]),
                    }
                )
                if split == "test":
                    summary = self.error_summary(pd.DataFrame(test_rows_for_summary[combo_name]))
                    summary.to_csv(self.output_dir / f"test_error_summary_{combo_name}.csv", index=False)
        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(self.output_dir / "metrics.csv", index=False)
        write_json(self.output_dir / "metrics.json", {"rows": metric_rows})
        self.write_combined_prediction_files()
        for combo_name, items in worst.items():
            overlay_dir = self.output_dir / f"worst_test_overlays_{combo_name}"
            overlay_dir.mkdir(parents=True, exist_ok=True)
            for rank, (value, idx, image_np, gt, pr) in enumerate(sorted(items, key=lambda item: item[0])[:save_worst], start=1):
                overlay_image(image_np, gt, pr).save(overlay_dir / f"{rank:03d}_idx{idx}_dice{value:.4f}.png")

    @torch.inference_mode()
    def evaluate_default_threshold(self, threshold: float = 0.5, min_pred_area: float = 0.0, splits: tuple[str, ...] = ("train", "val", "test")) -> None:
        metric_rows = []
        for split in splits:
            datasets = self.make_datasets(split)
            n = len(next(iter(datasets.values())))
            stats = {combo: {"dice_sum": 0.0, "count": 0, "empty": 0, "empty_fp": 0, "pred_area_sum": 0.0, "gt_area_sum": 0.0} for combo in self.combos}
            for indices in self.iter_indices(n):
                probs, masks, _raw_images = self.batch_probs(datasets, indices)
                gt_area = masks.mean(dim=tuple(range(1, masks.dim())))
                empty = gt_area <= 1e-8
                for combo_name, labels in self.combos.items():
                    combo_probs = self.combo_prob(probs, labels)
                    pred = apply_threshold(combo_probs, threshold, min_pred_area)
                    values = dice_values(pred, masks)
                    pred_area = pred.mean(dim=tuple(range(1, pred.dim())))
                    empty_fp = empty & (pred_area > 0.001)
                    item = stats[combo_name]
                    item["dice_sum"] += float(values.sum().detach().cpu())
                    item["count"] += int(values.numel())
                    item["empty"] += int(empty.sum().detach().cpu())
                    item["empty_fp"] += int(empty_fp.sum().detach().cpu())
                    item["pred_area_sum"] += float(pred_area.sum().detach().cpu())
                    item["gt_area_sum"] += float(gt_area.sum().detach().cpu())
            for combo_name, item in stats.items():
                metric_rows.append(
                    {
                        "combo": combo_name,
                        "split": split,
                        "dice": item["dice_sum"] / max(item["count"], 1),
                        "samples": int(item["count"]),
                        "nonempty_mean_dice": float("nan"),
                        "empty_count": int(item["empty"]),
                        "empty_false_positive_count": int(item["empty_fp"]),
                        "empty_false_positive_rate": item["empty_fp"] / max(item["empty"], 1),
                        "pred_area_ratio_mean": item["pred_area_sum"] / max(item["count"], 1),
                        "gt_area_ratio_mean": item["gt_area_sum"] / max(item["count"], 1),
                        "threshold": float(threshold),
                        "min_pred_area": float(min_pred_area),
                        "threshold_source": "default_0.5",
                    }
                )
        metrics_path = self.output_dir / "metrics.csv"
        if metrics_path.exists():
            metrics = pd.concat([pd.read_csv(metrics_path), pd.DataFrame(metric_rows)], ignore_index=True)
        else:
            metrics = pd.DataFrame(metric_rows)
        if "threshold_source" not in metrics.columns:
            metrics["threshold_source"] = "val_selected"
        metrics["threshold_source"] = metrics["threshold_source"].fillna("val_selected")
        metrics.to_csv(metrics_path, index=False)
        write_json(self.output_dir / "metrics.json", {"rows": json.loads(metrics.to_json(orient="records"))})

    def write_combined_prediction_files(self) -> None:
        for split in ("train", "val", "test"):
            parts = []
            for combo_name in self.combos:
                path = self.output_dir / f"localizer_predictions_{combo_name}_{split}.csv"
                if path.exists():
                    df = pd.read_csv(path)
                    df.insert(0, "combo", combo_name)
                    parts.append(df)
            if parts:
                pd.concat(parts, ignore_index=True).to_csv(self.output_dir / f"localizer_predictions_{split}.csv", index=False)

    @staticmethod
    def error_summary(df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame["area_bin"] = area_bin(frame["gt_area_ratio"])
        frame["size_bin"] = size_bin(frame)
        rows = []
        for group_name, keys in {
            "empty_gt": ["empty_gt"],
            "area_bin": ["area_bin"],
            "size_bin": ["size_bin"],
            "area_x_size": ["area_bin", "size_bin"],
        }.items():
            grouped = frame.groupby(keys, dropna=False, observed=False)
            for key, part in grouped:
                if not isinstance(key, tuple):
                    key = (key,)
                rows.append(
                    {
                        "group": group_name,
                        "key": "|".join(str(v) for v in key),
                        "samples": int(len(part)),
                        "mean_dice": float(part["per_sample_dice"].mean()),
                        "mean_fp": float(part["fp"].mean()),
                        "mean_fn": float(part["fn"].mean()),
                        "empty_false_positive_rate": float(part["empty_false_positive"].mean()) if "empty_false_positive" in part else float("nan"),
                        "gt_area_ratio_mean": float(part["gt_area_ratio"].mean()),
                        "pred_area_ratio_mean": float(part["pred_area_ratio"].mean()),
                    }
                )
        return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="name=/path/to/roi_localizer_best.pth")
    parser.add_argument("--combo", action="append", required=True, help="combo_name=checkpoint_name[:weight],checkpoint_name[:weight]")
    parser.add_argument("--output-dir", type=Path, default=ROI_RESULTS_DIR / "localizer_ensemble_eval")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--min-pred-areas", default="0,0.00025,0.0005,0.001,0.002,0.004")
    parser.add_argument("--save-worst-overlays", type=int, default=100)
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated splits to export after val threshold selection.")
    parser.add_argument("--only-default", action="store_true", help="Only append default 0.5-threshold metrics; skip val scan and prediction CSVs.")
    parser.add_argument("--skip-default", action="store_true", help="Skip the default 0.5-threshold metric pass.")
    args = parser.parse_args()

    checkpoints = dict(parse_named_path(value) for value in args.checkpoint)
    combos = dict(parse_combo(value) for value in args.combo)
    missing = sorted({label for labels in combos.values() for label, _weight in labels if label not in checkpoints})
    if missing:
        raise ValueError(f"Combo references unknown checkpoints: {missing}")
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    min_areas = [float(item) for item in args.min_pred_areas.split(",") if item.strip()]
    device = resolve_device(args.device)
    cache_device = resolve_device(args.cache_device)
    if cache_device.type == "cpu":
        cache_device = None
    output_dir = ensure_dir(args.output_dir)
    manifest = pd.read_csv(args.manifest)
    evaluator = EnsembleEvaluator(
        manifest=manifest,
        checkpoints=checkpoints,
        combos=combos,
        output_dir=output_dir,
        device=device,
        cache_device=cache_device,
        batch_size=int(args.batch_size),
    )
    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    if not args.only_default:
        selected = evaluator.scan_thresholds(thresholds=thresholds, min_areas=min_areas)
        evaluator.evaluate_selected(selected=selected, save_worst=int(args.save_worst_overlays), splits=splits)
    if not args.skip_default:
        evaluator.evaluate_default_threshold(threshold=0.5, min_pred_area=0.0, splits=splits)
    print(f"Ensemble evaluation written to {output_dir}")


if __name__ == "__main__":
    main()
