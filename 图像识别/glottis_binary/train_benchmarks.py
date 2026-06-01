#!/usr/bin/env python3
"""Train glottis/non-glottis binary gate benchmarks."""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SPLIT_PATH,
    MODEL_REGISTRY,
    FeatureBinaryClassifier,
    GPUAugment,
    SupConLoss,
    WarmupCosineScheduler,
    archive_source_files,
    binary_metrics,
    build_binary_split,
    checkpoint_payload,
    choose_gate_threshold,
    collect_outputs,
    default_train_config,
    git_provenance,
    gpu_normalise,
    iter_eval_batches,
    iter_train_batches,
    load_split_dataframe,
    merge_config,
    preload_split_cache,
    save_confusion_matrix_png,
    save_predictions,
    save_roc_pr_png,
    seed_everything,
    setup_device,
    threshold_table,
    write_json,
)

sys.dont_write_bytecode = True


def parse_args() -> argparse.Namespace:
    defaults = default_train_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--build-split", action="store_true")
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet50", "vit_base", "swin_base", "supcon_swin_base"],
        help=f"Model keys. Available: {sorted(MODEL_REGISTRY)}",
    )
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--patience", type=int, default=defaults["patience"])
    parser.add_argument("--supcon-epochs", type=int, default=defaults["supcon_epochs"])
    parser.add_argument("--supcon-patience", type=int, default=defaults["supcon_patience"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=defaults["eval_batch_size"])
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--supcon-learning-rate", type=float, default=defaults["supcon_learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--label-smoothing", type=float, default=defaults["label_smoothing"])
    parser.add_argument("--drop-rate", type=float, default=defaults["drop_rate"])
    parser.add_argument("--drop-path-rate", type=float, default=defaults["drop_path_rate"])
    parser.add_argument("--sampler-balance-alpha", type=float, default=defaults["sampler_balance_alpha"])
    parser.add_argument("--image-size", type=int, default=defaults["image_size"])
    parser.add_argument("--resize-size", type=int, default=defaults["resize_size"])
    parser.add_argument("--cache-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--quick-smoke", action="store_true", help="Run tiny epochs for script validation.")
    return parser.parse_args()


def train_ce_epoch(
    model: FeatureBinaryClassifier,
    train_cache,
    optimizer,
    criterion,
    scaler,
    gpu_aug,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    amp_context = torch.amp.autocast(device_type=device.type) if device.type == "cuda" else nullcontext()
    for images, labels in iter_train_batches(
        train_cache,
        batch_size=int(cfg["batch_size"]),
        balance_alpha=float(cfg["sampler_balance_alpha"]),
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images = gpu_aug(images)
        optimizer.zero_grad(set_to_none=True)
        with amp_context:
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        if cfg.get("grad_clip_norm", 0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.numel()
        total_loss += float(loss.detach().item()) * batch_size
        total += batch_size
        correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
    return {"train_loss": total_loss / max(total, 1), "train_sampled_acc": correct / max(total, 1)}


@torch.inference_mode()
def evaluate_loss_and_metrics(
    model: FeatureBinaryClassifier,
    cache,
    criterion,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    outputs = collect_outputs(model, cache, int(cfg["eval_batch_size"]), device)
    y_true = outputs["y_true"]
    probs = outputs["probs_glottis"]

    model.eval()
    total_loss = 0.0
    total = 0
    amp_context = torch.amp.autocast(device_type=device.type) if device.type == "cuda" else nullcontext()
    for images, labels in iter_eval_batches(cache, int(cfg["eval_batch_size"])):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images = gpu_normalise(images)
        with amp_context:
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += float(loss.detach().item()) * labels.numel()
        total += labels.numel()
    metrics = binary_metrics(y_true, probs, threshold=0.5)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


def train_supcon_phase(
    model: FeatureBinaryClassifier,
    train_cache,
    cfg: dict[str, Any],
    device: torch.device,
    run_dir: Path,
) -> int:
    criterion = SupConLoss(float(cfg["supcon_temperature"]))
    optimizer = torch.optim.AdamW(
        list(model.backbone.parameters()) + list(model.projector.parameters()),
        lr=float(cfg["supcon_learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(cfg["supcon_epochs"]), 1),
        eta_min=float(cfg["supcon_learning_rate"]) * 0.1,
    )
    scaler = torch.amp.GradScaler(device.type)
    gpu_aug = GPUAugment(cfg).to(device)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    amp_context = torch.amp.autocast(device_type=device.type) if device.type == "cuda" else nullcontext()
    checkpoint_path = run_dir / "supcon_checkpoint.pth"

    for epoch in range(1, int(cfg["supcon_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for images, labels in iter_train_batches(
            train_cache,
            batch_size=int(cfg["batch_size"]),
            balance_alpha=float(cfg["sampler_balance_alpha"]),
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            x1 = gpu_aug(images)
            x2 = gpu_aug(images)
            all_images = torch.cat([x1, x2], dim=0)
            all_labels = torch.cat([labels, labels], dim=0)
            optimizer.zero_grad(set_to_none=True)
            with amp_context:
                features = model.project(all_images)
                loss = criterion(features, all_labels)
            scaler.scale(loss).backward()
            if cfg.get("grad_clip_norm", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.backbone.parameters()) + list(model.projector.parameters()),
                    float(cfg["grad_clip_norm"]),
                )
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().item()) * labels.numel()
            total += labels.numel()

        train_loss = total_loss / max(total, 1)
        scheduler.step()
        improved = train_loss < best_loss - float(cfg.get("early_stopping_min_delta", 0.0005))
        if improved:
            best_loss = train_loss
            best_epoch = epoch
            stale = 0
            torch.save({"state_dict": model.state_dict(), "cfg": cfg}, checkpoint_path)
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "train_supcon_loss": train_loss,
                "best_train_supcon_loss": best_loss,
                "improved": improved,
            }
        )
        pd.DataFrame(history).to_csv(run_dir / "supcon_history.csv", index=False)
        print(f"    SupCon epoch {epoch:02d}: loss={train_loss:.5f}{' *' if improved else ''}")
        if stale >= int(cfg["supcon_patience"]):
            print(f"    SupCon early stop at epoch {epoch}; best epoch {best_epoch}.")
            break

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
    return best_epoch


def finalise_run_outputs(
    model: FeatureBinaryClassifier,
    model_key: str,
    model_info: dict[str, Any],
    caches,
    cfg: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    best_epoch: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    outputs_by_split = {
        split_name: collect_outputs(model, caches[split_name], int(cfg["eval_batch_size"]), device)
        for split_name in ("train", "val", "test")
    }
    val_thresholds = threshold_table(outputs_by_split["val"]["y_true"], outputs_by_split["val"]["probs_glottis"])
    threshold_info = choose_gate_threshold(
        val_thresholds,
        min_specificity=float(cfg["recommended_min_non_glottis_specificity"]),
        min_glottis_recall=float(cfg["recommended_min_glottis_recall"]),
    )
    threshold = float(threshold_info["threshold"])
    test_thresholds = threshold_table(outputs_by_split["test"]["y_true"], outputs_by_split["test"]["probs_glottis"])
    val_thresholds.to_csv(run_dir / "threshold_metrics_val.csv", index=False)
    test_thresholds.to_csv(run_dir / "threshold_metrics_test.csv", index=False)

    rows = []
    for split_name, outputs in outputs_by_split.items():
        default_metrics = binary_metrics(outputs["y_true"], outputs["probs_glottis"], threshold=0.5)
        gate_metrics = binary_metrics(outputs["y_true"], outputs["probs_glottis"], threshold=threshold)
        default_metrics.update(
            {
                "split": split_name,
                "threshold_mode": "default_0.5",
                "model_key": model_key,
                "timm_model": model_info["timm_name"],
                "best_epoch": best_epoch,
            }
        )
        gate_metrics.update(
            {
                "split": split_name,
                "threshold_mode": "recommended_gate",
                "model_key": model_key,
                "timm_model": model_info["timm_name"],
                "best_epoch": best_epoch,
            }
        )
        rows.extend([default_metrics, gate_metrics])
        save_predictions(split_name, outputs, caches[split_name], threshold, run_dir)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)

    test_outputs = outputs_by_split["test"]
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(
        test_outputs["y_true"],
        (test_outputs["probs_glottis"] >= threshold).astype(int),
        labels=[0, 1],
    )
    pd.DataFrame(
        cm,
        index=["true_non_glottis", "true_glottis"],
        columns=["pred_non_glottis", "pred_glottis"],
    ).to_csv(run_dir / "confusion_matrix_test.csv")
    save_confusion_matrix_png(
        test_outputs["y_true"],
        test_outputs["probs_glottis"],
        threshold,
        run_dir / "confusion_matrix_test.png",
    )
    save_roc_pr_png(test_outputs["y_true"], test_outputs["probs_glottis"], run_dir / "roc_pr_test.png")

    threshold_info["test_metrics_at_threshold"] = binary_metrics(
        test_outputs["y_true"], test_outputs["probs_glottis"], threshold=threshold
    )
    write_json(run_dir / "recommended_threshold.json", threshold_info)

    torch.save(
        checkpoint_payload(model, model_key, model_info, cfg, best_epoch, threshold=threshold),
        run_dir / "best_model.pth",
    )
    write_json(
        run_dir / "provenance.json",
        {
            "model_key": model_key,
            "model_info": model_info,
            "cfg": cfg,
            "git": git_provenance(),
            "command": " ".join(sys.argv),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "pretrained_source": model.pretrained_source,
        },
    )
    (run_dir / "run_command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    return {
        "model_key": model_key,
        "timm_model": model_info["timm_name"],
        "method": model_info["method"],
        "pretrained_source": model.pretrained_source,
        "best_epoch": best_epoch,
        "recommended_threshold": threshold,
        **{
            f"test_{key}": value
            for key, value in threshold_info["test_metrics_at_threshold"].items()
            if key != "threshold"
        },
    }


def train_one_model(
    model_key: str,
    model_info: dict[str, Any],
    caches,
    cfg: dict[str, Any],
    device: torch.device,
    run_root: Path,
    pretrained: bool,
    resume_existing: bool,
) -> dict[str, Any]:
    run_dir = run_root / model_key
    if resume_existing and (run_dir / "metrics.csv").exists():
        print(f"Skipping existing completed model: {model_key}")
        df = pd.read_csv(run_dir / "metrics.csv")
        test = df[(df["split"] == "test") & (df["threshold_mode"] == "recommended_gate")].iloc[0]
        threshold = (run_dir / "recommended_threshold.json").read_text(encoding="utf-8")
        threshold = float(__import__("json").loads(threshold)["threshold"])
        return {
            "model_key": model_key,
            "timm_model": model_info["timm_name"],
            "method": model_info["method"],
            "best_epoch": int(test["best_epoch"]),
            "recommended_threshold": threshold,
            **{
                f"test_{col}": test[col]
                for col in test.index
                if col not in {"split", "model_key", "timm_model", "threshold_mode"}
            },
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    model = FeatureBinaryClassifier(
        model_name=model_info["timm_name"],
        pretrained=pretrained,
        drop_rate=float(cfg["drop_rate"]),
        drop_path_rate=float(cfg["drop_path_rate"]),
        projection_dim=int(cfg["supcon_projection_dim"]),
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\n=== {model_key}: {model_info['description']} ===\n"
        f"    timm={model_info['timm_name']} pretrained_source={model.pretrained_source}\n"
        f"    parameters trainable/total={trainable:,}/{params:,}"
    )
    write_json(run_dir / "config_effective.json", cfg)

    if model_info["method"] == "supcon_ce":
        train_supcon_phase(model, caches["train"], cfg, device, run_dir)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["label_smoothing"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        fused=device.type == "cuda",
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=max(min(3, int(cfg["epochs"]) // 4), 1),
        total_epochs=max(int(cfg["epochs"]), 1),
        warmup_lr=float(cfg["learning_rate"]) * 0.05,
        min_lr=float(cfg["learning_rate"]) * 0.1,
    )
    scaler = torch.amp.GradScaler(device.type)
    gpu_aug = GPUAugment(cfg).to(device)

    best_score = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    best_path = run_dir / "ce_best_state.pth"
    started = time.time()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_stats = train_ce_epoch(model, caches["train"], optimizer, criterion, scaler, gpu_aug, cfg, device)
        val = evaluate_loss_and_metrics(model, caches["val"], criterion, cfg, device)
        scheduler.step()
        gate_score = (
            0.40 * val["balanced_accuracy"]
            + 0.40 * val["specificity_non_glottis"]
            + 0.20 * (0.0 if np.isnan(val["auroc"]) else val["auroc"])
        )
        improved = gate_score > best_score + float(cfg["early_stopping_min_delta"])
        if improved:
            best_score = gate_score
            best_epoch = epoch
            stale = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "score": best_score}, best_path)
        else:
            stale += 1
        row = {
            "epoch": epoch,
            "lr": lr,
            **train_stats,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_recall_glottis": val["recall_glottis_sensitivity"],
            "val_specificity_non_glottis": val["specificity_non_glottis"],
            "val_auroc": val["auroc"],
            "val_gate_score": gate_score,
            "best_val_gate_score": best_score,
            "improved": improved,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(
            f"    CE epoch {epoch:02d}: train_acc={train_stats['train_sampled_acc']:.4f} "
            f"val_acc={val['accuracy']:.4f} val_spec={val['specificity_non_glottis']:.4f} "
            f"val_rec={val['recall_glottis_sensitivity']:.4f} val_auc={val['auroc']:.4f}"
            f"{' *' if improved else ''}"
        )
        if stale >= int(cfg["patience"]):
            print(f"    CE early stop at epoch {epoch}; best epoch {best_epoch}.")
            break

    if best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"], strict=True)
    return finalise_run_outputs(model, model_key, model_info, caches, cfg, device, run_dir, best_epoch, history)


def main() -> None:
    args = parse_args()
    cfg = merge_config(
        default_train_config(),
        {
            "seed": args.seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "supcon_epochs": args.supcon_epochs,
            "supcon_patience": args.supcon_patience,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "supcon_learning_rate": args.supcon_learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "drop_rate": args.drop_rate,
            "drop_path_rate": args.drop_path_rate,
            "sampler_balance_alpha": args.sampler_balance_alpha,
            "image_size": args.image_size,
            "resize_size": args.resize_size,
        },
    )
    if args.quick_smoke:
        cfg.update({"epochs": 1, "supcon_epochs": 1, "patience": 1, "supcon_patience": 1})
    seed_everything(int(cfg["seed"]))

    if args.build_split or not args.split.exists():
        build_binary_split(
            dataset_root=args.dataset_root,
            output_path=args.split,
            manifest_path=args.manifest,
            seed=int(cfg["seed"]),
            force=args.force_split or not args.split.exists(),
        )

    df = load_split_dataframe(args.split)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    archive_source_files(run_root)
    write_json(run_root / "run_config.json", cfg)
    write_json(run_root / "split_summary.json", __import__("json").loads(args.split.read_text(encoding="utf-8")))
    (run_root / "run_command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    device = setup_device()
    print(f"Device: {device}")
    print(f"Run root: {run_root}")
    print("Preloading image tensors...")
    caches = preload_split_cache(df, cfg, device=device, cache_device=args.cache_device)

    summary_rows = []
    for model_key in args.models:
        if model_key not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model '{model_key}'. Available: {sorted(MODEL_REGISTRY)}")
        try:
            row = train_one_model(
                model_key=model_key,
                model_info=MODEL_REGISTRY[model_key],
                caches=caches,
                cfg=cfg,
                device=device,
                run_root=run_root,
                pretrained=not args.no_pretrained,
                resume_existing=args.resume_existing,
            )
            summary_rows.append(row)
            pd.DataFrame(summary_rows).to_csv(run_root / "benchmark_summary.csv", index=False)
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_root / "benchmark_summary.csv", index=False)
    if not summary.empty:
        sort_cols = ["test_specificity_non_glottis", "test_accuracy", "test_recall_glottis_sensitivity"]
        print("\nFinal benchmark summary:")
        print(summary.sort_values(sort_cols, ascending=False).to_string(index=False))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()
