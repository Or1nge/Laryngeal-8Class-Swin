#!/usr/bin/env python3
"""Train the BAGLS glottis localizer used for ROI cropping."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim

from common import (
    ROI_RESULTS_DIR,
    BaglsSegmentationDataset,
    checkpoint_payload,
    copy_notice,
    default_localizer_config,
    ensure_dir,
    iter_batches,
    localizer_batch_metrics,
    localizer_loss,
    load_json_config,
    make_localizer_model,
    normalise_localizer_batch,
    resolve_device,
    save_run_provenance,
    set_seed,
    write_json,
)


def evaluate(model, dataset, cfg, device) -> dict[str, float]:
    model.eval()
    losses, dices = [], []
    nonempty_dices, area_maes, fp_ratios, fn_ratios = [], [], [], []
    empty_count = 0
    empty_fp_count = 0
    channels_last = bool(cfg.get("channels_last", False)) and device.type == "cuda"
    with torch.inference_mode():
        for images, masks in iter_batches(
            dataset,
            int(cfg.get("eval_batch_size", cfg["batch_size"])),
            shuffle=False,
            device=device,
            augment=False,
            channels_last=channels_last,
        ):
            images = normalise_localizer_batch(images)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            logits = model(images)
            loss = localizer_loss(logits, masks, cfg)
            metrics = localizer_batch_metrics(logits, masks, cfg)
            losses.append(float(loss.detach().cpu()))
            dices.append(metrics["dice"])
            if metrics["nonempty_dice"] == metrics["nonempty_dice"]:
                nonempty_dices.append(metrics["nonempty_dice"])
            area_maes.append(metrics["area_mae"])
            fp_ratios.append(metrics["fp_ratio"])
            fn_ratios.append(metrics["fn_ratio"])
            empty_count += int(metrics["empty_count"])
            empty_fp_count += int(metrics["empty_fp_count"])
    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "dice": float(sum(dices) / max(len(dices), 1)),
        "nonempty_dice": float(sum(nonempty_dices) / max(len(nonempty_dices), 1)),
        "empty_fp_rate": float(empty_fp_count / max(empty_count, 1)),
        "empty_count": int(empty_count),
        "empty_fp_count": int(empty_fp_count),
        "area_mae": float(sum(area_maes) / max(len(area_maes), 1)),
        "fp_ratio": float(sum(fp_ratios) / max(len(fp_ratios), 1)),
        "fn_ratio": float(sum(fn_ratios) / max(len(fn_ratios), 1)),
    }


def selection_score(row: dict[str, float], cfg: dict) -> tuple[float, bool]:
    metric = str(cfg.get("selection_metric", "val_dice")).lower()
    if metric == "val_loss":
        return -float(row["val_loss"]), False
    if metric == "val_loss_dice":
        loss_weight = float(cfg.get("selection_loss_weight", 0.25))
        return float(row["val_dice"]) - loss_weight * float(row["val_loss"]), True
    if metric in {"val_dice_empty_fp", "val_generalization"}:
        return (
            float(row["val_dice"])
            - float(cfg.get("selection_empty_fp_weight", 0.5)) * float(row.get("val_empty_fp_rate", 0.0))
            - float(cfg.get("selection_area_mae_weight", 0.0)) * float(row.get("val_area_mae", 0.0)),
            True,
        )
    return float(row["val_dice"]), True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "config_localizer.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-device", choices=["auto", "cuda", "cpu"], default=None)
    args = parser.parse_args()

    cfg = load_json_config(args.config, default_localizer_config())
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.cache_device is not None:
        cfg["cache_device"] = args.cache_device
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device("auto")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision(str(cfg.get("float32_matmul_precision", "high")))
    cache_device = resolve_device(cfg.get("cache_device", "auto"))
    if cache_device.type == "cpu":
        cache_device = None
    manifest = pd.read_csv(args.manifest)
    if args.limit:
        split_count = max(int(manifest["split"].nunique()), 1) if "split" in manifest else 1
        per_split = max(1, int(args.limit) // split_count)
        manifest = manifest.groupby("split", group_keys=False).head(per_split).copy()

    train_ds = BaglsSegmentationDataset(manifest, cfg, split="train", cache_device=cache_device)
    val_ds = BaglsSegmentationDataset(manifest, cfg, split="val", cache_device=cache_device)
    if len(val_ds) == 0:
        val_ds = train_ds
    model = make_localizer_model(cfg).to(device)
    channels_last = bool(cfg.get("channels_last", False)) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if bool(cfg.get("compile_model", False)) and device.type == "cuda":
        model = torch.compile(model, mode=str(cfg.get("compile_mode", "reduce-overhead")))
    cfg["model_impl"] = str(getattr(model, "model_impl", cfg.get("model_arch", "unknown")))
    cfg["pretrained_source"] = str(getattr(model, "pretrained_source", "none"))

    if args.output_dir:
        run_dir = ensure_dir(args.output_dir)
    else:
        run_dir = ensure_dir(ROI_RESULTS_DIR / f"localizer_upgrade_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    write_json(run_dir / "config_effective.json", cfg)
    save_run_provenance(run_dir, cfg, [sys.executable, *sys.argv])
    copy_notice(run_dir)

    optimizer = optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler_name = str(cfg.get("scheduler", "cosine")).lower()
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(cfg["epochs"])),
            eta_min=float(cfg.get("min_learning_rate", 1e-6)),
        )
    elif scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(cfg.get("scheduler_factor", 0.5)),
            patience=int(cfg.get("scheduler_patience", 3)),
        )
    amp_enabled = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_score = -float("inf")
    best_row: dict[str, float] | None = None
    patience = int(cfg.get("early_stopping_patience", 8))
    min_delta = float(cfg.get("early_stopping_min_delta", 0.001))
    grad_clip = float(cfg.get("grad_clip_norm", 0.0))
    wait = 0
    history = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_area_mae = 0.0
        empty_count = 0
        empty_fp_count = 0
        steps = 0
        for images, masks in iter_batches(
            train_ds,
            int(cfg["batch_size"]),
            shuffle=True,
            device=device,
            augment=True,
            channels_last=channels_last,
        ):
            images = normalise_localizer_batch(images)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = localizer_loss(logits, masks, cfg)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            batch_metrics = localizer_batch_metrics(logits.detach(), masks, cfg)
            total_loss += float(loss.detach().cpu())
            total_dice += float(batch_metrics["dice"])
            total_area_mae += float(batch_metrics["area_mae"])
            empty_count += int(batch_metrics["empty_count"])
            empty_fp_count += int(batch_metrics["empty_fp_count"])
            steps += 1
        val = evaluate(model, val_ds, cfg, device)
        epoch_wall_sec = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(steps, 1),
            "train_dice": total_dice / max(steps, 1),
            "train_area_mae": total_area_mae / max(steps, 1),
            "train_empty_fp_rate": empty_fp_count / max(empty_count, 1),
            "train_empty_count": empty_count,
            "train_empty_fp_count": empty_fp_count,
            "val_loss": val["loss"],
            "val_dice": val["dice"],
            "val_nonempty_dice": val["nonempty_dice"],
            "val_empty_fp_rate": val["empty_fp_rate"],
            "val_empty_count": val["empty_count"],
            "val_empty_fp_count": val["empty_fp_count"],
            "val_area_mae": val["area_mae"],
            "val_fp_ratio": val["fp_ratio"],
            "val_fn_ratio": val["fn_ratio"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_wall_sec": epoch_wall_sec,
            "train_images_per_sec": len(train_ds) / max(epoch_wall_sec, 1e-9),
            "train_val_images_per_sec": (len(train_ds) + len(val_ds)) / max(epoch_wall_sec, 1e-9),
        }
        history.append(row)
        print(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        score, _higher_is_better = selection_score(row, cfg)
        improved = score > best_score + min_delta
        if improved:
            best_score = score
            best_row = row
            wait = 0
            torch.save(
                checkpoint_payload(
                    model,
                    cfg,
                    role="localizer",
                    extra={"best_epoch": epoch, "best_selection_score": best_score, "best_metrics": best_row},
                ),
                run_dir / "roi_localizer_best.pth",
            )
        else:
            wait += 1
            if wait >= patience:
                break
        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(score)
            else:
                scheduler.step()
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    write_json(
        run_dir / "metrics.json",
        {
            "selection_metric": str(cfg.get("selection_metric", "val_dice")),
            "best_selection_score": best_score,
            "best_metrics": best_row or {},
            "history_last": history[-1] if history else {},
        },
    )
    print(f"Best localizer: {run_dir / 'roi_localizer_best.pth'}")


if __name__ == "__main__":
    main()
