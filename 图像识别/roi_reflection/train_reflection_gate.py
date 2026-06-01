#!/usr/bin/env python3
"""Train the BAGLS ROI validity/reflection dual-head gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from common import (
    ROI_RESULTS_DIR,
    ReflectionDataset,
    binary_metrics,
    checkpoint_payload,
    choose_threshold,
    copy_notice,
    default_reflection_config,
    ensure_dir,
    iter_batches,
    load_json_config,
    make_reflection_model,
    normalise_reflection_batch,
    resolve_device,
    save_run_provenance,
    set_seed,
    threshold_table,
    timestamped_run_dir,
    write_json,
)


def focal_bce(logits: torch.Tensor, targets: torch.Tensor, gamma: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, prob, 1.0 - prob)
    return (bce * (1.0 - pt).pow(float(gamma))).mean()


@torch.inference_mode()
def collect_outputs(model, dataset, cfg, device) -> dict[str, np.ndarray]:
    model.eval()
    y_valid, y_reflect, p_valid, p_reflect = [], [], [], []
    for images, valid, reflect in iter_batches(dataset, int(cfg.get("eval_batch_size", cfg["batch_size"])), shuffle=False):
        images = normalise_reflection_batch(images.to(device, non_blocking=True))
        out = model(images)
        y_valid.extend(valid.cpu().numpy())
        y_reflect.extend(reflect.cpu().numpy())
        p_valid.extend(out["roi_valid_prob"].detach().cpu().numpy())
        p_reflect.extend(out["reflect_prob"].detach().cpu().numpy())
    return {
        "y_valid": np.asarray(y_valid, dtype=np.float32),
        "y_reflect": np.asarray(y_reflect, dtype=np.float32),
        "p_valid": np.asarray(p_valid, dtype=np.float32),
        "p_reflect": np.asarray(p_reflect, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--localizer-checkpoint", type=Path, default=None, help="Recorded for provenance; manifest already contains BAGLS ROI boxes.")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "config_reflection.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-device", choices=["auto", "cuda", "cpu"], default=None)
    args = parser.parse_args()

    cfg = load_json_config(args.config, default_reflection_config())
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.cache_device is not None:
        cfg["cache_device"] = args.cache_device
    cfg["localizer_checkpoint"] = str(args.localizer_checkpoint) if args.localizer_checkpoint else ""
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device("auto")
    cache_device = resolve_device(cfg.get("cache_device", "auto"))
    if cache_device.type == "cpu":
        cache_device = None
    manifest = pd.read_csv(args.manifest)
    if args.limit:
        manifest = manifest.head(int(args.limit)).copy()

    run_dir = ensure_dir(args.output_dir) if args.output_dir else timestamped_run_dir(ROI_RESULTS_DIR, "reflection_gate")
    write_json(run_dir / "config_effective.json", cfg)
    save_run_provenance(run_dir, cfg, [sys.executable, *sys.argv])
    copy_notice(run_dir)

    train_ds = ReflectionDataset(manifest, cfg, split="train", cache_device=cache_device)
    val_ds = ReflectionDataset(manifest, cfg, split="val", cache_device=cache_device)
    test_ds = ReflectionDataset(manifest, cfg, split="test", cache_device=cache_device)
    if len(val_ds) == 0:
        val_ds = train_ds
    model = make_reflection_model(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    gamma = float(cfg.get("focal_gamma", 2.0))

    best_score = -1.0
    patience = int(cfg.get("early_stopping_patience", 8))
    wait = 0
    history = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        losses = []
        for images, valid, reflect in iter_batches(train_ds, int(cfg["batch_size"]), shuffle=True):
            images = normalise_reflection_batch(images.to(device, non_blocking=True))
            valid = valid.to(device, non_blocking=True)
            reflect = reflect.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            valid_loss = F.binary_cross_entropy_with_logits(out["valid_logits"], valid)
            reflect_loss = focal_bce(out["reflect_logits"], reflect, gamma=gamma)
            loss = float(cfg.get("valid_loss_weight", 0.5)) * valid_loss + float(cfg.get("reflect_loss_weight", 1.0)) * reflect_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_out = collect_outputs(model, val_ds, cfg, device)
        valid_m = binary_metrics(val_out["y_valid"], val_out["p_valid"], threshold=float(cfg.get("valid_threshold", 0.55)), prefix="valid_")
        reflect_m = binary_metrics(val_out["y_reflect"], val_out["p_reflect"], threshold=float(cfg.get("reflect_threshold", 0.65)), prefix="reflect_")
        score = float(reflect_m.get("reflect_f1", 0.0) or 0.0) + 0.25 * float(valid_m.get("valid_f1", 0.0) or 0.0)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"), **valid_m, **reflect_m}
        history.append(row)
        print(row)
        if score > best_score + 1e-5:
            best_score = score
            wait = 0
            torch.save(
                checkpoint_payload(model, cfg, role="reflection", extra={"best_epoch": epoch, "best_score": best_score}),
                run_dir / "roi_reflection_best.pth",
            )
        else:
            wait += 1
            if wait >= patience:
                break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    model_ckpt = torch.load(run_dir / "roi_reflection_best.pth", map_location=device)
    model.load_state_dict(model_ckpt["model_state_dict"], strict=True)
    rows = []
    threshold_payload = {}
    for split, ds in (("val", val_ds), ("test", test_ds if len(test_ds) else val_ds)):
        outputs = collect_outputs(model, ds, cfg, device)
        valid_choice = choose_threshold(outputs["y_valid"], outputs["p_valid"], prefix="valid_")
        reflect_choice = choose_threshold(outputs["y_reflect"], outputs["p_reflect"], prefix="reflect_")
        threshold_table(outputs["y_reflect"], outputs["p_reflect"], prefix="reflect_").to_csv(run_dir / f"threshold_metrics_reflect_{split}.csv", index=False)
        threshold_payload[split] = {"valid": valid_choice, "reflect": reflect_choice}
        rows.append({"split": split, **binary_metrics(outputs["y_valid"], outputs["p_valid"], threshold=float(cfg["valid_threshold"]), prefix="valid_")})
        rows.append({"split": split, **binary_metrics(outputs["y_reflect"], outputs["p_reflect"], threshold=float(cfg["reflect_threshold"]), prefix="reflect_")})
        pred = pd.DataFrame(outputs)
        pred.to_csv(run_dir / f"predictions_{split}.csv", index=False)
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    write_json(run_dir / "recommended_threshold.json", threshold_payload)
    print(f"Best reflection gate: {run_dir / 'roi_reflection_best.pth'}")


if __name__ == "__main__":
    main()
