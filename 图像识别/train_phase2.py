"""Phase 2: Cross-Entropy Fine-tuning.

Usage:
    python train_phase2.py [--config CONFIG_PATH]

Loads Phase 1 checkpoint (if configured) and fine-tunes with cross-entropy loss.

Outputs:
    <workspace>/Results/<worktree>/best_model.pth       — best model weights
    <workspace>/Results/<worktree>/history.csv           — per-epoch metrics
    <workspace>/Results/<worktree>/metrics.csv           — final classification report
    <workspace>/Results/<worktree>/training_curves.png   — training curves
    <workspace>/Results/<worktree>/gradcam_maps.png      — GradCAM attention maps
    <workspace>/Results/<worktree>/logs_phase2/          — TensorBoard logs
"""

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from shared import (
    BASE_DIR, RESULTS_ROOT, RESULTS_DIR, WORKTREE_NAME,
    BEST_MODEL_PATH, PHASE2_BEST_MODEL_PATH, PHASE2_FINAL_METRICS_PATH,
    PHASE1_CHECKPOINT_PATH, PHASE1_HISTORY_PATH,
    init_label_mapping, seed_everything, setup_device, load_config,
    discover_images, load_dataset_split, preload_image_cache, print_data_summary,
    HierarchicalImageClassifier, LABEL_DICT,
    create_loaders, build_balanced_sampler,
    GPUAugment,
    build_optimizer_param_groups, WarmupCosineScheduler,
    train_one_epoch, evaluate, evaluate_hierarchical,
    save_training_curves, save_metrics_csv, generate_attention_maps,
    load_history_from_tensorboard,
    HISTORY_CSV_PATH,
    AsyncCheckpointSaver, create_classification_metrics,
    resolve_project_path,
)


def _git_value(args, default="unknown"):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=BASE_DIR,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        value = result.stdout.strip()
        return value or default
    except (OSError, subprocess.CalledProcessError):
        return default


def _git_provenance():
    branch = _git_value(["branch", "--show-current"], default="")
    if not branch:
        branch = _git_value(["rev-parse", "--abbrev-ref", "HEAD"], default="detached")
    commit = _git_value(["rev-parse", "--short", "HEAD"], default="unknown")
    full_commit = _git_value(["rev-parse", "HEAD"], default="unknown")
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=BASE_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.strip()
    except OSError:
        dirty = ""
    return {
        "branch": branch or "detached",
        "commit": commit,
        "full_commit": full_commit,
        "dirty": bool(dirty),
    }


def _safe_name(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    value = value.strip(".-_")
    return value or "unknown"


def _unique_dir(path):
    path = Path(path)
    if not path.exists():
        return path
    for idx in range(2, 1000):
        candidate = path.with_name(f"{path.name}_{idx}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find a free archive path for {path}")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _hier_metric_summary(metrics):
    return {
        "hier_acc": metrics.get("hier_acc"),
        "voc_f1": metrics.get("voc_f1"),
    }


def write_final_metrics(path, cfg, final_train, final_val, final_test, hier_val, hier_test, best_epoch):
    payload = {
        "best_epoch": best_epoch,
        "metrics": {
            "train": final_train,
            "val": final_val,
            "test": final_test,
            "hierarchical_val": _hier_metric_summary(hier_val),
            "hierarchical_test": _hier_metric_summary(hier_test),
        },
        "config": cfg,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)
    return payload


def archive_finished_run(cfg, final_train, final_val, final_test, hier_val, hier_test, best_epoch):
    if str(os.environ.get("LARYNX_ARCHIVE_RUNS", "1")).lower() in {"0", "false", "no"}:
        print("Run archive disabled by LARYNX_ARCHIVE_RUNS=0.")
        return None

    provenance = _git_provenance()
    archive_root = Path(os.environ.get("LARYNX_RUNS_DIR", os.path.join(RESULTS_ROOT, "runs")))
    archive_root.mkdir(parents=True, exist_ok=True)

    finished_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = provenance["commit"] + ("-dirty" if provenance["dirty"] else "")
    folder_name = (
        f"{finished_at}_{_safe_name(provenance['branch'])}_{_safe_name(version)}_"
        f"testacc{final_test['acc']:.4f}_testauc{final_test['auc']:.4f}"
    )
    archive_dir = _unique_dir(archive_root / folder_name)
    shutil.copytree(RESULTS_DIR, archive_dir)

    metadata = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "source_results_dir": os.path.abspath(RESULTS_DIR),
        "archive_dir": str(archive_dir),
        "worktree_name": WORKTREE_NAME,
        "git": provenance,
        "command": [sys.executable, *sys.argv],
        "best_epoch": best_epoch,
        "metrics": {
            "train": final_train,
            "val": final_val,
            "test": final_test,
            "hierarchical_val": _hier_metric_summary(hier_val),
            "hierarchical_test": _hier_metric_summary(hier_test),
        },
        "config": cfg,
    }
    with open(archive_dir / "run_metadata.json", "w") as f:
        json.dump(_json_safe(metadata), f, indent=2, ensure_ascii=False)

    return archive_dir


def main():
    parser = argparse.ArgumentParser(description="Phase 2: CE Fine-tuning")
    parser.add_argument("--config", type=str, default=os.path.join(BASE_DIR, "config_phase2.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    init_label_mapping(cfg)

    seed = cfg["seed"]
    seed_everything(seed)
    device = setup_device()

    print(f"Using device: {device}")
    print(f"Config: {json.dumps(cfg, indent=2)}")
    print("=" * 80)

    df = discover_images()
    train_df, val_df, test_df = load_dataset_split(df)
    print_data_summary(train_df, val_df, test_df)

    image_cache = preload_image_cache(train_df, val_df, test_df, cfg=cfg, device=device)

    print("Building Phase 2 (CE) balanced sampler:")
    ce_sampler = build_balanced_sampler(
        train_df,
        hierarchical=False,
        balance_alpha=cfg.get("sampler_balance_alpha", 1.0),
    )
    loaders = create_loaders(train_df, val_df, test_df, cfg, image_cache=image_cache,
                             train_sampler=ce_sampler)
    print("Class weights: disabled (using sampler-only balancing).")

    num_classes = len(LABEL_DICT)
    model = HierarchicalImageClassifier(num_classes=num_classes, cfg=cfg).to(device)

    # ── Load Phase 1 checkpoint ──────────────────────────────────────────
    phase1_ckpt = cfg.get("phase1_checkpoint", "")
    if phase1_ckpt and not os.path.isabs(phase1_ckpt):
        phase1_ckpt = resolve_project_path(phase1_ckpt)

    if not phase1_ckpt:
        phase1_ckpt = PHASE1_CHECKPOINT_PATH

    if os.path.exists(phase1_ckpt):
        state_dict = torch.load(phase1_ckpt, map_location=device)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load Phase 1 checkpoint at {phase1_ckpt}. "
                f"The configured class mapping has {num_classes} classes: {list(LABEL_DICT.keys())}. "
                "If this checkpoint came from a different class mapping, such as the old 4-class "
                "project or a previous 9-class run before exclusions, rerun train_phase1.py "
                "with the current config before starting Phase 2."
            ) from exc
        print(f"Loaded Phase 1 checkpoint: {phase1_ckpt}")
    else:
        raise FileNotFoundError(
            f"Phase 1 checkpoint not found at {phase1_ckpt}. "
            "Run train_phase1.py first or set config['phase1_checkpoint'] to a valid checkpoint."
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters — trainable: {trainable:,} / total: {total:,} ({100 * trainable / total:.1f}%)")

    # ── CE training setup ────────────────────────────────────────────────
    print("=" * 80)
    print("Phase 2: Cross-Entropy Fine-tuning")
    print("=" * 80)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    param_groups = build_optimizer_param_groups(model, cfg)
    optimizer = optim.AdamW(
        param_groups,
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        fused=torch.cuda.is_available(),
    )
    ce_min_lr = cfg.get("min_lr", cfg["learning_rate"] * 0.1)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg["warmup_epochs"],
        total_epochs=cfg["epochs"],
        warmup_lr=1e-6,
        min_lr=ce_min_lr,
    )
    scaler = torch.amp.GradScaler(device.type)
    grad_accum = cfg["grad_accum"]
    gpu_aug = GPUAugment(cfg).to(device)

    tb_log_dir = os.path.join(RESULTS_DIR, "logs_phase2")
    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    compiled_model = model
    ckpt_saver = AsyncCheckpointSaver(device)
    cls_metrics = create_classification_metrics(num_classes, device)

    # ── Training loop ────────────────────────────────────────────────────
    # NOTE: Test metrics are tracked every epoch purely for diagnostics and
    # CSV/TensorBoard review. They are NEVER used for model selection — only
    # the configurable val composite score drives early stopping and
    # checkpoint saving.
    best_val_score = -1.0
    best_val_f1 = -1.0
    best_val_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    score_f1_weight = float(cfg.get("selection_f1_weight", 0.5))
    score_auc_weight = float(cfg.get("selection_auc_weight", 0.5))
    weight_sum = score_f1_weight + score_auc_weight
    if weight_sum <= 0:
        raise ValueError("selection_f1_weight + selection_auc_weight must be positive.")
    score_f1_weight /= weight_sum
    score_auc_weight /= weight_sum
    early_min_delta = cfg.get("early_stopping_min_delta", 0.0)
    print(
        f"Validation score: {score_f1_weight:.2f}*macro-F1 + "
        f"{score_auc_weight:.2f}*AUROC (min_delta={early_min_delta})"
    )

    for epoch in range(1, cfg["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_f1, train_acc, train_auc = train_one_epoch(
            compiled_model, loaders["train"], optimizer, criterion, scaler, device, grad_accum,
            num_classes=num_classes, cfg=cfg, cls_metrics=cls_metrics, gpu_aug=gpu_aug,
        )
        val_metrics = evaluate(
            compiled_model, loaders["val"], criterion, device, num_classes, cls_metrics=cls_metrics
        )
        current_test_metrics = evaluate(
            compiled_model, loaders["test"], criterion, device, num_classes, cls_metrics=cls_metrics
        )

        scheduler.step()

        writer.add_scalar("F1/train", train_f1, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("AUC/train", train_auc, epoch)
        writer.add_scalar("F1/val", val_metrics["f1"], epoch)
        writer.add_scalar("Acc/val", val_metrics["acc"], epoch)
        writer.add_scalar("AUC/val", val_metrics["auc"], epoch)
        writer.add_scalar("F1/test", current_test_metrics["f1"], epoch)
        writer.add_scalar("Acc/test", current_test_metrics["acc"], epoch)
        writer.add_scalar("AUC/test", current_test_metrics["auc"], epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)

        val_score = score_f1_weight * val_metrics["f1"] + score_auc_weight * val_metrics["auc"]
        improved = val_score > best_val_score + early_min_delta
        if improved:
            best_val_score = val_score
            best_val_f1 = val_metrics["f1"]
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            ckpt_saver.save(model, BEST_MODEL_PATH)
        else:
            epochs_without_improvement += 1

        writer.add_scalar("Score/val_composite", val_score, epoch)
        writer.add_scalar("EarlyStopping/best_val_score", best_val_score, epoch)
        writer.add_scalar("EarlyStopping/best_val_f1", best_val_f1, epoch)
        writer.add_scalar("EarlyStopping/best_val_auc", best_val_auc, epoch)
        writer.add_scalar("EarlyStopping/epochs_without_improvement", epochs_without_improvement, epoch)

        if epoch % 5 == 0 or epoch == 1:
            star = "*" if improved else " "
            test_str = (f"  Test — F1: {current_test_metrics['f1']:.4f} "
                        f"Acc: {current_test_metrics['acc']:.4f} "
                        f"AUC: {current_test_metrics['auc']:.4f}")
            print(f" {star} Epoch {epoch}/{cfg['epochs']} — LR: {current_lr:.6f} | "
                  f"Train — F1: {train_f1:.4f} Acc: {train_acc:.4f} AUC: {train_auc:.4f} Loss: {train_loss:.4f} | "
                  f"Val — F1: {val_metrics['f1']:.4f} Acc: {val_metrics['acc']:.4f} "
                  f"AUC: {val_metrics['auc']:.4f} Loss: {val_metrics['loss']:.4f} Score: {val_score:.4f} | "
                  f"Best: {best_val_score:.4f}(F1={best_val_f1:.4f},AUC={best_val_auc:.4f})@ep{best_epoch} "
                  f"NoImpr: {epochs_without_improvement}/{cfg['early_stopping_patience']}"
                  f"{test_str}")

        gc.collect()
        torch.cuda.empty_cache()

        if epochs_without_improvement >= cfg["early_stopping_patience"]:
            print(f"  Early stopping at epoch {epoch} "
                  f"(best Val Score: {best_val_score:.4f} "
                  f"= {score_f1_weight:.2f}*F1[{best_val_f1:.4f}] "
                  f"+ {score_auc_weight:.2f}*AUC[{best_val_auc:.4f}] at epoch {best_epoch})")
            break

    ckpt_saver.wait()
    shutil.copyfile(BEST_MODEL_PATH, PHASE2_BEST_MODEL_PATH)
    print(f"Phase 2 best checkpoint preserved: {PHASE2_BEST_MODEL_PATH}")

    # ── Post-training ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"Training finished. Loading best model from epoch {best_epoch} ...")

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    writer.flush()
    history_df, _ = load_history_from_tensorboard(tb_log_dir)
    history_df.to_csv(HISTORY_CSV_PATH, index=False)

    supcon_history = None
    if os.path.exists(PHASE1_HISTORY_PATH):
        with open(PHASE1_HISTORY_PATH, "r") as f:
            supcon_history = json.load(f)
        print(f"Loaded Phase 1 history from {PHASE1_HISTORY_PATH} for combined curves.")

    save_training_curves(history_df, best_epoch, supcon_history=supcon_history, ce_phase_name="Phase 2")
    save_metrics_csv(compiled_model, loaders, criterion, device, num_classes)
    generate_attention_maps(model, test_df, loaders["eval_tf"], cfg, device)

    final_train = evaluate(
        compiled_model, loaders["train_eval"], criterion, device, num_classes, cls_metrics=cls_metrics
    )
    final_val = evaluate(
        compiled_model, loaders["val"], criterion, device, num_classes, cls_metrics=cls_metrics
    )
    final_test = evaluate(
        compiled_model, loaders["test"], criterion, device, num_classes, cls_metrics=cls_metrics
    )

    hier_val = evaluate_hierarchical(model, loaders["val"], device, num_classes)
    hier_test = evaluate_hierarchical(model, loaders["test"], device, num_classes)

    print(f"Best epoch: {best_epoch}")
    print(f"  Train — F1: {final_train['f1']:.4f}  Acc: {final_train['acc']:.4f}  AUC: {final_train['auc']:.4f}  Loss: {final_train['loss']:.4f}")
    print(f"  Val   — F1: {final_val['f1']:.4f}  Acc: {final_val['acc']:.4f}  AUC: {final_val['auc']:.4f}  Loss: {final_val['loss']:.4f}")
    print(f"  Test  — F1: {final_test['f1']:.4f}  Acc: {final_test['acc']:.4f}  AUC: {final_test['auc']:.4f}  Loss: {final_test['loss']:.4f}")
    print(f"\nHierarchical Classification (VOC vs Non-VOC):")
    print(f"  Val   — VOC Acc: {hier_val['hier_acc']:.4f}")
    print(f"  Test  — VOC Acc: {hier_test['hier_acc']:.4f}")
    write_final_metrics(
        PHASE2_FINAL_METRICS_PATH,
        cfg,
        final_train,
        final_val,
        final_test,
        hier_val,
        hier_test,
        best_epoch,
    )
    print(f"Phase 2 final metrics saved: {PHASE2_FINAL_METRICS_PATH}")
    writer.add_hparams(
        {
            "lr": cfg["learning_rate"],
            "batch_size": cfg["batch_size"],
            "grad_accum": cfg["grad_accum"],
            "dropout": cfg["dropout_rate"],
            "drop_path": cfg["drop_path_rate"],
            "label_smoothing": cfg["label_smoothing"],
            "unfreeze_blocks": cfg["unfreeze_last_n_blocks"],
            "layer_decay": cfg["layer_decay"],
            "sampler_alpha": cfg.get("sampler_balance_alpha", 1.0),
            "selection_f1_weight": score_f1_weight,
            "selection_auc_weight": score_auc_weight,
        },
        {
            "hparam/best_epoch": best_epoch,
            "hparam/val_f1": final_val["f1"],
            "hparam/val_acc": final_val["acc"],
            "hparam/val_auc": final_val["auc"],
            "hparam/test_f1": final_test["f1"],
            "hparam/test_acc": final_test["acc"],
            "hparam/test_auc": final_test["auc"],
            "hparam/hier_val_acc": hier_val["hier_acc"],
            "hparam/hier_test_acc": hier_test["hier_acc"],
        },
    )

    print(f"\nSaved: {RESULTS_DIR}/training_curves.png")
    print(f"Saved: {RESULTS_DIR}/metrics.csv")
    print(f"Saved: {RESULTS_DIR}/gradcam_maps.png")
    print(f"TensorBoard: tensorboard --logdir {tb_log_dir}")
    writer.flush()
    writer.close()

    archive_dir = archive_finished_run(
        cfg, final_train, final_val, final_test, hier_val, hier_test, best_epoch
    )
    if archive_dir is not None:
        print(f"Archived run: {archive_dir}")
        print(f"Archive TensorBoard: tensorboard --logdir {archive_dir / 'logs_phase2'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
