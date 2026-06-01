"""Phase 4: CE classifier retraining after Phase 3 focused SupCon."""

import argparse
import gc
import json
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from shared import (
    BASE_DIR, RESULTS_DIR,
    BEST_MODEL_PATH, PHASE1_HISTORY_PATH, PHASE3_CHECKPOINT_PATH,
    PHASE4_BEST_MODEL_PATH, HISTORY_CSV_PATH, PHASE4_HISTORY_CSV_PATH,
    init_label_mapping, seed_everything, setup_device, load_config,
    discover_images, load_dataset_split, preload_image_cache, print_data_summary,
    HierarchicalImageClassifier, LABEL_DICT,
    create_loaders, build_balanced_sampler, GPUAugment,
    WarmupCosineScheduler, train_one_epoch, evaluate, evaluate_hierarchical,
    save_training_curves, save_metrics_csv, generate_attention_maps,
    load_history_from_tensorboard, AsyncCheckpointSaver, create_classification_metrics,
    resolve_project_path,
)
from train_phase2 import archive_finished_run


def reset_module_parameters(module):
    for child in module.modules():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()


def resolve_phase3_checkpoint(cfg):
    configured = cfg.get("phase3_checkpoint", "")
    if configured:
        checkpoint = configured if os.path.isabs(configured) else resolve_project_path(configured)
        if os.path.exists(checkpoint):
            return checkpoint
    return PHASE3_CHECKPOINT_PATH


def set_phase4_trainable(model, train_backbone):
    if train_backbone:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.projector.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True


def main():
    parser = argparse.ArgumentParser(description="Phase 4: CE classifier retraining")
    parser.add_argument("--config", type=str, default=os.path.join(BASE_DIR, "config_phase4.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    init_label_mapping(cfg)
    seed_everything(cfg["seed"])
    device = setup_device()

    print(f"Using device: {device}")
    print(f"Config: {json.dumps(cfg, indent=2, ensure_ascii=False)}")
    print("=" * 80)

    df = discover_images()
    train_df, val_df, test_df = load_dataset_split(df)
    print_data_summary(train_df, val_df, test_df)

    image_cache = preload_image_cache(train_df, val_df, test_df, cfg=cfg, device=device)
    print("Building Phase 4 (CE) balanced sampler:")
    ce_sampler = build_balanced_sampler(
        train_df,
        hierarchical=False,
        balance_alpha=cfg.get("sampler_balance_alpha", 1.0),
    )
    loaders = create_loaders(train_df, val_df, test_df, cfg, image_cache=image_cache, train_sampler=ce_sampler)
    print("Class weights: disabled (using sampler-only balancing).")

    num_classes = len(LABEL_DICT)
    model = HierarchicalImageClassifier(num_classes=num_classes, cfg=cfg).to(device)
    phase3_checkpoint = resolve_phase3_checkpoint(cfg)
    if not os.path.exists(phase3_checkpoint):
        raise FileNotFoundError(
            f"Phase 3 checkpoint not found at {phase3_checkpoint}. Run train_phase3.py first."
        )
    model.load_state_dict(torch.load(phase3_checkpoint, map_location=device), strict=True)
    print(f"Loaded Phase 3 checkpoint: {phase3_checkpoint}")

    if cfg.get("phase4_reset_classifier", True):
        reset_module_parameters(model.classifier)
        print("Reinitialized Phase 4 classifier.")
    set_phase4_trainable(model, train_backbone=cfg.get("phase4_train_backbone", False))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters — trainable: {trainable:,} / total: {total:,} ({100 * trainable / total:.1f}%)")

    print("=" * 80)
    print("Phase 4: Cross-Entropy Classifier Retraining")
    print("=" * 80)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters for Phase 4.")
    optimizer = optim.AdamW(
        trainable_params,
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        fused=torch.cuda.is_available(),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg["warmup_epochs"],
        total_epochs=cfg["epochs"],
        warmup_lr=1e-6,
        min_lr=cfg.get("min_lr", cfg["learning_rate"] * 0.1),
    )
    scaler = torch.amp.GradScaler(device.type)
    gpu_aug = GPUAugment(cfg).to(device)
    grad_accum = cfg["grad_accum"]

    tb_log_dir = os.path.join(RESULTS_DIR, "logs_phase4")
    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)
    ckpt_saver = AsyncCheckpointSaver(device)
    cls_metrics = create_classification_metrics(num_classes, device)

    score_f1_weight = float(cfg.get("selection_f1_weight", 0.7))
    score_auc_weight = float(cfg.get("selection_auc_weight", 0.3))
    weight_sum = score_f1_weight + score_auc_weight
    if weight_sum <= 0:
        raise ValueError("selection_f1_weight + selection_auc_weight must be positive.")
    score_f1_weight /= weight_sum
    score_auc_weight /= weight_sum
    early_min_delta = cfg.get("early_stopping_min_delta", 0.0)

    best_val_score = -1.0
    best_val_f1 = -1.0
    best_val_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    print(
        f"Validation score: {score_f1_weight:.2f}*macro-F1 + "
        f"{score_auc_weight:.2f}*AUROC (min_delta={early_min_delta})"
    )

    for epoch in range(1, cfg["epochs"] + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_f1, train_acc, train_auc = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            scaler,
            device,
            grad_accum,
            num_classes=num_classes,
            cfg=cfg,
            cls_metrics=cls_metrics,
            gpu_aug=gpu_aug,
        )
        val_metrics = evaluate(model, loaders["val"], criterion, device, num_classes, cls_metrics=cls_metrics)
        current_test_metrics = evaluate(model, loaders["test"], criterion, device, num_classes, cls_metrics=cls_metrics)
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
            ckpt_saver.save(model, PHASE4_BEST_MODEL_PATH)
        else:
            epochs_without_improvement += 1

        writer.add_scalar("Score/val_composite", val_score, epoch)
        writer.add_scalar("EarlyStopping/best_val_score", best_val_score, epoch)

        if epoch == 1 or epoch % 5 == 0:
            star = "*" if improved else " "
            print(
                f" {star} Phase4 Epoch {epoch}/{cfg['epochs']} — LR: {current_lr:.6f} | "
                f"Train — F1: {train_f1:.4f} Acc: {train_acc:.4f} AUC: {train_auc:.4f} Loss: {train_loss:.4f} | "
                f"Val — F1: {val_metrics['f1']:.4f} Acc: {val_metrics['acc']:.4f} "
                f"AUC: {val_metrics['auc']:.4f} Loss: {val_metrics['loss']:.4f} Score: {val_score:.4f} | "
                f"Best: {best_val_score:.4f}(F1={best_val_f1:.4f},AUC={best_val_auc:.4f})@ep{best_epoch} "
                f"NoImpr: {epochs_without_improvement}/{cfg['early_stopping_patience']} "
                f"Test — F1: {current_test_metrics['f1']:.4f} Acc: {current_test_metrics['acc']:.4f} "
                f"AUC: {current_test_metrics['auc']:.4f}"
            )

        gc.collect()
        torch.cuda.empty_cache()
        if epochs_without_improvement >= cfg["early_stopping_patience"]:
            print(f"  Phase 4 early stopping at epoch {epoch} (best val score {best_val_score:.4f})")
            break

    ckpt_saver.wait()
    print("\n" + "=" * 80)
    print(f"Phase 4 finished. Loading best classifier from epoch {best_epoch} ...")
    model.load_state_dict(torch.load(PHASE4_BEST_MODEL_PATH, map_location=device))
    shutil.copyfile(PHASE4_BEST_MODEL_PATH, BEST_MODEL_PATH)
    print(f"Final best checkpoint saved: {BEST_MODEL_PATH}")

    writer.flush()
    history_df, _ = load_history_from_tensorboard(tb_log_dir)
    history_df.to_csv(PHASE4_HISTORY_CSV_PATH, index=False)
    history_df.to_csv(HISTORY_CSV_PATH, index=False)

    supcon_history = None
    if os.path.exists(PHASE1_HISTORY_PATH):
        with open(PHASE1_HISTORY_PATH, "r", encoding="utf-8") as f:
            supcon_history = json.load(f)
        print(f"Loaded Phase 1 history from {PHASE1_HISTORY_PATH} for combined curves.")

    save_training_curves(history_df, best_epoch, supcon_history=supcon_history, ce_phase_name="Phase 4")
    save_metrics_csv(model, loaders, criterion, device, num_classes)
    generate_attention_maps(model, test_df, loaders["eval_tf"], cfg, device)

    final_train = evaluate(model, loaders["train_eval"], criterion, device, num_classes, cls_metrics=cls_metrics)
    final_val = evaluate(model, loaders["val"], criterion, device, num_classes, cls_metrics=cls_metrics)
    final_test = evaluate(model, loaders["test"], criterion, device, num_classes, cls_metrics=cls_metrics)
    hier_val = evaluate_hierarchical(model, loaders["val"], device, num_classes)
    hier_test = evaluate_hierarchical(model, loaders["test"], device, num_classes)

    print(f"Best Phase 4 epoch: {best_epoch}")
    print(f"  Train — F1: {final_train['f1']:.4f}  Acc: {final_train['acc']:.4f}  AUC: {final_train['auc']:.4f}  Loss: {final_train['loss']:.4f}")
    print(f"  Val   — F1: {final_val['f1']:.4f}  Acc: {final_val['acc']:.4f}  AUC: {final_val['auc']:.4f}  Loss: {final_val['loss']:.4f}")
    print(f"  Test  — F1: {final_test['f1']:.4f}  Acc: {final_test['acc']:.4f}  AUC: {final_test['auc']:.4f}  Loss: {final_test['loss']:.4f}")
    print(f"  Val VOC Acc: {hier_val['hier_acc']:.4f}")
    print(f"  Test VOC Acc: {hier_test['hier_acc']:.4f}")

    writer.add_hparams(
        {
            "lr": cfg["learning_rate"],
            "batch_size": cfg["batch_size"],
            "phase4_reset_classifier": int(bool(cfg.get("phase4_reset_classifier", True))),
            "phase4_train_backbone": int(bool(cfg.get("phase4_train_backbone", False))),
            "sampler_alpha": cfg.get("sampler_balance_alpha", 1.0),
        },
        {
            "hparam/best_epoch": best_epoch,
            "hparam/val_f1": final_val["f1"],
            "hparam/val_acc": final_val["acc"],
            "hparam/val_auc": final_val["auc"],
            "hparam/test_f1": final_test["f1"],
            "hparam/test_acc": final_test["acc"],
            "hparam/test_auc": final_test["auc"],
        },
    )
    writer.flush()
    writer.close()

    archive_dir = archive_finished_run(
        cfg, final_train, final_val, final_test, hier_val, hier_test, best_epoch
    )
    if archive_dir is not None:
        print(f"Archived final run: {archive_dir}")
        print(f"Archive TensorBoard: tensorboard --logdir {archive_dir / 'logs_phase4'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
