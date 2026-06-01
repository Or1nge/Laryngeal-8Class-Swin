"""Phase 3: train-set confusion-focused SupCon refinement.

Phase 3 intentionally uses only Phase 2 predictions on the training split to
select confused class pairs. Validation and test predictions are not used for
pair selection or contrastive refinement.
"""

import argparse
import gc
import json
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from shared import (
    BASE_DIR, RESULTS_DIR,
    BEST_MODEL_PATH, PHASE2_BEST_MODEL_PATH, PHASE3_CHECKPOINT_PATH,
    PHASE3_HISTORY_PATH, PHASE3_CONFUSION_MATRIX_PATH,
    PHASE3_CONFUSION_PAIRS_PATH, PHASE3_MISCLASSIFIED_PATH,
    init_label_mapping, seed_everything, setup_device, load_config,
    discover_images, load_dataset_split, preload_image_cache, print_data_summary,
    HierarchicalImageClassifier, LABEL_DICT, DISPLAY_NAMES,
    VRAMDataLoader, build_balanced_sampler,
    GPUAugment, build_optimizer_param_groups, WarmupCosineScheduler,
    evaluate, create_classification_metrics, resolve_project_path,
)


class PairFocusedSupConLoss(nn.Module):
    def __init__(self, pairs, temperature=0.1, pair_margin=0.05, pair_margin_weight=1.0):
        super().__init__()
        self.pairs = [tuple(sorted((int(a), int(b)))) for a, b in pairs]
        self.temperature = temperature
        self.pair_margin = pair_margin
        self.pair_margin_weight = pair_margin_weight
        self.last_base_loss = 0.0
        self.last_pair_loss = 0.0

    def forward(self, features, labels):
        labels = labels.view(-1)
        sim = torch.matmul(features, features.T)
        logits = sim / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        batch_size = labels.shape[0]
        self_mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
        logits_mask = ~self_mask

        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        pos_count = positive_mask.float().sum(dim=1)
        valid = pos_count > 0
        if valid.any():
            base_loss = -(
                positive_mask.float() * log_prob
            ).sum(dim=1)[valid] / pos_count[valid]
            base_loss = base_loss.mean()
        else:
            base_loss = torch.zeros((), device=features.device)

        pair_losses = []
        for a, b in self.pairs:
            pair_mask = (
                (labels[:, None].eq(a) & labels[None, :].eq(b))
                | (labels[:, None].eq(b) & labels[None, :].eq(a))
            )
            if pair_mask.any():
                pair_losses.append(F.relu(sim[pair_mask] - self.pair_margin).mean())

        if pair_losses:
            pair_loss = torch.stack(pair_losses).mean()
        else:
            pair_loss = torch.zeros((), device=features.device)

        self.last_base_loss = base_loss.detach().item()
        self.last_pair_loss = pair_loss.detach().item()
        return base_loss + self.pair_margin_weight * pair_loss


def reset_module_parameters(module):
    for child in module.modules():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()


def resolve_phase2_checkpoint(cfg):
    configured = cfg.get("phase2_checkpoint", "")
    if configured:
        checkpoint = configured if os.path.isabs(configured) else resolve_project_path(configured)
        if os.path.exists(checkpoint):
            return checkpoint
    if os.path.exists(PHASE2_BEST_MODEL_PATH):
        return PHASE2_BEST_MODEL_PATH
    return BEST_MODEL_PATH


def analyze_train_confusions(train_df, y_true, y_pred, threshold):
    labels = list(range(len(LABEL_DICT)))
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1

    matrix_df = pd.DataFrame(
        matrix,
        index=[DISPLAY_NAMES[idx] for idx in labels],
        columns=[DISPLAY_NAMES[idx] for idx in labels],
    )
    matrix_df.to_csv(PHASE3_CONFUSION_MATRIX_PATH)

    pred_df = train_df.copy()
    pred_df["phase2_pred_label"] = [DISPLAY_NAMES[int(p)] for p in y_pred]
    pred_df["phase2_pred_id"] = [int(p) for p in y_pred]
    misclassified_df = pred_df[pred_df["label"].to_numpy() != np.asarray(y_pred)].copy()
    misclassified_df.to_csv(PHASE3_MISCLASSIFIED_PATH, index=False)

    supports = matrix.sum(axis=1)
    rows = []
    selected_pairs = set()
    for true_label in labels:
        for pred_label in labels:
            if true_label == pred_label:
                continue
            count = int(matrix[true_label, pred_label])
            support = int(supports[true_label])
            rate = count / support if support else 0.0
            selected = rate > threshold
            if selected:
                selected_pairs.add(tuple(sorted((true_label, pred_label))))
            rows.append(
                {
                    "true_label": DISPLAY_NAMES[true_label],
                    "pred_label": DISPLAY_NAMES[pred_label],
                    "true_id": true_label,
                    "pred_id": pred_label,
                    "count": count,
                    "true_support": support,
                    "confusion_rate": rate,
                    "selected": selected,
                }
            )

    pairs_df = pd.DataFrame(rows).sort_values(
        ["selected", "confusion_rate", "count"], ascending=[False, False, False]
    )
    pairs_df.to_csv(PHASE3_CONFUSION_PAIRS_PATH, index=False)
    return sorted(selected_pairs), pairs_df, misclassified_df


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Train-confusion-focused SupCon")
    parser.add_argument("--config", type=str, default=os.path.join(BASE_DIR, "config_phase3.json"))
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
    print("Phase 3 selection uses train split predictions only. Val/test are not evaluated here.")

    image_cache = preload_image_cache(train_df, cfg=cfg, device=device)
    train_eval_loader = VRAMDataLoader(
        train_df, image_cache, batch_size=cfg["eval_batch_size"], shuffle=False
    )

    num_classes = len(LABEL_DICT)
    model = HierarchicalImageClassifier(num_classes=num_classes, cfg=cfg).to(device)
    phase2_checkpoint = resolve_phase2_checkpoint(cfg)
    if not os.path.exists(phase2_checkpoint):
        raise FileNotFoundError(
            f"Phase 2 checkpoint not found at {phase2_checkpoint}. Run train_phase2.py first."
        )
    model.load_state_dict(torch.load(phase2_checkpoint, map_location=device), strict=True)
    print(f"Loaded Phase 2 checkpoint: {phase2_checkpoint}")

    ce_for_eval = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    train_metrics = evaluate(
        model,
        train_eval_loader,
        ce_for_eval,
        device,
        num_classes,
        return_preds=True,
        cls_metrics=create_classification_metrics(num_classes, device),
    )
    threshold = float(cfg.get("phase3_confusion_threshold", 0.1))
    selected_pairs, pairs_df, misclassified_df = analyze_train_confusions(
        train_df, train_metrics["y_true"], train_metrics["y_pred"], threshold
    )
    print(f"Train confusion threshold: > {threshold:.1%}")
    print(f"Misclassified train samples: {len(misclassified_df)}")
    print(f"Selected confusion pairs: {[(DISPLAY_NAMES[a], DISPLAY_NAMES[b]) for a, b in selected_pairs]}")

    metadata = {
        "phase2_checkpoint": phase2_checkpoint,
        "train_metrics": {k: v for k, v in train_metrics.items() if k not in {"y_true", "y_pred", "is_voc"}},
        "confusion_threshold": threshold,
        "selected_pairs": [
            {"a": DISPLAY_NAMES[a], "b": DISPLAY_NAMES[b], "a_id": a, "b_id": b}
            for a, b in selected_pairs
        ],
        "confusion_matrix_path": PHASE3_CONFUSION_MATRIX_PATH,
        "confusion_pairs_path": PHASE3_CONFUSION_PAIRS_PATH,
        "misclassified_path": PHASE3_MISCLASSIFIED_PATH,
    }

    if not selected_pairs:
        shutil.copyfile(phase2_checkpoint, PHASE3_CHECKPOINT_PATH)
        with open(PHASE3_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump([{**metadata, "skipped": True, "reason": "no train confusion pair exceeded threshold"}], f, indent=2, ensure_ascii=False)
        print(f"No selected pairs. Copied Phase 2 checkpoint to {PHASE3_CHECKPOINT_PATH}")
        return

    focus_classes = sorted({label for pair in selected_pairs for label in pair})
    focus_df = train_df[train_df["label"].isin(focus_classes)].reset_index(drop=True)
    if focus_df["label"].nunique() < 2:
        shutil.copyfile(phase2_checkpoint, PHASE3_CHECKPOINT_PATH)
        with open(PHASE3_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump([{**metadata, "skipped": True, "reason": "fewer than two focused classes"}], f, indent=2, ensure_ascii=False)
        print(f"Not enough focused classes. Copied Phase 2 checkpoint to {PHASE3_CHECKPOINT_PATH}")
        return

    if cfg.get("phase3_reinit_projector", True):
        reset_module_parameters(model.projector)
        print("Reinitialized Phase 3 projector.")
    else:
        print("Keeping projector weights from the Phase 2 checkpoint.")
    for param in model.classifier.parameters():
        param.requires_grad = False

    sampler = build_balanced_sampler(focus_df, hierarchical=False, balance_alpha=1.0)
    focus_loader = VRAMDataLoader(
        focus_df,
        image_cache,
        batch_size=int(cfg.get("phase3_batch_size", cfg["batch_size"])),
        sampler=sampler,
        shuffle=False,
    )

    phase3_lr = float(cfg.get("phase3_learning_rate", cfg.get("supcon_learning_rate", 2e-4)))
    phase3_cfg = {**cfg, "learning_rate": phase3_lr}
    optimizer = optim.AdamW(
        build_optimizer_param_groups(model, phase3_cfg),
        lr=phase3_lr,
        weight_decay=cfg["weight_decay"],
        fused=torch.cuda.is_available(),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=int(cfg.get("phase3_warmup_epochs", 3)),
        total_epochs=int(cfg.get("phase3_epochs", 40)),
        warmup_lr=1e-6,
        min_lr=float(cfg.get("phase3_min_lr", phase3_lr * 0.1)),
    )
    scaler = torch.amp.GradScaler(device.type)
    gpu_aug = GPUAugment(cfg).to(device)
    criterion = PairFocusedSupConLoss(
        selected_pairs,
        temperature=float(cfg.get("phase3_temperature", 0.1)),
        pair_margin=float(cfg.get("phase3_pair_margin", 0.05)),
        pair_margin_weight=float(cfg.get("phase3_pair_margin_weight", 1.0)),
    ).to(device)

    tb_log_dir = os.path.join(RESULTS_DIR, "logs_phase3")
    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    epochs = int(cfg.get("phase3_epochs", 40))
    patience = int(cfg.get("phase3_early_stopping_patience", 8))
    min_delta = float(cfg.get("phase3_early_stopping_min_delta", 0.001))
    grad_accum = int(cfg.get("grad_accum", 1))
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = torch.tensor(0.0, device=device)
        total = 0
        optimizer.zero_grad()
        for step, (images, labels, _is_voc) in enumerate(focus_loader, start=1):
            images = gpu_aug(images)
            with torch.amp.autocast(device_type=device.type):
                projections = model(images, return_projection=True)
                loss = criterion(projections, labels) / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == len(focus_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running_loss += loss.detach() * grad_accum * labels.shape[0]
            total += labels.shape[0]

        scheduler.step()
        avg_loss = (running_loss / max(total, 1)).item()
        current_lr = optimizer.param_groups[0]["lr"]
        improved = avg_loss < best_loss - min_delta
        if improved:
            best_loss = avg_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        row = {
            "epoch": epoch,
            "loss": avg_loss,
            "base_loss": criterion.last_base_loss,
            "pair_loss": criterion.last_pair_loss,
            "lr": current_lr,
            "best_loss": best_loss,
            "improved": improved,
        }
        history.append(row)
        writer.add_scalar("Phase3/loss", avg_loss, epoch)
        writer.add_scalar("Phase3/base_loss", criterion.last_base_loss, epoch)
        writer.add_scalar("Phase3/pair_loss", criterion.last_pair_loss, epoch)
        writer.add_scalar("Phase3/lr", current_lr, epoch)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"  Phase 3 Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f} "
                f"Base: {criterion.last_base_loss:.4f} Pair: {criterion.last_pair_loss:.4f} "
                f"LR: {current_lr:.6f} Best: {best_loss:.4f}@ep{best_epoch} "
                f"NoImpr: {no_improve}/{patience}"
            )

        gc.collect()
        torch.cuda.empty_cache()
        if no_improve >= patience:
            print(f"  Phase 3 early stopping at epoch {epoch} (best train loss {best_loss:.4f})")
            break

    writer.flush()
    writer.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), PHASE3_CHECKPOINT_PATH)
    with open(PHASE3_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump([{**metadata, "skipped": False, "focus_classes": [DISPLAY_NAMES[c] for c in focus_classes]}, *history], f, indent=2, ensure_ascii=False)
    print(f"Phase 3 checkpoint saved: {PHASE3_CHECKPOINT_PATH}")
    print(f"Phase 3 history saved: {PHASE3_HISTORY_PATH}")
    print(f"TensorBoard: tensorboard --logdir {tb_log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
