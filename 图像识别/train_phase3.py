"""Phase 3: validation-confusion-focused SupCon refinement.

Phase 3 selects directed confusions from Phase 2 validation predictions. If
class A is often predicted as class B, training uses only the involved train
classes and applies a directional prototype margin to A anchors against B.
Test predictions are not used.
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
    def __init__(
        self,
        pairs,
        pair_weights=None,
        temperature=0.1,
        pair_margin=0.05,
        pair_margin_weight=1.0,
    ):
        super().__init__()
        self.pairs = [tuple(sorted((int(a), int(b)))) for a, b in pairs]
        self.pair_weights = {
            tuple(sorted((int(a), int(b)))): float(weight)
            for (a, b), weight in (pair_weights or {}).items()
        }
        self.temperature = temperature
        self.pair_margin = pair_margin
        self.pair_margin_weight = pair_margin_weight
        self.last_base_loss = 0.0
        self.last_pair_loss = 0.0
        self.last_focus_loss = 0.0

    def forward(self, features, labels):
        labels = labels.view(-1)
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        logits_mask = torch.ones_like(sim, dtype=torch.bool)
        logits_mask.fill_diagonal_(False)

        positive_mask = labels[:, None].eq(labels[None, :]) & logits_mask
        mask_value = torch.finfo(sim.dtype).min
        log_prob = sim - torch.logsumexp(
            sim.masked_fill(~logits_mask, mask_value),
            dim=1,
            keepdim=True,
        )
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
        weights = []
        for pair in self.pairs:
            a, b = pair
            pair_mask = (
                (labels[:, None].eq(a) & labels[None, :].eq(b))
                | (labels[:, None].eq(b) & labels[None, :].eq(a))
            ) & logits_mask
            if pair_mask.any():
                pair_losses.append(F.relu(sim[pair_mask] - self.pair_margin).mean())
                weights.append(
                    torch.as_tensor(
                        self.pair_weights.get(pair, 1.0),
                        device=features.device,
                        dtype=features.dtype,
                    )
                )

        if pair_losses:
            pair_loss_tensor = torch.stack(pair_losses)
            weight_tensor = torch.stack(weights)
            pair_loss = (pair_loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-8)
        else:
            pair_loss = torch.zeros((), device=features.device)

        self.last_base_loss = base_loss.detach().item()
        self.last_pair_loss = pair_loss.detach().item()
        self.last_focus_loss = self.last_pair_loss
        return base_loss + self.pair_margin_weight * pair_loss


class DirectionFocusedSupConLoss(nn.Module):
    def __init__(
        self,
        direction_weights,
        temperature=0.1,
        direction_margin=0.12,
        direction_margin_weight=0.6,
    ):
        super().__init__()
        self.direction_weights = {
            (int(src), int(dst)): float(weight)
            for (src, dst), weight in direction_weights.items()
        }
        self.temperature = temperature
        self.direction_margin = direction_margin
        self.direction_margin_weight = direction_margin_weight
        self.last_base_loss = 0.0
        self.last_direction_loss = 0.0
        self.last_focus_loss = 0.0

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

        direction_losses = []
        weights = []
        for (src, dst), weight in self.direction_weights.items():
            src_mask = labels.eq(src)
            dst_mask = labels.eq(dst)
            if not src_mask.any() or not dst_mask.any():
                continue

            src_center = F.normalize(features[src_mask].mean(dim=0), dim=0).detach()
            dst_center = F.normalize(features[dst_mask].mean(dim=0), dim=0).detach()
            src_features = features[src_mask]
            src_sim = torch.matmul(src_features, src_center)
            dst_sim = torch.matmul(src_features, dst_center)
            direction_losses.append(F.relu(dst_sim - src_sim + self.direction_margin).mean())
            weights.append(torch.as_tensor(weight, device=features.device, dtype=features.dtype))

        if direction_losses:
            direction_loss_tensor = torch.stack(direction_losses)
            weight_tensor = torch.stack(weights)
            direction_loss = (
                direction_loss_tensor * weight_tensor
            ).sum() / weight_tensor.sum().clamp_min(1e-8)
        else:
            direction_loss = torch.zeros((), device=features.device)

        self.last_base_loss = base_loss.detach().item()
        self.last_direction_loss = direction_loss.detach().item()
        self.last_focus_loss = self.last_direction_loss
        return base_loss + self.direction_margin_weight * direction_loss


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


def analyze_val_confusions(val_df, y_true, y_pred, threshold, direction_weight_max):
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

    pred_df = val_df.copy()
    pred_df["phase2_pred_label"] = [DISPLAY_NAMES[int(p)] for p in y_pred]
    pred_df["phase2_pred_id"] = [int(p) for p in y_pred]
    misclassified_df = pred_df[pred_df["label"].to_numpy() != np.asarray(y_pred)].copy()
    misclassified_df.to_csv(PHASE3_MISCLASSIFIED_PATH, index=False)

    supports = matrix.sum(axis=1)
    rows = []
    selected_pairs = set()
    selected_pair_rates = {}
    selected_pair_directions = {}
    selected_directions = []
    direction_weights = {}
    for true_label in labels:
        for pred_label in labels:
            if true_label == pred_label:
                continue
            count = int(matrix[true_label, pred_label])
            support = int(supports[true_label])
            rate = count / support if support else 0.0
            selected = rate > threshold
            pair = tuple(sorted((true_label, pred_label)))
            if selected:
                selected_pairs.add(pair)
                selected_pair_rates[pair] = max(selected_pair_rates.get(pair, 0.0), rate)
                direction_weights[(true_label, pred_label)] = min(
                    direction_weight_max,
                    max(1.0, rate / max(threshold, 1e-8)),
                )
                selected_directions.append(
                    {
                        "true_label": DISPLAY_NAMES[true_label],
                        "pred_label": DISPLAY_NAMES[pred_label],
                        "true_id": true_label,
                        "pred_id": pred_label,
                        "count": count,
                        "true_support": support,
                        "confusion_rate": rate,
                        "weight": direction_weights[(true_label, pred_label)],
                    }
                )
                selected_pair_directions.setdefault(pair, []).append(
                    {
                        "true_label": DISPLAY_NAMES[true_label],
                        "pred_label": DISPLAY_NAMES[pred_label],
                        "count": count,
                        "true_support": support,
                        "confusion_rate": rate,
                    }
                )
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
    pair_weights = {
        pair: min(direction_weight_max, max(1.0, selected_pair_rates[pair] / max(threshold, 1e-8)))
        for pair in selected_pairs
    }
    selected_directions = sorted(
        selected_directions,
        key=lambda row: (row["confusion_rate"], row["count"]),
        reverse=True,
    )
    return (
        sorted(selected_pairs),
        pair_weights,
        direction_weights,
        selected_pair_directions,
        selected_directions,
        pairs_df,
        misclassified_df,
    )


def val_composite_score(metrics, cfg):
    f1_weight = float(cfg.get("selection_f1_weight", 0.7))
    auc_weight = float(cfg.get("selection_auc_weight", 0.3))
    return f1_weight * metrics.get("f1", 0.0) + auc_weight * metrics.get("auc", 0.0)


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Validation-confusion-focused SupCon")
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
    print("Phase 3 selection uses validation split predictions only. Test predictions are not used.")

    image_cache = preload_image_cache(train_df, val_df, cfg=cfg, device=device)
    val_eval_loader = VRAMDataLoader(
        val_df, image_cache, batch_size=cfg["eval_batch_size"], shuffle=False
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
    val_metrics = evaluate(
        model,
        val_eval_loader,
        ce_for_eval,
        device,
        num_classes,
        return_preds=True,
        cls_metrics=create_classification_metrics(num_classes, device),
    )
    threshold = float(cfg.get("phase3_confusion_threshold", 0.1))
    direction_weight_max = float(
        cfg.get("phase3_direction_weight_max", cfg.get("phase3_pair_weight_max", 3.0))
    )
    (
        selected_pairs,
        pair_weights,
        direction_weights,
        selected_pair_directions,
        selected_directions,
        pairs_df,
        misclassified_df,
    ) = analyze_val_confusions(
        val_df,
        val_metrics["y_true"],
        val_metrics["y_pred"],
        threshold,
        direction_weight_max,
    )
    print(f"Validation directed-confusion threshold: > {threshold:.1%}")
    print(f"Misclassified validation samples: {len(misclassified_df)}")
    print(f"Selected confusion pairs: {[(DISPLAY_NAMES[a], DISPLAY_NAMES[b]) for a, b in selected_pairs]}")
    print(
        "Selected directional weights: "
        f"{[(row['true_label'], row['pred_label'], round(row['weight'], 3)) for row in selected_directions]}"
    )
    baseline_val_score = val_composite_score(val_metrics, cfg)
    print(f"Phase 2 baseline validation score: {baseline_val_score:.4f}")

    metadata = {
        "phase2_checkpoint": phase2_checkpoint,
        "selection_split": "val",
        "selection_rule": "directed class confusion rate > threshold; pair weight = min(max_weight, rate / threshold)",
        "val_metrics": {k: v for k, v in val_metrics.items() if k not in {"y_true", "y_pred", "is_voc"}},
        "val_score_baseline": baseline_val_score,
        "confusion_threshold": threshold,
        "direction_weight_max": direction_weight_max,
        "selected_directions": selected_directions,
        "selected_pairs": [
            {
                "a": DISPLAY_NAMES[a],
                "b": DISPLAY_NAMES[b],
                "a_id": a,
                "b_id": b,
                "weight": pair_weights[(a, b)],
                "directions": selected_pair_directions.get((a, b), []),
            }
            for a, b in selected_pairs
        ],
        "confusion_matrix_path": PHASE3_CONFUSION_MATRIX_PATH,
        "confusion_pairs_path": PHASE3_CONFUSION_PAIRS_PATH,
        "misclassified_path": PHASE3_MISCLASSIFIED_PATH,
        "loss_mode": cfg.get("phase3_loss_mode", "direction"),
    }

    if not selected_pairs:
        shutil.copyfile(phase2_checkpoint, PHASE3_CHECKPOINT_PATH)
        with open(PHASE3_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump([{**metadata, "skipped": True, "reason": "no validation confusion pair exceeded threshold"}], f, indent=2, ensure_ascii=False)
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
    loss_mode = str(cfg.get("phase3_loss_mode", "direction")).lower()
    if loss_mode in {"pair", "pair_margin", "symmetric_pair"}:
        criterion = PairFocusedSupConLoss(
            selected_pairs,
            pair_weights=pair_weights,
            temperature=float(cfg.get("phase3_temperature", 0.1)),
            pair_margin=float(cfg.get("phase3_pair_margin", 0.05)),
            pair_margin_weight=float(cfg.get("phase3_pair_margin_weight", 1.0)),
        ).to(device)
        focus_loss_name = "pair_loss"
        print(
            "Phase 3 loss mode: pair_margin "
            f"(margin={cfg.get('phase3_pair_margin', 0.05)}, "
            f"weight={cfg.get('phase3_pair_margin_weight', 1.0)})"
        )
    elif loss_mode in {"direction", "directed", "prototype_direction"}:
        criterion = DirectionFocusedSupConLoss(
            direction_weights,
            temperature=float(cfg.get("phase3_temperature", 0.1)),
            direction_margin=float(cfg.get("phase3_direction_margin", 0.12)),
            direction_margin_weight=float(cfg.get("phase3_direction_margin_weight", 0.6)),
        ).to(device)
        focus_loss_name = "direction_loss"
        print(
            "Phase 3 loss mode: direction "
            f"(margin={cfg.get('phase3_direction_margin', 0.12)}, "
            f"weight={cfg.get('phase3_direction_margin_weight', 0.6)})"
        )
    else:
        raise ValueError(f"Unknown phase3_loss_mode: {loss_mode}")

    tb_log_dir = os.path.join(RESULTS_DIR, "logs_phase3")
    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    epochs = int(cfg.get("phase3_epochs", 40))
    patience = int(cfg.get("phase3_early_stopping_patience", 8))
    min_delta = float(cfg.get("phase3_early_stopping_min_delta", 0.001))
    grad_accum = int(cfg.get("grad_accum", 1))
    best_score = baseline_val_score
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0
    history = [
        {
            "epoch": 0,
            "loss": None,
            "base_loss": None,
            "pair_loss": None,
            "direction_loss": None,
            "val_f1": val_metrics["f1"],
            "val_acc": val_metrics["acc"],
            "val_auc": val_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_score": baseline_val_score,
            "lr": None,
            "best_score": best_score,
            "improved": True,
            "checkpoint_source": "phase2_baseline",
        }
    ]

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
        val_epoch_metrics = evaluate(
            model,
            val_eval_loader,
            ce_for_eval,
            device,
            num_classes,
            return_preds=False,
            cls_metrics=create_classification_metrics(num_classes, device),
        )
        val_score = val_composite_score(val_epoch_metrics, cfg)
        improved = val_score > best_score + min_delta
        if improved:
            best_score = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        row = {
            "epoch": epoch,
            "loss": avg_loss,
            "base_loss": criterion.last_base_loss,
            "pair_loss": criterion.last_pair_loss if hasattr(criterion, "last_pair_loss") else None,
            "direction_loss": criterion.last_direction_loss if hasattr(criterion, "last_direction_loss") else None,
            "val_f1": val_epoch_metrics["f1"],
            "val_acc": val_epoch_metrics["acc"],
            "val_auc": val_epoch_metrics["auc"],
            "val_loss": val_epoch_metrics["loss"],
            "val_score": val_score,
            "lr": current_lr,
            "best_score": best_score,
            "improved": improved,
        }
        history.append(row)
        writer.add_scalar("Phase3/loss", avg_loss, epoch)
        writer.add_scalar("Phase3/base_loss", criterion.last_base_loss, epoch)
        writer.add_scalar(f"Phase3/{focus_loss_name}", criterion.last_focus_loss, epoch)
        writer.add_scalar("Phase3/val_f1", val_epoch_metrics["f1"], epoch)
        writer.add_scalar("Phase3/val_acc", val_epoch_metrics["acc"], epoch)
        writer.add_scalar("Phase3/val_auc", val_epoch_metrics["auc"], epoch)
        writer.add_scalar("Phase3/val_score", val_score, epoch)
        writer.add_scalar("Phase3/lr", current_lr, epoch)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"  Phase 3 Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f} "
                f"Base: {criterion.last_base_loss:.4f} {focus_loss_name}: {criterion.last_focus_loss:.4f} "
                f"ValF1: {val_epoch_metrics['f1']:.4f} ValAUC: {val_epoch_metrics['auc']:.4f} "
                f"Score: {val_score:.4f} LR: {current_lr:.6f} "
                f"BestScore: {best_score:.4f}@ep{best_epoch} "
                f"NoImpr: {no_improve}/{patience}"
            )

        gc.collect()
        torch.cuda.empty_cache()
        if no_improve >= patience:
            print(f"  Phase 3 early stopping at epoch {epoch} (best val score {best_score:.4f})")
            break

    writer.flush()
    writer.close()
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), PHASE3_CHECKPOINT_PATH)
    with open(PHASE3_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    **metadata,
                    "skipped": False,
                    "focus_classes": [DISPLAY_NAMES[c] for c in focus_classes],
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "phase3_improved_over_phase2": best_epoch > 0,
                },
                *history,
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )
    if best_epoch == 0:
        print("Phase 3 did not beat the Phase 2 validation baseline; checkpoint remains Phase 2 weights.")
    print(f"Phase 3 checkpoint saved: {PHASE3_CHECKPOINT_PATH}")
    print(f"Phase 3 history saved: {PHASE3_HISTORY_PATH}")
    print(f"TensorBoard: tensorboard --logdir {tb_log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
