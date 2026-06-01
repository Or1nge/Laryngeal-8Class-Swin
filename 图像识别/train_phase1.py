"""Phase 1: Hierarchical Supervised Contrastive Learning Pretraining.

Usage:
    python train_phase1.py [--config CONFIG_PATH]

Outputs:
    <workspace>/Results/<worktree>/phase1_checkpoint.pth  — backbone + projector weights
    <workspace>/Results/<worktree>/phase1_history.json    — per-epoch loss/lr history
    <workspace>/Results/<worktree>/logs_phase1/           — TensorBoard logs
"""

import argparse
import gc
import json
import os
import shutil

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from shared import (
    BASE_DIR, RESULTS_DIR, PHASE1_CHECKPOINT_PATH, PHASE1_HISTORY_PATH,
    init_label_mapping, seed_everything, setup_device, load_config,
    discover_images, load_dataset_split, preload_image_cache, print_data_summary,
    HierarchicalImageClassifier, LABEL_DICT,
    LaryngealDataset, build_transforms, build_balanced_sampler,
    GPUAugment,
    build_optimizer_param_groups, WarmupCosineScheduler,
    HierarchicalSupConLoss, KnowledgeGuidedSupConLoss,
    build_kg_similarity_matrix, supcon_train_one_epoch,
    supcon_train_one_epoch_bilevel, CyclingDataLoaderIter,
    VRAMDataLoader,
)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: SupCon Pretraining")
    parser.add_argument("--config", type=str, default=os.path.join(BASE_DIR, "config_phase1.json"))
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

    num_classes = len(LABEL_DICT)
    model = HierarchicalImageClassifier(num_classes=num_classes, cfg=cfg).to(device)
    for param in model.classifier.parameters():
        param.requires_grad = False
    print("Phase 1 classifier frozen; training backbone + projector only.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters — trainable: {trainable:,} / total: {total:,} ({100 * trainable / total:.1f}%)")

    # ── SupCon training setup ────────────────────────────────────────────
    supcon_epochs = cfg.get("supcon_epochs", 50)
    supcon_lr = cfg.get("supcon_learning_rate", 5e-4)
    supcon_warmup = cfg.get("supcon_warmup_epochs", 3)
    supcon_temp = cfg.get("supcon_temperature", 0.1)
    voc_margin = cfg.get("supcon_voc_margin", 0.3)
    supcon_patience = cfg.get("supcon_early_stopping_patience", 10)
    supcon_min_delta = cfg.get("supcon_early_stopping_min_delta", 0.0)
    grad_accum = cfg.get("grad_accum", 2)

    print("=" * 80)
    print("Phase 1: Hierarchical Supervised Contrastive Learning Pretraining")
    print(f"  Epochs: {supcon_epochs}, LR: {supcon_lr}, Temperature: {supcon_temp}")
    print(f"  VOC Margin: {voc_margin}, Early Stopping Patience: {supcon_patience}")
    print("=" * 80)

    train_tf, eval_tf = build_transforms(cfg)
    gpu_aug = GPUAugment(cfg).to(device)
    supcon_dataset = LaryngealDataset(train_df, train_tf, cfg, image_cache=image_cache)
    supcon_bs = cfg.get("supcon_batch_size", cfg.get("batch_size", 64))
    nw = 0
    pin_memory = False
    loader_common = {
        "num_workers": nw,
        "pin_memory": pin_memory,
    }

    print("Building Phase 1 (SupCon) balanced sampler:")
    supcon_sampler = build_balanced_sampler(train_df, hierarchical=True)
    roi_mix_prob = 0.0
    if image_cache is not None and image_cache.get("roi_images") is not None:
        roi_mix_prob = float(cfg.get("roi_mix_prob", image_cache.get("roi_mix_prob", 0.0)) or 0.0)
        if roi_mix_prob > 0:
            print(f"Phase 1 training ROI replacement enabled: prob={roi_mix_prob:.2f}")

    if image_cache is not None and "images" in image_cache:
        supcon_loader = VRAMDataLoader(
            train_df,
            image_cache,
            batch_size=supcon_bs,
            sampler=supcon_sampler,
            shuffle=False,
            roi_mix_prob=roi_mix_prob,
        )
    else:
        supcon_loader = DataLoader(
            supcon_dataset,
            batch_size=supcon_bs,
            sampler=supcon_sampler,
            **loader_common,
        )

    kg_cfg = cfg.get("knowledge_graph", {})
    kg_learnable = False
    kg_delayed_activation = False
    kg_activation_epoch = 1
    zero_supcon_criterion = None
    if kg_cfg.get("enabled", False):
        kg_sim_matrix = build_kg_similarity_matrix(cfg)
        kg_weight = kg_cfg.get("kg_weight", 1.0)
        kg_learnable = kg_cfg.get("learnable", False)
        kg_activation_epoch = max(1, int(kg_cfg.get("activate_epoch", 1)))
        supcon_criterion = KnowledgeGuidedSupConLoss(
            temperature=supcon_temp,
            similarity_matrix=kg_sim_matrix,
            kg_weight=kg_weight,
            learnable=kg_learnable,
        )
        mode_str = "learnable" if kg_learnable else "fixed"
        print(f"  Using KnowledgeGuidedSupConLoss (kg_weight={kg_weight}, {mode_str})")
        if kg_learnable:
            init_sims = supcon_criterion.get_similarity_dict()
            print(f"  KG init: {init_sims}")
            if kg_activation_epoch > 1:
                kg_delayed_activation = True
                zero_supcon_criterion = KnowledgeGuidedSupConLoss(
                    temperature=supcon_temp,
                    similarity_matrix=torch.zeros_like(kg_sim_matrix),
                    kg_weight=kg_weight,
                    learnable=False,
                )
                print(
                    "  KG schedule: epochs 1-"
                    f"{kg_activation_epoch - 1} use fixed zero class similarity; "
                    f"learnable KG activates at epoch {kg_activation_epoch}."
                )
    else:
        supcon_criterion = HierarchicalSupConLoss(temperature=supcon_temp, voc_margin=voc_margin)
        print(f"  Using HierarchicalSupConLoss (voc_margin={voc_margin})")

    supcon_criterion = supcon_criterion.to(device)
    if zero_supcon_criterion is not None:
        zero_supcon_criterion = zero_supcon_criterion.to(device)

    supcon_cfg = {**cfg, "learning_rate": supcon_lr}
    supcon_param_groups = build_optimizer_param_groups(model, supcon_cfg)

    supcon_optimizer = optim.AdamW(
        supcon_param_groups,
        lr=supcon_lr,
        weight_decay=cfg["weight_decay"],
        fused=torch.cuda.is_available(),
    )

    kg_optimizer = None
    kg_supcon_iter = None
    kg_anchor_strength = kg_cfg.get("kg_anchor_strength", 0.0)
    if kg_learnable:
        kg_lr = supcon_lr * kg_cfg.get("kg_lr_multiplier", 0.5)
        kg_optimizer = optim.Adam(
            supcon_criterion.parameters(),
            lr=kg_lr,
        )
        print(f"  KG bilevel optimizer (lr={kg_lr:.6f}, anchor={kg_anchor_strength}, updated on train data)")

        print("Building Phase 1 (SupCon) train loader for bilevel KG:")
        kg_supcon_sampler = build_balanced_sampler(train_df, hierarchical=True)

        if image_cache is not None and "images" in image_cache:
            kg_supcon_loader = VRAMDataLoader(
                train_df,
                image_cache,
                batch_size=supcon_bs,
                sampler=kg_supcon_sampler,
                shuffle=False,
                roi_mix_prob=roi_mix_prob,
            )
        else:
            kg_supcon_dataset = LaryngealDataset(train_df, eval_tf, cfg, image_cache=image_cache)
            kg_supcon_loader = DataLoader(
                kg_supcon_dataset,
                batch_size=supcon_bs,
                sampler=kg_supcon_sampler,
                **loader_common,
            )
        kg_supcon_iter = CyclingDataLoaderIter(kg_supcon_loader)

    supcon_monitor = cfg.get("supcon_monitor")
    if supcon_monitor is None:
        supcon_monitor = "kg_loss" if kg_learnable else "loss"
    if supcon_monitor == "val_loss":
        print("  supcon_monitor='val_loss' is deprecated; KG updates now use train data, so using 'kg_loss'.")
        supcon_monitor = "kg_loss" if kg_learnable else "loss"
    if supcon_monitor == "kg_loss" and not kg_learnable:
        print("  supcon_monitor='kg_loss' requested without learnable KG; falling back to train loss.")
        supcon_monitor = "loss"
    if supcon_monitor not in {"loss", "kg_loss"}:
        raise ValueError("supcon_monitor must be either 'loss' or 'kg_loss'.")
    print(f"  Phase 1 checkpoint monitor: {supcon_monitor} (min_delta={supcon_min_delta})")

    supcon_min_lr = cfg.get("supcon_min_lr", supcon_lr * 0.1)
    supcon_scheduler = WarmupCosineScheduler(
        supcon_optimizer,
        warmup_epochs=supcon_warmup,
        total_epochs=supcon_epochs,
        warmup_lr=1e-6,
        min_lr=supcon_min_lr,
    )
    supcon_scaler = torch.amp.GradScaler(device.type)

    tb_log_dir = os.path.join(RESULTS_DIR, "logs_phase1")
    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # ── Training loop ────────────────────────────────────────────────────
    best_supcon_metric = float("inf")
    best_supcon_epoch = 0
    best_supcon_state = None
    best_kg_state = None
    supcon_epochs_no_improve = 0
    actual_supcon_epochs = 0
    history = []

    for epoch in range(1, supcon_epochs + 1):
        kg_loss = None
        kg_update_active = kg_learnable and (not kg_delayed_activation or epoch >= kg_activation_epoch)
        epoch_criterion = supcon_criterion
        if kg_learnable and not kg_update_active and zero_supcon_criterion is not None:
            epoch_criterion = zero_supcon_criterion
        if kg_delayed_activation and kg_update_active and epoch == kg_activation_epoch:
            print(f"  Activating learnable KG similarity at epoch {epoch}.")

        if kg_learnable and kg_update_active:
            train_loss, kg_loss = supcon_train_one_epoch_bilevel(
                model, supcon_loader, kg_supcon_iter,
                supcon_optimizer, kg_optimizer,
                epoch_criterion, supcon_scaler, device, grad_accum,
                kg_anchor_strength=kg_anchor_strength,
                gpu_aug=gpu_aug,
            )
            loss = train_loss
            writer.add_scalar("SupCon/kg_loss", kg_loss, epoch)
        else:
            loss = supcon_train_one_epoch(
                model, supcon_loader, supcon_optimizer,
                epoch_criterion, supcon_scaler, device, grad_accum,
                gpu_aug=gpu_aug,
            )

        supcon_scheduler.step()
        current_lr = supcon_optimizer.param_groups[0]["lr"]
        writer.add_scalar("SupCon/loss", loss, epoch)
        writer.add_scalar("SupCon/lr", current_lr, epoch)

        kg_sims = {}
        if kg_learnable:
            kg_sims = epoch_criterion.get_similarity_dict()
            for pair_name, sim_val in kg_sims.items():
                writer.add_scalar(f"KG_Similarity/{pair_name}", sim_val, epoch)

        actual_supcon_epochs = epoch

        history_entry = {
            "epoch": epoch,
            "loss": loss,
            "lr": current_lr,
            "kg_update_active": bool(kg_update_active),
            "kg_activation_epoch": kg_activation_epoch if kg_learnable else None,
        }
        if kg_sims:
            history_entry["kg_similarity"] = kg_sims
        if kg_learnable:
            history_entry["kg_loss"] = kg_loss
        monitor_ready = not (kg_learnable and kg_delayed_activation and not kg_update_active)
        monitor_loss = kg_loss if supcon_monitor == "kg_loss" and kg_loss is not None else loss
        history_entry["monitor"] = supcon_monitor if monitor_ready else "zero_similarity_warmup"
        history_entry["monitor_loss"] = monitor_loss
        history_entry["early_stopping_active"] = bool(monitor_ready)
        history.append(history_entry)
        writer.add_scalar("SupCon/monitor_loss", monitor_loss, epoch)

        if monitor_ready and monitor_loss < best_supcon_metric - supcon_min_delta:
            best_supcon_metric = monitor_loss
            best_supcon_epoch = epoch
            best_supcon_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if kg_learnable:
                best_kg_state = {
                    key: value.detach().cpu().clone()
                    for key, value in supcon_criterion.state_dict().items()
                }
            supcon_epochs_no_improve = 0
        elif monitor_ready:
            supcon_epochs_no_improve += 1
        else:
            supcon_epochs_no_improve = 0

        if epoch % 5 == 0 or epoch == 1:
            best_text = (
                f"{best_supcon_metric:.4f}"
                if best_supcon_state is not None
                else "pending KG activation"
            )
            line = (f"  SupCon Epoch {epoch}/{supcon_epochs} — Loss: {loss:.4f}, "
                    f"LR: {current_lr:.6f}, Best {supcon_monitor}: {best_text}, "
                    f"No improve: {supcon_epochs_no_improve}/{supcon_patience}")
            if kg_learnable and kg_update_active and kg_loss is not None:
                line += f", KGLoss: {kg_loss:.4f}"
            elif kg_learnable:
                line += ", KG: zero-sim frozen"
            if kg_sims:
                sim_str = ", ".join(f"{k}={v:.3f}" for k, v in kg_sims.items())
                line += f"\n    KG: [{sim_str}]"
            print(line)

        gc.collect()
        torch.cuda.empty_cache()

        if monitor_ready and supcon_epochs_no_improve >= supcon_patience:
            print(f"  SupCon early stopping at epoch {epoch} "
                  f"(no improvement for {supcon_patience} epochs, "
                  f"best {supcon_monitor}: {best_supcon_metric:.4f})")
            break

    writer.flush()
    writer.close()

    # ── Save checkpoint & history ────────────────────────────────────────
    if best_supcon_state is not None:
        model.load_state_dict(best_supcon_state)
        if best_kg_state is not None:
            supcon_criterion.load_state_dict(best_kg_state)
        print(f"Restored best Phase 1 state from epoch {best_supcon_epoch} before saving.")

    torch.save(model.state_dict(), PHASE1_CHECKPOINT_PATH)
    print(f"Phase 1 checkpoint saved: {PHASE1_CHECKPOINT_PATH}")

    with open(PHASE1_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Phase 1 history saved: {PHASE1_HISTORY_PATH}")

    if kg_learnable:
        final_sims = supcon_criterion.get_similarity_dict()
        print(f"\nLearned KG similarities: {final_sims}")

    best_summary = (
        f"{best_supcon_metric:.4f} at epoch {best_supcon_epoch}"
        if best_supcon_state is not None
        else "not selected"
    )
    print(f"\nPhase 1 complete. Ran {actual_supcon_epochs}/{supcon_epochs} epochs, "
          f"best {supcon_monitor}: {best_summary}")
    print(f"TensorBoard: tensorboard --logdir {tb_log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
