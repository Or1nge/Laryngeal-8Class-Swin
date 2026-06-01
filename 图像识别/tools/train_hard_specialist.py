#!/usr/bin/env python3
"""Train a small hard-class specialist from an existing 8-class checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import (  # noqa: E402
    BEST_MODEL_PATH,
    DISPLAY_NAMES,
    GPUAugment,
    HierarchicalImageClassifier,
    LABEL_DICT,
    VRAMDataLoader,
    WarmupCosineScheduler,
    build_optimizer_param_groups,
    create_classification_metrics,
    discover_images,
    evaluate,
    gpu_normalise,
    init_label_mapping,
    load_config,
    load_dataset_split,
    preload_image_cache,
    seed_everything,
    setup_device,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", nargs="+", required=True, help="DISPLAY_NAMES to keep, e.g. Reinke-Edema Vocal-Cord-Cyst Vocal-Cord-Polyp")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config_phase2.json")
    parser.add_argument("--base-model", type=Path, default=Path(BEST_MODEL_PATH))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--balance-alpha", type=float, default=1.0)
    parser.add_argument("--reuse-classifier-hidden", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def remap_split(df: pd.DataFrame, original_to_specialist: dict[int, int]) -> pd.DataFrame:
    out = df[df["label"].isin(original_to_specialist)].copy().reset_index(drop=True)
    out["original_label"] = out["label"]
    out["label"] = out["label"].map(original_to_specialist).astype(int)
    return out


def make_weighted_sampler(labels: np.ndarray, balance_alpha: float) -> torch.utils.data.WeightedRandomSampler:
    counts = pd.Series(labels).value_counts().to_dict()
    weights = torch.tensor(
        [1.0 / (counts[int(label)] ** balance_alpha) for label in labels],
        dtype=torch.float64,
    )
    return torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(labels), replacement=True)


def replace_classifier(model: HierarchicalImageClassifier, num_classes: int, reuse_hidden: bool) -> None:
    if reuse_hidden and isinstance(model.classifier, nn.Sequential) and len(model.classifier) >= 5:
        hidden_dim = model.classifier[-1].in_features
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1], nn.Linear(hidden_dim, num_classes))
        return
    hidden_dim = model.classifier[1].out_features
    dropout_feature = model.classifier[0].p
    dropout_classifier = model.classifier[3].p
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_feature),
        nn.Linear(model.feature_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout_classifier),
        nn.Linear(hidden_dim, num_classes),
    )


@torch.inference_mode()
def collect_predictions(model, loader, df: pd.DataFrame, device, class_names: list[str]) -> pd.DataFrame:
    model.eval()
    rows = []
    for images, labels, _is_voc in loader:
        images = gpu_normalise(images)
        labels = labels.to(device)
        with torch.amp.autocast(device_type=device.type):
            logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        for label, pred, prob_row in zip(labels.cpu().numpy(), preds.cpu().numpy(), probs.cpu().numpy()):
            rows.append(
                {
                    "true_label": class_names[int(label)],
                    "pred_label": class_names[int(pred)],
                    "confidence": float(prob_row[int(pred)]),
                    **{f"prob_{name}": float(prob_row[idx]) for idx, name in enumerate(class_names)},
                }
            )
    pred_df = df[["image_path", "patient_name", "source_folder", "original_label"]].copy().reset_index(drop=True)
    pred_df["original_label_name"] = pred_df["original_label"].map(DISPLAY_NAMES)
    return pd.concat([pred_df, pd.DataFrame(rows)], axis=1)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(args.config))
    cfg["learning_rate"] = args.learning_rate
    cfg["min_lr"] = args.min_lr
    cfg["epochs"] = args.epochs
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay
    init_label_mapping(cfg)
    seed_everything(cfg["seed"])
    device = setup_device()

    name_to_label = {name: idx for idx, name in DISPLAY_NAMES.items()}
    missing = [name for name in args.classes if name not in name_to_label]
    if missing:
        raise ValueError(f"Unknown class names: {missing}")
    original_labels = [name_to_label[name] for name in args.classes]
    original_to_specialist = {label: idx for idx, label in enumerate(original_labels)}
    class_names = list(args.classes)

    df = discover_images()
    train_df, val_df, test_df = load_dataset_split(df)
    train_df = remap_split(train_df, original_to_specialist)
    val_df = remap_split(val_df, original_to_specialist)
    test_df = remap_split(test_df, original_to_specialist)
    print(f"Specialist classes: {class_names}")
    print(f"Rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    image_cache = preload_image_cache(train_df, val_df, test_df, cfg=cfg, device=device)
    sampler = make_weighted_sampler(train_df["label"].to_numpy(), args.balance_alpha)
    train_loader = VRAMDataLoader(train_df, image_cache, cfg["batch_size"], sampler=sampler, shuffle=False)
    train_eval_loader = VRAMDataLoader(train_df, image_cache, cfg["eval_batch_size"], shuffle=False)
    val_loader = VRAMDataLoader(val_df, image_cache, cfg["eval_batch_size"], shuffle=False)
    test_loader = VRAMDataLoader(test_df, image_cache, cfg["eval_batch_size"], shuffle=False)

    model = HierarchicalImageClassifier(num_classes=len(LABEL_DICT), cfg=cfg).to(device)
    state_dict = torch.load(args.base_model, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    replace_classifier(model, len(class_names), args.reuse_classifier_hidden)
    model.to(device)
    for param in model.projector.parameters():
        param.requires_grad = False
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters trainable: {trainable:,}/{total:,} ({100 * trainable / total:.1f}%)")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    optimizer = optim.AdamW(
        build_optimizer_param_groups(model, cfg),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        fused=torch.cuda.is_available(),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg["warmup_epochs"],
        total_epochs=args.epochs,
        warmup_lr=1e-6,
        min_lr=args.min_lr,
    )
    scaler = torch.amp.GradScaler(device.type)
    gpu_aug = GPUAugment(cfg).to(device)
    cls_metrics = create_classification_metrics(len(class_names), device)

    best_score = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1, train_acc, train_auc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            cfg["grad_accum"],
            num_classes=len(class_names),
            cfg=cfg,
            cls_metrics=cls_metrics,
            gpu_aug=gpu_aug,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, len(class_names), cls_metrics=cls_metrics)
        scheduler.step()
        score = val_metrics["f1"]
        improved = score > best_score + cfg.get("early_stopping_min_delta", 0.0)
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save(best_state, output_dir / "best_model.pth")
        else:
            no_improve += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_f1": train_f1,
                "train_acc": train_acc,
                "train_auc": train_auc,
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["f1"],
                "val_acc": val_metrics["acc"],
                "val_auc": val_metrics["auc"],
                "best_val_f1": best_score,
                "improved": improved,
            }
        )
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch}/{args.epochs} "
                f"train_f1={train_f1:.4f} val_f1={val_metrics['f1']:.4f} "
                f"val_acc={val_metrics['acc']:.4f} val_auc={val_metrics['auc']:.4f} "
                f"best={best_score:.4f}@{best_epoch} no_improve={no_improve}/{args.patience}"
            )
        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("No specialist checkpoint was saved.")
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    loaders = {"train": train_eval_loader, "val": val_loader, "test": test_loader}
    metrics = {}
    for split, loader in loaders.items():
        metrics[split] = evaluate(model, loader, criterion, device, len(class_names), cls_metrics=cls_metrics)
        pred_df = collect_predictions(
            model,
            loader,
            {"train": train_df, "val": val_df, "test": test_df}[split],
            device,
            class_names,
        )
        pred_df.to_csv(output_dir / f"predictions_{split}.csv", index=False)
        report = classification_report(
            pred_df["true_label"],
            pred_df["pred_label"],
            labels=class_names,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).T.to_csv(output_dir / f"classification_report_{split}.csv")
        cm = confusion_matrix(pred_df["true_label"], pred_df["pred_label"], labels=class_names)
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / f"confusion_matrix_{split}.csv")

    summary = {
        "classes": class_names,
        "original_labels": original_labels,
        "base_model": str(args.base_model.resolve()),
        "config": cfg,
        "best_epoch": best_epoch,
        "best_val_f1": best_score,
        "metrics": metrics,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print(f"Saved specialist outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
