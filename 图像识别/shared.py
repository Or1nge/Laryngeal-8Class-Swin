"""Shared utilities for Phase 1 (SupCon) and Phase 2 (CE) training scripts."""

import json
import math
import os
import random
import threading
from collections import Counter
from contextlib import nullcontext

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchmetrics
from sklearn.metrics import classification_report

from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

torch.set_num_threads(torch.get_num_threads())
torch.set_num_interop_threads(torch.get_num_interop_threads())

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)


def _detect_workspace_dir(base_dir):
    """Return the no-space workspace root when running from worktrees/<name>."""
    parent = os.path.dirname(base_dir)
    if os.path.basename(parent) == "worktrees":
        return os.path.dirname(parent)
    grandparent = os.path.dirname(parent)
    if os.path.basename(grandparent) == "worktrees":
        return os.path.dirname(grandparent)
    return base_dir


def _detect_worktree_name(base_dir):
    parent = os.path.dirname(base_dir)
    if os.path.basename(parent) == "worktrees":
        return os.path.basename(base_dir)
    grandparent = os.path.dirname(parent)
    if os.path.basename(grandparent) == "worktrees":
        return os.path.basename(parent)
    return os.path.basename(base_dir)


def _first_existing_dir(*paths):
    for path in paths:
        if os.path.isdir(path):
            return path
    return paths[0]


WORKSPACE_DIR = os.environ.get("LARYNX_WORKSPACE_DIR", _detect_workspace_dir(BASE_DIR))
WORKTREE_NAME = os.environ.get("LARYNX_WORKTREE_NAME", _detect_worktree_name(BASE_DIR))
WORKSPACE_PARENT_DIR = os.path.dirname(WORKSPACE_DIR)

IMAGE_DIR = os.environ.get(
    "LARYNX_IMAGE_DIR",
    _first_existing_dir(
        os.path.join(WORKSPACE_DIR, "Laryngeal_Dataset_Processed"),
        os.path.join(WORKSPACE_PARENT_DIR, "Laryngeal_Dataset_Processed"),
        os.path.join(PARENT_DIR, "Laryngeal_Dataset_Processed"),
    ),
)
RESULTS_ROOT = os.environ.get("LARYNX_RESULTS_ROOT", os.path.join(WORKSPACE_DIR, "Results"))
RESULTS_DIR = os.environ.get("LARYNX_RESULTS_DIR", os.path.join(RESULTS_ROOT, WORKTREE_NAME))
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")
PHASE2_BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "phase2_best_model.pth")
PHASE2_FINAL_METRICS_PATH = os.path.join(RESULTS_DIR, "phase2_final_metrics.json")
PHASE3_CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "phase3_checkpoint.pth")
PHASE3_HISTORY_PATH = os.path.join(RESULTS_DIR, "phase3_history.json")
PHASE3_CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "phase3_train_confusion_matrix.csv")
PHASE3_CONFUSION_PAIRS_PATH = os.path.join(RESULTS_DIR, "phase3_train_confusion_pairs.csv")
PHASE3_MISCLASSIFIED_PATH = os.path.join(RESULTS_DIR, "phase3_train_misclassified_samples.csv")
PHASE4_BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "phase4_best_model.pth")
PHASE4_HISTORY_CSV_PATH = os.path.join(RESULTS_DIR, "phase4_history.csv")
ONNX_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.onnx")
HISTORY_CSV_PATH = os.path.join(RESULTS_DIR, "history.csv")
METRICS_CSV_PATH = os.path.join(RESULTS_DIR, "metrics.csv")
TRAINING_CURVE_PATH = os.path.join(RESULTS_DIR, "training_curves.png")
ATTENTION_MAP_PATH = os.path.join(RESULTS_DIR, "gradcam_maps.png")
PHASE1_CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "phase1_checkpoint.pth")
PHASE1_HISTORY_PATH = os.path.join(RESULTS_DIR, "phase1_history.json")
DATASET_SPLIT_PATH = os.path.join(BASE_DIR, "dataset_split.json")

LOCAL_WEIGHT_CANDIDATES = [
    os.path.join(WORKSPACE_PARENT_DIR, "pretrained_weights",
                 "swin_base_patch4_window7_224.ms_in22k_ft_in1k.safetensors"),
    os.path.join(WORKSPACE_DIR, "pretrained_weights",
                 "swin_base_patch4_window7_224.ms_in22k_ft_in1k.safetensors"),
    os.path.join("/home/or1ngelinux/CVProjects/pretrained_weights",
                 "swin_base_patch4_window7_224.ms_in22k_ft_in1k.safetensors"),
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_project_path(path):
    """Resolve legacy relative paths against the current workspace layout."""
    if not path:
        return path
    path = os.fspath(path)
    if os.path.isabs(path):
        return path
    normalized = path.replace("\\", "/")
    if normalized == "best_model.pth":
        return BEST_MODEL_PATH
    if normalized.startswith("Results/"):
        return os.path.join(RESULTS_DIR, normalized[len("Results/"):])
    return os.path.join(BASE_DIR, path)

# ── Hierarchical Label Mapping ───────────────────────────────────────────────
# Use mutable containers so that `from shared import LABEL_DICT` etc.
# see updates after init_label_mapping() mutates them in-place.
CLASS_TO_FOLDERS = {}
LABEL_DICT = {}
LABEL_NAMES = {}
DISPLAY_NAMES = {}
VOC_LABELS = set()
NON_VOC_LABEL = None
FOLDER_TO_LABEL = {}
DEFAULT_NON_VOC_CLASS = "Non-Vocal-Cord"

DEFAULT_CLASS_FOLDERS = {
    "Non-Vocal-Cord": ["混杂图片"],
    "Normal": ["正常"],
    "Reinke-Edema": ["声带任克水肿"],
    "Vocal-Cord-Cyst": ["声带囊肿"],
    "Vocal-Cord-Polyp": ["声带息肉"],
    "Vocal-Cord-Leukoplakia": ["声带白斑"],
    "Vocal-Cord-Granuloma": ["声带肉芽肿"],
    "Cancer": ["喉癌"],
}


def init_label_mapping(cfg):
    """Initialize label mapping from config."""
    global NON_VOC_LABEL

    configured_folders = cfg.get("class_folders", DEFAULT_CLASS_FOLDERS)
    non_voc_class = cfg.get("non_voc_class", DEFAULT_NON_VOC_CLASS)
    if non_voc_class not in configured_folders:
        raise KeyError(
            f"non_voc_class='{non_voc_class}' is not present in class_folders. "
            f"Available classes: {list(configured_folders.keys())}"
        )

    CLASS_TO_FOLDERS.clear()
    CLASS_TO_FOLDERS.update(configured_folders)

    LABEL_DICT.clear()
    LABEL_DICT.update({name: idx for idx, name in enumerate(CLASS_TO_FOLDERS.keys())})

    LABEL_NAMES.clear()
    LABEL_NAMES.update({idx: name for name, idx in LABEL_DICT.items()})

    display_cfg = cfg.get("class_display_names", {})
    DISPLAY_NAMES.clear()
    DISPLAY_NAMES.update({
        idx: display_cfg.get(name, name)
        for name, idx in LABEL_DICT.items()
    })

    NON_VOC_LABEL = LABEL_DICT[non_voc_class]
    VOC_LABELS.clear()
    VOC_LABELS.update({idx for name, idx in LABEL_DICT.items() if name != non_voc_class})

    FOLDER_TO_LABEL.clear()
    seen_folders = {}
    for class_name, folders in CLASS_TO_FOLDERS.items():
        for folder in folders:
            if folder in seen_folders:
                raise ValueError(
                    f"Folder '{folder}' is mapped to both '{seen_folders[folder]}' "
                    f"and '{class_name}'. Each source folder must belong to one class."
                )
            seen_folders[folder] = class_name
            FOLDER_TO_LABEL[folder] = LABEL_DICT[class_name]


def is_voc_label(label):
    return label in VOC_LABELS


def get_voc_label_indices():
    return sorted(VOC_LABELS)


# ── config ───────────────────────────────────────────────────────────────────
def _read_split_class_folders(split_path=DATASET_SPLIT_PATH):
    if not os.path.exists(split_path):
        return None

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    split_class_folders = split.get("class_folders")
    if split_class_folders is None:
        return None
    if not isinstance(split_class_folders, dict) or not split_class_folders:
        raise ValueError(
            f"dataset split at {split_path} has an invalid class_folders field."
        )
    return split_class_folders


def _sync_config_with_dataset_split(cfg, split_path=DATASET_SPLIT_PATH):
    split_class_folders = _read_split_class_folders(split_path)
    if split_class_folders is None:
        return cfg

    requested_exclusions = cfg.get("excluded_classes_from_split", [])
    if isinstance(requested_exclusions, str):
        requested_exclusions = [requested_exclusions]
    requested_exclusions = list(dict.fromkeys(requested_exclusions))

    non_voc_class = cfg.get("non_voc_class", DEFAULT_NON_VOC_CLASS)
    if non_voc_class not in split_class_folders:
        raise KeyError(
            f"non_voc_class='{non_voc_class}' is not present in {split_path} class_folders. "
            f"Available classes: {list(split_class_folders.keys())}"
        )
    if non_voc_class in requested_exclusions:
        raise ValueError(
            f"non_voc_class='{non_voc_class}' cannot be listed in excluded_classes_from_split."
        )

    absent_exclusions = [
        name for name in requested_exclusions if name not in split_class_folders
    ]
    if absent_exclusions:
        print(
            f"Note: excluded_classes_from_split entries are already absent from "
            f"{split_path}: {absent_exclusions}"
        )

    active_exclusions = [
        name for name in requested_exclusions if name in split_class_folders
    ]
    if active_exclusions:
        split_class_folders = {
            name: folders
            for name, folders in split_class_folders.items()
            if name not in active_exclusions
        }
        if not split_class_folders:
            raise ValueError("excluded_classes_from_split removed every class.")
        print(f"Excluding classes from active training split: {active_exclusions}")

    configured_class_folders = cfg.get("class_folders")
    if configured_class_folders != split_class_folders:
        configured_classes = (
            list(configured_class_folders.keys())
            if isinstance(configured_class_folders, dict)
            else []
        )
        split_classes = list(split_class_folders.keys())
        dropped = [name for name in configured_classes if name not in split_class_folders]
        added = [name for name in split_classes if name not in configured_classes]
        notes = []
        if dropped:
            notes.append(f"dropped from config: {dropped}")
        if added:
            notes.append(f"added from split: {added}")
        suffix = f" ({'; '.join(notes)})" if notes else ""
        print(
            f"Using class_folders from {split_path} "
            f"({len(split_class_folders)} classes){suffix}."
        )

    cfg["class_folders"] = split_class_folders
    cfg["excluded_classes_from_split"] = requested_exclusions
    return cfg


def load_config(config_path=None, sync_with_split=True):
    defaults = {
        "seed": 42,
        "image_size": 384,
        "resize_size": 400,
        "batch_size": 32,
        "eval_batch_size": 48,
        "grad_accum": 2,
        "epochs": 100,
        "warmup_epochs": 5,
        "learning_rate": 3e-5,
        "weight_decay": 0.05,
        "dropout_rate": 0.3,
        "drop_path_rate": 0.15,
        "label_smoothing": 0.1,
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 0.0,
        "selection_f1_weight": 0.5,
        "selection_auc_weight": 0.5,
        "sampler_balance_alpha": 1.0,
        "excluded_classes_from_split": [],
        "num_workers": 7,
        "prefetch_factor": 4,
        "persistent_workers": True,
        "gpu_augment_enabled": True,
        "unfreeze_last_n_blocks": 4,
        "unfreeze_blocks": None,
        "layer_decay": 0.65,
        "classifier_hidden_dim": 128,
        "crop_black_threshold": 15,
        "color_jitter_brightness": 0.2,
        "color_jitter_contrast": 0.2,
        "color_jitter_saturation": 0.1,
        "color_jitter_hue": 0.02,
        "random_resized_crop_scale_min": 0.8,
        "random_resized_crop_scale_max": 1.0,
        "random_affine_degrees": 15,
        "random_affine_translate": [0.1, 0.1],
        "random_affine_scale": [0.9, 1.1],
        "random_horizontal_flip_prob": 0.5,
        "random_vertical_flip_prob": 0.5,
        "gaussian_blur_prob": 0.5,
        "gaussian_blur_sigma_max": 5.0,
        "random_adjust_sharpness_prob": 0.5,
        "random_adjust_sharpness_factor": 2.0,
        "supcon_enabled": True,
        "supcon_batch_size": 64,
        "supcon_epochs": 50,
        "supcon_early_stopping_patience": 10,
        "supcon_early_stopping_min_delta": 0.0,
        "supcon_monitor": None,
        "supcon_learning_rate": 5e-4,
        "supcon_temperature": 0.1,
        "supcon_projection_dim": 128,
        "supcon_warmup_epochs": 3,
        "supcon_voc_margin": 0.3,
    }
    if config_path is None:
        config_path = os.path.join(BASE_DIR, "config_phase2.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        defaults.update(user_cfg)
        print(f"Loaded config from {config_path}")
    else:
        print(f"Config not found at {config_path}, using defaults")
    if sync_with_split:
        _sync_config_with_dataset_split(defaults)
    return defaults


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    return device


# ── helpers ──────────────────────────────────────────────────────────────────
class CropBlackBorders:
    def __init__(self, threshold=15):
        self.threshold = threshold

    def __call__(self, img):
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np
        mask = gray > self.threshold
        coords = np.argwhere(mask)
        if coords.size == 0:
            return img
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return Image.fromarray(img_np[y0:y1, x0:x1])


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _as_float_image_batch(x):
    """Return batched CHW images as float tensors in [0, 1]."""
    if torch.is_floating_point(x):
        return x
    return x.to(dtype=torch.float32).div_(255.0)


_IMAGENET_STATS_CACHE = {}


def _imagenet_stats_for(x):
    key = (x.device.type, x.device.index, x.dtype)
    stats = _IMAGENET_STATS_CACHE.get(key)
    if stats is None:
        stats = (
            _IMAGENET_MEAN.to(device=x.device, dtype=x.dtype),
            _IMAGENET_STD.to(device=x.device, dtype=x.dtype),
        )
        _IMAGENET_STATS_CACHE[key] = stats
    return stats


def gpu_normalise(x):
    """Normalize a batched image tensor with ImageNet statistics."""
    x = _as_float_image_batch(x)
    mean, std = _imagenet_stats_for(x)
    return (x - mean) / std


class GPUAugment(nn.Module):
    """Batched tensor augmentation used after DataLoader prefetch.

    Input tensors are unnormalized images in CHW format, either uint8 [0, 255]
    or float [0, 1]. Output tensors are ImageNet-normalized and ready for the
    Swin backbone.
    """

    def __init__(self, cfg):
        super().__init__()
        self.enabled = cfg.get("gpu_augment_enabled", True)
        self.img_size = cfg["image_size"]
        self.crop_scale_min = cfg.get("random_resized_crop_scale_min", 0.8)
        self.crop_scale_max = cfg.get("random_resized_crop_scale_max", 1.0)
        self.crop_ratio_min = cfg.get("random_resized_crop_ratio_min", 0.9)
        self.crop_ratio_max = cfg.get("random_resized_crop_ratio_max", 1.1)
        self.hflip_p = cfg.get("random_horizontal_flip_prob", 0.5)
        self.vflip_p = cfg.get("random_vertical_flip_prob", 0.5)
        self.affine_degrees = cfg.get("random_affine_degrees", 15)
        self.affine_translate = tuple(cfg.get("random_affine_translate", [0.1, 0.1]))
        self.affine_scale = tuple(cfg.get("random_affine_scale", [0.9, 1.1]))
        self.cj_brightness = cfg.get("color_jitter_brightness", 0.2)
        self.cj_contrast = cfg.get("color_jitter_contrast", 0.2)
        self.cj_saturation = cfg.get("color_jitter_saturation", 0.1)
        self.cj_hue = cfg.get("color_jitter_hue", 0.02)
        self.blur_prob = cfg.get("gaussian_blur_prob", 0.5)
        self.blur_kernel = int(cfg.get("gaussian_blur_kernel", 7))
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1
        self.blur_sigma_min = cfg.get("gaussian_blur_sigma_min", 0.1)
        self.blur_sigma_max = cfg.get("gaussian_blur_sigma_max", 5.0)
        self.sharpness_prob = cfg.get("random_adjust_sharpness_prob", 0.5)
        self.sharpness_factor = cfg.get("random_adjust_sharpness_factor", 2.0)
        self.register_buffer(
            "_blur_coords",
            torch.arange(self.blur_kernel, dtype=torch.float32) - self.blur_kernel // 2,
        )

    @torch.no_grad()
    def _random_resized_crop(self, x):
        bsz, channels, height, width = x.shape
        log_ratio_min = math.log(self.crop_ratio_min)
        log_ratio_max = math.log(self.crop_ratio_max)

        scales = torch.empty(bsz, device=x.device).uniform_(
            self.crop_scale_min, self.crop_scale_max
        )
        log_ratios = torch.empty(bsz, device=x.device).uniform_(log_ratio_min, log_ratio_max)
        ratios = log_ratios.exp()

        crop_h = (height * scales.sqrt() / ratios.sqrt()).clamp(1, height).int()
        crop_w = (width * scales.sqrt() * ratios.sqrt()).clamp(1, width).int()

        max_y = (height - crop_h).clamp(min=0).float()
        max_x = (width - crop_w).clamp(min=0).float()
        top = (torch.rand(bsz, device=x.device) * (max_y + 1)).int()
        left = (torch.rand(bsz, device=x.device) * (max_x + 1)).int()

        crop_h_f = crop_h.float()
        crop_w_f = crop_w.float()
        top_f = top.float()
        left_f = left.float()

        theta = torch.zeros(bsz, 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = crop_w_f / width
        theta[:, 1, 1] = crop_h_f / height
        theta[:, 0, 2] = (2 * left_f + crop_w_f) / width - 1.0
        theta[:, 1, 2] = (2 * top_f + crop_h_f) / height - 1.0

        grid = F.affine_grid(
            theta, [bsz, channels, self.img_size, self.img_size], align_corners=False
        )
        return F.grid_sample(
            x, grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )

    @torch.no_grad()
    def _random_flip(self, x):
        bsz = x.size(0)
        if self.hflip_p > 0:
            mask = torch.rand(bsz, device=x.device) < self.hflip_p
            if mask.any():
                x[mask] = x[mask].flip(-1)
        if self.vflip_p > 0:
            mask = torch.rand(bsz, device=x.device) < self.vflip_p
            if mask.any():
                x[mask] = x[mask].flip(-2)
        return x

    @torch.no_grad()
    def _random_affine(self, x):
        if self.affine_degrees <= 0 and self.affine_translate == (0, 0) and self.affine_scale == (1, 1):
            return x

        bsz = x.size(0)
        angles = torch.empty(bsz, device=x.device).uniform_(
            -self.affine_degrees, self.affine_degrees
        )
        rad = angles * (math.pi / 180.0)
        cos_a = rad.cos()
        sin_a = rad.sin()
        scale = torch.empty(bsz, device=x.device).uniform_(*self.affine_scale)
        tx = torch.empty(bsz, device=x.device).uniform_(
            -self.affine_translate[0], self.affine_translate[0]
        )
        ty = torch.empty(bsz, device=x.device).uniform_(
            -self.affine_translate[1], self.affine_translate[1]
        )

        theta = torch.zeros(bsz, 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = cos_a * scale
        theta[:, 0, 1] = sin_a * scale
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = -sin_a * scale
        theta[:, 1, 1] = cos_a * scale
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(
            x, grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )

    @torch.no_grad()
    def _color_jitter(self, x):
        bsz = x.size(0)
        if self.cj_brightness > 0:
            factor = torch.empty(bsz, 1, 1, 1, device=x.device).uniform_(
                1 - self.cj_brightness, 1 + self.cj_brightness
            )
            x = x * factor
        if self.cj_contrast > 0:
            factor = torch.empty(bsz, 1, 1, 1, device=x.device).uniform_(
                1 - self.cj_contrast, 1 + self.cj_contrast
            )
            gray_mean = x.mean(dim=[1, 2, 3], keepdim=True)
            x = gray_mean + factor * (x - gray_mean)
        if self.cj_saturation > 0:
            factor = torch.empty(bsz, 1, 1, 1, device=x.device).uniform_(
                1 - self.cj_saturation, 1 + self.cj_saturation
            )
            gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
            x = gray + factor * (x - gray)
        if self.cj_hue > 0:
            angle = torch.empty(bsz, device=x.device).uniform_(
                -self.cj_hue, self.cj_hue
            ) * (2 * math.pi)
            cos_h = angle.cos().view(bsz, 1, 1, 1)
            sin_h = angle.sin().view(bsz, 1, 1, 1)
            gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
            color_axis = torch.stack(
                [x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]],
                dim=1,
            ) * 0.5
            x = gray + cos_h * (x - gray) + sin_h * color_axis
        return x.clamp(0.0, 1.0)

    @torch.no_grad()
    def _gaussian_blur(self, x):
        if self.blur_prob <= 0:
            return x
        bsz = x.size(0)
        mask = torch.rand(bsz, device=x.device) < self.blur_prob
        if not mask.any():
            return x

        subset = x[mask]
        n_subset = subset.size(0)
        sigma = torch.empty(n_subset, device=x.device).uniform_(
            self.blur_sigma_min, self.blur_sigma_max
        )
        coords = self._blur_coords.to(x.device)
        g1d = (-coords.unsqueeze(0) ** 2 / (2 * sigma.unsqueeze(1) ** 2)).exp()
        g1d = g1d / g1d.sum(dim=1, keepdim=True)

        kernel = self.blur_kernel
        pad = kernel // 2
        subset_padded = F.pad(subset, [pad, pad, 0, 0], mode="reflect")
        subset_flat = subset_padded.view(1, n_subset * 3, subset_padded.shape[2], -1)
        kernel_h = g1d.unsqueeze(1).unsqueeze(1).repeat(1, 3, 1, 1)
        kernel_h = kernel_h.view(n_subset * 3, 1, 1, kernel)
        blurred_h = F.conv2d(subset_flat, kernel_h, groups=n_subset * 3)
        blurred_h = blurred_h.view(n_subset, 3, subset_padded.shape[2], -1)

        blurred_h_padded = F.pad(blurred_h, [0, 0, pad, pad], mode="reflect")
        blurred_flat = blurred_h_padded.view(1, n_subset * 3, -1, blurred_h_padded.shape[3])
        kernel_v = g1d.unsqueeze(1).unsqueeze(-1).repeat(1, 3, 1, 1)
        kernel_v = kernel_v.view(n_subset * 3, 1, kernel, 1)
        blurred = F.conv2d(blurred_flat, kernel_v, groups=n_subset * 3)
        blurred = blurred.view(n_subset, 3, -1, blurred_h_padded.shape[3])

        x = x.clone()
        x[mask] = blurred
        return x

    @torch.no_grad()
    def _random_sharpness(self, x):
        if self.sharpness_prob <= 0 or self.sharpness_factor == 1.0:
            return x
        bsz = x.size(0)
        mask = torch.rand(bsz, device=x.device) < self.sharpness_prob
        if not mask.any():
            return x
        subset = x[mask]
        blurred = F.avg_pool2d(
            F.pad(subset, [1, 1, 1, 1], mode="reflect"),
            kernel_size=3,
            stride=1,
        )
        sharpened = subset + (self.sharpness_factor - 1.0) * (subset - blurred)
        x = x.clone()
        x[mask] = sharpened.clamp(0.0, 1.0)
        return x

    def forward(self, x):
        x = _as_float_image_batch(x)
        if self.enabled:
            x = self._random_resized_crop(x)
            x = self._random_flip(x)
            x = self._random_affine(x)
            x = self._color_jitter(x)
            x = self._gaussian_blur(x)
            x = self._random_sharpness(x)
        return gpu_normalise(x)


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_lr=1e-6, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(self.warmup_epochs, 1)
            return [self.warmup_lr + alpha * (base_lr - self.warmup_lr) for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        return [
            self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
            for base_lr in self.base_lrs
        ]


class HierarchicalSupConLoss(nn.Module):
    """Hierarchical Supervised Contrastive Learning Loss.

    Enforces structure in projection space:
    - Non-Vocal-Cord samples form one cluster
    - Vocal-Cord samples form another region
    - Within Vocal-Cord, separates all configured disease subclasses
    - VOC vs Non-VOC margin constraint: forces Non-VOC to stay away from VOC region
    """

    def __init__(self, temperature=0.1, voc_margin=0.3):
        super().__init__()
        self.temperature = temperature
        self.voc_margin = voc_margin

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_dot_contrast = torch.matmul(contrast_feature, contrast_feature.T) / self.temperature
        logits_max, _ = anchor_dot_contrast.max(dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device).repeat(1, 1)

        self_mask = torch.eye(batch_size, device=device)
        mask = mask - self_mask
        logits_mask = 1.0 - self_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        pos_count = mask.sum(dim=1).clamp(min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_count

        base_loss = -mean_log_prob_pos.mean()

        all_labels = labels.squeeze(-1)
        non_voc_mask = torch.eq(all_labels, NON_VOC_LABEL).float().unsqueeze(-1)
        voc_mask = 1.0 - non_voc_mask

        similarity_matrix = torch.matmul(contrast_feature, contrast_feature.T)

        margin_loss = torch.tensor(0.0, device=device)
        non_voc_indices = (all_labels == NON_VOC_LABEL).nonzero(as_tuple=True)[0]
        voc_indices = (all_labels != NON_VOC_LABEL).nonzero(as_tuple=True)[0]

        if len(non_voc_indices) > 0 and len(voc_indices) > 0:
            non_voc_to_voc_sim = similarity_matrix[non_voc_indices][:, voc_indices]
            margin_loss = torch.relu(self.voc_margin - non_voc_to_voc_sim).mean()

        return base_loss + margin_loss


def build_kg_similarity_matrix(cfg):
    """Build a [num_classes x num_classes] similarity matrix from knowledge_graph config.

    The matrix encodes inter-class soft similarity (0=unrelated, 1=identical).
    Diagonal is always 0 (same-class handled separately by hard positive mask).
    """
    kg_cfg = cfg.get("knowledge_graph", {})
    if not kg_cfg.get("enabled", False):
        return None

    class_sim = kg_cfg.get("class_similarity", {})
    excluded_classes = set(cfg.get("excluded_classes_from_split", []))
    num_classes = len(LABEL_DICT)
    sim_matrix = torch.zeros(num_classes, num_classes)

    class_names = list(LABEL_DICT.keys())
    for pair_key, sim_val in class_sim.items():
        matched = False
        for c1 in class_names:
            for c2 in class_names:
                if c1 == c2:
                    continue
                if pair_key == f"{c1}-{c2}" or pair_key == f"{c2}-{c1}":
                    i, j = LABEL_DICT[c1], LABEL_DICT[c2]
                    sim_matrix[i, j] = sim_val
                    sim_matrix[j, i] = sim_val
                    matched = True
                    break
            if matched:
                break
        if not matched and excluded_classes and any(name in pair_key for name in excluded_classes):
            continue
        if not matched:
            print(f"  WARNING: knowledge_graph pair '{pair_key}' could not be matched to any class pair")

    print(f"  Knowledge graph similarity matrix:\n{sim_matrix}")
    return sim_matrix


class KnowledgeGuidedSupConLoss(nn.Module):
    """Knowledge Graph-Guided Supervised Contrastive Loss.

    Replaces the binary positive mask in standard SupCon with soft weights
    derived from a medical knowledge graph. Classes with medical similarity
    (e.g. Normal vs specific benign disease vs Cancer progression) receive partial positive weight,
    so the loss doesn't push them apart as aggressively as unrelated classes.

    When ``learnable=True``, the inter-class similarities become trainable
    ``nn.Parameter`` values (constrained to [0,1] via sigmoid), initialized
    from the prior knowledge graph.  Only C*(C-1)/2 free parameters are
    stored (upper-triangular); the full symmetric matrix is reconstructed in
    each forward pass.

    Args:
        temperature: contrastive temperature scaling
        similarity_matrix: [C, C] tensor of inter-class soft similarity (0-1),
            diagonal should be 0 (same-class handled by hard mask)
        kg_weight: scaling factor for soft positive contributions
        learnable: if True, similarities become optimizable parameters
    """

    def __init__(self, temperature=0.1, similarity_matrix=None, kg_weight=1.0,
                 learnable=False):
        super().__init__()
        self.temperature = temperature
        self.kg_weight = kg_weight
        self.learnable = learnable
        self.num_classes = similarity_matrix.shape[0] if similarity_matrix is not None else 0

        if learnable and similarity_matrix is not None:
            n = self.num_classes
            triu_r, triu_c = torch.triu_indices(n, n, offset=1)
            prior_vals = similarity_matrix[triu_r, triu_c].clamp(0.01, 0.99)
            raw_init = torch.log(prior_vals / (1.0 - prior_vals))
            self._raw_sim = nn.Parameter(raw_init.clone())
            self.register_buffer("_raw_sim_init", raw_init.clone())
            self.register_buffer("_triu_r", triu_r)
            self.register_buffer("_triu_c", triu_c)
        elif similarity_matrix is not None:
            self.register_buffer("similarity_matrix", similarity_matrix)
        else:
            self.similarity_matrix = None

    def get_similarity_matrix(self):
        """Return the current [C, C] similarity matrix (differentiable when learnable)."""
        if self.learnable and hasattr(self, "_raw_sim"):
            n = self.num_classes
            device = self._raw_sim.device
            sim = torch.zeros(n, n, device=device)
            vals = torch.sigmoid(self._raw_sim)
            sim[self._triu_r, self._triu_c] = vals
            sim[self._triu_c, self._triu_r] = vals
            return sim
        return getattr(self, "similarity_matrix", None)

    def get_similarity_dict(self):
        """Return learned similarities as {(i,j): float} for logging."""
        sim = self.get_similarity_matrix()
        if sim is None:
            return {}
        n = self.num_classes
        result = {}
        for i in range(n):
            for j in range(i + 1, n):
                ci = LABEL_NAMES.get(i, str(i))
                cj = LABEL_NAMES.get(j, str(j))
                result[f"{ci}-{cj}"] = sim[i, j].item()
        return result

    def anchor_loss(self):
        """L2 penalty pulling _raw_sim back towards its initial values."""
        if self.learnable and hasattr(self, "_raw_sim_init"):
            return ((self._raw_sim - self._raw_sim_init) ** 2).mean()
        return torch.tensor(0.0)

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_dot_contrast = torch.matmul(contrast_feature, contrast_feature.T) / self.temperature
        logits_max, _ = anchor_dot_contrast.max(dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        labels = labels.contiguous().view(-1, 1)

        hard_mask = torch.eq(labels, labels.T).float().to(device)

        sim_matrix = self.get_similarity_matrix()
        if sim_matrix is not None:
            labels_flat = labels.squeeze(-1).long()
            soft_mask = sim_matrix[labels_flat][:, labels_flat]
            mask = hard_mask + self.kg_weight * soft_mask * (1.0 - hard_mask)
        else:
            mask = hard_mask

        self_mask = torch.eye(batch_size, device=device)
        mask = mask * (1.0 - self_mask)
        logits_mask = 1.0 - self_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        pos_weight = mask.sum(dim=1).clamp(min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_weight

        return -mean_log_prob_pos.mean()


class StandardSupConLoss(nn.Module):
    """Standard Supervised Contrastive Learning loss (Khosla et al., 2020)."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        n_views = features.shape[1]

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_dot_contrast = torch.matmul(contrast_feature, contrast_feature.T) / self.temperature
        logits_max, _ = anchor_dot_contrast.max(dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device).repeat(n_views, n_views)

        self_mask = torch.eye(batch_size * n_views, device=device)
        mask = mask - self_mask
        logits_mask = 1.0 - self_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)

        pos_count = mask.sum(dim=1).clamp(min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_count

        return -mean_log_prob_pos.mean()


def get_patient_name(filename):
    basename = os.path.basename(filename)
    if "_" in basename:
        return basename.split("_")[0]
    stem = os.path.splitext(basename)[0]
    if len(stem) >= 8 and stem[:8].isdigit():
        return stem[:8]
    return stem


def is_supported_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def is_valid_image_file(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


# ── data ─────────────────────────────────────────────────────────────────────
def discover_images():
    image_rows = []
    skipped_non_image = 0
    skipped_invalid_image = 0
    for root, _, files in os.walk(IMAGE_DIR):
        folder_name = os.path.basename(root)
        for filename in sorted(files):
            if not is_supported_image_file(filename):
                skipped_non_image += 1
                continue
            path = os.path.join(root, filename)
            if not is_valid_image_file(path):
                skipped_invalid_image += 1
                continue
            if folder_name not in FOLDER_TO_LABEL:
                continue
            label = FOLDER_TO_LABEL[folder_name]
            image_rows.append(
                {
                    "image_path": path,
                    "patient_name": get_patient_name(path),
                    "source_folder": folder_name,
                    "label": label,
                    "label_name": DISPLAY_NAMES[label],
                    "is_voc": is_voc_label(label),
                }
            )

    if skipped_non_image or skipped_invalid_image:
        print(f"Skipped non-image files: {skipped_non_image}, invalid images: {skipped_invalid_image}")

    df = pd.DataFrame(image_rows)
    if df.empty:
        raise RuntimeError(f"No images found under {IMAGE_DIR}")
    return df





def load_dataset_split(df, split_path=None):
    """Load the patient-level train/val/test split from JSON."""
    if split_path is None:
        split_path = DATASET_SPLIT_PATH

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Dataset split file not found at {split_path}.")

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    split_class_folders = split.get("class_folders")
    if split_class_folders is not None:
        split_classes = list(split_class_folders.keys())
        configured_classes = list(LABEL_DICT.keys())
        active_split_classes = [name for name in split_classes if name in LABEL_DICT]
        ignored_split_classes = [name for name in split_classes if name not in LABEL_DICT]
        if active_split_classes != configured_classes:
            raise ValueError(
                f"Configured classes do not match dataset split classes. "
                f"configured={configured_classes}, active_split={active_split_classes}. "
                "Load config through load_config(..., sync_with_split=True) "
                "or regenerate dataset_split.json."
            )
        if ignored_split_classes:
            print(f"Ignoring inactive classes from dataset split: {ignored_split_classes}")

    patient_split = split.get("patients", split)
    train_patients = set(patient_split.get("train", []))
    val_patients = set(patient_split.get("val", []))
    test_patients = set(patient_split.get("test", []))
    if not train_patients or not val_patients or not test_patients:
        raise ValueError(
            f"Dataset split at {split_path} must contain non-empty train/val/test patient lists."
        )

    overlap_tv = train_patients & val_patients
    overlap_tt = train_patients & test_patients
    overlap_vt = val_patients & test_patients
    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            f"Dataset split has overlapping patients across subsets: "
            f"train∩val={overlap_tv}, train∩test={overlap_tt}, val∩test={overlap_vt}"
        )

    df_patients = set(df["patient_name"].unique().tolist())
    split_patients_all = train_patients | val_patients | test_patients
    missing_in_df = split_patients_all - df_patients
    missing_in_split = df_patients - split_patients_all
    if missing_in_df:
        active_hint = "active " if split_class_folders is not None and ignored_split_classes else ""
        print(f"Warning: {len(missing_in_df)} patients listed in split are missing from the {active_hint}dataset "
              f"(e.g. {sorted(missing_in_df)[:3]}).")
    if missing_in_split:
        print(f"Warning: {len(missing_in_split)} patients in the dataset are not present in the split "
              f"and will be ignored (e.g. {sorted(missing_in_split)[:3]}).")

    train_df = df[df["patient_name"].isin(train_patients)].reset_index(drop=True)
    val_df = df[df["patient_name"].isin(val_patients)].reset_index(drop=True)
    test_df = df[df["patient_name"].isin(test_patients)].reset_index(drop=True)

    all_label_ids = set(LABEL_NAMES.keys())
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        missing_labels = all_label_ids - set(split_df["label"].unique().tolist())
        if missing_labels:
            missing_names = [DISPLAY_NAMES[idx] for idx in sorted(missing_labels)]
            print(
                f"Warning: {split_name} split has no samples for classes {missing_names}. "
                "For rare diseases, patient-level splitting may require more data or an explicit split file."
            )

    print(f"Loaded dataset split from {split_path}.")
    return train_df, val_df, test_df


class LaryngealDataset(Dataset):
    def __init__(self, dataframe, transform, cfg, return_visual=False, image_cache=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.return_visual = return_visual
        self.img_size = cfg["image_size"]
        self._cache = image_cache
        self.visual_transform = transforms.Compose(
            [
                CropBlackBorders(threshold=cfg["crop_black_threshold"]),
                transforms.Resize(cfg["resize_size"]),
                transforms.CenterCrop(cfg["image_size"]),
            ]
        )

    def __len__(self):
        return len(self.dataframe)

    def _load_image(self, img_path):
        if self._cache is not None and img_path in self._cache.get("path_to_idx", {}):
            idx = self._cache["path_to_idx"][img_path]
            return self._cache["images"][idx]
        if self._cache is not None and img_path in self._cache:
            # Fallback for old cache dict if needed
            cached = self._cache[img_path]
            if isinstance(cached, torch.Tensor):
                return cached
            return Image.fromarray(cached)
        return Image.open(img_path).convert("RGB")

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row["image_path"]
        label = int(row["label"])
        is_voc = row["is_voc"]
        try:
            img = self._load_image(img_path)
            image_tensor = img if isinstance(img, torch.Tensor) else self.transform(img)
            if self.return_visual:
                with Image.open(img_path) as visual_src:
                    visual_image = np.array(self.visual_transform(visual_src.convert("RGB")))
                return image_tensor, label, is_voc, visual_image, img_path
            return image_tensor, label, is_voc
        except Exception as exc:
            print(f"Failed to load {img_path}: {exc}")
            fallback = torch.zeros(3, self.img_size, self.img_size)
            if self.return_visual:
                return fallback, label, is_voc, np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8), img_path
            return fallback, label, is_voc


# ── model ────────────────────────────────────────────────────────────────────
def _resolve_unfreeze_block_specs(specs, stage_block_counts):
    selected = []
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError(f"unfreeze_blocks entries must be objects, got: {spec!r}")
        stage_idx = int(spec["stage"])
        block_idx = int(spec["block"])
        if stage_idx < 0:
            stage_idx += len(stage_block_counts)
        if stage_idx < 0 or stage_idx >= len(stage_block_counts):
            raise ValueError(f"Invalid unfreeze block stage index: {spec!r}")
        blocks_in_stage = stage_block_counts[stage_idx]
        if block_idx < 0:
            block_idx += blocks_in_stage
        if block_idx < 0 or block_idx >= blocks_in_stage:
            raise ValueError(f"Invalid unfreeze block index: {spec!r}")
        selected.append((stage_idx, block_idx))
    return sorted(set(selected))


class HierarchicalImageClassifier(nn.Module):
    SWIN_MODEL_NAME = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"

    def __init__(self, num_classes, cfg):
        super().__init__()
        drop_path = cfg.get("drop_path_rate", 0.0)
        self.backbone = timm.create_model(
            self.SWIN_MODEL_NAME,
            pretrained=False,
            num_classes=0,
            drop_path_rate=drop_path,
        )
        loaded_local = False
        for weight_path in LOCAL_WEIGHT_CANDIDATES:
            if os.path.exists(weight_path):
                try:
                    from safetensors.torch import load_file

                    state_dict = load_file(weight_path)
                    self.backbone.load_state_dict(state_dict, strict=False)
                    loaded_local = True
                    print(f"Loaded local pretrained weights: {weight_path}")
                    break
                except Exception as exc:
                    print(f"Failed to load local weights from {weight_path}: {exc}")
        if not loaded_local:
            self.backbone = timm.create_model(
                self.SWIN_MODEL_NAME,
                pretrained=True,
                num_classes=0,
                drop_path_rate=drop_path,
            )
            print("Loaded timm pretrained weights.")

        self.feature_dim = self.backbone.num_features

        all_blocks = []
        for stage in self.backbone.layers:
            for blk in stage.blocks:
                all_blocks.append(blk)
        total_blocks = len(all_blocks)
        unfreeze_specs = cfg.get("unfreeze_blocks")
        if unfreeze_specs:
            for param in self.backbone.parameters():
                param.requires_grad = False
            stage_block_counts = [len(stage.blocks) for stage in self.backbone.layers]
            selected_blocks = _resolve_unfreeze_block_specs(unfreeze_specs, stage_block_counts)
            for stage_idx, block_idx in selected_blocks:
                for param in self.backbone.layers[stage_idx].blocks[block_idx].parameters():
                    param.requires_grad = True
            labels = ", ".join(f"stage{stage_idx}.block{block_idx}" for stage_idx, block_idx in selected_blocks)
            print(
                f"Swin blocks: {total_blocks} total, explicitly trainable: "
                f"{len(selected_blocks)} ({labels})"
            )
        else:
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = False
            unfreeze_n = cfg["unfreeze_last_n_blocks"]
            freeze_upto = max(0, total_blocks - unfreeze_n)
            for i in range(freeze_upto):
                for param in all_blocks[i].parameters():
                    param.requires_grad = False
            cumulative = 0
            for stage in self.backbone.layers:
                stage_end = cumulative + len(stage.blocks)
                if stage_end <= freeze_upto and hasattr(stage, "downsample") and stage.downsample is not None:
                    for param in stage.downsample.parameters():
                        param.requires_grad = False
                cumulative = stage_end
            print(f"Swin blocks: {total_blocks} total, frozen: {freeze_upto}, trainable: {total_blocks - freeze_upto}")

        hidden_dim = cfg["classifier_hidden_dim"]
        dr = cfg["dropout_rate"]
        self.classifier = nn.Sequential(
            nn.Dropout(dr * 0.5),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hidden_dim, num_classes),
        )

        proj_dim = cfg.get("supcon_projection_dim", 128)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, proj_dim),
        )

    def forward(self, x, return_projection=False):
        features = self.backbone(x)
        if return_projection:
            return F.normalize(self.projector(features), dim=1)
        return self.classifier(features)

    def predict_hierarchical(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        voc_indices = get_voc_label_indices()

        non_voc_prob = probs[:, NON_VOC_LABEL]
        voc_prob = probs[:, voc_indices].sum(dim=1)

        is_vocal_cord = voc_prob > non_voc_prob

        final_preds = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        final_labels = [["", ""] for _ in range(logits.size(0))]

        for i in range(logits.size(0)):
            if is_vocal_cord[i]:
                voc_logits = logits[i, voc_indices]
                pred_voc = voc_indices[torch.argmax(voc_logits).item()]
                final_preds[i] = pred_voc
                final_labels[i] = ["Vocal-Cord", DISPLAY_NAMES[pred_voc]]
            else:
                final_preds[i] = NON_VOC_LABEL
                final_labels[i] = ["Non-Vocal-Cord", "N/A"]

        return final_preds, final_labels, is_vocal_cord


def _build_base_preprocess(cfg, to_tensor=True):
    steps = [
        CropBlackBorders(threshold=cfg["crop_black_threshold"]),
        transforms.Resize(cfg["resize_size"]),
        transforms.CenterCrop(cfg["image_size"]),
    ]
    if to_tensor:
        steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


def _pil_to_uint8_chw(img):
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def preload_image_cache(*dataframes, cfg=None, max_size=400, device=None):
    all_paths = []
    seen_paths = set()
    for df in dataframes:
        for path in df["image_path"].tolist():
            if path not in seen_paths:
                seen_paths.add(path)
                all_paths.append(path)
    path_to_idx = {p: i for i, p in enumerate(all_paths)}
    N = len(all_paths)

    if cfg is not None:
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        C, H, W = 3, cfg["image_size"], cfg["image_size"]
        images_tensor = torch.empty((N, C, H, W), dtype=torch.uint8, device=target_device)
        preprocess = _build_base_preprocess(cfg, to_tensor=False)
        memory_name = "VRAM" if target_device.type == "cuda" else "RAM"
        print(f"Caching {N} preprocessed images into one contiguous {target_device} {memory_name} tensor ...")

        for i, path in enumerate(all_paths):
            try:
                img = Image.open(path).convert("RGB")
                tensor = _pil_to_uint8_chw(preprocess(img)).to(target_device)
                images_tensor[i] = tensor
            except Exception as exc:
                print(f"  Cache failed: {path}: {exc}")
                images_tensor[i].zero_()
        
        # Convert to channels_last for optimal Tensor Core & memory bandwidth utilization
        images_tensor = images_tensor.to(memory_format=torch.channels_last)
        total_bytes = images_tensor.nelement() * images_tensor.element_size()
        print(f"  Done — unified cache size ~{total_bytes / 1024**2:.0f} MiB (Channels Last)")
        return {"path_to_idx": path_to_idx, "images": images_tensor, "ordered_paths": all_paths}
    else:
        cache = {}
        print(f"Caching {N} images into memory (max_size={max_size}) ...")
        for path in all_paths:
            try:
                img = Image.open(path).convert("RGB")
                w, h = img.size
                if max(w, h) > max_size:
                    scale = max_size / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
                cache[path] = np.array(img)
            except Exception as exc:
                print(f"  Cache failed: {path}: {exc}")
        return cache


# ── transforms & loaders ─────────────────────────────────────────────────────
def build_transforms(cfg):
    train_tf = _build_base_preprocess(cfg, to_tensor=True)
    eval_tf = _build_base_preprocess(cfg, to_tensor=True)
    return train_tf, eval_tf


class VRAMDataLoader:
    def __init__(self, df, cache, batch_size, sampler=None, shuffle=False):
        self.batch_size = batch_size
        self.sampler = sampler
        self.shuffle = shuffle
        
        self.images_tensor = cache["images"]
        self.device = self.images_tensor.device
        self.already_on_device = self.device.type == "cuda"
        self.dataset_size = len(df)
        path_index_list = [cache["path_to_idx"][p] for p in df["image_path"]]
        self._contiguous_start = None
        if path_index_list:
            start = path_index_list[0]
            if path_index_list == list(range(start, start + len(path_index_list))):
                self._contiguous_start = start
        self.path_indices = None
        if self._contiguous_start is None:
            self.path_indices = torch.tensor(path_index_list, dtype=torch.long, device=self.device)
        self.labels = torch.tensor(df["label"].values, dtype=torch.long, device=self.device)
        self.is_voc = torch.tensor(df["is_voc"].values, dtype=torch.bool, device=self.device)
        self.num_samples = self.dataset_size
        self.sample_weights = None
        self.replacement = True
        if sampler is not None:
            self.num_samples = int(getattr(sampler, "num_samples", self.dataset_size))
            weights = getattr(sampler, "weights", None)
            if weights is not None:
                self.sample_weights = weights.to(device=self.device, dtype=torch.float32)
                self.replacement = bool(getattr(sampler, "replacement", True))
        self._sequential_indices = torch.arange(self.dataset_size, device=self.device)

    def _global_indices(self, local_indices):
        if self._contiguous_start is not None:
            return local_indices + self._contiguous_start
        return self.path_indices[local_indices]
        
    def __iter__(self):
        if self.sample_weights is not None:
            indices = torch.multinomial(
                self.sample_weights,
                self.num_samples,
                replacement=self.replacement,
            )
        elif self.sampler is not None:
            indices = torch.tensor(list(self.sampler), dtype=torch.long, device=self.device)
        elif self.shuffle:
            indices = torch.randperm(self.dataset_size, device=self.device)
        else:
            indices = self._sequential_indices
            
        for i in range(0, self.num_samples, self.batch_size):
            end = min(i + self.batch_size, self.num_samples)
            if (
                self._contiguous_start is not None
                and self.sampler is None
                and not self.shuffle
            ):
                image_start = self._contiguous_start + i
                image_end = self._contiguous_start + end
                local_slice = slice(i, end)
                yield (
                    self.images_tensor[image_start:image_end],
                    self.labels[local_slice],
                    self.is_voc[local_slice],
                )
                continue

            batch_idx = indices[i:end]
            global_idx = self._global_indices(batch_idx)
            # CUDA advanced indexing preserves channels_last layout for the image batch.
            yield self.images_tensor[global_idx], self.labels[batch_idx], self.is_voc[batch_idx]

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def create_loaders(train_df, val_df, test_df, cfg, image_cache=None, train_sampler=None):
    _, eval_tf = build_transforms(cfg)
    bs = cfg["batch_size"]
    ebs = cfg["eval_batch_size"]
    
    use_shuffle = train_sampler is None
    
    if image_cache is not None and "images" in image_cache:
        train_loader = VRAMDataLoader(train_df, image_cache, bs, sampler=train_sampler, shuffle=use_shuffle)
        train_eval_loader = VRAMDataLoader(train_df, image_cache, ebs, shuffle=False)
        val_loader = VRAMDataLoader(val_df, image_cache, ebs, shuffle=False)
        test_loader = VRAMDataLoader(test_df, image_cache, ebs, shuffle=False)
    else:
        # Fallback to normal DataLoader if not using unified VRAM cache
        train_tf = _build_base_preprocess(cfg, to_tensor=True)
        train_dataset = LaryngealDataset(train_df, train_tf, cfg, image_cache=image_cache)
        train_eval_dataset = LaryngealDataset(train_df, eval_tf, cfg, image_cache=image_cache)
        val_dataset = LaryngealDataset(val_df, eval_tf, cfg, image_cache=image_cache)
        test_dataset = LaryngealDataset(test_df, eval_tf, cfg, image_cache=image_cache)
        common = {"num_workers": 0, "pin_memory": False}
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=use_shuffle, sampler=train_sampler, **common)
        train_eval_loader = DataLoader(train_eval_dataset, batch_size=ebs, shuffle=False, **common)
        val_loader = DataLoader(val_dataset, batch_size=ebs, shuffle=False, **common)
        test_loader = DataLoader(test_dataset, batch_size=ebs, shuffle=False, **common)

    return {
        "train": train_loader,
        "train_eval": train_eval_loader,
        "val": val_loader,
        "test": test_loader,
        "eval_tf": eval_tf,
    }


def compute_class_weights(train_df, device):
    counts = train_df["label"].value_counts().reindex(range(len(LABEL_DICT)), fill_value=0).sort_index()
    safe_counts = counts.replace(0, np.nan)
    weights = len(train_df) / (len(LABEL_DICT) * safe_counts.values)
    weights = np.nan_to_num(weights, nan=0.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_balanced_sampler(df, hierarchical=False, balance_alpha=1.0):
    """Build a WeightedRandomSampler for class-balanced training.

    When hierarchical=True (Phase 1): VOC total = Non-VOC total,
    then within VOC each configured subclass gets equal share.
    When hierarchical=False (Phase 2): inverse-frequency weighting with
    ``balance_alpha``. alpha=1.0 gives class-equal sampling; alpha=0.0
    recovers the natural dataset distribution.
    """
    labels = df["label"].values
    class_counts = Counter(labels)

    if hierarchical:
        n_nonvoc = class_counts.get(NON_VOC_LABEL, 1)
        voc_classes = [c for c in class_counts if c != NON_VOC_LABEL]
        n_voc_sub = len(voc_classes) or 1
        weight_map = {}
        weight_map[NON_VOC_LABEL] = 0.5 / n_nonvoc
        for c in voc_classes:
            weight_map[c] = (0.5 / n_voc_sub) / class_counts[c]
        sample_weights = [weight_map[l] for l in labels]
    else:
        balance_alpha = float(balance_alpha)
        if not 0.0 <= balance_alpha <= 1.0:
            raise ValueError(f"balance_alpha must be in [0, 1], got {balance_alpha}")
        sample_weights = [1.0 / (class_counts[l] ** balance_alpha) for l in labels]

    sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(df),
        replacement=True,
    )

    mode = (
        "hierarchical (VOC=Non-VOC, within-VOC equal)"
        if hierarchical
        else f"{len(class_counts)}-class damped balance (alpha={balance_alpha:.2f})"
    )
    effective = {}
    total_w = sample_weights.sum().item()
    for c in sorted(class_counts):
        mask = torch.tensor(labels == c)
        effective[DISPLAY_NAMES[c]] = f"{sample_weights[mask].sum().item() / total_w * 100:.1f}%"
    print(f"  Balanced sampler ({mode}): {effective}")

    return sampler


def _swin_param_depth(name, stage_cumulative_blocks, total_blocks):
    if name.startswith("classifier") or name.startswith("projector"):
        return 0
    if name.startswith("backbone.norm"):
        return 1

    if name.startswith("backbone.layers."):
        parts = name.split(".")
        stage_idx = int(parts[2])
        offset = stage_cumulative_blocks[stage_idx]
        if parts[3] == "blocks":
            block_idx = int(parts[4])
            flat_idx = offset + block_idx
            return total_blocks - flat_idx + 1
        if parts[3] == "downsample":
            flat_idx = stage_cumulative_blocks[stage_idx + 1] if stage_idx + 1 < len(stage_cumulative_blocks) else total_blocks
            return total_blocks - flat_idx + 1

    return total_blocks + 2


def build_optimizer_param_groups(model, cfg):
    lr = cfg["learning_rate"]
    wd = cfg["weight_decay"]
    layer_decay = cfg.get("layer_decay", 1.0)

    if layer_decay >= 1.0:
        return [
            {"params": list(filter(lambda p: p.requires_grad, model.parameters())), "lr": lr, "weight_decay": wd}
        ]

    stage_block_counts = [len(stage.blocks) for stage in model.backbone.layers]
    stage_cumulative = [0]
    for c in stage_block_counts:
        stage_cumulative.append(stage_cumulative[-1] + c)
    total_blocks = stage_cumulative[-1]

    no_decay_kw = {"bias", "norm"}
    groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        depth = _swin_param_depth(name, stage_cumulative, total_blocks)
        scale = layer_decay ** depth
        use_wd = 0.0 if any(kw in name for kw in no_decay_kw) else wd
        groups.append({"params": [param], "lr": lr * scale, "weight_decay": use_wd})

    return groups


# ── CUDA prefetch / async IO ─────────────────────────────────────────────────
def _batch_to_device(batch, device):
    """Move every Tensor in a nested batch to device with non_blocking when possible."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        mapped = tuple(
            _batch_to_device(b, device) if isinstance(b, torch.Tensor) else b for b in batch
        )
        return mapped
    return batch


class CUDAPrefetcher:
    """Overlap H2D transfer (prefetch stream) with default-stream compute."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self._it = None
        self._next = None

    def __iter__(self):
        self._it = iter(self.loader)
        self._preload_next()
        return self

    def __len__(self):
        return len(self.loader)

    def _preload_next(self):
        try:
            batch = next(self._it)
        except StopIteration:
            self._next = None
            return
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self._next = _batch_to_device(batch, self.device)
        else:
            self._next = _batch_to_device(batch, self.device)

    def __next__(self):
        if self._next is None:
            raise StopIteration
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self._next
        self._preload_next()
        return batch


class AsyncCheckpointSaver:
    """Copy state_dict to CPU on a dedicated stream, then torch.save in a background thread."""

    def __init__(self, device):
        self.device = device
        self.stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self._thread = None
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            if self._thread is not None:
                self._thread.join()
                self._thread = None

    def save(self, model, path):
        self.wait()
        state = model.state_dict()
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                cpu_state = {}
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        cpu_state[k] = v.detach().to("cpu", non_blocking=True)
                    else:
                        cpu_state[k] = v
            self.stream.synchronize()
        else:
            cpu_state = {
                k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in state.items()
            }

        def _write():
            torch.save(cpu_state, path)

        t = threading.Thread(target=_write, daemon=True)
        with self._lock:
            self._thread = t
        t.start()


def maybe_prefetch_loader(loader, device):
    """Wrap CPU-backed loaders with CUDAPrefetcher; pass through GPU-resident loaders."""
    if getattr(loader, "already_on_device", False):
        return loader
    if device.type == "cuda":
        return CUDAPrefetcher(loader, device)
    return loader


def create_classification_metrics(num_classes, device):
    """Reusable torchmetrics for multiclass train/eval (call .reset() each epoch)."""
    f1_metric = torchmetrics.F1Score(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)
    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    auroc_metric = torchmetrics.AUROC(task="multiclass", num_classes=num_classes).to(device)
    return f1_metric, acc_metric, auroc_metric


class CyclingDataLoaderIter:
    """Infinite iterator over a DataLoader, restarting when exhausted."""

    def __init__(self, loader):
        self.loader = loader
        self._iter = iter(loader)

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


# ── train / eval ─────────────────────────────────────────────────────────────
def supcon_train_one_epoch(model, loader, optimizer, criterion, scaler, device, grad_accum, gpu_aug=None):
    model.train()
    running_loss = torch.tensor(0.0, device=device)
    total_samples = 0
    optimizer.zero_grad()

    prefetch_loader = maybe_prefetch_loader(loader, device)
    for step, (images, labels, _is_voc) in enumerate(prefetch_loader, start=1):
        images = gpu_aug(images) if gpu_aug is not None else gpu_normalise(images)
        bsz = labels.shape[0]

        with torch.amp.autocast(device_type=device.type):
            projections = model(images, return_projection=True)
            projections = projections.unsqueeze(1)
            loss = criterion(projections, labels) / grad_accum

        scaler.scale(loss).backward()
        if step % grad_accum == 0 or step == len(prefetch_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.detach() * grad_accum * bsz
        total_samples += bsz

    del prefetch_loader
    return (running_loss / max(total_samples, 1)).item()


def supcon_train_one_epoch_bilevel(
    model, train_loader, val_iter,
    model_optimizer, kg_optimizer,
    criterion, scaler, device, grad_accum,
    kg_anchor_strength=0.0,
    gpu_aug=None,
):
    """Bilevel SupCon: model params updated on train data, KG params on val data.

    First-order approximation: after every model optimizer step (every
    ``grad_accum`` train mini-batches), fetch one val batch, run a forward
    pass with *detached* model outputs so gradients only reach KG parameters,
    and perform a single KG optimizer step.

    ``kg_anchor_strength`` adds an L2 penalty pulling KG similarities back
    towards their initial prior values, preventing unconstrained drift.
    """
    model.train()
    running_train_loss = torch.tensor(0.0, device=device)
    running_val_loss = torch.tensor(0.0, device=device)
    total_train = 0
    total_val = 0
    model_optimizer.zero_grad()

    prefetch_loader = maybe_prefetch_loader(train_loader, device)
    for step, (images, labels, _is_voc) in enumerate(prefetch_loader, start=1):
        images = gpu_aug(images) if gpu_aug is not None else gpu_normalise(images)
        bsz = labels.shape[0]

        # ── Inner step: train batch → update model θ ─────────────
        with torch.amp.autocast(device_type=device.type):
            projections = model(images, return_projection=True)
            projections = projections.unsqueeze(1)
            train_loss = criterion(projections, labels) / grad_accum

        scaler.scale(train_loss).backward()

        if step % grad_accum == 0 or step == len(prefetch_loader):
            scaler.step(model_optimizer)
            scaler.update()
            model_optimizer.zero_grad()

            # ── Outer step: val batch → update KG φ ──────────────
            val_batch = next(val_iter)
            val_images, val_labels = val_batch[0], val_batch[1]
            if not val_images.is_cuda:
                val_images = val_images.to(device, non_blocking=True)
                val_labels = val_labels.to(device, non_blocking=True)
            val_images = gpu_normalise(val_images)

            kg_optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                with torch.no_grad():
                    val_proj = model(val_images, return_projection=True)
                val_proj = val_proj.detach().unsqueeze(1)
                val_loss = criterion(val_proj, val_labels)

            if kg_anchor_strength > 0:
                val_loss = val_loss + kg_anchor_strength * criterion.anchor_loss()

            val_loss.backward()
            kg_optimizer.step()

            running_val_loss += val_loss.detach() * val_labels.shape[0]
            total_val += val_labels.shape[0]

        running_train_loss += train_loss.detach() * grad_accum * bsz
        total_train += bsz

    del prefetch_loader
    avg_train = (running_train_loss / max(total_train, 1)).item()
    avg_val = (running_val_loss / max(total_val, 1)).item()
    return avg_train, avg_val


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    grad_accum,
    num_classes,
    cfg=None,
    cls_metrics=None,
    gpu_aug=None,
):
    model.train()
    running_loss = torch.tensor(0.0, device=device)
    total_samples = 0
    if cls_metrics is None:
        f1_metric, acc_metric, auroc_metric = create_classification_metrics(num_classes, device)
    else:
        f1_metric, acc_metric, auroc_metric = cls_metrics
        f1_metric.reset()
        acc_metric.reset()
        auroc_metric.reset()
    optimizer.zero_grad()

    prefetch_loader = maybe_prefetch_loader(loader, device)
    for step, (images, labels, _is_voc) in enumerate(prefetch_loader, start=1):
        images = gpu_aug(images) if gpu_aug is not None else gpu_normalise(images)
        with torch.amp.autocast(device_type=device.type):
            logits = model(images)
            loss = criterion(logits, labels) / grad_accum
        scaler.scale(loss).backward()

        if step % grad_accum == 0 or step == len(prefetch_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.detach() * grad_accum * images.size(0)
        total_samples += images.size(0)
        logits_det = logits.detach()
        preds = torch.argmax(logits_det, dim=1)
        probs = F.softmax(logits_det, dim=1)
        f1_metric.update(preds, labels)
        acc_metric.update(preds, labels)
        auroc_metric.update(probs, labels)

    del prefetch_loader
    epoch_loss = (running_loss / max(total_samples, 1)).item()
    epoch_f1 = f1_metric.compute().item()
    epoch_acc = acc_metric.compute().item()
    epoch_auc = auroc_metric.compute().item()
    return epoch_loss, epoch_f1, epoch_acc, epoch_auc


def evaluate(model, loader, criterion, device, num_classes, return_preds=False, cls_metrics=None):
    model.eval()
    running_loss = torch.tensor(0.0, device=device)
    total_samples = 0
    if cls_metrics is None:
        f1_metric, acc_metric, auroc_metric = create_classification_metrics(num_classes, device)
    else:
        f1_metric, acc_metric, auroc_metric = cls_metrics
        f1_metric.reset()
        acc_metric.reset()
        auroc_metric.reset()
    all_preds, all_labels, all_is_voc, all_preds_hier = [], [], [], []

    prefetch_loader = maybe_prefetch_loader(loader, device)
    with torch.no_grad():
        for batch in prefetch_loader:
            if len(batch) == 5:
                images, labels, is_voc, _, _ = batch
            else:
                images, labels, is_voc = batch

            images = gpu_normalise(images)
            with torch.amp.autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, labels) if criterion is not None else None

            if loss is not None:
                running_loss += loss.detach() * images.size(0)
            total_samples += images.size(0)
            logits_det = logits.detach()
            preds = torch.argmax(logits_det, dim=1)
            probs = F.softmax(logits_det, dim=1)
            f1_metric.update(preds, labels)
            acc_metric.update(preds, labels)
            auroc_metric.update(probs, labels)

            if return_preds:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_is_voc.extend(is_voc.cpu().numpy() if hasattr(is_voc, 'cpu') else is_voc)

    del prefetch_loader
    result = {
        "loss": (running_loss / max(total_samples, 1)).item() if criterion is not None else None,
        "f1": f1_metric.compute().item(),
        "acc": acc_metric.compute().item(),
        "auc": auroc_metric.compute().item(),
    }
    if return_preds:
        result["y_true"] = all_labels
        result["y_pred"] = all_preds
        result["is_voc"] = all_is_voc
    return result


def _eval_test_worker(model, loader, device, num_classes, result_holder, lock):
    """Run test evaluation in a background thread."""
    try:
        model.eval()
        stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        ctx = torch.cuda.stream(stream) if stream else nullcontext()
        with ctx, torch.no_grad():
            f1_metric = torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            ).to(device)
            acc_metric = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            ).to(device)
            auroc_metric = torchmetrics.AUROC(
                task="multiclass", num_classes=num_classes
            ).to(device)
            for batch in loader:
                images, labels = batch[0], batch[1]
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                images = gpu_normalise(images)
                with torch.amp.autocast(device_type=device.type):
                    logits = model(images)
                logits_det = logits.detach()
                preds = torch.argmax(logits_det, dim=1)
                probs = F.softmax(logits_det, dim=1)
                f1_metric.update(preds, labels)
                acc_metric.update(preds, labels)
                auroc_metric.update(probs, labels)
            if stream:
                stream.synchronize()
            metrics = {
                "f1": f1_metric.compute().item(),
                "acc": acc_metric.compute().item(),
                "auc": auroc_metric.compute().item(),
            }
        with lock:
            result_holder[0] = metrics
    except Exception as exc:
        print(f"  [test-eval-thread] error: {exc}")


def evaluate_hierarchical(model, loader, device, num_classes):
    model.eval()
    all_preds_hier, all_labels_hier, all_is_voc_true = [], [], []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                images, labels, is_voc, _, _ = batch
            else:
                images, labels, is_voc = batch

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = gpu_normalise(images)

            final_preds, _hier_labels, _is_voc_pred = model.predict_hierarchical(images)

            if torch.is_tensor(is_voc):
                all_is_voc_true.extend(is_voc.detach().cpu().numpy())
            else:
                all_is_voc_true.extend(is_voc)
            all_labels_hier.extend(labels.cpu().numpy())
            all_preds_hier.extend(final_preds.cpu().numpy())

    voc_correct = sum(
        1 for v, p in zip(all_is_voc_true, all_preds_hier)
        if p != NON_VOC_LABEL and v
    )
    non_voc_correct = sum(
        1 for v, p in zip(all_is_voc_true, all_preds_hier)
        if p == NON_VOC_LABEL and not v
    )
    hier_acc = (voc_correct + non_voc_correct) / max(len(all_is_voc_true), 1)

    voc_only_labels = [l for l, v in zip(all_labels_hier, all_is_voc_true) if v]
    voc_only_preds = [p for p, v in zip(all_preds_hier, all_is_voc_true) if v]

    voc_f1 = None
    if len(voc_only_labels) > 0:
        try:
            from sklearn.metrics import f1_score
            voc_f1 = f1_score(
                voc_only_labels,
                voc_only_preds,
                labels=get_voc_label_indices(),
                average="macro",
                zero_division=0,
            )
        except Exception:
            pass

    return {
        "hier_acc": hier_acc,
        "voc_f1": voc_f1,
        "voc_only_labels": voc_only_labels,
        "voc_only_preds": voc_only_preds,
    }


# ── visualisation ────────────────────────────────────────────────────────────
def save_training_curves(history_df, best_epoch, supcon_history=None, ce_phase_name="Phase 2"):
    best_row = history_df.loc[history_df["epoch"] == best_epoch].iloc[0]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    colors = {"train": "#2563EB", "val": "#F59E0B"}
    has_supcon = supcon_history is not None and len(supcon_history) > 0

    if has_supcon:
        fig = plt.figure(figsize=(22, 14), facecolor="white")
        gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30,
                              height_ratios=[0.8, 1])
        ax_supcon = fig.add_subplot(gs[0, :])
        ax_f1 = fig.add_subplot(gs[1, 0])
        ax_acc = fig.add_subplot(gs[1, 1])
        ax_auc = fig.add_subplot(gs[1, 2])
    else:
        fig, (ax_f1, ax_acc, ax_auc) = plt.subplots(
            1, 3, figsize=(22, 6), facecolor="white")
        ax_supcon = None

    epochs = history_df["epoch"]

    if ax_supcon is not None:
        sc_df = pd.DataFrame(supcon_history)
        has_supcon_val = "val_loss" in sc_df.columns and sc_df["val_loss"].notna().any()
        if "monitor_loss" in sc_df.columns and sc_df["monitor_loss"].notna().any():
            monitor_col = "monitor_loss"
            monitor_name = sc_df.get("monitor", pd.Series(["monitor"])).dropna().iloc[-1]
            monitor_label = "Val loss" if monitor_name == "val_loss" else "Train loss"
        elif has_supcon_val:
            monitor_col = "val_loss"
            monitor_label = "Val loss"
        else:
            monitor_col = "loss"
            monitor_label = "Train loss"

        best_sc_idx = sc_df[monitor_col].idxmin()
        best_sc_epoch = sc_df.loc[best_sc_idx, "epoch"]
        best_sc_loss = sc_df.loc[best_sc_idx, monitor_col]

        ax_supcon.plot(sc_df["epoch"], sc_df["loss"], color="#7C3AED",
                       linewidth=2.2, alpha=0.9, label="Train SupCon Loss")
        if has_supcon_val:
            ax_supcon.plot(sc_df["epoch"], sc_df["val_loss"], color=colors["val"],
                           linewidth=2.0, alpha=0.9, label="Val SupCon Loss")
        ax_supcon.axvline(best_sc_epoch, color="#EF4444", linestyle="--",
                          alpha=0.6, linewidth=1.2, label="Best Epoch")
        ax_supcon.scatter([best_sc_epoch], [best_sc_loss], color="#EF4444",
                          s=70, zorder=5, edgecolors="white", linewidths=1.5)
        ax_supcon.annotate(
            f"Best epoch {best_sc_epoch}\n{monitor_label} = {best_sc_loss:.4f}",
            xy=(best_sc_epoch, best_sc_loss),
            xytext=(best_sc_epoch + max(1, len(sc_df) * 0.05),
                    best_sc_loss + (sc_df[monitor_col].max() - sc_df[monitor_col].min()) * 0.15),
            arrowprops={"arrowstyle": "->", "color": "#6B7280", "lw": 1.2},
            fontsize=9, color="#374151",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white",
                  "ec": "#D1D5DB", "alpha": 0.9},
        )
        ax_supcon.set_title("Phase 1 — Supervised Contrastive Loss",
                            fontsize=14, fontweight="bold", pad=12, color="#1F2937")
        ax_supcon.set_xlabel("Epoch", fontsize=11, color="#4B5563")
        ax_supcon.set_ylabel("Loss", fontsize=11, color="#4B5563")
        ax_supcon.grid(alpha=0.2, linestyle="--")
        ax_supcon.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#E5E7EB")
        ax_supcon.tick_params(colors="#6B7280", labelsize=9)

    def _annotate_best(ax, metric_name, val_col, y_val):
        ax.axvline(best_epoch, color="#EF4444", linestyle="--",
                   alpha=0.6, linewidth=1.2, label="Best Epoch")
        ax.scatter([best_epoch], [y_val], color="#EF4444",
                   s=60, zorder=5, edgecolors="white", linewidths=1.5)
        total_epochs = len(epochs)
        txt_x = best_epoch + max(1, total_epochs * 0.05)
        txt_y = min(1.0, y_val + 0.06)
        ax.annotate(
            f"Best ep {best_epoch}\nVal {metric_name} = {y_val:.4f}",
            xy=(best_epoch, y_val),
            xytext=(txt_x, txt_y),
            arrowprops={"arrowstyle": "->", "color": "#6B7280", "lw": 1.2},
            fontsize=8.5, color="#374151",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white",
                  "ec": "#D1D5DB", "alpha": 0.9},
        )

    def _format_metric_axis(ax):
        ax.set_ylim(0.0, 1.02)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))

    ax_f1.plot(epochs, history_df["train_f1"], color=colors["train"],
               label="Train", linewidth=2.2, alpha=0.9)
    ax_f1.plot(epochs, history_df["val_f1"], color=colors["val"],
               label="Val", linewidth=2.2, alpha=0.9)
    _annotate_best(ax_f1, "F1", "val_f1", best_row["val_f1"])
    ax_f1.set_title(f"{ce_phase_name} — Macro F1", fontsize=14, fontweight="bold",
                     pad=12, color="#1F2937")
    ax_f1.set_xlabel("Epoch", fontsize=11, color="#4B5563")
    ax_f1.set_ylabel("Macro F1", fontsize=11, color="#4B5563")
    _format_metric_axis(ax_f1)
    ax_f1.grid(alpha=0.2, linestyle="--")
    ax_f1.legend(loc="lower right", fontsize=9, framealpha=0.9, edgecolor="#E5E7EB")
    ax_f1.tick_params(colors="#6B7280", labelsize=9)

    ax_acc.plot(epochs, history_df["train_acc"], color=colors["train"],
                label="Train", linewidth=2.2, alpha=0.9)
    ax_acc.plot(epochs, history_df["val_acc"], color=colors["val"],
                label="Val", linewidth=2.2, alpha=0.9)
    best_val_acc = best_row.get("val_acc", 0)
    _annotate_best(ax_acc, "Acc", "val_acc", best_val_acc)
    ax_acc.set_title(f"{ce_phase_name} — Accuracy", fontsize=14, fontweight="bold",
                      pad=12, color="#1F2937")
    ax_acc.set_xlabel("Epoch", fontsize=11, color="#4B5563")
    ax_acc.set_ylabel("Accuracy", fontsize=11, color="#4B5563")
    _format_metric_axis(ax_acc)
    ax_acc.grid(alpha=0.2, linestyle="--")
    ax_acc.legend(loc="lower right", fontsize=9, framealpha=0.9, edgecolor="#E5E7EB")
    ax_acc.tick_params(colors="#6B7280", labelsize=9)

    has_auc = "val_auc" in history_df.columns and history_df["val_auc"].notna().any()
    if has_auc:
        has_train_auc = "train_auc" in history_df.columns and history_df["train_auc"].notna().any()
        if has_train_auc:
            ax_auc.plot(epochs, history_df["train_auc"], color=colors["train"],
                        label="Train", linewidth=2.2, alpha=0.9)
        ax_auc.plot(epochs, history_df["val_auc"], color=colors["val"],
                    label="Val", linewidth=2.2, alpha=0.9)
        best_val_auc = best_row.get("val_auc", 0)
        _annotate_best(ax_auc, "AUC", "val_auc", best_val_auc)
    ax_auc.set_title(f"{ce_phase_name} — AUROC", fontsize=14, fontweight="bold",
                      pad=12, color="#1F2937")
    ax_auc.set_xlabel("Epoch", fontsize=11, color="#4B5563")
    ax_auc.set_ylabel("AUROC", fontsize=11, color="#4B5563")
    _format_metric_axis(ax_auc)
    ax_auc.grid(alpha=0.2, linestyle="--")
    ax_auc.legend(loc="lower right", fontsize=9, framealpha=0.9, edgecolor="#E5E7EB")
    ax_auc.tick_params(colors="#6B7280", labelsize=9)

    title = f"Training Curves (Phase 1 + {ce_phase_name})" if has_supcon else f"Training Curves ({ce_phase_name})"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98, color="#111827")
    fig.subplots_adjust(top=0.88 if has_supcon else 0.82)
    plt.savefig(TRAINING_CURVE_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def save_metrics_csv(model, loaders, criterion, device, num_classes):
    rows = []
    cls_metrics = create_classification_metrics(num_classes, device)
    for split_name, loader in (("train", loaders["train_eval"]), ("val", loaders["val"]), ("test", loaders["test"])):
        metrics = evaluate(
            model, loader, criterion, device, num_classes, return_preds=True, cls_metrics=cls_metrics
        )
        hier_metrics = evaluate_hierarchical(model, loader, device, num_classes)

        report = classification_report(
            metrics["y_true"],
            metrics["y_pred"],
            labels=list(DISPLAY_NAMES.keys()),
            target_names=[DISPLAY_NAMES[idx] for idx in DISPLAY_NAMES],
            output_dict=True,
            zero_division=0,
        )
        for label_name, values in report.items():
            if isinstance(values, dict):
                row = {"split": split_name, "label": label_name}
                row.update(values)
                rows.append(row)
        rows.append(
            {
                "split": split_name,
                "label": "overall",
                "precision": np.nan,
                "recall": np.nan,
                "f1-score": metrics["f1"],
                "support": len(metrics["y_true"]),
                "accuracy": metrics["acc"],
                "loss": metrics["loss"],
            }
        )
        rows.append(
            {
                "split": split_name,
                "label": "hierarchical_voc_acc",
                "precision": np.nan,
                "recall": hier_metrics["hier_acc"],
                "f1-score": hier_metrics["hier_acc"],
                "support": len(metrics["y_true"]),
                "accuracy": hier_metrics["hier_acc"],
                "loss": np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(METRICS_CSV_PATH, index=False)


def sample_visualization_dataframe(df, n_samples=8, seed=42):
    if len(df) <= n_samples:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    label_order = sorted(df["label"].unique().tolist())
    sampled_parts = []
    remaining_budget = n_samples
    base_quota = max(1, n_samples // max(len(label_order), 1))

    for label in label_order:
        group = df[df["label"] == label]
        take_n = min(len(group), base_quota)
        if take_n > 0:
            sampled_parts.append(group.sample(n=take_n, random_state=seed))
            remaining_budget -= take_n

    if remaining_budget > 0:
        used_paths = set(pd.concat(sampled_parts)["image_path"].tolist()) if sampled_parts else set()
        remaining_df = df[~df["image_path"].isin(used_paths)]
        if not remaining_df.empty:
            sampled_parts.append(
                remaining_df.sample(n=min(remaining_budget, len(remaining_df)), random_state=seed)
            )

    sampled = pd.concat(sampled_parts, ignore_index=True).drop_duplicates(subset=["image_path"])
    if len(sampled) > n_samples:
        sampled = sampled.sample(n=n_samples, random_state=seed)
    return sampled.reset_index(drop=True)


def generate_attention_maps(model, dataframe, eval_tf, cfg, device, writer=None, epoch=None):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if dataframe.empty:
        return

    vis_df = sample_visualization_dataframe(dataframe, n_samples=8, seed=cfg["seed"])
    vis_dataset = LaryngealDataset(vis_df, eval_tf, cfg, return_visual=True)
    vis_loader = DataLoader(vis_dataset, batch_size=len(vis_dataset), shuffle=False)

    batch = next(iter(vis_loader))
    if len(batch) == 5:
        images, labels, _is_voc, original_images, _paths = batch
    else:
        images, labels, _is_voc = batch
        original_images = None

    images = gpu_normalise(images.to(device))
    img_size = cfg["image_size"]

    last_stage = model.backbone.layers[-1]
    target_layers = [last_stage.blocks[-1]]

    def reshape_transform(tensor):
        if tensor.dim() == 3:
            B, HW, C = tensor.shape
            h = w = int(HW ** 0.5)
            return tensor.reshape(B, h, w, C).permute(0, 3, 1, 2)
        if tensor.dim() == 4 and tensor.shape[-1] != tensor.shape[-2]:
            return tensor.permute(0, 3, 1, 2)
        return tensor

    model.eval()
    with torch.inference_mode():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu()

    # GradCAM backward needs a leaf tensor with grad enabled (pred pass stays inference-only).
    cam_input = images.detach().clone().requires_grad_(True)
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    targets = [ClassifierOutputTarget(int(p)) for p in preds]
    grayscale_cams = cam(input_tensor=cam_input, targets=targets)
    cam.activations_and_grads.release()

    fig, axes = plt.subplots(4, 4, figsize=(22, 22))
    axes = axes.flatten()

    for idx in range(len(vis_df)):
        if original_images is not None:
            original = original_images[idx].numpy().astype(np.uint8)
        else:
            original = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        cam_map = grayscale_cams[idx]
        rgb_img = cv2.resize(original, (img_size, img_size)).astype(np.float32) / 255.0
        cam_resized = cv2.resize(cam_map, (img_size, img_size))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        overlay = 0.5 * heatmap + 0.5 * rgb_img
        overlay = np.clip(overlay, 0, 1)
        overlay = np.uint8(255 * overlay)

        gt_name = DISPLAY_NAMES[int(labels[idx].item())]
        pred_name = DISPLAY_NAMES[int(preds[idx].item())]

        axes[2 * idx].imshow(original)
        axes[2 * idx].set_title(f"S{idx + 1} Original")
        axes[2 * idx].axis("off")

        axes[2 * idx + 1].imshow(overlay)
        axes[2 * idx + 1].set_title(f"T:{gt_name} | P:{pred_name}")
        axes[2 * idx + 1].axis("off")

    for idx in range(2 * len(vis_df), 16):
        axes[idx].axis("off")

    plt.tight_layout()
    if writer is not None and epoch is not None:
        writer.add_figure("GradCAM_Maps/Eval", fig, epoch)
    else:
        plt.savefig(ATTENTION_MAP_PATH, dpi=180)
    plt.close(fig)


def load_history_from_tensorboard(log_dir):
    """Read training history from TensorBoard event files."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(log_dir)
    ea.Reload()

    tag_data = {}
    for tag in ea.Tags().get("scalars", []):
        tag_data[tag] = {e.step: e.value for e in ea.Scalars(tag)}

    ce_epochs = sorted(tag_data.get("F1/train", {}).keys())
    rows = []
    for ep in ce_epochs:
        rows.append({
            "epoch": ep,
            "lr": tag_data.get("LearningRate", {}).get(ep),
            "train_loss": tag_data.get("Loss/train", {}).get(ep),
            "train_f1": tag_data.get("F1/train", {}).get(ep),
            "train_acc": tag_data.get("Acc/train", {}).get(ep),
            "train_auc": tag_data.get("AUC/train", {}).get(ep),
            "val_loss": tag_data.get("Loss/val", {}).get(ep),
            "val_f1": tag_data.get("F1/val", {}).get(ep),
            "val_acc": tag_data.get("Acc/val", {}).get(ep),
            "val_auc": tag_data.get("AUC/val", {}).get(ep),
            "test_f1": tag_data.get("F1/test", {}).get(ep),
            "test_acc": tag_data.get("Acc/test", {}).get(ep),
            "test_auc": tag_data.get("AUC/test", {}).get(ep),
        })
    history_df = pd.DataFrame(rows)

    sc_epochs = sorted(tag_data.get("SupCon/loss", {}).keys())
    supcon_history = [
        {"epoch": ep, "loss": tag_data["SupCon/loss"][ep],
         "lr": tag_data.get("SupCon/lr", {}).get(ep)}
        for ep in sc_epochs
    ] if sc_epochs else None

    return history_df, supcon_history


# ── ONNX export ──────────────────────────────────────────────────────────────
class _ClassifierWrapper(nn.Module):
    """Thin wrapper exposing only the classification path for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.classifier = model.classifier

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return F.softmax(logits, dim=1)


def export_to_onnx(model, cfg, device):
    """Export the trained model to ONNX with embedded preprocessing metadata."""
    import onnx

    img_size = cfg["image_size"]
    wrapper = _ClassifierWrapper(model).to(device)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    batch_dim = torch.export.Dim("batch_size", min=1, max=128)
    torch.onnx.export(
        wrapper,
        dummy_input,
        ONNX_MODEL_PATH,
        opset_version=18,
        input_names=["image"],
        output_names=["probabilities"],
        dynamic_shapes={
            "x": {0: batch_dim},
        },
    )

    onnx_model = onnx.load(ONNX_MODEL_PATH)

    metadata = {
        "model_name": "HierarchicalImageClassifier",
        "backbone": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        "num_classes": str(len(LABEL_DICT)),
        "image_size": str(img_size),
        "resize_size": str(cfg["resize_size"]),
        "normalize_mean": "0.485,0.456,0.406",
        "normalize_std": "0.229,0.224,0.225",
        "class_names": ",".join(LABEL_DICT.keys()),
        "output_type": "softmax_probabilities",
        "crop_black_threshold": str(cfg["crop_black_threshold"]),
        "preprocessing_steps": (
            "1) CropBlackBorders(threshold={thr}) "
            "2) Resize({rsz}) "
            "3) CenterCrop({img}) "
            "4) ToTensor (HWC uint8 → CHW float32 /255) "
            "5) Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"
        ).format(thr=cfg["crop_black_threshold"], rsz=cfg["resize_size"], img=img_size),
    }

    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value

    onnx.save(onnx_model, ONNX_MODEL_PATH)
    onnx.checker.check_model(onnx_model)

    file_size_mb = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024)
    print(f"ONNX model exported: {ONNX_MODEL_PATH} ({file_size_mb:.1f} MB)")
    print(f"  Input:  image [batch_size, 3, {img_size}, {img_size}] float32")
    print(f"  Output: probabilities [batch_size, {len(LABEL_DICT)}] float32 (softmax)")
    print(f"  Classes: {list(LABEL_DICT.keys())}")

    return ONNX_MODEL_PATH


def print_data_summary(train_df, val_df, test_df):
    """Print dataset statistics."""
    print(f"Image counts  — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print(f"Patient counts — train: {train_df['patient_name'].nunique()}, "
          f"val: {val_df['patient_name'].nunique()}, test: {test_df['patient_name'].nunique()}")
    print(f"Train dist: {dict(Counter(train_df['label_name']))}")
    print(f"Val   dist: {dict(Counter(val_df['label_name']))}")
    print(f"Test  dist: {dict(Counter(test_df['label_name']))}")
    print(f"\nVocal Cord distribution:")
    print(f"  Train VOC: {train_df['is_voc'].sum()}, Non-VOC: {(~train_df['is_voc']).sum()}")
    print(f"  Val   VOC: {val_df['is_voc'].sum()}, Non-VOC: {(~val_df['is_voc']).sum()}")
    print(f"  Test  VOC: {test_df['is_voc'].sum()}, Non-VOC: {(~test_df['is_voc']).sum()}")
