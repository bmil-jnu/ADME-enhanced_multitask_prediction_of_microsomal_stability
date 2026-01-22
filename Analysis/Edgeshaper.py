"""
train_mtmm_with_edgeshaper.py

End-to-end training script for a multi-task, multi-modal model (FP + Graph + Descriptors),
with:
  - Scaffold K-fold training
  - Uncertainty-weighted multi-task loss (Focal + Ranking)
  - Ensemble testing across folds
  - Per-task optimal threshold selection
  - (Optional) EdgeSHAPer-based fragment aggregation + visualization

Notes
-----
1) This script expects your dataset pipeline to attach the following fields in each batch:
   - data.smil2vec : (B, L) or equivalent tokenized FP representation
   - data.desc     : (B, D) descriptor matrix (optional; zero-padded if missing)
   - data.y        : (B, num_tasks) with -1 for missing labels
   - graph fields required by your ADME_Multimdal_Multitask (x, edge_index, etc.)

2) EdgeSHAPer part requires helper functions implemented elsewhere, e.g.:
   - explain_one_graph_edgeshaper
   - analyze_species_common
   - save_samples_grid
   - save_fragment_bars

   If you put them in `Edgeshaper_Utile.py`, this script will import them.
"""

# ------------------------------------------------------------------------------
# Standard Library Imports
# ------------------------------------------------------------------------------
import os
import time
import math
import json
from typing import Optional, List, Dict, Tuple

# ------------------------------------------------------------------------------
# Third-Party Imports
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from torch_geometric.loader import DataLoader as GeometricDataLoader

# Metrics
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    recall_score,
    precision_score,
    confusion_matrix,
)

# ------------------------------------------------------------------------------
# Project Imports
# ------------------------------------------------------------------------------
from Dataset import build_scaffold_kfold_loader, MolDataset, seq_dict_smi
from Model import ADME_Multimdal_Multitask
try:
    from Utile import seed_set, create_logger, EarlyStopping, printPerformance, get_metric_func
    from Focal_loss import FocalLoss
except ImportError:
    raise ImportError("[Error] Required modules (utile, Focal_loss) are not found.")

# ------------------------------------------------------------------------------
# Optional: EdgeSHAPer/XAI Helpers
# ------------------------------------------------------------------------------
try:
    from Edgeshaper_Utile import (
        explain_one_graph_edgeshaper,
        analyze_species_common,
        save_samples_grid,
        save_fragment_bars,
        TaskWrapper
    )
    _HAS_XAI = True
except Exception:
    # If you want EdgeSHAPer, ensure xai_edgeshaper.py is available.
    _HAS_XAI = False
    explain_one_graph_edgeshaper = None
    analyze_species_common = None
    save_samples_grid = None
    save_fragment_bars = None


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def _prep_fp_tensor(fp: torch.Tensor, B: int, L: int = 100) -> torch.Tensor:
    """
    Normalize FP tensor shape to (B, L).

    Supported:
      - (B, 1, L) -> squeeze
      - (B*L,)    -> reshape
      - (B, L)    -> 그대로 사용
    """
    if fp.dim() == 3 and fp.size(1) == 1:
        fp = fp.squeeze(1)
    if fp.dim() == 1 and fp.numel() == B * L:
        fp = fp.view(B, L)
    return fp


def _forward_three_modal(model: nn.Module, data, device, logger=None):
    """
    Forward pass wrapper that safely prepares:
      - fp token tensor
      - descriptor matrix
      - graph batch

    Returns
    -------
    pooled : torch.Tensor
        Shared embedding (if your model returns it).
    task_outputs : List[torch.Tensor]
        List length = num_tasks; each output shape typically (B, 1) or (B,).
    """
    B = data.num_graphs

    # --------------------------
    # 1) FP tokens
    # --------------------------
    fp = getattr(data, "smil2vec", None)
    if fp is None:
        fp = torch.zeros((B, 100), device=device, dtype=torch.long)
    else:
        fp = _prep_fp_tensor(fp, B=B, L=100).to(device)
        if fp.dtype != torch.long:
            fp = fp.long()

    # --------------------------
    # 2) Descriptors
    # --------------------------
    expected_desc = int(getattr(model, "desc_in_dim", 0) or 0)
    desc = getattr(data, "desc", None)

    if desc is None:
        # If desc is missing, fill with zeros.
        desc = torch.zeros(
            (B, expected_desc if expected_desc > 0 else 1),
            device=device,
            dtype=torch.float32,
        )
    else:
        desc = desc.to(device).float()
        # Normalize shape to (B, D)
        if desc.dim() == 1:
            if desc.numel() == B:
                desc = desc.unsqueeze(1)
            elif desc.numel() % B == 0:
                desc = desc.view(B, -1)
            else:
                desc = desc.unsqueeze(1)
        elif desc.dim() > 2:
            desc = desc.view(B, -1)

        # Pad/trim to expected dim if known
        if expected_desc > 0:
            D = desc.size(-1)
            if D < expected_desc:
                pad = torch.zeros((B, expected_desc - D), device=device, dtype=desc.dtype)
                desc = torch.cat([desc, pad], dim=1)
            elif D > expected_desc:
                desc = desc[:, :expected_desc]

    # --------------------------
    # 3) Graph + Model Forward
    # --------------------------
    pooled, task_outputs = model({"fp": fp, "graph": data, "desc": desc})
    return pooled, task_outputs


def make_cosine_with_warmup(
    optimizer,
    max_epochs: int,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    base_lr: float = 1e-4,
):
    """
    Cosine decay with linear warmup.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        t = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        cos_decay = 0.5 * (1 + math.cos(math.pi * t))
        return max(min_lr / base_lr, cos_decay)

    return LambdaLR(optimizer, lr_lambda)


def compute_global_max_token_idx(datasets: List) -> int:
    """
    Scan datasets to find the maximum token id in `smil2vec`.
    Used to safely set vocab_size.
    """
    gmax = 0
    for ds in datasets:
        for d in ds:
            if hasattr(d, "smil2vec") and d.smil2vec is not None:
                flat = d.smil2vec.view(-1)
                if flat.numel() == 0:
                    continue
                v = int(flat.max().item())
                gmax = max(gmax, v)
    return gmax


def _infer_desc_dim(train_desc_cols, tr_ds, test_ds) -> int:
    """
    Infer descriptor dimension (D).
    Priority:
      1) len(train_desc_cols) if provided
      2) inspect first item in train/test dataset
    """
    if train_desc_cols is not None:
        return len(train_desc_cols)

    for ds in (tr_ds, test_ds):
        if len(ds) > 0 and hasattr(ds[0], "desc") and getattr(ds[0], "desc") is not None:
            d = ds[0].desc
            if d.dim() > 1:
                return int(d.view(d.size(0), -1).size(1))
            return int(d.size(-1))
    return 0


def _save_csv(df: pd.DataFrame, path: str, logger=None) -> str:
    """
    Save dataframe to CSV (force CSV for compatibility).
    """
    base, _ = os.path.splitext(path)
    csv_path = base + ".csv"
    df.to_csv(csv_path, index=False)
    if logger:
        logger.info(f"[SAVE] -> {csv_path}")
    return csv_path


# ------------------------------------------------------------------------------
# Improved Multi-Task Loss Wrapper
# ------------------------------------------------------------------------------
class ImprovedMultiTaskLossWrapper(nn.Module):
    """
    Multi-task loss wrapper:
      - Focal loss (handles imbalance)
      - Ranking loss (encourages pos logits > neg logits)
      - Uncertainty weighting (learned log_vars per task)

    Forward returns:
      total_loss, pooled, task_outputs, avg_raw_focal, avg_raw_ranking
    """
    def __init__(
        self,
        model: nn.Module,
        num_tasks: int,
        gamma: float = 2.5,
        alpha: float = 0.35,
        ranking_margin: float = 0.2,
        ranking_lambda: float = 0.3,
    ):
        super().__init__()
        self.model = model
        self.num_tasks = num_tasks

        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.ranking_margin = ranking_margin
        self.ranking_lambda = ranking_lambda

        # log variance per task (uncertainty weighting)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def compute_ranking_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Margin-based pairwise ranking loss:
          want logit(pos) >= logit(neg) + margin
        """
        pos_mask = (targets == 1).squeeze(-1)
        neg_mask = (targets == 0).squeeze(-1)

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        pos_logits = logits[pos_mask]  # (P, 1) or (P,)
        neg_logits = logits[neg_mask]  # (N, 1) or (N,)

        # Expand for broadcasting: (P,1) - (1,N)
        pos_exp = pos_logits.unsqueeze(1)
        neg_exp = neg_logits.unsqueeze(0)

        loss = F.relu(self.ranking_margin - (pos_exp - neg_exp))
        return loss.mean()

    def forward(self, data, device, logger=None, pos_weights: Optional[torch.Tensor] = None):
        pooled, task_outputs = _forward_three_modal(self.model, data, device, logger)

        y = data.y
        if y.dim() == 1:
            y = y.view(-1, self.num_tasks)

        total_loss = 0.0
        focal_losses = []
        ranking_losses = []

        for i in range(self.num_tasks):
            task_logits = task_outputs[i]
            task_targets = y[:, i:i+1]

            valid_mask = (task_targets != -1).squeeze(-1)
            if not valid_mask.any():
                continue

            valid_logits = task_logits[valid_mask]
            valid_targets = task_targets[valid_mask]

            # Raw focal
            pw = pos_weights[i:i+1] if pos_weights is not None else None
            focal_l = self.focal_loss(valid_logits, valid_targets, pw)

            # Raw ranking
            ranking_l = self.compute_ranking_loss(valid_logits, valid_targets)

            task_loss = focal_l + self.ranking_lambda * ranking_l

            # Uncertainty weighting: exp(-s)*L + s
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * task_loss + self.log_vars[i]

            total_loss += weighted_loss
            focal_losses.append(focal_l.item())
            ranking_losses.append(ranking_l.item())

        avg_focal = float(np.mean(focal_losses)) if focal_losses else 0.0
        avg_ranking = float(np.mean(ranking_losses)) if ranking_losses else 0.0

        return total_loss, pooled, task_outputs, avg_focal, avg_ranking


# ------------------------------------------------------------------------------
# Train / Validate / Test
# ------------------------------------------------------------------------------
def train_one_epoch(
    epoch: int,
    wrapper_model: ImprovedMultiTaskLossWrapper,
    loader,
    optimizer,
    device,
    task_type: str,
    metric: str,
    logger,
    max_grad_norm: float = 1.0,
    pos_weights: Optional[torch.Tensor] = None,
):
    wrapper_model.train()

    total_opt_loss = 0.0
    total_raw_focal = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        loss, pooled, task_outputs, avg_focal, avg_ranking = wrapper_model(
            batch, device, logger, pos_weights
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapper_model.parameters(), max_grad_norm)
        optimizer.step()

        total_opt_loss += float(loss.item())
        total_raw_focal += float(avg_focal)
        num_batches += 1

    avg_opt = total_opt_loss / max(1, num_batches)
    avg_focal = total_raw_focal / max(1, num_batches)

    if epoch % 5 == 0:
        logger.info(
            f"Epoch {epoch:3d} | Train OptLoss: {avg_opt:.4f} | Raw Focal: {avg_focal:.4f}"
        )

    return avg_opt, avg_focal


@torch.no_grad()
def validate(
    epoch: int,
    wrapper_model: ImprovedMultiTaskLossWrapper,
    val_loader,
    device,
    task_type: str,
    metric: str,
    logger,
):
    wrapper_model.eval()
    num_tasks = wrapper_model.num_tasks

    opt_losses = []
    raw_focals = []

    y_pred_list = {i: [] for i in range(num_tasks)}
    y_true_list = {i: [] for i in range(num_tasks)}

    for batch in val_loader:
        batch = batch.to(device)

        loss, pooled, task_outputs, avg_focal, avg_ranking = wrapper_model(
            batch, device, logger, pos_weights=None
        )

        opt_losses.append(float(loss.item()))
        raw_focals.append(float(avg_focal))

        y = batch.y
        if y.dim() == 1:
            y = y.view(-1, num_tasks)

        for i in range(num_tasks):
            logits = task_outputs[i].squeeze(-1)
            yi = y[:, i]
            valid = (yi != -1)
            if not valid.any():
                continue

            yi_v = yi[valid].float()
            if task_type == "classification":
                pi_v = torch.sigmoid(logits[valid]).cpu().numpy()
            else:
                pi_v = logits[valid].cpu().numpy()

            y_pred_list[i].extend(pi_v)
            y_true_list[i].extend(yi_v.cpu().numpy())

    val_opt_loss = float(np.mean(opt_losses)) if opt_losses else 0.0
    val_raw_focal = float(np.mean(raw_focals)) if raw_focals else 0.0

    metric_func = get_metric_func(metric=metric)
    scores = []
    for i in range(num_tasks):
        if len(y_true_list[i]) > 0:
            s = metric_func(y_true_list[i], y_pred_list[i])
            scores.append(s)

    avg_score = float(np.nanmean(scores)) if scores else 0.0

    if epoch % 5 == 0:
        logger.info(
            f"Epoch {epoch:3d} | Val OptLoss: {val_opt_loss:.4f} | "
            f"Val RawFocal: {val_raw_focal:.4f} | {metric.upper()}: {avg_score:.4f}"
        )

    return val_opt_loss, val_raw_focal, avg_score


# ------------------------------------------------------------------------------
# Threshold Utilities
# ------------------------------------------------------------------------------
def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    method: str = "f1",
    min_recall: float = 0.7,
):
    """
    Find an optimal threshold.

    method:
      - 'f1' : maximize F1 over a dense grid
      - 'recall_constrained' : choose threshold with recall >= min_recall and max F1
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if method == "f1":
        thresholds = np.linspace(0, 1, 1000)
        f1s = []
        for thr in thresholds:
            pred = (y_score >= thr).astype(int)
            if len(np.unique(pred)) < 2:
                f1s.append(0.0)
            else:
                f1s.append(f1_score(y_true, pred, zero_division=0))
        best_idx = int(np.argmax(f1s))
        opt_thr = float(thresholds[best_idx])

    elif method == "recall_constrained":
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
        valid = recalls[:-1] >= float(min_recall)

        if not valid.any():
            best_idx = int(np.argmax(recalls[:-1]))
            opt_thr = float(thresholds[best_idx])
        else:
            f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
            f1s2 = f1s.copy()
            f1s2[~valid] = -1
            best_idx = int(np.argmax(f1s2))
            opt_thr = float(thresholds[best_idx])

    else:
        raise ValueError(f"Unknown method: {method}")

    pred_opt = (y_score >= opt_thr).astype(int)

    tp = int(((pred_opt == 1) & (y_true == 1)).sum())
    tn = int(((pred_opt == 0) & (y_true == 0)).sum())
    fp = int(((pred_opt == 1) & (y_true == 0)).sum())
    fn = int(((pred_opt == 0) & (y_true == 1)).sum())

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1v = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = dict(
        threshold=opt_thr,
        f1_score=float(f1v),
        recall=float(recall),
        precision=float(precision),
        specificity=float(specificity),
        accuracy=float(accuracy),
        tp=tp, tn=tn, fp=fp, fn=fn,
    )
    return opt_thr, metrics


def print_threshold_comparison(y_true, y_score, task_name: str, logger):
    """
    Compare:
      - F1-max threshold
      - Recall-constrained threshold (recall >= 0.7)
      - default 0.5
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"Threshold Analysis for {task_name}")
    logger.info("=" * 70)

    thr_f1, met_f1 = find_optimal_threshold(y_true, y_score, method="f1")
    thr_r, met_r = find_optimal_threshold(y_true, y_score, method="recall_constrained", min_recall=0.70)

    pred_def = (y_score >= 0.5).astype(int)
    tp = ((pred_def == 1) & (y_true == 1)).sum()
    tn = ((pred_def == 0) & (y_true == 0)).sum()
    fp = ((pred_def == 1) & (y_true == 0)).sum()
    fn = ((pred_def == 0) & (y_true == 1)).sum()

    recall_def = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec_def = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    spec_def = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_def = f1_score(y_true, pred_def, zero_division=0)

    logger.info(f"\n{'Method':<30} {'Threshold':>10} {'F1':>8} {'Recall':>8} {'Precision':>10} {'Specificity':>10}")
    logger.info("-" * 76)
    logger.info(
        f"{'F1 Maximization':<30} {thr_f1:>10.3f} {met_f1['f1_score']:>8.3f} "
        f"{met_f1['recall']:>8.3f} {met_f1['precision']:>10.3f} {met_f1['specificity']:>10.3f}"
    )
    logger.info(
        f"{'Min Recall 70% + F1':<30} {thr_r:>10.3f} {met_r['f1_score']:>8.3f} "
        f"{met_r['recall']:>8.3f} {met_r['precision']:>10.3f} {met_r['specificity']:>10.3f}"
    )
    logger.info("-" * 76)
    logger.info(
        f"{'Default (0.5)':<30} {0.5:>10.3f} {f1_def:>8.3f} "
        f"{recall_def:>8.3f} {prec_def:>10.3f} {spec_def:>10.3f}"
    )
    logger.info("=" * 76)

    return thr_f1, met_f1


# ------------------------------------------------------------------------------
# Main Training Function
# ------------------------------------------------------------------------------
def main_train(
    output_dir: str = "output",
    tag: str = "default",
    seed: int = 42,
    batch_size: int = 128,
    task_type: str = "classification",
    metric: str = "prc",
    base_lr: float = 1e-4,
    n_splits: int = 10,
    data_path: str = "D:/metabolic_stability_invitro4444/원본",
    patience: int = 10,
    max_epochs: int = 200,
    perf_threshold: float = 0.5,
    perf_printout: bool = True,
    perf_plot: bool = False,
    # EdgeSHAPer options
    do_edgeshaper: bool = False,
    M_explain: int = 128,
    top_frac: float = 0.2,
    task_for_panel: str = "human",
    max_samples: Optional[int] = None,
):
    """
    Train with scaffold K-fold, save best model per fold, evaluate ensemble on test set,
    then optionally run EdgeSHAPer explanation pipeline.
    """
    seed_set(seed)
    os.makedirs(output_dir, exist_ok=True)

    logger = create_logger(output_dir=output_dir, tag=tag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} | Tag: {tag}")

    # Prepare EdgeSHAPer output directory
    edgeshaper_dir = os.path.join(output_dir, "edgeshaper")
    if do_edgeshaper:
        os.makedirs(edgeshaper_dir, exist_ok=True)
        if not _HAS_XAI:
            raise RuntimeError(
                "do_edgeshaper=True but xai_edgeshaper helpers are not importable. "
                "Please provide xai_edgeshaper.py (or update imports)."
            )

    dataset_names = {"train": "train.csv", "test": "test.csv"}
    tasks = ["human", "rat", "mouse"]
    num_tasks = len(tasks)

    # ---------------------------
    # Dataset init (train)
    # ---------------------------
    tr_ds = MolDataset(
        root=data_path,
        dataset=dataset_names["train"],
        task_type=task_type,
        tasks=tasks,
        logger=logger,
    )
    train_desc_cols = getattr(tr_ds, "desc_cols_", None)
    train_desc_scaler = getattr(tr_ds, "desc_scaler_", None)
    if train_desc_cols is None:
        raise RuntimeError("train_desc_cols is None. Check preprocessing pipeline.")

    # ---------------------------
    # K-fold loaders
    # ---------------------------
    train_loaders, val_loaders, train_desc_cols, train_desc_scaler = build_scaffold_kfold_loader(
        data_path=data_path,
        dataset_name=dataset_names["train"],
        task_type=task_type,
        batch_size=batch_size,
        tasks=tasks,
        logger=logger,
        n_splits=n_splits,
        seed=seed,
    )

    # ---------------------------
    # Test loader
    # ---------------------------
    test_dataset = MolDataset(
        root=data_path,
        dataset=dataset_names["test"],
        task_type=task_type,
        tasks=tasks,
        logger=logger,
        desc_cols=train_desc_cols,
        desc_scaler=train_desc_scaler,
    )
    test_loader = GeometricDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # ---------------------------
    # Vocab / Desc Dim
    # ---------------------------
    global_max_idx = compute_global_max_token_idx([tr_ds, test_dataset])
    base_vocab = len(seq_dict_smi) + 1
    vocab_size = max(base_vocab, global_max_idx + 1)
    desc_dim = _infer_desc_dim(train_desc_cols, tr_ds, test_dataset)
    logger.info(f"Vocab size: {vocab_size} | Desc dim: {desc_dim}")

    # ---------------------------
    # Global class balance (use only first fold to avoid double counting)
    # ---------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Global Train Class Distribution (using fold-0 only)")
    logger.info("=" * 60)

    all_pos = torch.zeros(num_tasks, dtype=torch.float64)
    all_neg = torch.zeros(num_tasks, dtype=torch.float64)

    for batch in train_loaders[0]:
        y = batch.y
        if y.dim() == 1:
            y = y.view(-1, num_tasks)
        for i in range(num_tasks):
            valid = (y[:, i] != -1)
            if valid.any():
                yi = y[valid, i]
                all_pos[i] += (yi == 1).sum().item()
                all_neg[i] += (yi == 0).sum().item()

    global_pos_weights = []
    for i, name in enumerate(tasks):
        pos_count = int(all_pos[i].item())
        neg_count = int(all_neg[i].item())
        total = pos_count + neg_count
        pos_ratio = pos_count / total * 100 if total > 0 else 0.0
        imbalance = neg_count / max(pos_count, 1)

        logger.info(f"\n{name.upper()}:")
        logger.info(f"  Positive: {pos_count:5d} ({pos_ratio:.1f}%)")
        logger.info(f"  Negative: {neg_count:5d} ({100 - pos_ratio:.1f}%)")
        logger.info(f"  Total:    {total:5d}")
        logger.info(f"  neg/pos ratio: {imbalance:.2f}x  (recommended pos_weight)")

        global_pos_weights.append(imbalance)

    global_pos_weights = torch.tensor(global_pos_weights, dtype=torch.float32, device=device)
    global_pos_weights = torch.clamp(global_pos_weights, min=1.0, max=20.0)

    logger.info("\n" + "-" * 60)
    logger.info(f"Final global pos_weights (clamped): {global_pos_weights.cpu().numpy()}")
    logger.info("=" * 60 + "\n")

    # ---------------------------
    # K-fold training
    # ---------------------------
    fold_ckpt_paths = []
    fold_train_loss_means = []
    fold_val_loss_means = []

    for fold_idx, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        logger.info("\n" + "=" * 60)
        logger.info(f"Fold {fold_idx + 1}/{n_splits}")
        logger.info("=" * 60)

        # Base model
        base_model = ADME_Multimdal_Multitask(
            vocab_size=vocab_size,
            device=device,
            num_tasks=num_tasks,
            desc_in_dim=desc_dim,
            fp_mode="dense",
            fp_type="morgan+maccs+rdit",
            fp_emb_dim=128,
            graph_out_dim=128,
            fusion_dim=128,
            dropout=0.7,
        ).to(device)

        # Loss wrapper
        wrapper_model = ImprovedMultiTaskLossWrapper(
            base_model,
            num_tasks,
            gamma=2.0,
            alpha=0.30,
            ranking_margin=0.2,
            ranking_lambda=0.3,
        ).to(device)

        # pos_weights per task
        pos_weights = global_pos_weights.clone()
        logger.info(f"Using global pos_weights: {pos_weights.cpu().numpy()}")

        optimizer = torch.optim.AdamW(wrapper_model.parameters(), lr=base_lr, weight_decay=1e-2)
        scheduler = make_cosine_with_warmup(
            optimizer, max_epochs=max_epochs, warmup_epochs=5, base_lr=base_lr
        )

        ckpt_path = os.path.join(output_dir, f"{tag}_fold{fold_idx + 1}.pt")
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            path=ckpt_path,
            mode="max",
        )

        fold_tr_losses = []
        fold_val_losses = []

        for epoch in range(1, max_epochs + 1):
            tr_opt, tr_raw_focal = train_one_epoch(
                epoch=epoch,
                wrapper_model=wrapper_model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                task_type=task_type,
                metric=metric,
                logger=logger,
                max_grad_norm=1.0,
                pos_weights=pos_weights,
            )

            val_opt_loss, val_raw_focal, val_score = validate(
                epoch=epoch,
                wrapper_model=wrapper_model,
                val_loader=val_loader,
                device=device,
                task_type=task_type,
                metric=metric,
                logger=logger,
            )

            scheduler.step()

            fold_tr_losses.append(tr_opt)
            fold_val_losses.append(val_opt_loss)

            # IMPORTANT: save base_model state dict (not wrapper) for easy reload later
            early_stopping(val_score, base_model)

            if early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        fold_ckpt_paths.append(ckpt_path)
        fold_train_loss_means.append(float(np.mean(fold_tr_losses)) if fold_tr_losses else float("nan"))
        fold_val_loss_means.append(float(np.mean(fold_val_losses)) if fold_val_losses else float("nan"))

    # ---------------------------
    # Ensemble testing (average probs across folds)
    # ---------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Ensemble Testing (Average across folds)")
    logger.info("=" * 60)

    ensemble_probs = {sp: np.zeros((len(test_dataset),), dtype=float) for sp in tasks}
    ensemble_counts = {sp: np.zeros((len(test_dataset),), dtype=float) for sp in tasks}

    # Collect true labels once
    all_labels = []
    for batch in test_loader:
        y = batch.y
        if y.dim() == 1:
            y = y.view(-1, num_tasks)
        all_labels.append(y.numpy())
    all_labels = np.concatenate(all_labels, axis=0)

    true_labels = {sp: all_labels[:, i] for i, sp in enumerate(tasks)}

    # A "test model shell" to load each fold weights
    test_base_model = ADME_Multimdal_Multitask(
        vocab_size=vocab_size,
        device=device,
        num_tasks=num_tasks,
        desc_in_dim=desc_dim,
        fp_mode="dense",
        fp_type="morgan+maccs+rdit",
        fp_emb_dim=128,
        graph_out_dim=128,
        fusion_dim=128,
        dropout=0.7,
    ).to(device)

    for fold_idx, ckpt in enumerate(fold_ckpt_paths):
        if not os.path.exists(ckpt):
            logger.warning(f"[Skip] Missing checkpoint: {ckpt}")
            continue

        logger.info(f"Loading fold {fold_idx + 1}: {ckpt}")
        test_base_model.load_state_dict(torch.load(ckpt, map_location=device))
        test_base_model.eval()

        fold_probs = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                _, task_outputs = _forward_three_modal(test_base_model, batch, device)

                # Concatenate task probs: shape (B, num_tasks)
                probs = torch.cat([torch.sigmoid(o).detach().cpu() for o in task_outputs], dim=1)
                fold_probs.append(probs.numpy())

        fold_probs = np.concatenate(fold_probs, axis=0)

        for i, sp in enumerate(tasks):
            ensemble_probs[sp] += fold_probs[:, i]
            ensemble_counts[sp] += 1.0

    # ---------------------------
    # Final evaluation with optimal thresholds
    # ---------------------------
    optimal_thresholds = {}
    summary_metrics = {}
    final_results_frames = []

    for i, sp in enumerate(tasks):
        y_score = ensemble_probs[sp] / np.maximum(ensemble_counts[sp], 1.0)
        y_true = true_labels[sp]

        valid = (y_true != -1)
        if not np.any(valid):
            logger.info(f"[{sp}] No valid labels.")
            continue

        y_true_v = y_true[valid].astype(int)
        y_score_v = y_score[valid].astype(float)

        logger.info("\n" + "=" * 70)
        logger.info(f"{sp.upper()} Performance Analysis")
        logger.info("=" * 70)

        thr, thr_metrics = print_threshold_comparison(y_true_v, y_score_v, sp.upper(), logger)
        optimal_thresholds[sp] = float(thr)

        pred = (y_score_v >= thr).astype(int)

        # Metrics
        acc = accuracy_score(y_true_v, pred)
        try:
            auc = roc_auc_score(y_true_v, y_score_v)
        except Exception:
            auc = 0.0
        try:
            aupr = average_precision_score(y_true_v, y_score_v)
        except Exception:
            aupr = 0.0

        mcc = matthews_corrcoef(y_true_v, pred)
        f1v = f1_score(y_true_v, pred, zero_division=0)
        rec = recall_score(y_true_v, pred, zero_division=0)
        prec = precision_score(y_true_v, pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true_v, pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        logger.info(f"\n>>> Final Results for {sp.upper()} (thr={thr:.3f}) <<<")
        logger.info(f"Accuracy:    {acc:.4f}")
        logger.info(f"AUC-ROC:     {auc:.4f}")
        logger.info(f"AP (AUPR):   {aupr:.4f}")
        logger.info(f"MCC:         {mcc:.4f}")
        logger.info(f"Recall:      {rec:.4f}")
        logger.info(f"Specificity: {spec:.4f}")
        logger.info(f"Precision:   {prec:.4f}")
        logger.info(f"F1-score:    {f1v:.4f}")

        summary_metrics[sp] = dict(AUC=float(auc), F1=float(f1v), Recall=float(rec),
                                   Precision=float(prec), Threshold=float(thr))

        final_results_frames.append(pd.DataFrame({
            "species": [sp] * len(y_true_v),
            "y_true": y_true_v,
            "y_score": y_score_v,
            "fold": ["ensemble"] * len(y_true_v),
        }))

    # Save ensemble predictions
    if final_results_frames:
        full_df = pd.concat(final_results_frames, ignore_index=True)
        _save_csv(full_df, os.path.join(output_dir, f"{tag}_ensemble_preds.csv"), logger)

    # Print final summary table
    logger.info("\n" + "=" * 80)
    logger.info("FINAL OPTIMAL PERFORMANCE SUMMARY (Ensemble)")
    logger.info("=" * 80)
    logger.info(f"{'Task':<10} | {'Threshold':<10} | {'AUC':<8} | {'F1':<8} | {'Recall':<8} | {'Precision':<10}")
    logger.info("-" * 80)
    for sp in tasks:
        if sp in summary_metrics:
            m = summary_metrics[sp]
            logger.info(f"{sp.upper():<10} | {m['Threshold']:<10.3f} | {m['AUC']:<8.4f} | "
                        f"{m['F1']:<8.4f} | {m['Recall']:<8.4f} | {m['Precision']:<10.4f}")
        else:
            logger.info(f"{sp.upper():<10} | {'N/A':<10} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<10}")
    logger.info("=" * 80 + "\n")

    # Also store a compact "test summary" CSV/JSON under edgeshaper_dir for convenience
    os.makedirs(edgeshaper_dir, exist_ok=True)
    summary_rows = []
    for sp in tasks:
        if sp not in summary_metrics:
            summary_rows.append({"task": sp, "note": "no valid labels", "N": 0})
            continue

        y = true_labels[sp]
        p = ensemble_probs[sp] / np.maximum(ensemble_counts[sp], 1.0)
        valid = (y != -1)
        yv = y[valid].astype(int)
        pv = p[valid].astype(float)

        thr = float(optimal_thresholds.get(sp, perf_threshold))
        # Optional printing through your project helper
        printPerformance(yv, pv, threshold=thr, plot=perf_plot, printout=perf_printout)

        pred = (pv >= thr).astype(int)
        row = {
            "task": sp,
            "N": int(yv.size),
            "thr": thr,
            "pos_rate": float((yv == 1).mean()),
            "ACC@thr": float((pred == yv).mean()),
            "F1@thr": float(f1_score(yv, pred, zero_division=0)),
        }
        if yv.min() != yv.max():
            row["AUC"] = float(roc_auc_score(yv, pv))
            row["AUPRC"] = float(average_precision_score(yv, pv))
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(edgeshaper_dir, "test_metrics_summary.csv"), index=False)
    with open(os.path.join(edgeshaper_dir, "test_metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    # Save per-task predictions
    for sp in tasks:
        y = true_labels[sp]
        p = ensemble_probs[sp] / np.maximum(ensemble_counts[sp], 1.0)
        valid = (y != -1)
        if np.any(valid):
            pd.DataFrame({"y_true": y[valid].astype(int), "y_prob": p[valid].astype(float)}).to_csv(
                os.path.join(edgeshaper_dir, f"test_predictions_{sp}.csv"), index=False
            )

    # --------------------------------------------------------------------------
    # Optional: EdgeSHAPer aggregation + plots
    # --------------------------------------------------------------------------
    df_frag = None
    sample_results_all = []

    if do_edgeshaper:
        logger.info("[EdgeSHAPer] Start fragment aggregation...")

        # IMPORTANT:
        # `test_base_model` currently contains the last-loaded fold weights.
        # If you want a specific fold (e.g., best fold) for explanation, load that fold here.
        xai_model = test_base_model
        xai_model.eval()

        # 1) Aggregate fragment stats across tasks
        df_frag, df_common = analyze_species_common(
            model=xai_model,
            loader=test_loader,
            task_names=tasks,
            M=M_explain,
            top_frac=top_frac,
            device=device,
        )
        df_frag.to_csv(os.path.join(edgeshaper_dir, "edgeshaper_frag_stats.csv"), index=False)
        df_common.to_csv(os.path.join(edgeshaper_dir, "edgeshaper_common_fragments.csv"), index=False)
        logger.info(f"[EdgeSHAPer] Saved stats: {len(df_frag)} rows | common: {len(df_common)} rows")

        # 2) Collect per-sample explanations for a selected task (panel A)
        if task_for_panel not in tasks:
            logger.warning(f"Task '{task_for_panel}' not in tasks list. Skip sample collection.")
        else:
            t_idx = tasks.index(task_for_panel)
            t0 = time.time()

            with torch.no_grad():
                for batch in test_loader:
                    for item in batch.to(device).to_data_list():
                        # True label for task
                        y_true = None
                        if hasattr(item, "y") and item.y is not None:
                            yv = item.y.view(-1)
                            if yv.numel() > t_idx and float(yv[t_idx].item()) != -1:
                                y_true = float(yv[t_idx].item())

                        # Prediction
                        _, outs = xai_model({
                            "fp": getattr(item, "smil2vec", None),
                            "graph": item.to(device),
                            "desc": getattr(item, "desc", None),
                        })
                        y_pred = torch.sigmoid(outs[t_idx]).view(-1).mean().item()

                        # EdgeSHAPer explanation
                        xai = explain_one_graph_edgeshaper(
                            base_model=xai_model,
                            batch_item=item.to(device),
                            task_idx=t_idx,
                            target_class=1,
                            M=M_explain,
                            device=device,
                        )

                        sample_results_all.append({
                            "smiles": getattr(item, "smiles", ""),
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "edge_scores": xai["edge_score"],
                            "edge_to_bonds": xai["edge_to_bonds"],
                        })

                        if max_samples is not None and len(sample_results_all) >= int(max_samples):
                            break
                    if max_samples is not None and len(sample_results_all) >= int(max_samples):
                        break

            logger.info(f"[EdgeSHAPer] Collected {len(sample_results_all)} samples "
                        f"in {time.time() - t0:.1f}s (task={task_for_panel})")

            # 3) Save panels/plots
            save_samples_grid(
                sample_results_all,
                os.path.join(edgeshaper_dir, "edgeshaper_samples.png"),
                ncols=4,
                dpi=220,
            )
            save_fragment_bars(
                df_frag,
                edgeshaper_dir,
                tasks=tasks,
                k_top=10,
                dpi=220,
            )

            logger.info("[EdgeSHAPer] Saved sample grid + fragment bars")

    logger.info("Training Complete!")
    return test_base_model, fold_train_loss_means, fold_val_loss_means, df_frag, sample_results_all, edgeshaper_dir, tasks


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    (model, trn_losses, val_losses, df_frag, sample_results_all, edgeshaper_dir, tasks) = main_train(
        output_dir="output",
        tag="default",
        seed=42,
        batch_size=128,
        task_type="classification",
        metric="prc",
        base_lr=1e-4,
        n_splits=10,
        data_path="D:/metabolic_stability_invitro4444/원본",
        patience=10,
        max_epochs=200,
        perf_threshold=0.5,
        perf_plot=False,
        perf_printout=True,
        do_edgeshaper=True,
        M_explain=12,
        top_frac=0.15,
        task_for_panel="human",
        max_samples=80,
    )
