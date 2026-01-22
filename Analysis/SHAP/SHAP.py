"""
Train + ensemble evaluation script for cross-species (multi-task) liver microsomal stability.

Pipeline
--------
1) Load train/test datasets (MolDataset).
2) Build scaffold K-fold loaders.
3) Compute global class imbalance (neg/pos) per task from the train dataset (no fold duplication).
4) Train K models (one per fold) with:
   - Focal loss (+ optional pos_weight)
   - Pairwise margin ranking loss
   - Uncertainty weighting across tasks (learned log_vars)
   - Cosine schedule with warmup
   - Early stopping (saves base_model checkpoints)
5) Ensemble inference on test set by averaging fold probabilities.
6) Per-task threshold search:
   - F1-max (grid search)
   - Recall-constrained + best F1 (PR-curve thresholds)
   - Default 0.5 comparison
7) Save ensemble predictions as CSV and print final summary.
8) (Optional) SHAP analysis on descriptor features via shap_utils.

Notes
-----
- Label convention: y in {0,1}; y == -1 means missing label for that task.
- This script depends on your project modules:
  dataset_scaffold.py, model.py, utile.py, Focal_loss.py, shap_utils.py
"""

import os
import time
from math import cos, pi
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Subset

from torch_geometric.loader import DataLoader as GeometricDataLoader

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
try:
    from Dataset import build_scaffold_kfold_loader, MolDataset, seq_dict_smi
    from Model import ADME_Multimdal_Multitask
    from Utile import seed_set, create_logger, EarlyStopping, get_metric_func
    from Focal_loss import FocalLoss

    # SHAP utilities (optional; only used when shap_enable=True)
    from Shap_utils import (
        shap_rank_descriptors_for_task,
        collapse_binary_dummy_pairs,
        rank_from_shap_values,
        save_shap_violin_from_rank,
        save_barh_from_rank,
        plot_shap_beeswarm,
    )
except ImportError as e:
    raise ImportError(
        "[Error] Required project modules not found. "
        "Check your repo structure and PYTHONPATH."
    ) from e


# =====================================================================
# Helper functions: data preparation & misc
# =====================================================================
def _prep_fp_tensor(fp: torch.Tensor, batch_size: int, seq_len: int = 100) -> torch.Tensor:
    """
    Normalize fingerprint tensor shape to (B, L) of dtype long.
    Accepts (B,1,L), (B,L), or a flat 1D tensor with length B*L.
    """
    if fp.dim() == 3 and fp.size(1) == 1:
        fp = fp.squeeze(1)
    if fp.dim() == 1 and fp.numel() == batch_size * seq_len:
        fp = fp.view(batch_size, seq_len)
    return fp


def _prepare_desc_tensor(
    desc: Optional[torch.Tensor],
    batch_size: int,
    expected_desc_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Prepare descriptor tensor to shape (B, expected_desc_dim) float32.
    Pads/truncates if needed.
    """
    if expected_desc_dim <= 0:
        # Keep at least 1 dimension to avoid shape errors downstream
        expected_desc_dim = 1

    if desc is None:
        return torch.zeros((batch_size, expected_desc_dim), device=device, dtype=torch.float32)

    desc = desc.to(device).float()

    if desc.dim() == 1:
        # Possible shapes: (B,) or (B*D,)
        if desc.numel() == batch_size:
            desc = desc.unsqueeze(1)
        elif desc.numel() % batch_size == 0:
            desc = desc.view(batch_size, -1)
        else:
            desc = desc.unsqueeze(1)
    elif desc.dim() > 2:
        desc = desc.view(batch_size, -1)

    d = desc.size(-1)
    if d < expected_desc_dim:
        pad = torch.zeros((batch_size, expected_desc_dim - d), device=device, dtype=desc.dtype)
        desc = torch.cat([desc, pad], dim=1)
    elif d > expected_desc_dim:
        desc = desc[:, :expected_desc_dim]

    return desc


def _forward_three_modal(
    model: nn.Module,
    data,
    device: torch.device,
    logger=None,
):
    """
    Forward pass for the tri-modal ADME_Multimdal_Multitask:
      - fp: smil2vec tokens
      - graph: PyG batch
      - desc: descriptor vector

    Returns
    -------
    pooled : torch.Tensor
    task_outputs : List[torch.Tensor]  (each is (B,1) logits)
    """
    batch_size = data.num_graphs

    # Fingerprint tokens (smil2vec)
    fp = getattr(data, "smil2vec", None)
    if fp is None:
        fp = torch.zeros((batch_size, 100), device=device, dtype=torch.long)
    else:
        fp = _prep_fp_tensor(fp, batch_size=batch_size, seq_len=100).to(device)
        if fp.dtype != torch.long:
            fp = fp.long()

    # Descriptors
    expected_desc = int(getattr(model, "desc_in_dim", 0) or 0)
    desc_raw = getattr(data, "desc", None)
    desc = _prepare_desc_tensor(desc_raw, batch_size, expected_desc, device)

    # Graph is the PyG batch itself
    pooled, task_outputs = model({"fp": fp, "graph": data, "desc": desc})
    return pooled, task_outputs


def make_cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    base_lr: float = 1e-4,
) -> LambdaLR:
    """
    Cosine annealing with linear warmup.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        t = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        cos_decay = 0.5 * (1.0 + cos(pi * t))
        return max(min_lr / base_lr, cos_decay)

    return LambdaLR(optimizer, lr_lambda)


def compute_global_max_token_idx(datasets: List[MolDataset]) -> int:
    """
    Compute the maximum token index in smil2vec across provided datasets.
    Used to determine vocab_size safely.
    """
    gmax = 0
    for ds in datasets:
        for d in ds:
            if hasattr(d, "smil2vec") and d.smil2vec is not None:
                flat = d.smil2vec.view(-1)
                if flat.numel() == 0:
                    continue
                gmax = max(gmax, int(flat.max().item()))
    return gmax


def infer_desc_dim(train_desc_cols, tr_ds: MolDataset, test_ds: MolDataset) -> int:
    """
    Infer descriptor input dimension.
    Prefer train_desc_cols; otherwise inspect the first sample that has 'desc'.
    """
    if train_desc_cols is not None:
        return int(len(train_desc_cols))

    for ds in (tr_ds, test_ds):
        if len(ds) > 0 and hasattr(ds[0], "desc") and getattr(ds[0], "desc") is not None:
            d = ds[0].desc
            if isinstance(d, torch.Tensor):
                if d.dim() == 1:
                    return int(d.size(0))
                return int(d.view(d.size(0), -1).size(1))
    return 0


def save_csv(df: pd.DataFrame, path: str, logger=None) -> str:
    """
    Save dataframe as CSV. If input path is .parquet or any extension, enforce .csv.
    """
    base, _ = os.path.splitext(path)
    csv_path = base + ".csv"
    df.to_csv(csv_path, index=False)
    if logger:
        logger.info(f"[SAVE] Predictions -> {csv_path}")
    return csv_path


def compute_pos_weights_from_dataset(
    dataset: MolDataset,
    num_tasks: int,
    logger=None,
    task_names: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    Compute per-task pos_weight (= neg/pos) by iterating the dataset ONCE.
    This avoids the fold-duplication issue.

    Returns
    -------
    pos_weights : torch.Tensor (num_tasks,)
    """
    pos = np.zeros(num_tasks, dtype=np.float64)
    neg = np.zeros(num_tasks, dtype=np.float64)

    for sample in dataset:
        y = getattr(sample, "y", None)
        if y is None:
            continue
        if isinstance(y, torch.Tensor):
            y = y.view(-1).detach().cpu().numpy()
        else:
            y = np.asarray(y).reshape(-1)

        if y.size < num_tasks:
            continue

        for i in range(num_tasks):
            if y[i] == -1:
                continue
            if y[i] == 1:
                pos[i] += 1
            elif y[i] == 0:
                neg[i] += 1

    pos = np.maximum(pos, 1.0)
    weights = neg / pos

    if logger:
        logger.info("\n" + "=" * 70)
        logger.info("Global class distribution (computed from train dataset once)")
        logger.info("=" * 70)
        for i in range(num_tasks):
            name = task_names[i] if task_names else f"task{i}"
            total = int(pos[i] + neg[i])
            pos_ratio = (pos[i] / total * 100.0) if total > 0 else 0.0
            logger.info(
                f"{name.upper():>8} | pos={int(pos[i]):5d} ({pos_ratio:5.1f}%) "
                f"| neg={int(neg[i]):5d} ({100.0 - pos_ratio:5.1f}%) "
                f"| total={total:5d} | neg/pos={weights[i]:.2f}x"
            )
        logger.info("=" * 70 + "\n")

    return torch.tensor(weights, dtype=torch.float32)


# =====================================================================
# Loss wrapper: Focal + Ranking + Uncertainty weighting
# =====================================================================
class ImprovedMultiTaskLossWrapper(nn.Module):
    """
    Multi-task loss wrapper:
      - Focal loss per task (optional pos_weight)
      - Pairwise margin ranking loss per task
      - Uncertainty weighting across tasks (learned log_vars)

    Forward returns:
      (total_opt_loss, pooled, task_outputs, avg_raw_focal, avg_raw_ranking)
    """

    def __init__(
        self,
        model: nn.Module,
        num_tasks: int,
        gamma: float = 2.0,
        alpha: float = 0.75,
        ranking_margin: float = 0.2,
        ranking_lambda: float = 0.3,
    ):
        super().__init__()
        self.model = model
        self.num_tasks = num_tasks

        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.ranking_margin = ranking_margin
        self.ranking_lambda = ranking_lambda

        # Uncertainty parameters (one per task)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def compute_ranking_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Pairwise margin ranking loss:
        encourages logits(pos) > logits(neg) by a margin.
        """
        pos_mask = (targets == 1).squeeze(-1)
        neg_mask = (targets == 0).squeeze(-1)

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        pos_logits = logits[pos_mask]  # (P, 1) or (P,)
        neg_logits = logits[neg_mask]  # (N, 1) or (N,)

        pos_logits = pos_logits.view(-1)
        neg_logits = neg_logits.view(-1)

        # (P, 1) - (1, N) => (P, N)
        diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
        loss = F.relu(self.ranking_margin - diff)
        return loss.mean()

    def _focal_loss_call(self, logits: torch.Tensor, targets: torch.Tensor, pos_weight: Optional[torch.Tensor]):
        """
        Call focal loss with a defensive fallback depending on the implementation signature.
        """
        if pos_weight is None:
            return self.focal_loss(logits, targets)

        try:
            # Your current code uses this signature.
            return self.focal_loss(logits, targets, pos_weight)
        except TypeError:
            # Fallback if the implementation does not accept pos_weight.
            return self.focal_loss(logits, targets)

    def forward(self, data, device: torch.device, logger=None, pos_weights: Optional[torch.Tensor] = None):
        pooled, task_outputs = _forward_three_modal(self.model, data, device, logger)

        y = data.y
        if y.dim() == 1:
            y = y.view(-1, self.num_tasks)

        total_loss = 0.0
        focal_losses: List[float] = []
        ranking_losses: List[float] = []

        for i in range(self.num_tasks):
            task_logits = task_outputs[i]          # (B, 1)
            task_targets = y[:, i:i + 1]           # (B, 1)

            valid_mask = (task_targets != -1).squeeze(-1)
            if not valid_mask.any():
                continue

            valid_logits = task_logits[valid_mask]
            valid_targets = task_targets[valid_mask]

            pw = pos_weights[i:i + 1] if pos_weights is not None else None

            # Raw focal loss (for monitoring)
            focal_l = self._focal_loss_call(valid_logits, valid_targets, pw)

            # Raw ranking loss (for monitoring)
            ranking_l = self.compute_ranking_loss(valid_logits, valid_targets)

            # Task loss (raw)
            task_loss = focal_l + self.ranking_lambda * ranking_l

            # Uncertainty weighting: exp(-s) * L + s
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * task_loss + self.log_vars[i]

            total_loss = total_loss + weighted_loss

            focal_losses.append(float(focal_l.item()))
            ranking_losses.append(float(ranking_l.item()))

        avg_focal = float(np.mean(focal_losses)) if focal_losses else 0.0
        avg_ranking = float(np.mean(ranking_losses)) if ranking_losses else 0.0
        return total_loss, pooled, task_outputs, avg_focal, avg_ranking


# =====================================================================
# Train / Validate / Test
# =====================================================================
def train_one_epoch(
    epoch: int,
    wrapper_model: ImprovedMultiTaskLossWrapper,
    loader: GeometricDataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_type: str,
    metric: str,
    logger,
    max_grad_norm: float = 1.0,
    pos_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Returns
    -------
    avg_opt_loss : float
        Optimization loss (uncertainty weighted).
    avg_raw_focal : float
        Average raw focal loss (more interpretable training signal).
    """
    wrapper_model.train()

    total_opt = 0.0
    total_raw_focal = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        opt_loss, _, _, avg_focal, _ = wrapper_model(batch, device, logger, pos_weights)
        opt_loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapper_model.parameters(), max_grad_norm)
        optimizer.step()

        total_opt += float(opt_loss.item())
        total_raw_focal += float(avg_focal)
        n_batches += 1

    avg_opt = total_opt / max(n_batches, 1)
    avg_focal = total_raw_focal / max(n_batches, 1)

    if logger and (epoch % 5 == 0):
        logger.info(f"Epoch {epoch:3d} | Train OptLoss: {avg_opt:.4f} | Raw Focal: {avg_focal:.4f}")

    return avg_opt, avg_focal


@torch.no_grad()
def validate_one_epoch(
    epoch: int,
    wrapper_model: ImprovedMultiTaskLossWrapper,
    val_loader: GeometricDataLoader,
    device: torch.device,
    task_type: str,
    metric: str,
    logger,
) -> Tuple[float, float, float]:
    """
    Validate for one epoch.

    Returns
    -------
    val_opt_loss : float
    val_raw_focal : float
    avg_val_score : float
        Average metric across tasks (ignoring missing labels).
    """
    wrapper_model.eval()
    num_tasks = wrapper_model.num_tasks

    opt_losses: List[float] = []
    raw_focals: List[float] = []

    y_pred_list: Dict[int, List[float]] = {i: [] for i in range(num_tasks)}
    y_true_list: Dict[int, List[float]] = {i: [] for i in range(num_tasks)}

    for batch in val_loader:
        batch = batch.to(device)

        opt_loss, _, task_outputs, avg_focal, _ = wrapper_model(
            batch, device, logger, pos_weights=None
        )
        opt_losses.append(float(opt_loss.item()))
        raw_focals.append(float(avg_focal))

        y = batch.y
        if y.dim() == 1:
            y = y.view(-1, num_tasks)

        for i in range(num_tasks):
            logits = task_outputs[i].squeeze(-1)
            labels = y[:, i]
            valid = (labels != -1)
            if not valid.any():
                continue

            y_true = labels[valid].float().detach().cpu().numpy()
            if task_type == "classification":
                y_pred = torch.sigmoid(logits[valid]).detach().cpu().numpy()
            else:
                y_pred = logits[valid].detach().cpu().numpy()

            y_true_list[i].extend(y_true.tolist())
            y_pred_list[i].extend(y_pred.tolist())

    val_opt_loss = float(np.mean(opt_losses)) if opt_losses else 0.0
    val_raw_focal = float(np.mean(raw_focals)) if raw_focals else 0.0

    metric_func = get_metric_func(metric=metric)
    scores = []
    for i in range(num_tasks):
        if len(y_true_list[i]) > 0:
            scores.append(float(metric_func(y_true_list[i], y_pred_list[i])))

    avg_val_score = float(np.nanmean(scores)) if scores else 0.0

    if logger and (epoch % 5 == 0):
        logger.info(
            f"Epoch {epoch:3d} | Val OptLoss: {val_opt_loss:.4f} | "
            f"Val Raw Focal: {val_raw_focal:.4f} | {metric.upper()}: {avg_val_score:.4f}"
        )

    return val_opt_loss, val_raw_focal, avg_val_score


@torch.no_grad()
def test_model(
    model: nn.Module,
    criterion: nn.Module,
    test_loader: GeometricDataLoader,
    device: torch.device,
    task_type: str = "classification",
    metric: str = "auc",
    logger=None,
    criterion_list: Optional[List[nn.Module]] = None,
) -> Tuple[float, float]:
    """
    Generic test function (kept for compatibility).
    """
    model.eval()
    start = time.time()

    losses: List[float] = []
    num_tasks = getattr(model, "num_tasks", 3)

    y_pred_list: Dict[int, List[float]] = {i: [] for i in range(num_tasks)}
    y_true_list: Dict[int, List[float]] = {i: [] for i in range(num_tasks)}

    for batch in test_loader:
        batch = batch.to(device)
        _, task_outputs = _forward_three_modal(model, batch, device, logger)

        y = batch.y
        if y.dim() == 1:
            y = y.view(-1, num_tasks)

        for i in range(num_tasks):
            logits = task_outputs[i].squeeze(-1)
            labels = y[:, i]
            valid = (labels != -1)
            if not valid.any():
                continue

            logits_v = logits[valid]
            labels_v = labels[valid].float()

            crit = criterion_list[i] if criterion_list is not None else criterion
            loss = crit(logits_v, labels_v)
            losses.append(float(loss.item()))

            if task_type == "classification":
                y_pred = torch.sigmoid(logits_v).detach().cpu().numpy()
            else:
                y_pred = logits_v.detach().cpu().numpy()

            y_true = labels_v.detach().cpu().numpy()
            y_pred_list[i].extend(y_pred.tolist())
            y_true_list[i].extend(y_true.tolist())

    test_loss = float(np.mean(losses)) if losses else float("nan")

    metric_func = get_metric_func(metric=metric)
    scores = []
    for i in range(num_tasks):
        if len(y_true_list[i]) > 0:
            scores.append(float(metric_func(y_true_list[i], y_pred_list[i])))

    avg_score = float(np.nanmean(scores)) if scores else float("nan")
    elapsed = time.time() - start

    if logger:
        logger.info(f"[Test] Loss {test_loss:.4f} | {metric}: {avg_score:.4f} | {elapsed:.2f}s")
    else:
        print(f"Test Loss: {test_loss:.4f}, {metric}: {avg_score:.4f}")

    return test_loss, avg_score


# =====================================================================
# Threshold search utilities
# =====================================================================
def _binary_metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, float]:
    """
    Compute metrics at a given threshold.
    """
    y_pred = (y_score >= thr).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": float(thr),
        "f1_score": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    method: str = "f1",
    min_recall: float = 0.70,
    grid_size: int = 1000,
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal threshold.

    method:
      - "f1": maximize F1 by grid search over [0,1]
      - "recall_constrained": among thresholds with recall >= min_recall, maximize F1 (using PR curve)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if y_true.size == 0:
        thr = 0.5
        return thr, _binary_metrics_at_threshold(y_true, y_score, thr)

    if method == "f1":
        thresholds = np.linspace(0.0, 1.0, grid_size)
        best_thr, best_f1 = 0.5, -1.0

        for thr in thresholds:
            y_pred = (y_score >= thr).astype(int)
            if np.unique(y_pred).size < 2:
                f1 = 0.0
            else:
                f1 = f1_score(y_true, y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        return best_thr, _binary_metrics_at_threshold(y_true, y_score, best_thr)

    if method == "recall_constrained":
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
        if thresholds.size == 0:
            thr = 0.5
            return thr, _binary_metrics_at_threshold(y_true, y_score, thr)

        pre = precisions[:-1]
        rec = recalls[:-1]
        f1s = (2 * pre * rec) / (pre + rec + 1e-12)

        valid = rec >= float(min_recall)
        if not np.any(valid):
            idx = int(np.argmax(rec))
            best_thr = float(thresholds[idx])
        else:
            f1_masked = np.where(valid, f1s, -1.0)
            idx = int(np.argmax(f1_masked))
            best_thr = float(thresholds[idx])

        return best_thr, _binary_metrics_at_threshold(y_true, y_score, best_thr)

    raise ValueError(f"Unknown method: {method}")


def log_threshold_comparison(y_true: np.ndarray, y_score: np.ndarray, task_name: str, logger, min_recall: float = 0.70):
    """
    Log multiple threshold strategies and return the F1-max threshold.
    """
    logger.info("\n" + "=" * 78)
    logger.info(f"Threshold Analysis | {task_name}")
    logger.info("=" * 78)

    thr_f1, m_f1 = find_optimal_threshold(y_true, y_score, method="f1", min_recall=min_recall)
    thr_rc, m_rc = find_optimal_threshold(y_true, y_score, method="recall_constrained", min_recall=min_recall)

    thr_def = 0.5
    m_def = _binary_metrics_at_threshold(y_true, y_score, thr_def)

    logger.info(f"\n{'Method':<28} {'Thr':>8} {'F1':>8} {'Recall':>8} {'Prec':>8} {'Spec':>8}")
    logger.info("-" * 72)
    logger.info(f"{'F1 Maximization':<28} {thr_f1:>8.3f} {m_f1['f1_score']:>8.3f} {m_f1['recall']:>8.3f} {m_f1['precision']:>8.3f} {m_f1['specificity']:>8.3f}")
    logger.info(f"{f'Min Recall {min_recall:.0%} + Best F1':<28} {thr_rc:>8.3f} {m_rc['f1_score']:>8.3f} {m_rc['recall']:>8.3f} {m_rc['precision']:>8.3f} {m_rc['specificity']:>8.3f}")
    logger.info(f"{'Default (0.5)':<28} {thr_def:>8.3f} {m_def['f1_score']:>8.3f} {m_def['recall']:>8.3f} {m_def['precision']:>8.3f} {m_def['specificity']:>8.3f}")
    logger.info("=" * 78)

    return thr_f1, m_f1


# =====================================================================
# SHAP helper
# =====================================================================
def build_balanced_task_loader(
    dataset: MolDataset,
    task_index: int,
    batch_size: int,
    n_total: int,
    pos_frac: float = 0.5,
    seed: int = 42,
) -> GeometricDataLoader:
    """
    Build a roughly class-balanced loader for a specific task from a dataset.
    If one class is missing, fallback to a random subset.

    This is used for SHAP evaluation to avoid severe imbalance in explanations.
    """
    rng = np.random.default_rng(seed)
    pos_idx: List[int] = []
    neg_idx: List[int] = []

    for i in range(len(dataset)):
        y = dataset[i].y
        if isinstance(y, torch.Tensor):
            yi = int(y.view(-1)[task_index].item())
        else:
            yi = int(np.asarray(y).reshape(-1)[task_index])

        if yi == 1:
            pos_idx.append(i)
        elif yi == 0:
            neg_idx.append(i)

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        all_idx = np.arange(len(dataset))
        choose = rng.choice(all_idx, size=min(n_total, len(all_idx)), replace=False).tolist()
        subset = Subset(dataset, choose)
        return GeometricDataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=False)

    n_pos = min(int(n_total * pos_frac), len(pos_idx))
    n_neg = min(n_total - n_pos, len(neg_idx))

    choose_pos = rng.choice(pos_idx, size=n_pos, replace=False).tolist()
    choose_neg = rng.choice(neg_idx, size=n_neg, replace=False).tolist()
    choose = choose_pos + choose_neg
    rng.shuffle(choose)

    subset = Subset(dataset, choose)
    return GeometricDataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=False)


# =====================================================================
# main_train
# =====================================================================
def main_train(
    output_dir: str = "output",
    tag: str = "default",
    seed: int = 42,
    batch_size: int = 128,
    task_type: str = "classification",
    metric: str = "prc",
    base_lr: float = 1e-4,
    n_splits: int = 10,
    data_path: str = r"root/dataset",
    patience: int = 10,
    max_epochs: int = 200,
    perf_threshold: float = 0.5,
    perf_printout: bool = True,
    perf_plot: bool = False,
    # --- SHAP options ---
    shap_enable: bool = False,
    shap_backend: str = "kernel",
    shap_background_n: int = 100,
    shap_eval_n: int = 256,
    shap_topk: int = 20,
    kernel_nsamples: int = 2048,
    kernel_chunk: int = 1024,
):
    """
    Main training entry.

    Returns
    -------
    best_model : ADME_Multimdal_Multitask
        A model instance (loaded from the last fold during ensemble stage).
    fold_train_loss_means : List[float]
    fold_val_loss_means : List[float]
    """
    seed_set(seed)
    os.makedirs(output_dir, exist_ok=True)

    logger = create_logger(output_dir=output_dir, tag=tag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} | Tag: {tag}")

    dataset_names = {"train": "train.csv", "test": "test.csv"}
    tasks = ["human", "rat", "mouse"]
    num_tasks = len(tasks)

    # ---------------------------
    # Load datasets
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
        raise RuntimeError("train_desc_cols is None. Check preprocessing in MolDataset.")

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
    # Vocab size & descriptor dim
    # ---------------------------
    global_max_idx = compute_global_max_token_idx([tr_ds, test_dataset])
    base_vocab = len(seq_dict_smi) + 1
    vocab_size = max(base_vocab, global_max_idx + 1)

    desc_dim = infer_desc_dim(train_desc_cols, tr_ds, test_dataset)
    logger.info(f"Vocab size: {vocab_size} | Desc dim: {desc_dim}")

    # ---------------------------
    # Global pos_weight (neg/pos)
    # ---------------------------
    global_pos_weights = compute_pos_weights_from_dataset(
        dataset=tr_ds,
        num_tasks=num_tasks,
        logger=logger,
        task_names=tasks,
    ).to(device)
    global_pos_weights = torch.clamp(global_pos_weights, min=1.0, max=20.0)
    logger.info(f"Final global pos_weights (clamped): {global_pos_weights.detach().cpu().numpy()}")

    # ---------------------------
    # K-fold training
    # ---------------------------
    fold_ckpt_paths: List[str] = []
    fold_train_loss_means: List[float] = []
    fold_val_loss_means: List[float] = []

    for fold_idx, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        logger.info("\n" + "=" * 70)
        logger.info(f"Fold {fold_idx + 1}/{n_splits}")
        logger.info("=" * 70)

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
            dropout=0.4,
        ).to(device)

        wrapper_model = ImprovedMultiTaskLossWrapper(
            base_model,
            num_tasks=num_tasks,
            gamma=2.0,
            alpha=0.75,
            ranking_margin=0.2,
            ranking_lambda=0.3,
        ).to(device)

        pos_weights = global_pos_weights.clone()
        logger.info(f"Using global pos_weights: {pos_weights.detach().cpu().numpy()}")

        optimizer = torch.optim.AdamW(wrapper_model.parameters(), lr=base_lr, weight_decay=1e-3)
        scheduler = make_cosine_with_warmup(
            optimizer,
            max_epochs=max_epochs,
            warmup_epochs=5,
            base_lr=base_lr,
        )

        ckpt_path = os.path.join(output_dir, f"{tag}_fold{fold_idx + 1}.pt")
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            path=ckpt_path,
            mode="max",
        )

        trn_losses: List[float] = []
        val_losses: List[float] = []

        for epoch in range(1, max_epochs + 1):
            trn_opt, trn_raw = train_one_epoch(
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

            val_opt, val_raw, val_score = validate_one_epoch(
                epoch=epoch,
                wrapper_model=wrapper_model,
                val_loader=val_loader,
                device=device,
                task_type=task_type,
                metric=metric,
                logger=logger,
            )

            scheduler.step()

            trn_losses.append(trn_opt)
            val_losses.append(val_opt)

            # Save base_model only (wrapper has extra parameters not needed for inference)
            early_stopping(val_score, base_model)
            if early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        fold_ckpt_paths.append(ckpt_path)
        fold_train_loss_means.append(float(np.mean(trn_losses)) if trn_losses else float("nan"))
        fold_val_loss_means.append(float(np.mean(val_losses)) if val_losses else float("nan"))

    # ---------------------------
    # Ensemble testing
    # ---------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Ensemble Testing (Averaging all folds)")
    logger.info("=" * 70)

    n_test = len(test_dataset)
    ensemble_sum = np.zeros((n_test, num_tasks), dtype=np.float64)
    ensemble_cnt = np.zeros((n_test, num_tasks), dtype=np.float64)

    # Collect true labels once
    all_labels = []
    for batch in test_loader:
        y = batch.y
        if y.dim() == 1:
            y = y.view(-1, num_tasks)
        all_labels.append(y.detach().cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0, num_tasks), dtype=float)
    true_labels = {tasks[i]: all_labels[:, i] for i in range(num_tasks)} if all_labels.size else {}

    # Model template for loading checkpoints
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
        dropout=0.4,
    ).to(device)

    for fold_idx, ckpt in enumerate(fold_ckpt_paths):
        if not os.path.exists(ckpt):
            logger.warning(f"[Skip] Missing checkpoint: {ckpt}")
            continue

        logger.info(f"Loading fold {fold_idx + 1} from {ckpt} ...")
        test_base_model.load_state_dict(torch.load(ckpt, map_location=device))
        test_base_model.eval()

        probs_list = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                _, task_outputs = _forward_three_modal(test_base_model, batch, device, logger)
                batch_probs = torch.cat([torch.sigmoid(o).detach().cpu() for o in task_outputs], dim=1)
                probs_list.append(batch_probs.numpy())

        fold_probs = np.concatenate(probs_list, axis=0) if probs_list else np.zeros((n_test, num_tasks))
        ensemble_sum += fold_probs
        ensemble_cnt += 1.0

    ensemble_probs = ensemble_sum / np.maximum(ensemble_cnt, 1.0)

    # ---------------------------
    # Final evaluation: thresholds + metrics + CSV
    # ---------------------------
    final_frames: List[pd.DataFrame] = []
    summary_metrics: Dict[str, Dict[str, float]] = {}

    for i, sp in enumerate(tasks):
        if sp not in true_labels:
            logger.info(f"[{sp.upper()}] Missing labels; skip.")
            continue

        y_true = true_labels[sp]
        y_score = ensemble_probs[:, i]

        valid = (y_true != -1)
        if not np.any(valid):
            logger.info(f"[{sp.upper()}] No valid labels; skip.")
            continue

        y_true_v = y_true[valid].astype(int)
        y_score_v = y_score[valid].astype(float)

        logger.info("\n" + "=" * 78)
        logger.info(f"{sp.upper()} Performance Analysis")
        logger.info("=" * 78)

        opt_thr, opt_m = log_threshold_comparison(y_true_v, y_score_v, sp.upper(), logger, min_recall=0.70)

        y_pred = (y_score_v >= opt_thr).astype(int)

        # Robust AUC/AP (may fail if single-class)
        try:
            auc = float(roc_auc_score(y_true_v, y_score_v))
        except Exception:
            auc = 0.0
        try:
            ap = float(average_precision_score(y_true_v, y_score_v))
        except Exception:
            ap = 0.0

        acc = float(accuracy_score(y_true_v, y_pred))
        mcc = float(matthews_corrcoef(y_true_v, y_pred))
        f1 = float(f1_score(y_true_v, y_pred, zero_division=0))
        rec = float(recall_score(y_true_v, y_pred, zero_division=0))
        pre = float(precision_score(y_true_v, y_pred, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(y_true_v, y_pred).ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        logger.info(f"\n>>> Final @ Threshold={opt_thr:.3f} <<<")
        logger.info(f"Accuracy:    {acc:.4f}")
        logger.info(f"ROC-AUC:     {auc:.4f}")
        logger.info(f"AP (AUPR):   {ap:.4f}")
        logger.info(f"MCC:         {mcc:.4f}")
        logger.info(f"Recall:      {rec:.4f}")
        logger.info(f"Specificity: {spec:.4f}")
        logger.info(f"Precision:   {pre:.4f}")
        logger.info(f"F1-score:    {f1:.4f}")

        summary_metrics[sp] = {
            "Threshold": float(opt_thr),
            "AUC": auc,
            "AP": ap,
            "F1": f1,
            "Recall": rec,
            "Precision": pre,
        }

        final_frames.append(
            pd.DataFrame(
                {
                    "species": [sp] * len(y_true_v),
                    "y_true": y_true_v,
                    "y_score": y_score_v,
                    "y_pred": y_pred,
                    "threshold": opt_thr,
                    "fold": ["ensemble"] * len(y_true_v),
                }
            )
        )

    if final_frames:
        out_df = pd.concat(final_frames, ignore_index=True)
        save_csv(out_df, os.path.join(output_dir, f"{tag}_ensemble_preds.csv"), logger=logger)

    logger.info("\n" + "=" * 92)
    logger.info("FINAL OPTIMAL PERFORMANCE SUMMARY (Ensemble)")
    logger.info("=" * 92)
    logger.info(f"{'Task':<10} | {'Thr':<7} | {'AUC':<8} | {'AP':<8} | {'F1':<8} | {'Recall':<8} | {'Precision':<9}")
    logger.info("-" * 92)
    for sp in tasks:
        if sp not in summary_metrics:
            logger.info(f"{sp.upper():<10} | {'N/A':<7} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<9}")
            continue
        m = summary_metrics[sp]
        logger.info(
            f"{sp.upper():<10} | {m['Threshold']:<7.3f} | {m['AUC']:<8.4f} | {m['AP']:<8.4f} | "
            f"{m['F1']:<8.4f} | {m['Recall']:<8.4f} | {m['Precision']:<9.4f}"
        )
    logger.info("=" * 92)
    logger.info("Training complete.")

    # ---------------------------
    # SHAP (optional)
    # ---------------------------
    best_model = test_base_model  # uses the model instance from ensemble stage

    if shap_enable:
        if desc_dim == 0 or train_desc_cols is None:
            logger.warning("[SHAP] desc_in_dim=0 or missing desc schema. Skipping SHAP.")
        else:
            shap_dir = os.path.join(output_dir, "shap")
            os.makedirs(shap_dir, exist_ok=True)
            logger.info(f"[SHAP] Start backend={shap_backend} | out_dir={shap_dir}")

            for t_idx, t_name in enumerate(tasks):
                n_total = max(shap_eval_n, shap_background_n)

                t_loader = build_balanced_task_loader(
                    dataset=test_dataset,
                    task_index=t_idx,
                    batch_size=batch_size,
                    n_total=n_total,
                    pos_frac=0.5,
                    seed=seed + 1000 + t_idx,
                )
                if len(t_loader.dataset) == 0:
                    logger.warning(f"[SHAP] {t_name}: no samples; skip.")
                    continue

                # Run SHAP ranking for a task
                df_rank, sv, Xe, diag = shap_rank_descriptors_for_task(
                    full_model=best_model,
                    eval_loader=t_loader,
                    desc_cols=list(train_desc_cols),
                    task_index=t_idx,
                    device=device,
                    background_n=shap_background_n,
                    eval_n=shap_eval_n,
                    pool_n=None,
                    backend=shap_backend,
                    return_values=True,
                    kernel_nsamples=kernel_nsamples,
                    kernel_chunk=kernel_chunk,
                    collapse_yesno=False,
                )

                # Resolve feature names as used internally by SHAP pipeline
                feature_names = diag["feature_names"]

                # Optional: collapse binary dummy pairs (Yes/No)
                collapsed_names, Xe_c, sv_c = collapse_binary_dummy_pairs(
                    feature_names=feature_names,
                    X_eval=Xe,
                    shap_values=sv,
                    prefer_keep="Yes",
                    mode="diff",
                )

                # Re-rank based on collapsed SHAP values
                df_rank = rank_from_shap_values(sv_c, collapsed_names)

                # Output paths
                csv_path = os.path.join(shap_dir, f"shap_rank_{t_name}.csv")
                bees_png = os.path.join(shap_dir, f"shap_beeswarm_top{shap_topk}_{t_name}.png")
                violin_png = os.path.join(shap_dir, f"shap_violin_top{shap_topk}_{t_name}.png")
                bar_png = os.path.join(shap_dir, f"shap_bar_top{shap_topk}_{t_name}.png")

                df_rank.to_csv(csv_path, index=False, encoding="utf-8-sig")

                logger.info(
                    "[SHAP][%s] backend=%s | N_eval=%d | N_bg=%d | base~%.4f | mean f(x)=%.4f (d=%+.4f) | "
                    "frac(Sum(phi)<0)=%.2f%% | additivity_MAE=%.6f",
                    t_name,
                    diag.get("backend_used", "unknown"),
                    diag.get("N_eval", -1),
                    diag.get("N_bg", -1),
                    diag.get("base_est", float("nan")),
                    diag.get("mean_fx", float("nan")),
                    diag.get("delta_mean", float("nan")),
                    float(diag.get("frac_sumphi_neg", 0.0)) * 100.0,
                    float(diag.get("additivity_mae", float("nan"))),
                )

                # Save plots
                save_shap_violin_from_rank(
                    task_name=t_name,
                    df_rank=df_rank,
                    shap_values=sv_c,
                    X_eval=Xe_c,
                    feature_names=collapsed_names,
                    out_png=violin_png,
                    topk=shap_topk,
                    plot_type="violin",
                    dpi=500,
                )

                plot_shap_beeswarm(
                    task_name=t_name,
                    shap_values=sv_c,
                    X_eval=Xe_c,
                    feature_names=collapsed_names,
                    out_png=bees_png,
                    max_display=shap_topk,
                    symmetric_xlim=True,
                    dpi=500,
                )

                save_barh_from_rank(
                    df_rank=df_rank,
                    out_png=bar_png,
                    topk=shap_topk,
                    title=None,
                    dpi=500,
                )

                logger.info(f"[SHAP] {t_name}: saved -> {violin_png} , {bar_png} , {bees_png}")

            logger.info("[SHAP] Done.")

    return best_model, fold_train_loss_means, fold_val_loss_means


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    model, trn_losses, val_losses = main_train(
        tag="improved_model",
        perf_threshold=0.5,
        perf_plot=True,
        perf_printout=True,
        shap_enable=True,
        shap_backend="kernel",
        shap_background_n=100,
        shap_eval_n=256,
        shap_topk=20,
        kernel_nsamples=2048,
        kernel_chunk=1024,
    )

    print("\n=== Training Summary ===")
    print(f"Average Train Loss: {np.nanmean(trn_losses):.4f}")
    print(f"Average Val Loss:   {np.nanmean(val_losses):.4f}")
