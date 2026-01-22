"""
Training utilities for a multi-modal, multi-task model (FP tokens + Graph + Descriptors).

This file provides:
- Input preparation helpers (fingerprint token tensor, descriptor tensor)
- A unified forward function for three modalities
- Cosine LR scheduler with warmup
- Task-wise pos_weight estimation for imbalance
- Improved multi-task loss wrapper:
    * Focal loss (with optional pos_weight)
    * Margin-based ranking loss
    * Uncertainty weighting (learnable log variances)
- Train / Validate / Test loops

Notes
-----
- This module depends on project-specific files:
    dataset_scaffold.py, model.py, utile.py, Focal_loss.py
- Label convention: y == -1 indicates missing/invalid labels for a given task.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from math import cos, pi
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

# -----------------------------------------------------------------------------
# Project-specific imports
# -----------------------------------------------------------------------------
try:
    from Dataset import build_scaffold_kfold_loader, MolDataset, seq_dict_smi  # noqa: F401
    from Model import ADME_Multimdal_Multitask  # noqa: F401
    from Utile import seed_set, create_logger, EarlyStopping, printPerformance, get_metric_func  # noqa: F401
    from Focal_loss import FocalLoss
except ImportError:
    # Keep the file importable even if project modules are not available.
    # Raise later when actually used.
    FocalLoss = None  # type: ignore
    print("[Error] Required project modules not found. "
          "Make sure dataset_scaffold.py/model.py/utile.py/Focal_loss.py are on PYTHONPATH.")


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _prep_fp_tensor(fp: torch.Tensor, batch_size: int, seq_len: int = 100) -> torch.Tensor:
    """
    Normalize fingerprint token tensor shapes.

    Expected output shape: (B, L)
      - If input is (B, 1, L) -> squeeze to (B, L)
      - If input is flat (B*L,) -> reshape to (B, L)
    """
    if fp.dim() == 3 and fp.size(1) == 1:
        fp = fp.squeeze(1)
    if fp.dim() == 1 and fp.numel() == batch_size * seq_len:
        fp = fp.view(batch_size, seq_len)
    return fp


def _forward_three_modal(
    model: nn.Module,
    data,
    device: torch.device,
    logger=None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Forward pass for three modalities:
      - fp: tokenized fingerprint sequence (default: zeros)
      - desc: descriptor vector (auto-pad/trim to model.desc_in_dim if provided)
      - graph: PyG batch object (data)

    Returns
    -------
    pooled : torch.Tensor
        A pooled/shared representation from the model.
    task_outputs : list[torch.Tensor]
        A list of logits per task. Each element is typically (B, 1).
    """
    batch_size = data.num_graphs

    # -------------------------
    # 1) Fingerprint tokens (FP)
    # -------------------------
    fp = getattr(data, "smil2vec", None)
    if fp is None:
        fp = torch.zeros((batch_size, 100), device=device, dtype=torch.long)
    else:
        fp = _prep_fp_tensor(fp, batch_size=batch_size, seq_len=100).to(device)
        if fp.dtype != torch.long:
            fp = fp.long()

    # -------------------------
    # 2) Descriptors (DESC)
    # -------------------------
    expected_desc = int(getattr(model, "desc_in_dim", 0) or 0)
    desc = getattr(data, "desc", None)

    if desc is None:
        # If expected_desc is unknown, keep at least 1 dim to avoid empty tensor issues.
        desc_dim = expected_desc if expected_desc > 0 else 1
        desc = torch.zeros((batch_size, desc_dim), device=device, dtype=torch.float32)
    else:
        desc = desc.to(device).float()

        # Normalize descriptor shape to (B, D)
        if desc.dim() == 1:
            if desc.numel() == batch_size:
                desc = desc.unsqueeze(1)
            elif desc.numel() % batch_size == 0:
                desc = desc.view(batch_size, -1)
            else:
                desc = desc.unsqueeze(1)
        elif desc.dim() > 2:
            desc = desc.view(batch_size, -1)

        # Pad/trim to match expected_desc if model declares it.
        if expected_desc > 0:
            d = desc.size(-1)
            if d < expected_desc:
                pad = torch.zeros((batch_size, expected_desc - d), device=device, dtype=desc.dtype)
                desc = torch.cat([desc, pad], dim=1)
            elif d > expected_desc:
                desc = desc[:, :expected_desc]

    # -------------------------
    # 3) Graph (PyG Batch)
    # -------------------------
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
    Cosine decay with linear warmup scheduler.

    During warmup:
        lr = base_lr * (epoch+1)/warmup_epochs
    After warmup:
        lr follows cosine decay down to min_lr.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)

        t = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        cos_decay = 0.5 * (1 + cos(pi * t))
        return max(min_lr / base_lr, cos_decay)

    return LambdaLR(optimizer, lr_lambda)


def compute_pos_weight_per_task(loaders, num_tasks: int, device: torch.device) -> torch.Tensor:
    """
    Estimate per-task pos_weight = (#neg / #pos) using labels from given loaders.

    Label convention:
      - y in {0, 1} for valid labels
      - y == -1 indicates missing label for that task

    Returns
    -------
    pos_weight : torch.Tensor, shape (num_tasks,)
        pos_weight for BCEWithLogits-style losses (or custom focal loss if supported).
    """
    pos = torch.zeros(num_tasks, dtype=torch.float64)
    neg = torch.zeros(num_tasks, dtype=torch.float64)

    for loader in loaders:
        for batch in loader:
            y = batch.y
            if y.dim() == 1:
                y = y.view(-1, num_tasks)

            for i in range(num_tasks):
                valid = (y[:, i] != -1)
                if not valid.any():
                    continue

                yi = y[valid, i]
                pos[i] += (yi == 1).sum().item()
                neg[i] += (yi == 0).sum().item()

    pos = torch.clamp(pos, min=1.0)  # avoid division by zero
    w = (neg / pos).to(torch.float32)
    return w.to(device)


def compute_global_max_token_idx(datasets) -> int:
    """
    Scan datasets to find the global maximum token index in `smil2vec`.

    Useful for setting vocab size dynamically, when tokens are integer IDs.
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


def _infer_desc_dim(train_desc_cols, tr_ds, test_ds) -> int:
    """
    Infer descriptor input dimension.

    Priority:
      1) train_desc_cols length (if provided)
      2) peek first sample's `desc` shape in train/test dataset
      3) fallback to 0
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


def _save_parquet_or_csv(df, path: str, logger=None) -> str:
    """
    Save predictions to CSV (parquet fallback).

    For portability in GitHub examples, this function always writes CSV.
    """
    base, _ = os.path.splitext(path)
    csv_path = base + ".csv"
    df.to_csv(csv_path, index=False)
    if logger:
        logger.info(f"[SAVE] Predictions -> {csv_path}")
    return csv_path


# -----------------------------------------------------------------------------
# Improved Multi-Task Loss Wrapper
# -----------------------------------------------------------------------------
class ImprovedMultiTaskLossWrapper(nn.Module):
    """
    Improved multi-task loss wrapper:
      - Focal loss per task (optionally with pos_weight)
      - Margin-based ranking loss per task
      - Uncertainty weighting across tasks using learnable log variances

    Total loss per task:
        L_task = L_focal + ranking_lambda * L_rank
        L_weighted = exp(-s_i) * L_task + s_i
    where s_i is learnable log variance for task i.
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
        if FocalLoss is None:
            raise ImportError("FocalLoss is not available. Check Focal_loss.py import.")

        self.model = model
        self.num_tasks = num_tasks

        # Focal loss implementation is expected to support:
        #   focal_loss(logits, targets, pos_weight_optional)
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)

        # Ranking loss hyperparameters
        self.ranking_margin = ranking_margin
        self.ranking_lambda = ranking_lambda

        # Uncertainty weighting parameters (one per task)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def compute_ranking_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Margin ranking loss: encourage logits(positive) > logits(negative) by a margin.

        logits: (N, 1) or (N,)
        targets: (N, 1) or (N,) with values in {0,1}
        """
        # Ensure shape is (N,)
        logits_ = logits.view(-1)
        targets_ = targets.view(-1)

        pos_mask = targets_ == 1
        neg_mask = targets_ == 0

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        pos_logits = logits_[pos_mask]  # (P,)
        neg_logits = logits_[neg_mask]  # (N,)

        # Compute pairwise margin violations: margin - (pos - neg)
        # Shapes: (P, 1) - (1, N) => (P, N)
        pos_exp = pos_logits.unsqueeze(1)
        neg_exp = neg_logits.unsqueeze(0)

        ranking_loss = F.relu(self.ranking_margin - (pos_exp - neg_exp))
        return ranking_loss.mean()

    def forward(
        self,
        data,
        device: torch.device,
        logger=None,
        pos_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], float, float]:
        """
        Forward pass with multi-task loss computation.

        Returns
        -------
        total_loss : torch.Tensor
            Optimization loss (includes uncertainty weighting; can be negative in rare cases).
        pooled : torch.Tensor
            Model pooled representation.
        task_outputs : list[torch.Tensor]
            Task logits.
        avg_focal : float
            Mean raw focal loss across valid tasks (for logging).
        avg_ranking : float
            Mean raw ranking loss across valid tasks (for logging).
        """
        pooled, task_outputs = _forward_three_modal(self.model, data, device, logger)

        y = data.y
        if y.dim() == 1:
            y = y.view(-1, self.num_tasks)

        total_loss = 0.0
        focal_losses: List[float] = []
        ranking_losses: List[float] = []

        for i in range(self.num_tasks):
            task_logits = task_outputs[i]         # typically (B, 1) or (B,)
            task_targets = y[:, i:i + 1]          # keep (B, 1) for masking

            valid_mask = (task_targets != -1).squeeze(-1)
            if not valid_mask.any():
                continue

            valid_logits = task_logits[valid_mask]
            valid_targets = task_targets[valid_mask]

            # 1) Raw focal loss
            pw = pos_weights[i:i + 1] if pos_weights is not None else None
            focal_l = self.focal_loss(valid_logits, valid_targets, pw)

            # 2) Raw ranking loss
            ranking_l = self.compute_ranking_loss(valid_logits, valid_targets)

            # 3) Task combined loss
            task_loss = focal_l + self.ranking_lambda * ranking_l

            # 4) Uncertainty weighting: exp(-s)*L + s
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * task_loss + self.log_vars[i]
            total_loss = total_loss + weighted_loss

            focal_losses.append(float(focal_l.item()))
            ranking_losses.append(float(ranking_l.item()))

        avg_focal = float(np.mean(focal_losses)) if focal_losses else 0.0
        avg_ranking = float(np.mean(ranking_losses)) if ranking_losses else 0.0

        return total_loss, pooled, task_outputs, avg_focal, avg_ranking


# -----------------------------------------------------------------------------
# Train / Validate / Test
# -----------------------------------------------------------------------------
def train_one_epoch(
    epoch: int,
    wrapper_model: ImprovedMultiTaskLossWrapper,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_type: str,
    metric: str,
    logger,
    max_grad_norm: float = 1.0,
    pos_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """
    Train for a single epoch.

    Returns
    -------
    avg_opt_loss : float
        Mean optimization loss (uncertainty-weighted).
    avg_raw_focal : float
        Mean raw focal loss (more interpretable training signal).
    """
    wrapper_model.train()

    total_opt_loss = 0.0
    total_raw_focal = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        opt_loss, _, _, raw_focal, _ = wrapper_model(
            batch, device=device, logger=logger, pos_weights=pos_weights
        )

        opt_loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapper_model.parameters(), max_grad_norm)
        optimizer.step()

        total_opt_loss += float(opt_loss.item())
        total_raw_focal += float(raw_focal)
        num_batches += 1

    avg_opt_loss = total_opt_loss / max(num_batches, 1)
    avg_raw_focal = total_raw_focal / max(num_batches, 1)

    if logger and epoch % 5 == 0:
        logger.info(
            f"Epoch {epoch:3d} | Train OptLoss: {avg_opt_loss:.4f} | Raw Focal: {avg_raw_focal:.4f}"
        )

    return avg_opt_loss, avg_raw_focal


@torch.no_grad()
def validate(
    epoch: int,
    wrapper_model: ImprovedMultiTaskLossWrapper,
    val_loader,
    device: torch.device,
    task_type: str,
    metric: str,
    logger,
) -> Tuple[float, float, float]:
    """
    Validate model and compute an average metric across tasks.

    Returns
    -------
    val_opt_loss : float
        Mean optimization loss (uncertainty-weighted).
    val_raw_focal : float
        Mean raw focal loss.
    avg_val_score : float
        Mean metric score across tasks (nan-robust).
    """
    wrapper_model.eval()
    num_tasks = wrapper_model.num_tasks

    opt_losses: List[float] = []
    raw_focals: List[float] = []

    y_pred_list: Dict[int, List[float]] = {i: [] for i in range(num_tasks)}
    y_label_list: Dict[int, List[float]] = {i: [] for i in range(num_tasks)}

    metric_func = get_metric_func(metric=metric)

    for batch in val_loader:
        batch = batch.to(device)

        opt_loss, _, task_outputs, raw_focal, _ = wrapper_model(
            batch, device=device, logger=logger, pos_weights=None
        )

        opt_losses.append(float(opt_loss.item()))
        raw_focals.append(float(raw_focal))

        y = batch.y
        if y.dim() == 1:
            y = y.view(-1, num_tasks)

        for i in range(num_tasks):
            y_pred = task_outputs[i].view(-1)
            y_label = y[:, i].view(-1)

            valid_idx = (y_label != -1)
            if not valid_idx.any():
                continue

            y_pred_v = y_pred[valid_idx]
            y_label_v = y_label[valid_idx].float()

            if task_type == "classification":
                y_pred_list[i].extend(torch.sigmoid(y_pred_v).cpu().numpy().tolist())
            else:
                y_pred_list[i].extend(y_pred_v.cpu().numpy().tolist())

            y_label_list[i].extend(y_label_v.cpu().numpy().tolist())

    val_opt_loss = float(np.mean(opt_losses)) if opt_losses else 0.0
    val_raw_focal = float(np.mean(raw_focals)) if raw_focals else 0.0

    task_scores: List[float] = []
    for i in range(num_tasks):
        if len(y_label_list[i]) > 0:
            task_scores.append(metric_func(y_label_list[i], y_pred_list[i]))

    avg_val_score = float(np.nanmean(task_scores)) if task_scores else 0.0

    if logger and epoch % 5 == 0:
        logger.info(
            f"Epoch {epoch:3d} | Val OptLoss: {val_opt_loss:.4f} | "
            f"Val Raw Focal: {val_raw_focal:.4f} | {metric.upper()}: {avg_val_score:.4f}"
        )

    return val_opt_loss, val_raw_focal, avg_val_score


@torch.no_grad()
def test(
    model: nn.Module,
    criterion: nn.Module,
    test_loader,
    device: torch.device,
    task_type: str = "classification",
    metric: str = "auc",
    logger=None,
    criterion_list: Optional[List[nn.Module]] = None,
) -> Tuple[float, float]:
    """
    Evaluate on test set.

    Parameters
    ----------
    model : nn.Module
        Underlying model (not the wrapper).
    criterion : nn.Module
        Default loss criterion for tasks.
    criterion_list : list[nn.Module] | None
        Optional per-task criterion list.

    Returns
    -------
    test_loss : float
        Mean loss over all task-sample pairs with valid labels.
    avg_metric : float
        Mean metric across tasks.
    """
    model.eval()
    start = time.time()

    losses: List[float] = []
    num_tasks = int(getattr(model, "num_tasks", 3))

    y_pred_list: Dict[int, List[float]] = {i: [] for i in range(num_tasks)}
    y_label_list: Dict[int, List[float]] = {i: [] for i in range(num_tasks)}

    metric_func = get_metric_func(metric=metric)

    for batch in test_loader:
        data = batch.to(device)
        _, task_outputs = _forward_three_modal(model, data, device, logger)

        y = data.y
        if y.dim() == 1:
            y = y.view(-1, num_tasks)

        # Ensure (B, T)
        assert y.dim() == 2 and y.size(1) == num_tasks, "Labels must be shaped (B, num_tasks)."

        for i in range(num_tasks):
            y_pred = task_outputs[i].view(-1)
            y_label = y[:, i].view(-1)

            valid_idx = (y_label != -1)
            if not valid_idx.any():
                continue

            y_pred_v = y_pred[valid_idx]
            y_label_v = y_label[valid_idx].float()

            crit = criterion_list[i] if criterion_list is not None else criterion
            loss = crit(y_pred_v, y_label_v)
            losses.append(float(loss.item()))

            if task_type == "classification":
                y_pred_list[i].extend(torch.sigmoid(y_pred_v).cpu().numpy().tolist())
            else:
                y_pred_list[i].extend(y_pred_v.cpu().numpy().tolist())

            y_label_list[i].extend(y_label_v.cpu().numpy().tolist())

    test_loss = float(np.mean(losses)) if losses else float("nan")

    task_scores: List[float] = []
    for i in range(num_tasks):
        if len(y_label_list[i]) > 0:
            task_scores.append(metric_func(y_label_list[i], y_pred_list[i]))

    avg_metric = float(np.nanmean(task_scores)) if task_scores else float("nan")
    duration = time.time() - start

    if logger:
        logger.info(f"[Test] Loss {test_loss:.4f} | {metric}: {avg_metric:.4f} | {duration:.2f}s")
    else:
        print(f"Test Loss: {test_loss:.4f}, {metric}: {avg_metric:.4f}, {duration:.2f}s")

    return test_loss, avg_metric
