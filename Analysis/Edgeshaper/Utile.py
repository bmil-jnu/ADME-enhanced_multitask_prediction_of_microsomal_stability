# utile.py
"""
Utility helpers for:
- Reproducibility (seed)
- Logging
- Optimizer / Scheduler
- Early stopping
- Safe metrics (ROC-AUC, PR-AUC, AP) with edge-case handling
- Reporting / plotting
- Class imbalance helpers (pos_weight)

Notes
-----
- This file is written to be robust against:
  * missing/invalid labels (-1)
  * NaN/Inf in predictions
  * single-class labels (AUC/AP undefined)
"""

from __future__ import annotations

import os
import time
import random
import math
import logging
from typing import Optional, Callable, Literal, Dict, Any, Tuple

import numpy as np
import torch

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib.pyplot as plt


# =============================================================================
# Reproducibility
# =============================================================================
def seed_set(seed: int = 2024) -> None:
    """Set random seeds for reproducibility across python/numpy/torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic settings (may reduce speed but improves reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Logger
# =============================================================================
def create_logger(output_dir: str = "output", tag: str = "default") -> logging.Logger:
    """
    Create a logger that writes both to console and to a log file.

    - Console: colorized format
    - File: plain format
    """
    log_name = f"training_{tag}_{time.strftime('%Y-%m-%d')}.log"
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Avoid duplicate handlers when re-imported in notebooks
    if logger.handlers:
        logger.handlers.clear()

    # Plain and colorized formats
    plain_fmt = "[%(asctime)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    color_fmt = (
        "\033[92m[%(asctime)s]\033[0m "
        "\033[93m(%(filename)s %(lineno)d):\033[0m "
        "\033[95m%(levelname)-5s\033[0m %(message)s"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, log_name))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt=plain_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    return logger


# =============================================================================
# Optimizer / Scheduler
# =============================================================================
def build_optimizer(
    model: torch.nn.Module,
    optimizer_type: Literal["sgd", "adam", "adamw", "ranger"] = "adamw",
    base_lr: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """
    Build an optimizer by name.

    Parameters
    ----------
    optimizer_type : {"sgd","adam","adamw","ranger"}
        - "ranger" requires: pip install ranger-adabelief
    """
    params = model.parameters()
    opt = optimizer_type.lower()

    if opt == "sgd":
        return torch.optim.SGD(
            params, lr=base_lr, momentum=momentum, weight_decay=weight_decay, nesterov=True
        )
    if opt == "adam":
        return torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
    if opt == "adamw":
        return torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)

    if opt == "ranger":
        try:
            from ranger_adabelief import Ranger
        except Exception as e:
            raise ImportError("Ranger optimizer not installed. `pip install ranger-adabelief`") from e
        return Ranger(params, lr=base_lr, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: Literal["reduce"] = "reduce",
    factor: float = 0.1,
    patience: int = 10,
    min_lr: float = 1e-5,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build LR scheduler.

    Currently supported:
    - "reduce": ReduceLROnPlateau (mode='min')
    """
    if scheduler_type == "reduce":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr
        )
    raise NotImplementedError(f"Unsupported scheduler_type: {scheduler_type}")


# =============================================================================
# Early Stopping
# =============================================================================
class EarlyStopping:
    """
    Early stopping utility.

    mode:
      - "min": smaller is better (e.g., loss)
      - "max": larger is better (e.g., AUROC / AUPR / PRC-AUC)
    """

    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 1e-3,
        path: str = "checkpoint.pt",
        trace_func: Callable[[str], None] = print,
        mode: Literal["min", "max"] = "max",
    ):
        self.patience = int(patience)
        self.verbose = bool(verbose)
        self.delta = float(delta)
        self.path = str(path)
        self.trace_func = trace_func
        self.mode = mode

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, val_score: float, model: torch.nn.Module) -> None:
        """Update early-stopping state given a new validation score."""
        score = float(val_score)

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(score, model)
            return

        improved = False
        if self.mode == "min":
            improved = score < (self.best_score - self.delta)
        else:  # "max"
            improved = score > (self.best_score + self.delta)

        if improved:
            self.best_score = score
            self._save_checkpoint(score, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                self.trace_func(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, score: float, model: torch.nn.Module) -> None:
        """Save model when validation score improves."""
        if self.verbose:
            self.trace_func(f"Validation score improved. Saving model to: {self.path} (score={score:.6f})")
        torch.save(model.state_dict(), self.path)


# =============================================================================
# Metrics (safe wrappers)
# =============================================================================
def _finite_mask(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Mask for finite labels & probabilities."""
    return np.isfinite(y) & np.isfinite(p)


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Safe ROC-AUC:
    - Returns NaN if not computable (e.g., single-class labels or empty).
    """
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs, dtype=float)
    m = _finite_mask(y, p)
    if m.sum() == 0:
        return float("nan")
    y = y[m]
    p = p[m]
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_ap(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Safe Average Precision (AP / AUPR):
    - Returns NaN if not computable.
    """
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs, dtype=float)
    m = _finite_mask(y, p)
    if m.sum() == 0:
        return float("nan")
    y = y[m]
    p = p[m]
    if np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def prc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Trapezoidal PR-AUC (area under PR curve by trapezoid rule).
    This is different from Average Precision (AP).
    """
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs, dtype=float)
    m = _finite_mask(y, p)
    if m.sum() == 0:
        return float("nan")
    y = y[m]
    p = p[m]
    if np.unique(y).size < 2:
        return float("nan")

    precision, recall, _ = precision_recall_curve(y, p)
    return float(auc(recall, precision))


def get_metric_func(metric: str = "auc") -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Return a metric function that is safe for edge cases.

    Supported metrics:
    - "auc": ROC-AUC
    - "prc": trapezoidal PR-AUC
    - "ap" : Average Precision (AP)
    - "rmse", "mae"
    """
    m = metric.lower()

    if m == "auc":
        return _safe_auc
    if m == "prc":
        return prc_auc
    if m == "ap":
        return _safe_ap
    if m == "rmse":
        return lambda y, p: float(math.sqrt(mean_squared_error(y, p)))
    if m == "mae":
        return lambda y, p: float(mean_absolute_error(y, p))

    raise ValueError(f'Metric "{metric}" not supported.')


def validate_loss_nan(loss: torch.Tensor, logger: logging.Logger, epoch: int) -> bool:
    """Return True if NaN loss is detected and log an error."""
    if torch.isnan(loss):
        logger.error(f"NaN loss detected at epoch {epoch}.")
        return True
    return False


# =============================================================================
# Thresholding helpers
# =============================================================================
def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Find the threshold maximizing F1 score using precision-recall curve.
    Returns: (best_threshold, best_f1)
    """
    p, r, thr = precision_recall_curve(y_true, y_prob)
    if len(thr) == 0:
        return 0.5, 0.0

    f1 = (2 * p * r) / (p + r + 1e-9)
    # precision_recall_curve returns p/r length = len(thr)+1
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i]), float(f1[i])


def binarize_with_thresholds(
    y_prob_dict: Dict[int, np.ndarray],
    thresholds: Optional[np.ndarray],
) -> Dict[int, np.ndarray]:
    """Binarize probabilities with per-task thresholds."""
    y_pred_bin: Dict[int, np.ndarray] = {}
    for t_idx, probs in y_prob_dict.items():
        thr = float(thresholds[t_idx]) if thresholds is not None and t_idx < len(thresholds) else 0.5
        y_pred_bin[t_idx] = (np.asarray(probs) >= thr).astype(int)
    return y_pred_bin


def op_metrics(
    y_true_dict: Dict[int, np.ndarray],
    y_pred_bin: Dict[int, np.ndarray],
) -> Dict[int, Optional[Dict[str, float]]]:
    """
    Compute operational metrics per task (thresholded):
    precision, recall, f1, mcc, acc, specificity
    """
    out: Dict[int, Optional[Dict[str, float]]] = {}
    for t_idx in y_true_dict.keys():
        y_t = np.asarray(y_true_dict[t_idx])
        y_p = np.asarray(y_pred_bin[t_idx])

        if y_t.size == 0:
            out[t_idx] = None
            continue

        prec = precision_score(y_t, y_p, zero_division=0)
        rec = recall_score(y_t, y_p, zero_division=0)
        f1v = f1_score(y_t, y_p, zero_division=0)
        mcc = matthews_corrcoef(y_t, y_p) if np.unique(y_t).size > 1 else 0.0
        acc = accuracy_score(y_t, y_p)

        tn = np.sum((y_t == 0) & (y_p == 0))
        fp = np.sum((y_t == 0) & (y_p == 1))
        spec = tn / (tn + fp + 1e-9)

        out[t_idx] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1v),
            "mcc": float(mcc),
            "acc": float(acc),
            "specificity": float(spec),
        }
    return out


# =============================================================================
# Visualization & Reporting
# =============================================================================
def printPerformance(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    printout: bool = True,
    plot: bool = True,
) -> list:
    """
    Print and (optionally) plot common classification metrics safely.

    Handles:
    - labels == -1 (ignored)
    - NaN/Inf in labels/probs
    - single-class labels (AUC/AP become NaN, confusion handled safely)
    """
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)

    # Mask invalid labels (-1) + non-finite
    valid = (labels != -1) & np.isfinite(labels) & np.isfinite(probs)
    labels = labels[valid]
    probs = probs[valid]

    if labels.size == 0:
        if printout:
            print("No valid labels to evaluate (all were -1 or non-finite).")
        return [float("nan")] * 8

    preds_bin = (probs >= float(threshold)).astype(int)

    # Confusion (safe even if labels are single-class)
    uniq = np.unique(labels.astype(int))
    if uniq.size < 2:
        # If only one class exists, build confusion counts manually
        if int(uniq[0]) == 1:
            tn = fp = fn = 0
            tp = int((preds_bin == 1).sum())
        else:
            tp = fp = fn = 0
            tn = int((preds_bin == 0).sum())
    else:
        tn, fp, fn, tp = confusion_matrix(labels.astype(int), preds_bin).ravel()

    def _sdiv(n: float, d: float) -> float:
        return float(n / d) if d else 0.0

    recall = _sdiv(tp, tp + fn)          # sensitivity
    specificity = _sdiv(tn, tn + fp)
    precision = _sdiv(tp, tp + fp)
    f1v = _sdiv(2 * precision * recall, precision + recall)

    acc = float(accuracy_score(labels, preds_bin))
    rocA = _safe_auc(labels, probs)
    ap = _safe_ap(labels, probs)
    mcc = float(matthews_corrcoef(labels, preds_bin)) if np.unique(labels).size > 1 else float("nan")

    metrics = [acc, rocA, ap, mcc, recall, specificity, precision, f1v]

    if printout:
        names = ["Accuracy", "AUC-ROC", "AP (AUPR)", "MCC", "Recall", "Specificity", "Precision", "F1-score"]
        for n, v in zip(names, metrics):
            print(f"{n}: {v:.4f}" if v == v else f"{n}: nan")

    # Plot curves only if ROC/AP are meaningful
    if plot and np.unique(labels).size > 1:
        # ROC curve
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure()
        plt.plot(fpr, tpr, lw=2)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC (AUC = {rocA:.3f})" if rocA == rocA else "ROC (AUC = nan)")
        plt.show()

        # PR curve
        prec, rec, _ = precision_recall_curve(labels, probs)
        plt.figure()
        plt.plot(rec, prec, lw=2)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR (AP = {ap:.3f})" if ap == ap else "PR (AP = nan)")
        plt.show()

    return metrics


# =============================================================================
# Class imbalance helper
# =============================================================================
@torch.no_grad()
def compute_pos_weight(dataset) -> torch.Tensor:
    """
    Compute pos_weight per task for BCEWithLogitsLoss style weighting.

    pos_weight[i] = (num_negative / num_positive) for task i.
    - Ignores missing labels (-1).
    - If a task has no valid samples, uses 1.0.
    """
    ylist = []
    for d in dataset:
        y = d.y.view(-1).detach().cpu().numpy()
        ylist.append(y)

    Y = np.vstack(ylist)  # (N, T)
    valid = (Y >= 0)      # keep {0,1}, ignore -1

    pos_w = []
    T = Y.shape[1]
    for i in range(T):
        m = valid[:, i]
        if m.sum() == 0:
            pos_w.append(1.0)
            continue

        yi = Y[m, i]
        pos = float((yi == 1).sum())
        neg = float((yi == 0).sum())
        w = (neg + 1e-6) / (pos + 1e-6)
        pos_w.append(float(w))

    return torch.tensor(pos_w, dtype=torch.float32)
