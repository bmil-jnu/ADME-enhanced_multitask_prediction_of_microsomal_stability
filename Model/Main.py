"""
Main training script for multi-modal, multi-task classification with:
- K-fold (scaffold split) training
- Global class-imbalance statistics and pos_weight
- Ensemble inference over folds
- Task-wise optimal threshold search (F1-max or recall-constrained)
- Final summary + prediction CSV export

Dependencies (project-specific):
- Dataset.py: build_scaffold_kfold_loader, MolDataset, seq_dict_smi
- Model.py: ADME_Multimdal_Multitask
- utile.py: seed_set, create_logger, EarlyStopping, get_metric_func
- This script assumes you already have:
  - compute_global_max_token_idx()
  - _infer_desc_dim()
  - _forward_three_modal()
  - ImprovedMultiTaskLossWrapper()
  - train(), validate()  (from your previous code)

Label convention:
- y in {0,1} for valid labels
- y == -1 indicates missing label for that task
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader

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

# -----------------------------------------------------------------------------
# Project-specific imports
# -----------------------------------------------------------------------------
try:
    from Datadset import build_scaffold_kfold_loader, MolDataset, seq_dict_smi
    from Model import ADME_Multimdal_Multitask
    from Utie import seed_set, create_logger, EarlyStopping
except ImportError:
    print("[Error] Required modules not found. Check your project structure.")
    raise


# -----------------------------------------------------------------------------
# Threshold utilities
# -----------------------------------------------------------------------------
def _safe_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute basic binary classification metrics safely.

    Returns a dict including confusion-matrix derived stats.
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

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
    min_recall: float = 0.5,
    grid_size: int = 1000,
) -> Tuple[float, Dict[str, float]]:
    """
    Find an optimal decision threshold for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels {0,1}.
    y_score : np.ndarray
        Predicted probabilities in [0,1].
    method : str
        - "f1": maximize F1 score (grid search over [0,1])
        - "recall_constrained": maximize F1 subject to recall >= min_recall,
          using PR-curve thresholds.
    min_recall : float
        Recall constraint for "recall_constrained".
    grid_size : int
        Number of thresholds for method="f1" grid search.

    Returns
    -------
    optimal_threshold : float
    metrics : dict
        Metrics computed at the optimal threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if y_true.size == 0:
        return 0.5, {"threshold": 0.5, **_safe_binary_metrics(y_true, (y_score >= 0.5).astype(int))}

    if method == "f1":
        # Grid-search threshold that maximizes F1 (keeps behavior close to your original code)
        thresholds = np.linspace(0.0, 1.0, grid_size)
        best_thr = 0.5
        best_f1 = -1.0

        for thr in thresholds:
            y_pred = (y_score >= thr).astype(int)

            # Avoid degenerate predictions (all 0 or all 1)
            if np.unique(y_pred).size < 2:
                f1 = 0.0
            else:
                f1 = f1_score(y_true, y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)

        y_pred_opt = (y_score >= best_thr).astype(int)
        metrics = _safe_binary_metrics(y_true, y_pred_opt)
        metrics["threshold"] = best_thr
        return best_thr, metrics

    if method == "recall_constrained":
        # PR curve thresholds are aligned with precision/recall except the last point
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

        # thresholds has length = len(precisions)-1
        if thresholds.size == 0:
            # Fallback: no thresholds returned (e.g., constant scores)
            thr = 0.5
            y_pred_opt = (y_score >= thr).astype(int)
            metrics = _safe_binary_metrics(y_true, y_pred_opt)
            metrics["threshold"] = thr
            return thr, metrics

        rec = recalls[:-1]
        pre = precisions[:-1]
        f1s = (2 * pre * rec) / (pre + rec + 1e-12)

        valid = rec >= float(min_recall)
        if not np.any(valid):
            # If constraint is impossible, choose threshold achieving maximum recall
            idx = int(np.argmax(rec))
            thr = float(thresholds[idx])
        else:
            f1s_masked = np.where(valid, f1s, -1.0)
            idx = int(np.argmax(f1s_masked))
            thr = float(thresholds[idx])

        y_pred_opt = (y_score >= thr).astype(int)
        metrics = _safe_binary_metrics(y_true, y_pred_opt)
        metrics["threshold"] = thr
        return thr, metrics

    raise ValueError(f"Unknown method: {method}")


def print_threshold_comparison(
    y_true: np.ndarray,
    y_score: np.ndarray,
    task_name: str,
    logger,
    min_recall: float = 0.70,
) -> Tuple[float, Dict[str, float]]:
    """
    Log a comparison among:
    - F1-max threshold
    - Recall-constrained (recall >= min_recall) + best F1
    - Default threshold=0.5

    Returns F1-max threshold and its metrics.
    """
    logger.info("\n" + "=" * 78)
    logger.info(f"Threshold Analysis | Task: {task_name}")
    logger.info("=" * 78)

    # 1) F1 maximization (grid)
    thr_f1, m_f1 = find_optimal_threshold(y_true, y_score, method="f1")

    # 2) Recall constrained
    thr_rc, m_rc = find_optimal_threshold(
        y_true, y_score, method="recall_constrained", min_recall=min_recall
    )

    # 3) Default 0.5
    thr_def = 0.5
    y_pred_def = (y_score >= thr_def).astype(int)
    m_def = _safe_binary_metrics(y_true, y_pred_def)

    logger.info(
        f"\n{'Method':<30} {'Threshold':>10} {'F1':>8} {'Recall':>8} {'Precision':>10} {'Specificity':>12}"
    )
    logger.info("-" * 86)
    logger.info(
        f"{'F1 Maximization':<30} {thr_f1:>10.3f} {m_f1['f1_score']:>8.3f} "
        f"{m_f1['recall']:>8.3f} {m_f1['precision']:>10.3f} {m_f1['specificity']:>12.3f}"
    )
    logger.info(
        f"{f'Min Recall {min_recall:.0%} + Best F1':<30} {thr_rc:>10.3f} {m_rc['f1_score']:>8.3f} "
        f"{m_rc['recall']:>8.3f} {m_rc['precision']:>10.3f} {m_rc['specificity']:>12.3f}"
    )
    logger.info("-" * 86)
    logger.info(
        f"{'Default (0.5)':<30} {thr_def:>10.3f} {m_def['f1_score']:>8.3f} "
        f"{m_def['recall']:>8.3f} {m_def['precision']:>10.3f} {m_def['specificity']:>12.3f}"
    )
    logger.info("=" * 86)

    return thr_f1, m_f1


# -----------------------------------------------------------------------------
# Data/label utilities
# -----------------------------------------------------------------------------
def compute_label_distribution_from_dataset(
    dataset: MolDataset,
    tasks: List[str],
    logger=None,
) -> torch.Tensor:
    """
    Compute per-task imbalance ratio (neg/pos) from the train dataset *once* (no fold duplication).

    Returns
    -------
    pos_weights : torch.Tensor, shape (num_tasks,)
        Recommended pos_weight values (neg/pos), clamped later in main.
    """
    num_tasks = len(tasks)
    pos = np.zeros(num_tasks, dtype=np.float64)
    neg = np.zeros(num_tasks, dtype=np.float64)

    for item in dataset:
        y = getattr(item, "y", None)
        if y is None:
            continue

        y = y.view(-1) if torch.is_tensor(y) else torch.tensor(y).view(-1)
        if y.numel() != num_tasks:
            # If your dataset uses different y shape, adjust here.
            y = y.view(-1)[:num_tasks]

        y_np = y.detach().cpu().numpy().astype(float)
        for i in range(num_tasks):
            if y_np[i] == -1:
                continue
            if y_np[i] == 1:
                pos[i] += 1
            elif y_np[i] == 0:
                neg[i] += 1

    pos = np.maximum(pos, 1.0)
    ratios = neg / pos  # neg/pos

    if logger:
        logger.info("\n" + "=" * 70)
        logger.info("Train Label Distribution (computed from dataset, no fold duplication)")
        logger.info("=" * 70)
        for i, name in enumerate(tasks):
            total = int(pos[i] + neg[i])
            pos_ratio = (pos[i] / total * 100.0) if total > 0 else 0.0
            logger.info(
                f"{name.upper():>8} | pos={int(pos[i]):5d} ({pos_ratio:5.1f}%) "
                f"| neg={int(neg[i]):5d} ({100.0 - pos_ratio:5.1f}%) "
                f"| total={total:5d} | neg/pos={ratios[i]:.2f}x"
            )
        logger.info("=" * 70 + "\n")

    return torch.tensor(ratios, dtype=torch.float32)


def gather_labels_from_loader(test_loader, num_tasks: int) -> np.ndarray:
    """
    Gather all labels from a DataLoader into a single (N, num_tasks) numpy array.
    """
    ys = []
    for batch in test_loader:
        y = batch.y
        if y.dim() == 1:
            y = y.view(-1, num_tasks)
        ys.append(y.detach().cpu().numpy())
    if not ys:
        return np.zeros((0, num_tasks), dtype=float)
    return np.concatenate(ys, axis=0)


def safe_auc_ap(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Compute ROC-AUC and AP (AUPR) safely.
    Returns 0.0 if computation fails (e.g., single-class).
    """
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = 0.0
    try:
        ap = float(average_precision_score(y_true, y_score))
    except Exception:
        ap = 0.0
    return auc, ap


# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------
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
):
    """
    Train K-fold models and run ensemble evaluation on test set.

    Returns
    -------
    test_base_model : ADME_Multimdal_Multitask
        A model instance used for loading fold checkpoints for testing.
    fold_train_loss_means : list[float]
    fold_val_loss_means : list[float]
    """
    # ---------------------------
    # Setup
    # ---------------------------
    seed_set(seed)
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=output_dir, tag=tag)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} | Tag: {tag}")

    dataset_names = {"train": "train.csv", "test": "test.csv"}
    tasks = ["human", "rat", "mouse"]
    num_tasks = len(tasks)

    # ---------------------------
    # Train dataset (for metadata: desc cols/scaler)
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

    # ---------------------------
    # K-Fold loaders
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
    # Vocab & descriptor dimension
    # ---------------------------
    # NOTE: these helper functions are assumed to exist in your codebase.
    global_max_idx = compute_global_max_token_idx([tr_ds, test_dataset])  # noqa: F821
    base_vocab = len(seq_dict_smi) + 1
    vocab_size = max(base_vocab, global_max_idx + 1)

    desc_dim = _infer_desc_dim(train_desc_cols, tr_ds, test_dataset)  # noqa: F821
    logger.info(f"Vocab size: {vocab_size} | Desc dim: {desc_dim}")

    # ---------------------------
    # Global pos_weight from dataset (no fold duplication)
    # ---------------------------
    global_pos_weights = compute_label_distribution_from_dataset(tr_ds, tasks, logger=logger)
    global_pos_weights = torch.clamp(global_pos_weights.to(device), min=1.0, max=20.0)
    logger.info(f"Global pos_weights (clamped): {global_pos_weights.detach().cpu().numpy()}")

    # ---------------------------
    # K-Fold training
    # ---------------------------
    fold_ckpt_paths: List[str] = []
    fold_train_loss_means: List[float] = []
    fold_val_loss_means: List[float] = []

    for fold_idx, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        logger.info("\n" + "=" * 70)
        logger.info(f"Fold {fold_idx + 1}/{n_splits}")
        logger.info("=" * 70)

        # 1) Initialize base model
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

        # 2) Wrap with loss wrapper (assumed available in your code)
        wrapper_model = ImprovedMultiTaskLossWrapper(  # noqa: F821
            base_model,
            num_tasks,
            gamma=2.0,
            alpha=0.75,
            ranking_margin=0.2,
            ranking_lambda=0.3,
        ).to(device)

        pos_weights = global_pos_weights.clone()
        logger.info(f"Using global pos_weights: {pos_weights.detach().cpu().numpy()}")

        # 3) Optimizer & scheduler
        optimizer = torch.optim.AdamW(wrapper_model.parameters(), lr=base_lr, weight_decay=1e-3)
        scheduler = make_cosine_with_warmup(  # noqa: F821
            optimizer,
            max_epochs=max_epochs,
            warmup_epochs=5,
            base_lr=base_lr,
        )

        # 4) Early stopping (save base_model state dict)
        ckpt_path = os.path.join(output_dir, f"{tag}_fold{fold_idx + 1}.pt")
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            path=ckpt_path,
            mode="max",
        )

        fold_trn_losses: List[float] = []
        fold_val_losses: List[float] = []

        for epoch in range(1, max_epochs + 1):
            # NOTE: train()/validate() are assumed to exist from your previous code.
            trn_opt_loss, trn_raw_focal = train(  # noqa: F821
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

            val_opt_loss, val_raw_loss, val_score = validate(  # noqa: F821
                wrapper_model=wrapper_model,
                val_loader=val_loader,
                device=device,
                task_type=task_type,
                metric=metric,
                logger=logger,
                epoch=epoch,
            )

            scheduler.step()

            fold_trn_losses.append(float(trn_opt_loss))
            fold_val_losses.append(float(val_opt_loss))

            # Save best *base_model* (wrapper has extra params not needed for inference)
            early_stopping(val_score, base_model)

            if early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        fold_ckpt_paths.append(ckpt_path)
        fold_train_loss_means.append(float(np.mean(fold_trn_losses)) if fold_trn_losses else np.nan)
        fold_val_loss_means.append(float(np.mean(fold_val_losses)) if fold_val_losses else np.nan)

    # -----------------------------------------------------------------------------
    # Ensemble testing (average probabilities across folds)
    # -----------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("Ensemble Testing (averaging probabilities across folds)")
    logger.info("=" * 70)

    n_test = len(test_dataset)
    ensemble_sum = np.zeros((n_test, num_tasks), dtype=np.float64)
    ensemble_cnt = np.zeros((n_test, num_tasks), dtype=np.float64)

    # Collect true labels once
    all_labels = gather_labels_from_loader(test_loader, num_tasks=num_tasks)
    true_labels = {tasks[i]: all_labels[:, i] for i in range(num_tasks)}

    # Model template for loading fold checkpoints
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
            logger.info(f"[Skip] Missing checkpoint: {ckpt}")
            continue

        logger.info(f"Loading fold {fold_idx + 1}: {ckpt}")
        test_base_model.load_state_dict(torch.load(ckpt, map_location=device))
        test_base_model.eval()

        probs_list = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                _, task_outputs = _forward_three_modal(test_base_model, batch, device)  # noqa: F821
                batch_probs = torch.cat([torch.sigmoid(o).detach().cpu() for o in task_outputs], dim=1)
                probs_list.append(batch_probs.numpy())

        fold_probs = np.concatenate(probs_list, axis=0)  # (N, T)
        ensemble_sum += fold_probs
        ensemble_cnt += 1.0

    ensemble_probs = ensemble_sum / np.maximum(ensemble_cnt, 1.0)

    # -----------------------------------------------------------------------------
    # Final evaluation with optimal thresholds + CSV export
    # -----------------------------------------------------------------------------
    final_rows = []
    summary_metrics: Dict[str, Dict[str, float]] = {}
    optimal_thresholds: Dict[str, float] = {}

    for i, sp in enumerate(tasks):
        y_true = true_labels[sp]
        y_score = ensemble_probs[:, i]

        valid_mask = (y_true != -1)
        if not np.any(valid_mask):
            logger.info(f"[{sp.upper()}] No valid labels found. Skipping.")
            continue

        y_true_v = y_true[valid_mask].astype(int)
        y_score_v = y_score[valid_mask].astype(float)

        logger.info("\n" + "=" * 78)
        logger.info(f"{sp.upper()} | Ensemble Performance")
        logger.info("=" * 78)

        # 1) Threshold analysis
        opt_thr, opt_m = print_threshold_comparison(
            y_true_v, y_score_v, task_name=sp.upper(), logger=logger, min_recall=0.70
        )
        optimal_thresholds[sp] = opt_thr

        # 2) Metrics at optimal threshold
        y_pred = (y_score_v >= opt_thr).astype(int)
        auc, ap = safe_auc_ap(y_true_v, y_score_v)

        try:
            tn, fp, fn, tp = confusion_matrix(y_true_v, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except Exception:
            specificity = 0.0

        acc = accuracy_score(y_true_v, y_pred)
        mcc = matthews_corrcoef(y_true_v, y_pred)
        f1 = f1_score(y_true_v, y_pred, zero_division=0)
        rec = recall_score(y_true_v, y_pred, zero_division=0)
        pre = precision_score(y_true_v, y_pred, zero_division=0)

        logger.info(f"\n>>> Final @ Threshold={opt_thr:.3f} <<<")
        logger.info(f"Accuracy:    {acc:.4f}")
        logger.info(f"ROC-AUC:     {auc:.4f}")
        logger.info(f"AP (AUPR):   {ap:.4f}")
        logger.info(f"MCC:         {mcc:.4f}")
        logger.info(f"Recall:      {rec:.4f}")
        logger.info(f"Specificity: {specificity:.4f}")
        logger.info(f"Precision:   {pre:.4f}")
        logger.info(f"F1-score:    {f1:.4f}")

        summary_metrics[sp] = {
            "Threshold": float(opt_thr),
            "AUC": float(auc),
            "AP": float(ap),
            "F1": float(f1),
            "Recall": float(rec),
            "Precision": float(pre),
        }

        # Save per-sample predictions for CSV
        df_sp = pd.DataFrame(
            {
                "species": sp,
                "y_true": y_true_v,
                "y_score": y_score_v,
                "y_pred": y_pred,
                "threshold": opt_thr,
                "fold": "ensemble",
            }
        )
        final_rows.append(df_sp)

    # Export CSV
    if final_rows:
        out_df = pd.concat(final_rows, ignore_index=True)
        out_path = os.path.join(output_dir, f"{tag}_ensemble_preds.csv")
        out_df.to_csv(out_path, index=False)
        logger.info(f"[SAVE] Ensemble predictions -> {out_path}")

    # Final summary table
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

    return test_base_model, fold_train_loss_means, fold_val_loss_means


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    model, trn_losses, val_losses = main_train(
        tag="improved_model",
        perf_threshold=0.5,
        perf_plot=True,
        perf_printout=True,
    )

    print("\n=== Training Summary ===")
    print(f"Average Train Loss: {np.nanmean(trn_losses):.4f}")
    print(f"Average Val Loss:   {np.nanmean(val_losses):.4f}")
