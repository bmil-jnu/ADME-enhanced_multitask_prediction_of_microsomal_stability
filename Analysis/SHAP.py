# -*- coding: utf-8 -*-

import os, re, importlib, shutil
from typing import Optional, List, Tuple, Dict
from itertools import cycle, islice

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.data import Data, InMemoryDataset, Batch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import shap

# Project modules
from dataset import build_multilabel_stratified_loader, MolDataset, seq_dict_smi
from model import MTMM
from train_shap import train, validate, test
from utile import seed_set, create_logger, EarlyStopping, printPerformance
# from Focal_loss import FocalLoss  # (unused here)

# Optional: EdgeSHAPer (guard if not installed)
try:
    from edgeshaper import Edgeshaper  # GPL-3.0, verify license compatibility in your project
except Exception:
    Edgeshaper = None

# ── Global font: Times New Roman (fallback to serif) ──────────────────
if any("Times New Roman" in f.name for f in fm.fontManager.ttflist):
    matplotlib.rcParams["font.family"] = "Times New Roman"
else:
    matplotlib.rcParams["font.family"] = "serif"

FONT_TITLE = 20
FONT_LABEL = 18
FONT_TICK  = 16
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.titlesize": FONT_TITLE,
    "axes.labelsize":  FONT_LABEL,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
})

# ---------------------------------------------------------------------
# SHAP utilities
# ---------------------------------------------------------------------
@torch.no_grad()
def _gather_samples_for_shap(loader, max_n: int):
    """Collect up to `max_n` samples from a loader; return (X_desc, X_fp, graphs)."""
    desc_list, fp_list, graph_list = [], [], []
    for b in loader:
        if not hasattr(b, 'desc') or b.desc is None:
            raise RuntimeError("Descriptor tensor `desc` required for SHAP (desc_in_dim > 0).")
        if not hasattr(b, 'smil2vec') or b.smil2vec is None:
            raise RuntimeError("Fingerprint `smil2vec` required for SHAP.")

        desc_list.append(b.desc.detach().cpu())
        fp_list.append(b.smil2vec.detach().cpu())
        graph_list.extend(b.to_data_list())

        if sum(x.size(0) for x in fp_list) >= max_n:
            break

    X_desc = torch.cat(desc_list, dim=0)[:max_n]   # (N, D)
    X_fp   = torch.cat(fp_list,   dim=0)[:max_n]   # (N, L)
    G_list = graph_list[:max_n]                    # len=N
    return X_desc, X_fp, G_list


def collapse_binary_dummy_pairs(
    feature_names: List[str],
    X_eval: np.ndarray,        # (N, D)
    shap_values: np.ndarray,   # (N, D)
    prefer_keep: str = "Yes",
    pattern: str = r"^(.*)_(Yes|No)$",
):
    """
    Merge one-hot dummy pairs like '<name>_Yes' and '<name>_No' into a single column:
    - New SHAP = SHAP(Yes) + SHAP(No)
    - New feature series = original '<name>_Yes' values (0/1)
    """
    name2idx = {n: i for i, n in enumerate(feature_names)}
    used = set()
    new_names, new_X_cols, new_S_cols = [], [], []

    for name in feature_names:
        if name in used:
            continue
        m = re.match(pattern, name)
        if m:
            base, _ = m.groups()
            yn = (f"{base}_Yes", f"{base}_No")
            i_yes = name2idx.get(yn[0], None)
            i_no  = name2idx.get(yn[1], None)
            if i_yes is not None and i_no is not None:
                phi = shap_values[:, i_yes] + shap_values[:, i_no]  # merge SHAP
                xcol = X_eval[:, i_yes]                              # keep Yes values for color
                keep_name = yn[0] if prefer_keep == "Yes" else yn[1]
                new_names.append(keep_name)
                new_X_cols.append(xcol)
                new_S_cols.append(phi)
                used.update([yn[0], yn[1]])
                continue

        # Not a pair: keep as is
        i = name2idx[name]
        new_names.append(name)
        new_X_cols.append(X_eval[:, i])
        new_S_cols.append(shap_values[:, i])
        used.add(name)

    X_new = np.column_stack(new_X_cols) if new_X_cols else np.empty((len(X_eval), 0))
    S_new = np.column_stack(new_S_cols) if new_S_cols else np.empty((len(shap_values), 0))
    return new_names, X_new, S_new


def rank_from_shap_values(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """Return features ranked by mean(|SHAP|)."""
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
          .sort_values("mean_abs_shap", ascending=False)
          .reset_index(drop=True)
    )


def _build_task_valid_loader(dataset, task_index: int, batch_size: int) -> GeometricDataLoader:
    """Filter dataset to samples with valid label (!= -1) for a specific task."""
    valid_idx = []
    for i in range(len(dataset)):
        y = dataset[i].y.view(-1)
        if y[task_index].item() != -1:
            valid_idx.append(i)
    sub = torch.utils.data.Subset(dataset, valid_idx) if len(valid_idx) > 0 else dataset
    return GeometricDataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)


class ModelForSHAPDesc(nn.Module):
    """
    Wrapper that fixes FP/graph inputs and exposes only descriptors to the explainer.
    Returns logits for the selected task.
    """
    def __init__(self, full_model, ref_fp, ref_graph_list, device, task_index: int):
        super().__init__()
        self.model = full_model
        self.ref_fp = ref_fp                    # (N, L) long
        self.ref_graph_list = ref_graph_list    # list[Data]
        self.device = device
        self.task_index = int(task_index)

    def forward(self, desc: torch.Tensor):
        if desc.device != self.device:
            desc = desc.to(self.device)
        B = desc.size(0)
        fp = self.ref_fp[:B].to(self.device).long()
        batch_graph = Batch.from_data_list(self.ref_graph_list[:B]).to(self.device)
        _, outs = self.model({'fp': fp, 'graph': batch_graph, 'desc': desc})
        return outs[self.task_index].view(-1)   # logits


def save_shap_violin_from_rank(
    task_name: str,
    df_rank: pd.DataFrame,
    shap_values: np.ndarray,      # (N, D)
    X_eval: np.ndarray,           # (N, D)
    feature_names: List[str],     # len D
    out_png: str,
    topk: int = 20,
    plot_type: str = "violin",    # "violin" or "dot"
    symmetric_xlim: bool = True,
    dpi: int = 300,
    font_main: str = "Times New Roman",
):
    """Save a clean SHAP summary (violin / dot) for top-k features."""
    feats_all = list(feature_names)
    top_feats = [f for f in df_rank["feature"].tolist()[:topk] if f in feats_all]
    idx = [feats_all.index(f) for f in top_feats]
    sv_top = shap_values[:, idx]
    Xe_top = X_eval[:, idx]

    with plt.rc_context({"font.family": [font_main, "DejaVu Sans"]}):
        X_df = pd.DataFrame(Xe_top, columns=top_feats)
        shap.summary_plot(
            sv_top, X_df, feature_names=top_feats,
            plot_type=plot_type, max_display=len(top_feats), show=False
        )
        ax = plt.gca()
        ax.set_title(task_name, fontsize=18, pad=8)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=16)
        ax.set_ylabel("")

        # Symmetric x-limits to reduce outlier dominance
        if symmetric_xlim:
            m = np.nanpercentile(np.abs(sv_top), 99.5)
            if np.isfinite(m) and m > 0:
                ax.set_xlim(-float(m), float(m))

        # Remove all axis lines / grid / collection edges for a clean look
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(False)
        ax.tick_params(axis='both', which='both', length=0)
        for ln in list(ax.lines):
            ln.set_visible(False)
        for coll in ax.collections:
            if hasattr(coll, "set_edgecolors"):
                try: coll.set_edgecolors("none")
                except Exception: pass
            if hasattr(coll, "set_linewidths"):
                try: coll.set_linewidths(0)
                except Exception: pass
            if hasattr(coll, "set_edgecolor"):
                try: coll.set_edgecolor("none")
                except Exception: pass
            if hasattr(coll, "set_linewidth"):
                try: coll.set_linewidth(0)
                except Exception: pass

        # Colorbar cleanup while keeping its label
        fig = plt.gcf()
        if len(fig.axes) > 1:
            cax = fig.axes[-1]
            try:
                cax.set_ylabel("Feature value", rotation=90)
                for sp in cax.spines.values():
                    sp.set_visible(False)
                cax.tick_params(length=0)
            except Exception:
                pass

        plt.tight_layout()
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close()


def save_barh_from_rank(
    df_rank: pd.DataFrame,
    out_png: str,
    topk: int = 20,
    title: Optional[str] = None,
    dpi: int = 300,
    font_main: str = "Times New Roman",
):
    """Save a horizontal bar chart for top-k mean(|SHAP|) features."""
    topk_df = df_rank.head(topk)
    with plt.rc_context({"font.family": [font_main, "DejaVu Sans"]}):
        plt.figure(figsize=(6.0, 0.45*len(topk_df)))
        plt.barh(topk_df["feature"][::-1], topk_df["mean_abs_shap"][::-1])
        plt.xlabel("Mean absolute SHAP value\n(impact on model output)")
        if title: plt.title(title, fontsize=12, pad=6)
        plt.tight_layout()
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close()


def plot_shap_beeswarm(
    task_name: str,
    shap_values: np.ndarray,   # (N, D)
    X_eval: np.ndarray,        # (N, D)
    feature_names: List[str],
    out_png: str,
    max_display: int = 20,
    symmetric_xlim: bool = True,
    dpi: int = 220,
):
    """Save a SHAP beeswarm (dot) plot."""
    X_df = pd.DataFrame(X_eval, columns=feature_names)
    shap.summary_plot(
        shap_values, X_df, feature_names=feature_names,
        plot_type="dot", max_display=max_display, show=False
    )
    ax = plt.gca()
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
    ax.set_ylabel("")
    ax.set_title(f"{task_name}", fontsize=12, pad=8)

    if symmetric_xlim:
        m = np.nanpercentile(np.abs(shap_values), 99.5)
        if np.isfinite(m) and m > 0:
            ax.set_xlim(-float(m), float(m))

    # Keep "Feature value" label on the colorbar, if present
    fig = plt.gcf()
    if len(fig.axes) > 1:
        cax = fig.axes[-1]
        cax.set_ylabel("Feature value", rotation=90)

    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()

# ---------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------
def make_cosine_with_warmup(optimizer, max_epochs, warmup_epochs=5, min_lr=1e-6, base_lr=1e-4):
    """Cosine LR with linear warmup; returns a LambdaLR scheduler."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        t = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        cos_decay = 0.5 * (1 + np.cos(np.pi * t))
        return max(min_lr / base_lr, cos_decay)
    return LambdaLR(optimizer, lr_lambda)


def compute_pos_weight_per_task(loaders, num_tasks, device):
    """Compute per-task positive class weights: pos_weight = (neg / pos), ignoring -1 labels."""
    pos = torch.zeros(num_tasks, dtype=torch.float64)
    neg = torch.zeros(num_tasks, dtype=torch.float64)
    for loader in loaders:
        for batch in loader:
            y = batch.y
            if y.dim() == 1:
                y = y.view(-1, num_tasks)
            for i in range(num_tasks):
                valid = (y[:, i] != -1)
                if valid.any():
                    yi = y[valid, i]
                    pos[i] += (yi == 1).sum().item()
                    neg[i] += (yi == 0).sum().item()
    pos = torch.clamp(pos, min=1.0)
    w = (neg / pos).to(torch.float32)
    return w.to(device)


def _save_parquet_or_csv(df: pd.DataFrame, parquet_path: str, logger=None) -> str:
    """Try Parquet (pyarrow/fastparquet); fallback to CSV; return saved path."""
    base, _ = os.path.splitext(parquet_path)
    try:
        if importlib.util.find_spec("pyarrow") is not None:
            df.to_parquet(parquet_path, index=False, engine="pyarrow")
            if logger: logger.info(f"[SAVE] Predictions -> {parquet_path} (engine=pyarrow)")
            return parquet_path
    except Exception as e:
        if logger: logger.warning(f"[WARN] pyarrow parquet save failed: {e}")

    try:
        if importlib.util.find_spec("fastparquet") is not None:
            df.to_parquet(parquet_path, index=False, engine="fastparquet")
            if logger: logger.info(f"[SAVE] Predictions -> {parquet_path} (engine=fastparquet)")
            return parquet_path
    except Exception as e:
        if logger: logger.warning(f"[WARN] fastparquet parquet save failed: {e}")

    csv_path = base + ".csv"
    df.to_csv(csv_path, index=False)
    if logger: logger.warning(f"[SAVE] Parquet engine not available; fell back to CSV -> {csv_path}")
    return csv_path


def compute_global_max_token_idx(datasets):
    """Scan datasets for max token id in `smil2vec` to derive a safe vocab size."""
    gmax = 0
    for ds in datasets:
        for d in ds:
            if hasattr(d, 'smil2vec') and d.smil2vec is not None:
                flat = d.smil2vec.view(-1)
                if flat.numel() == 0:
                    continue
                v = int(flat.max().item())
                gmax = max(gmax, v)
    return gmax


def _infer_desc_dim(train_desc_cols, tr_ds, test_ds) -> int:
    """Infer descriptor dimension from schema or the first sample that has `desc`."""
    if train_desc_cols is not None:
        return len(train_desc_cols)
    for ds in (tr_ds, test_ds):
        if len(ds) > 0 and hasattr(ds[0], "desc") and getattr(ds[0], "desc") is not None:
            d = ds[0].desc
            if d.dim() == 1:
                return 1
            return int(d.view(d.size(0), -1).size(1)) if d.dim() > 1 else int(d.size(-1))
    return 0

# ---------------------------------------------------------------------
# Descriptor-only SHAP (Deep → Kernel fallback, NumPy-safe)
# ---------------------------------------------------------------------
def shap_rank_descriptors_for_task(
    full_model,
    eval_loader,
    desc_cols: List[str],
    task_index: int,
    device,
    background_n: int = 64,
    eval_n: int = 256,
    backend: str = "deep",
    return_values: bool = True,
    kernel_nsamples: int = 2048,   # sample budget
    kernel_chunk: int = 512,       # internal batch size
):
    """Rank descriptors for a task via SHAP (mean|SHAP|). Return rank DF, SHAP values, and X_eval."""
    Xb_desc, Xb_fp, Xb_graphs = _gather_samples_for_shap(eval_loader, background_n)
    Xe_desc, Xe_fp, Xe_graphs = _gather_samples_for_shap(eval_loader, eval_n)

    desc_dim = Xb_desc.size(1)
    assert desc_dim == len(desc_cols), f"desc_cols({len(desc_cols)}) vs Xb_desc.size(1)({desc_dim})"

    shap_values = None

    # Try DeepExplainer first
    if backend == "deep":
        try:
            wrapper_bg = ModelForSHAPDesc(full_model, Xb_fp.to(device), Xb_graphs, device, task_index)
            explainer = shap.DeepExplainer(wrapper_bg, Xb_desc.to(device).float())
            sv = explainer.shap_values(Xe_desc.to(device).float())
            if isinstance(sv, list):
                sv = sv[0]
            shap_values = np.asarray(
                sv.detach().cpu().tolist() if isinstance(sv, torch.Tensor) else sv, dtype=float
            )
        except Exception as e:
            print(f"[WARN] DeepExplainer failed ({e}); fallback to KernelExplainer.")
            backend = "kernel"

    # Fallback: KernelExplainer
    if backend == "kernel":
        # Safe baseline / eval arrays
        Xb_desc_np = np.asarray(Xb_desc.detach().cpu().tolist(), dtype=float)
        Xe_desc_np = np.asarray(Xe_desc.detach().cpu().tolist(), dtype=float)

        fp_pool = Xb_fp.to(device).long()
        graphs_pool = Xb_graphs
        pool_len = fp_pool.size(0)

        def _build_batch(B: int):
            reps = (B + pool_len - 1) // pool_len
            fp = fp_pool.repeat(reps, 1)[:B]
            g_list = list(islice(cycle(graphs_pool), B))
            return fp, g_list

        def f_desc(X_np: np.ndarray):
            X = torch.tensor(X_np, dtype=torch.float32, device=device)
            outs = []
            for s in range(0, X.size(0), kernel_chunk):
                Xc = X[s:s+kernel_chunk]
                Bc = Xc.size(0)
                fp, g_list = _build_batch(Bc)
                batch_graph = Batch.from_data_list(g_list).to(device)
                with torch.no_grad():
                    _, o = full_model({'fp': fp, 'graph': batch_graph, 'desc': Xc})
                    out_chunk = np.asarray(o[task_index].view(-1).detach().cpu().tolist(), dtype=float)
                    outs.append(out_chunk)
            return np.concatenate(outs, axis=0)

        explainer = shap.KernelExplainer(f_desc, Xb_desc_np)
        sv = explainer.shap_values(Xe_desc_np, nsamples=kernel_nsamples)
        shap_values = sv[0] if isinstance(sv, list) else sv

    # Rank by mean absolute SHAP
    df_rank = rank_from_shap_values(shap_values, desc_cols)

    if return_values:
        Xe_desc_np = np.asarray(Xe_desc.detach().cpu().tolist(), dtype=float)
        return df_rank, shap_values, Xe_desc_np
    else:
        return df_rank

# ---------------------------------------------------------------------
# Main training + evaluation + optional SHAP
# ---------------------------------------------------------------------
def main_train(
    output_dir="output",
    tag="default",
    seed=42,
    batch_size=128,
    task_type='classification',
    metric='prc',
    base_lr=1e-4,
    n_splits=10,
    data_path='/home/shap/',
    patience: int = 5,
    perf_threshold: float = 0.5,
    perf_thresholds: Optional[Dict[str, float]] = None,
    perf_plot: bool = False,
    perf_printout: bool = True,
    use_cosine=True,
    max_epochs: int = 200,
    # --- SHAP options ---
    shap_enable: bool = False,
    shap_backend: str = "kernel",
    shap_background_n: int = 64,
    shap_eval_n: int = 256,
    shap_topk: int = 20,
    kernel_nsamples: int = 2048,
    kernel_chunk: int = 512,
):
    # Setup
    seed_set(seed)
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=output_dir, tag=tag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_names = {'train': 'train.csv', 'test': 'test.csv'}
    tasks = ['human', 'rat', 'mouse']

    # Build loaders (k-fold) and test set with the same descriptor schema
    train_loaders, val_loaders, train_desc_cols = build_multilabel_stratified_loader(
        data_path=data_path, dataset_name=dataset_names['train'],
        task_type=task_type, batch_size=batch_size, tasks=tasks,
        logger=logger, n_splits=n_splits
    )
    test_dataset = MolDataset(
        root=data_path, dataset=dataset_names['test'],
        task_type=task_type, tasks=tasks, logger=logger,
        desc_cols=train_desc_cols
    )
    test_loader = GeometricDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    )
    if hasattr(test_dataset, "desc_cols_") and train_desc_cols is not None:
        assert len(test_dataset.desc_cols_) == len(train_desc_cols), \
            f"Descriptor mismatch: train={len(train_desc_cols)} vs test={len(test_dataset.desc_cols_)}"

    # Global vocab size from data
    tr_ds = MolDataset(root=data_path, dataset=dataset_names['train'],
                       task_type=task_type, tasks=tasks, logger=logger)
    global_max_idx = compute_global_max_token_idx([tr_ds, test_dataset])
    base_vocab = len(seq_dict_smi) + 1  # include PAD
    vocab_size = max(base_vocab, global_max_idx + 1)
    logger.info(f"[Vocab] base={base_vocab}, global_max_idx={global_max_idx} -> vocab_size={vocab_size}")

    # Descriptor dimension (for model construction + SHAP)
    desc_dim = len(train_desc_cols) if train_desc_cols is not None else 0
    logger.info(f"[Desc] descriptor dim = {desc_dim}")

    fold_train_loss_means, fold_val_loss_means = [], []
    fold_best_val_losses, fold_ckpt_paths = [], []

    # K-fold training
    for fold_idx, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        logger.info(f"======== Fold {fold_idx+1}/{n_splits} ========")

        model = MTMM(
            vocab_size=vocab_size,
            device=device,
            num_tasks=len(tasks),
            desc_in_dim=desc_dim,
            fp_mode="dense",
            fp_type="morgan+maccs+rdit",
            fp_emb_dim=512,
            graph_out_dim=512,
            fusion_dim=512,
            dropout=0.3,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
        if use_cosine:
            scheduler = make_cosine_with_warmup(optimizer, max_epochs=max_epochs, warmup_epochs=5,
                                                min_lr=1e-6, base_lr=base_lr)
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False)

        # Per-task BCEWithLogits with pos_weight
        num_tasks = len(tasks)
        pos_w = compute_pos_weight_per_task([train_loader], num_tasks, device)
        criterion_list = [nn.BCEWithLogitsLoss(pos_weight=pos_w[i]) for i in range(num_tasks)]
        criterion = nn.BCEWithLogitsLoss()  # fallback/compat

        ckpt_path = os.path.join(output_dir, f'{tag}_best_model_loss_fold{fold_idx+1}.pt')
        early_stopping = EarlyStopping(
            patience=patience, delta=0.0, monitor='loss', path=ckpt_path, verbose=True
        )

        fold_train_losses, fold_val_losses = [], []
        best_val = float('inf')

        for epoch in range(1, max_epochs + 1):
            trn_loss, trn_score = train(
                epoch=epoch, model=model, criterion=criterion,
                train_loader=train_loader, optimizer=optimizer, lr_scheduler=None,
                device=device, task_type=task_type, metric=metric, logger=logger,
                criterion_list=criterion_list,   # pass per-task losses
            )
            val_loss, val_score = validate(
                model=model, criterion=criterion, val_loader=val_loader,
                device=device, task_type=task_type, metric=metric, logger=logger, epoch=epoch,
                criterion_list=criterion_list,   # pass per-task losses
            )
            fold_train_losses.append(trn_loss)
            fold_val_losses.append(val_loss)

            if use_cosine:
                scheduler.step()
            else:
                scheduler.step(val_loss)  # ReduceLROnPlateau

            # EarlyStopping monitors negative metric to maximize it
            metric_to_max = val_score
            early_stopping(-metric_to_max, model)

            if val_loss < best_val:
                best_val = val_loss
            if early_stopping.early_stop:
                logger.info(f"[Fold {fold_idx+1}] Early stopping at epoch {epoch}")
                break

        fold_train_loss_means.append(float(np.mean(fold_train_losses)) if fold_train_losses else float('nan'))
        fold_val_loss_means.append(float(np.mean(fold_val_losses)) if fold_val_losses else float('nan'))
        fold_best_val_losses.append(best_val)
        fold_ckpt_paths.append(ckpt_path)

    # Select best fold by lowest val loss and evaluate on test
    best_fold = int(np.nanargmin(np.array(fold_best_val_losses)))
    best_ckpt = fold_ckpt_paths[best_fold]
    logger.info(f"[Best Fold] #{best_fold+1} | best_val_loss={fold_best_val_losses[best_fold]:.6f} | ckpt={best_ckpt}")

    best_model = MTMM(
        vocab_size=vocab_size,
        device=device,
        num_tasks=len(tasks),
        desc_in_dim=desc_dim,
        fp_mode="dense",
        fp_type="morgan+maccs+rdit",
        fp_emb_dim=512,
        graph_out_dim=512,
        fusion_dim=512,
        dropout=0.3,
    ).to(device)

    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=device)
        best_model.load_state_dict(state)
        logger.info("Loaded best fold checkpoint for final test.")
    else:
        logger.warning("Best checkpoint not found; using last in-memory model from final fold.")

    logger.info("===== Test =====")
    test_loss, test_metric = test(
        model=best_model, criterion=criterion, test_loader=test_loader, device=device,
        task_type=task_type, metric=metric, logger=logger
    )

    # Optional: per-task performance table on test set
    if task_type == 'classification':
        test_labels, test_probs = defaultdict(list), defaultdict(list)
        best_model.eval()
        with torch.no_grad():
            for batch in test_loader:
                B = batch.num_graphs
                batch = batch.to(device)

                # Normalize FP tensor shapes
                fp = getattr(batch, 'smil2vec', None)
                if fp is not None:
                    if fp.dim() == 3 and fp.size(1) == 1:
                        fp = fp.squeeze(1)
                    if fp.dim() == 1 and fp.numel() % B == 0:
                        fp = fp.view(B, -1)
                    fp = fp.long().to(device)

                # Normalize descriptor shapes
                desc = getattr(batch, 'desc', None)
                if getattr(best_model, 'desc_in_dim', 0) > 0:
                    if desc is None:
                        desc = torch.zeros((B, best_model.desc_in_dim), device=device, dtype=torch.float32)
                    else:
                        desc = desc.to(device).float()
                        if desc.dim() == 1 and desc.numel() % B == 0:
                            desc = desc.view(B, -1)
                        elif desc.dim() == 1:
                            desc = desc.unsqueeze(1)
                        elif desc.dim() > 2:
                            desc = desc.view(B, -1)
                        if desc.size(-1) != best_model.desc_in_dim:
                            raise RuntimeError(f"desc dim mismatch: expected {best_model.desc_in_dim}, got {desc.size(-1)}")
                else:
                    desc = None

                _, preds = best_model({'fp': fp, 'graph': batch, 'desc': desc})
                _probs_t = torch.cat([torch.sigmoid(p).detach().cpu() for p in preds], dim=1)
                probs    = np.asarray(_probs_t.tolist(), dtype=float)
                _labels_t = batch.y.detach().cpu()
                labels    = np.asarray(_labels_t.tolist(), dtype=int)

                for i, t in enumerate(tasks):
                    lab_i = labels[:, i]
                    prb_i = probs[:, i]
                    valid_idx = (lab_i != -1)
                    if np.any(valid_idx):
                        test_labels[t].extend(lab_i[valid_idx].tolist())
                        test_probs[t].extend(np.asarray(prb_i)[valid_idx].tolist())

        for t in tasks:
            if len(test_labels[t]) == 0:
                logger.info(f"Performance for {t}: (no valid labels; all were -1)")
                logger.info("-" * 40)
                continue
            thr = perf_threshold if not (perf_thresholds and t in perf_thresholds) else float(perf_thresholds[t])
            logger.info(f"Performance for {t} (threshold={thr:.3f}):")
            printPerformance(
                test_labels[t],
                test_probs[t],
                threshold=thr,
                plot=perf_plot,
                printout=perf_printout,
            )
            logger.info("-" * 40)
    else:
        logger.info("Regression task detected – per-task classification report is skipped.")

    # Optional: SHAP analysis per task
    if shap_enable:
        if desc_dim == 0 or train_desc_cols is None:
            logger.warning("[SHAP] desc_in_dim=0 or missing schema; skip descriptor SHAP.")
        else:
            shap_dir = os.path.join(output_dir, "shap")
            os.makedirs(shap_dir, exist_ok=True)
            logger.info("[SHAP] Start (%s) -> %s", shap_backend, shap_dir)

            for t_idx, t_name in enumerate(tasks):
                t_loader = _build_task_valid_loader(test_dataset, task_index=t_idx, batch_size=batch_size)
                if len(t_loader.dataset) == 0:
                    logger.warning(f"[SHAP] {t_name}: no valid samples; skip")
                    continue

                # Rank + values + eval inputs
                df_rank, sv, Xe = shap_rank_descriptors_for_task(
                    full_model=best_model,
                    eval_loader=t_loader,
                    desc_cols=list(train_desc_cols),
                    task_index=t_idx,
                    device=device,
                    background_n=shap_background_n,
                    eval_n=shap_eval_n,
                    backend=shap_backend,
                    return_values=True,
                    kernel_nsamples=kernel_nsamples,
                    kernel_chunk=kernel_chunk,
                )

                # Quick diagnostics (logit scale)
                Xb_desc, Xb_fp, Xb_graphs = _gather_samples_for_shap(t_loader, shap_background_n)
                fp_pool = Xb_fp.to(device).long()
                graphs_pool = Xb_graphs
                pool_len = fp_pool.size(0)

                def _build_batch(B: int):
                    reps = (B + pool_len - 1) // pool_len
                    fp = fp_pool.repeat(reps, 1)[:B]
                    g_list = list(islice(cycle(graphs_pool), B))
                    return fp, g_list

                def _f_desc_np(X_np: np.ndarray):
                    X = torch.tensor(X_np, dtype=torch.float32, device=device)
                    outs = []
                    for s in range(0, X.size(0), kernel_chunk):
                        Xc = X[s:s+kernel_chunk]
                        B = Xc.size(0)
                        fp, g_list = _build_batch(B)
                        batch_graph = Batch.from_data_list(g_list).to(device)
                        with torch.no_grad():
                            _, o = best_model({'fp': fp, 'graph': batch_graph, 'desc': Xc})
                            outs.append(np.asarray(o[t_idx].view(-1).detach().cpu().tolist(), dtype=float))
                    return np.concatenate(outs, axis=0)

                fx = _f_desc_np(Xe)                 # logits
                sum_phi = sv.sum(axis=1)            # per-sample sum of SHAP
                base_est = float(np.mean(fx - sum_phi))
                frac_neg = float(np.mean(sum_phi < 0.0))

                # Positive rate in the eval subset (no -1 in valid loader)
                y_eval = []
                n = 0
                for b in t_loader:
                    y_eval.append(np.asarray(b.y[:, t_idx].detach().cpu().tolist(), dtype=float))
                    n += b.num_graphs
                    if n >= Xe.shape[0]: break
                y_eval = np.concatenate(y_eval)[:Xe.shape[0]] if len(y_eval) else np.asarray([])
                pos_rate = float(np.mean(y_eval)) if y_eval.size else float('nan')

                logger.info(
                    "[SHAP][%s] N=%d | base~%.4f | mean f(x)=%.4f (d=%+.4f) | frac(Sum(phi)<0)=%.2f%% | pos_rate~%.2f%%",
                    t_name, Xe.shape[0], base_est, float(np.mean(fx)),
                    float(np.mean(fx) - base_est), frac_neg*100.0, pos_rate*100.0
                )

                # Collapse Yes/No dummy pairs before plotting
                collapsed_names, Xe_c, sv_c = collapse_binary_dummy_pairs(
                    feature_names=list(train_desc_cols),
                    X_eval=Xe,
                    shap_values=sv,
                    prefer_keep="Yes",
                )
                df_rank = rank_from_shap_values(sv_c, collapsed_names)

                # Save artifacts
                csv_path = os.path.join(shap_dir, f"shap_rank_{t_name}.csv")
                df_rank.to_csv(csv_path, index=False, encoding="utf-8-sig")

                bees_png   = os.path.join(shap_dir, f"shap_beeswarm_top{shap_topk}_{t_name}.png")
                violin_png = os.path.join(shap_dir, f"shap_violin_top{shap_topk}_{t_name}.png")
                bar_png    = os.path.join(shap_dir, f"shap_bar_top{shap_topk}_{t_name}.png")

                save_shap_violin_from_rank(
                    task_name=t_name,
                    df_rank=df_rank,
                    shap_values=sv_c,
                    X_eval=Xe_c,
                    feature_names=collapsed_names,
                    out_png=violin_png,
                    topk=shap_topk,
                    plot_type="violin",
                    dpi=300,
                )
                save_barh_from_rank(
                    df_rank=df_rank,
                    out_png=bar_png,
                    topk=shap_topk,
                    title=None,
                    dpi=300,
                )
                plot_shap_beeswarm(
                    task_name=t_name,
                    shap_values=sv_c,
                    X_eval=Xe_c,
                    feature_names=collapsed_names,
                    out_png=bees_png,
                    max_display=shap_topk,
                    symmetric_xlim=True,
                    dpi=220,
                )
                logger.info(f"[SHAP] {t_name}: saved -> {violin_png} , {bar_png}")
            logger.info("[SHAP] Done.")

    return best_model, fold_train_loss_means, fold_val_loss_means


if __name__ == "__main__":
    model, trn_losses, val_losses = main_train(
        perf_threshold=0.5,
        perf_plot=True,
        perf_printout=True,
        shap_enable=True,
        shap_backend="kernel",
        shap_background_n=128,
        shap_eval_n=256,
        shap_topk=10,
        kernel_nsamples=2048,
        kernel_chunk=1024,
    )
