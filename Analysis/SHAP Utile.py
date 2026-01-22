"""
Descriptor-level SHAP utilities (KernelExplainer) for tri-modal models.

Assumptions
-----------
- Each batch/sample has:
  - desc:     (B, D_desc) float
  - smil2vec: (B, L_fp)   long/int tokens
  - graph:    PyG Data objects (batch.to_data_list())

Design
------
- SHAP explains descriptor effects under a *fixed context*:
  - fp and graph are fixed to a reference sample from the pooled set.
- Background/eval samples are chosen by quantiles of model logit values
  under that fixed context (improves coverage across score range).
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Batch

import shap

# Matplotlib is only needed for plotting helpers
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ============================================================
# Optional: global matplotlib style helper
# ============================================================
def set_global_matplotlib_style(
    font_main: str = "Times New Roman",
    font_fallback: str = "serif",
    font_title: int = 20,
    font_label: int = 18,
    font_tick: int = 16,
) -> None:
    """
    Apply a consistent plotting style globally (optional).
    Call this in your main script if you want unified styling.
    """
    if any(font_main in f.name for f in fm.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = font_main
    else:
        matplotlib.rcParams["font.family"] = font_fallback

    matplotlib.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": font_title,
            "axes.labelsize": font_label,
            "xtick.labelsize": font_tick,
            "ytick.labelsize": font_tick,
        }
    )


# ============================================================
# 1) Gather samples from loader
# ============================================================
@torch.no_grad()
def _gather_samples_for_shap(loader, max_n: int):
    """
    Collect up to max_n samples from a PyG loader.

    Requires batch to have:
      - batch.desc: (B, D_desc)
      - batch.smil2vec: (B, L_fp)
      - graph via batch.to_data_list()

    Returns
    -------
    X_desc : torch.FloatTensor (N, D_desc)
    X_fp   : torch.LongTensor  (N, L_fp)
    G_list : list[Data] length N
    """
    desc_list: List[torch.Tensor] = []
    fp_list: List[torch.Tensor] = []
    graph_list = []

    n_collected = 0
    for b in loader:
        if not hasattr(b, "desc") or b.desc is None:
            raise RuntimeError("SHAP requires 'desc' in batch.")
        if not hasattr(b, "smil2vec") or b.smil2vec is None:
            raise RuntimeError("SHAP requires 'smil2vec' in batch.")

        desc_cpu = b.desc.detach().cpu()
        fp_cpu = b.smil2vec.detach().cpu()

        desc_list.append(desc_cpu)
        fp_list.append(fp_cpu)
        graph_list.extend(b.to_data_list())

        n_collected += int(fp_cpu.size(0))
        if n_collected >= max_n:
            break

    if len(desc_list) == 0:
        raise RuntimeError("[SHAP] No samples gathered from loader.")

    X_desc = torch.cat(desc_list, dim=0)[:max_n]
    X_fp = torch.cat(fp_list, dim=0)[:max_n]
    G_list = graph_list[:max_n]
    return X_desc, X_fp, G_list


# ============================================================
# 2) Collapse Yes/No dummy pairs (optional)
# ============================================================
def collapse_binary_dummy_pairs(
    feature_names: List[str],
    X_eval: np.ndarray,        # (N, D)
    shap_values: np.ndarray,   # (N, D)
    prefer_keep: str = "Yes",
    pattern: str = r"^(.*)_(Yes|No)$",
    mode: str = "diff",        # "diff"(recommended) or "sum"
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Collapse (base_Yes, base_No) dummy pairs into a single column.

    mode="diff": SHAP = phi_yes - phi_no (recommended)
    mode="sum" : SHAP = phi_yes + phi_no

    Feature value (color) uses base_Yes dummy values (0/1) by default.
    """
    name2idx = {n: i for i, n in enumerate(feature_names)}
    used = set()

    new_names: List[str] = []
    new_X_cols: List[np.ndarray] = []
    new_S_cols: List[np.ndarray] = []

    for name in feature_names:
        if name in used:
            continue

        m = re.match(pattern, name)
        if m:
            base, _tag = m.groups()
            yes_name, no_name = f"{base}_Yes", f"{base}_No"
            i_yes = name2idx.get(yes_name, None)
            i_no = name2idx.get(no_name, None)

            if i_yes is not None and i_no is not None:
                if mode == "sum":
                    phi = shap_values[:, i_yes] + shap_values[:, i_no]
                else:
                    phi = shap_values[:, i_yes] - shap_values[:, i_no]

                # keep feature value column from preferred keep (default Yes)
                keep_name = yes_name if prefer_keep == "Yes" else no_name
                keep_idx = i_yes if prefer_keep == "Yes" else i_no
                xcol = X_eval[:, keep_idx]

                new_names.append(keep_name)
                new_X_cols.append(xcol)
                new_S_cols.append(phi)
                used.update([yes_name, no_name])
                continue

        # Not a pair -> keep as-is
        i = name2idx[name]
        new_names.append(name)
        new_X_cols.append(X_eval[:, i])
        new_S_cols.append(shap_values[:, i])
        used.add(name)

    X_new = np.column_stack(new_X_cols) if new_X_cols else np.empty((len(X_eval), 0))
    S_new = np.column_stack(new_S_cols) if new_S_cols else np.empty((len(shap_values), 0))
    return new_names, X_new, S_new


def rank_from_shap_values(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Rank features by mean absolute SHAP magnitude.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# 3) Quantile index selection helper
# ============================================================
def _select_indices_by_quantiles(values: np.ndarray, k: int, exclude: Optional[set] = None) -> List[int]:
    """
    Pick indices roughly evenly across sorted values by quantiles.
    Ensures uniqueness and respects exclude set.
    """
    values = np.asarray(values).reshape(-1)
    N = int(values.size)
    if N == 0:
        return []

    if exclude is None:
        exclude = set()

    k = int(min(k, N))
    if k <= 0:
        return []

    order = np.argsort(values)  # ascending
    # mid-point quantiles
    qs = np.linspace(0.0, 1.0, k, endpoint=False) + (0.5 / k)
    pos = (qs * (N - 1)).astype(int)

    picked: List[int] = []
    used = set(exclude)

    # Try around each target position to find unused index
    for p in pos:
        found = False
        for d in range(N):
            for cand in (p - d, p + d):
                if 0 <= cand < N:
                    idx = int(order[cand])
                    if idx not in used:
                        picked.append(idx)
                        used.add(idx)
                        found = True
                        break
            if found:
                break

    # Fill if short
    if len(picked) < k:
        for idx in order:
            idx = int(idx)
            if idx not in used:
                picked.append(idx)
                used.add(idx)
                if len(picked) == k:
                    break

    return picked


# ============================================================
# 4) Backward-compatible sampler (used in older pipelines)
# ============================================================
def gather_shap_background_and_eval(
    full_model,
    eval_loader,
    task_index: int,
    device,
    pool_n: int,
    background_n: int,
    eval_n: int,
    kernel_chunk: int = 512,
    *,
    clone_graph: bool = True,
):
    """
    Backward-compatible API.

    IMPORTANT
    ---------
    - Computes logits using a fixed context (ref_fp/ref_graph from first pooled sample).
    - Selects background/eval indices by quantiles of those logits.

    Returns
    -------
    (Xb_desc, Xb_fp, Xb_graphs), (Xe_desc, Xe_fp, Xe_graphs), fx_pool
    """
    full_model.eval()

    Xp_desc, Xp_fp, Xp_graphs = _gather_samples_for_shap(eval_loader, int(pool_n))
    Xp_desc_np = np.asarray(Xp_desc.detach().cpu().tolist(), dtype=float)

    ref_fp = Xp_fp[:1].to(device).long()
    ref_g0 = Xp_graphs[0]

    def f_desc_fixed(X_np: np.ndarray) -> np.ndarray:
        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        outs: List[np.ndarray] = []
        with torch.no_grad():
            for s in range(0, X.size(0), int(kernel_chunk)):
                Xc = X[s : s + int(kernel_chunk)]
                Bc = int(Xc.size(0))
                fp = ref_fp.repeat(Bc, 1)

                if clone_graph:
                    g_list = [ref_g0.clone() for _ in range(Bc)]
                else:
                    g_list = [ref_g0] * Bc

                batch_graph = Batch.from_data_list(g_list).to(device)
                _, o = full_model({"fp": fp, "graph": batch_graph, "desc": Xc})
                outs.append(o[int(task_index)].view(-1).detach().cpu().numpy())
        return np.concatenate(outs, axis=0)

    fx_pool = f_desc_fixed(Xp_desc_np)

    bg_idx = _select_indices_by_quantiles(fx_pool, int(background_n))
    ev_idx = _select_indices_by_quantiles(fx_pool, int(eval_n), exclude=set(bg_idx))

    Xb_desc = Xp_desc[bg_idx]
    Xb_fp = Xp_fp[bg_idx]
    Xb_g = [Xp_graphs[i] for i in bg_idx]

    Xe_desc = Xp_desc[ev_idx]
    Xe_fp = Xp_fp[ev_idx]
    Xe_g = [Xp_graphs[i] for i in ev_idx]

    return (Xb_desc, Xb_fp, Xb_g), (Xe_desc, Xe_fp, Xe_g), fx_pool


# ============================================================
# 5) Main SHAP rank function
# ============================================================
def shap_rank_descriptors_for_task(
    full_model,
    eval_loader,
    desc_cols: List[str],
    task_index: int,
    device,
    background_n: int = 64,
    eval_n: int = 256,
    pool_n: Optional[int] = None,
    backend: str = "kernel",
    return_values: bool = True,
    kernel_nsamples: int = 2048,
    kernel_chunk: int = 512,
    collapse_yesno: bool = False,
    collapse_mode: str = "diff",
    *,
    clone_graph: bool = True,
):
    """
    Descriptor SHAP (KernelExplainer) under a fixed fp/graph context.

    Steps
    -----
    1) Pool samples from eval_loader
    2) Fix context (fp/graph) to the first pooled sample
    3) Compute logits for pooled desc in that fixed context
    4) Pick background/eval indices by logit quantiles
    5) KernelExplainer(f_desc_fixed, X_bg) -> SHAP on X_eval

    Returns
    -------
    If return_values:
      df_rank, shap_values, X_eval, diag
    Else:
      df_rank, diag
    """
    full_model.eval()

    if pool_n is None:
        pool_n = max(int(background_n) * 10, int(eval_n) * 4)

    # Pool samples
    Xp_desc, Xp_fp, Xp_graphs = _gather_samples_for_shap(eval_loader, int(pool_n))
    Xp_desc_np = np.asarray(Xp_desc.detach().cpu().tolist(), dtype=float)

    desc_dim = int(Xp_desc.size(1))
    if desc_dim != len(desc_cols):
        raise ValueError(f"desc_cols({len(desc_cols)}) != X_desc_dim({desc_dim})")

    # Fixed context
    ref_fp = Xp_fp[:1].to(device).long()
    ref_g0 = Xp_graphs[0]

    def f_desc_fixed(X_np: np.ndarray) -> np.ndarray:
        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        outs: List[np.ndarray] = []
        with torch.no_grad():
            for s in range(0, X.size(0), int(kernel_chunk)):
                Xc = X[s : s + int(kernel_chunk)]
                Bc = int(Xc.size(0))

                fp = ref_fp.repeat(Bc, 1)
                if clone_graph:
                    g_list = [ref_g0.clone() for _ in range(Bc)]
                else:
                    g_list = [ref_g0] * Bc

                batch_graph = Batch.from_data_list(g_list).to(device)
                _, o = full_model({"fp": fp, "graph": batch_graph, "desc": Xc})
                outs.append(o[int(task_index)].view(-1).detach().cpu().numpy())
        return np.concatenate(outs, axis=0)

    # Quantile sampling based on pooled logits
    fx_pool = f_desc_fixed(Xp_desc_np)
    bg_idx = _select_indices_by_quantiles(fx_pool, int(background_n))
    ev_idx = _select_indices_by_quantiles(fx_pool, int(eval_n), exclude=set(bg_idx))

    Xb_desc_np = Xp_desc_np[bg_idx]
    Xe_desc_np = Xp_desc_np[ev_idx]

    # KernelExplainer
    # NOTE: shap may return list for multi-output; we normalize to array
    explainer = shap.KernelExplainer(f_desc_fixed, Xb_desc_np)
    sv = explainer.shap_values(Xe_desc_np, nsamples=int(kernel_nsamples))

    shap_values = sv[0] if isinstance(sv, list) else sv
    shap_values = np.asarray(shap_values, dtype=float)

    fx_eval = f_desc_fixed(Xe_desc_np)
    sum_phi = shap_values.sum(axis=1)

    ev = explainer.expected_value
    base_est = float(np.asarray(ev).reshape(-1)[0])  # robust to scalar/list

    add_mae = float(np.mean(np.abs((fx_eval - base_est) - sum_phi)))
    frac_neg = float(np.mean(sum_phi < 0.0))

    diag: Dict[str, object] = {
        "backend_used": backend,
        "N_eval": int(Xe_desc_np.shape[0]),
        "N_bg": int(Xb_desc_np.shape[0]),
        "base_est": base_est,
        "mean_fx": float(np.mean(fx_eval)),
        "delta_mean": float(np.mean(fx_eval) - base_est),
        "frac_sumphi_neg": frac_neg,
        "additivity_mae": add_mae,
        "collapsed": False,
        "feature_names": list(desc_cols),
    }

    out_feature_names = list(desc_cols)
    out_Xe = Xe_desc_np
    out_sv = shap_values

    if collapse_yesno:
        out_feature_names, out_Xe, out_sv = collapse_binary_dummy_pairs(
            feature_names=list(desc_cols),
            X_eval=out_Xe,
            shap_values=out_sv,
            prefer_keep="Yes",
            mode=collapse_mode,
        )
        diag["collapsed"] = True
        diag["collapse_mode"] = collapse_mode
        diag["feature_names"] = out_feature_names

        # Optional additivity check after collapsing
        fx2 = f_desc_fixed(out_Xe)
        sum_phi2 = out_sv.sum(axis=1)
        diag["additivity_mae_collapsed"] = float(np.mean(np.abs((fx2 - base_est) - sum_phi2)))

    df_rank = rank_from_shap_values(out_sv, out_feature_names)

    if return_values:
        return df_rank, out_sv, out_Xe, diag
    return df_rank, diag


# ============================================================
# 6) Plot helpers
# ============================================================
def save_shap_violin_from_rank(
    task_name: str,
    df_rank: pd.DataFrame,
    shap_values: np.ndarray,      # (N, D)
    X_eval: np.ndarray,           # (N, D)
    feature_names: List[str],
    out_png: str,
    topk: int = 20,
    plot_type: str = "violin",    # "violin" or "dot"
    symmetric_xlim: bool = True,
    dpi: int = 300,
    font_main: str = "Times New Roman",
) -> None:
    """
    Save a SHAP summary plot (violin/dot) restricted to top-k ranked features.
    """
    feats_all = list(feature_names)
    top_feats = [f for f in df_rank["feature"].tolist()[:topk] if f in feats_all]
    idx = [feats_all.index(f) for f in top_feats]

    sv_top = shap_values[:, idx]
    Xe_top = X_eval[:, idx]
    X_df = pd.DataFrame(Xe_top, columns=top_feats)

    with plt.rc_context({"font.family": [font_main, "DejaVu Sans"]}):
        shap.summary_plot(
            sv_top,
            X_df,
            feature_names=top_feats,
            plot_type=plot_type,
            max_display=len(top_feats),
            show=False,
        )
        ax = plt.gca()
        ax.set_xlabel("SHAP value (impact on model output)")
        ax.set_ylabel("")
        ax.set_title(f"{task_name}", pad=8)

        if symmetric_xlim:
            m = np.nanpercentile(np.abs(sv_top), 99.5)
            if np.isfinite(m) and m > 0:
                ax.set_xlim(-float(m), float(m))

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
    dpi: int = 500,
    font_main: str = "Times New Roman",
) -> None:
    """
    Save SHAP beeswarm (dot) plot.
    """
    X_df = pd.DataFrame(X_eval, columns=feature_names)

    with plt.rc_context({"font.family": [font_main, "DejaVu Sans"]}):
        shap.summary_plot(
            shap_values,
            X_df,
            feature_names=feature_names,
            plot_type="dot",
            max_display=max_display,
            show=False,
        )
        ax = plt.gca()
        ax.set_xlabel("SHAP value (impact on model output)")
        ax.set_ylabel("")
        ax.set_title(f"{task_name}", pad=8)

        if symmetric_xlim:
            m = np.nanpercentile(np.abs(shap_values), 99.5)
            if np.isfinite(m) and m > 0:
                ax.set_xlim(-float(m), float(m))

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
) -> None:
    """
    Save horizontal bar plot of mean(|SHAP|) for top-k features.
    """
    topk_df = df_rank.head(topk)
    features = topk_df["feature"].tolist()
    vals = topk_df["mean_abs_shap"].values

    with plt.rc_context({"font.family": [font_main, "DejaVu Sans"]}):
        plt.figure(figsize=(6, max(4, 0.35 * len(features))))
        y_pos = np.arange(len(features))

        plt.barh(y_pos, vals)
        plt.yticks(y_pos, features)
        plt.gca().invert_yaxis()
        plt.xlabel("Mean |SHAP value|")
        if title:
            plt.title(title)

        plt.tight_layout()
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close()


# ============================================================
# 7) Optional wrapper class (compatibility)
# ============================================================
class ModelForSHAPDesc(nn.Module):
    """
    Optional torch-module wrapper for explaining desc with fixed fp/graph context.

    Notes
    -----
    - This is not required for KernelExplainer (we use f_desc_fixed),
      but kept for compatibility with older code.
    """

    def __init__(self, full_model, ref_fp, ref_graph_list, device, task_index: int):
        super().__init__()
        self.model = full_model
        self.ref_fp = ref_fp
        self.ref_graph_list = ref_graph_list
        self.device = device
        self.task_index = int(task_index)

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        if desc.device != self.device:
            desc = desc.to(self.device)

        B = int(desc.size(0))
        fp = self.ref_fp[:B].to(self.device).long()
        batch_graph = Batch.from_data_list(self.ref_graph_list[:B]).to(self.device)

        _, outs = self.model({"fp": fp, "graph": batch_graph, "desc": desc})
        return outs[self.task_index].view(-1)
