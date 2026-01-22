# -*- coding: utf-8 -*-
"""
Heatmaps of ADME enrichment for Top-N EdgeSHAPer fragments (per task).

Input  (per task):  analysis_out/frag_adme_enrichment_{task}.csv
Outputs (per task): analysis_out/heatmap_cont_{TAG}.png
                    analysis_out/heatmap_bin_{TAG}.png

Notes
- The CSV is expected to contain columns like: task, fragment, kind (continuous/binary),
  adme_col (endpoint), es (EdgeSHAPer score), q (FDR), cohens_d (for continuous),
  logOR (for binary), n_present, n_absent, etc.
- Column names are resolved robustly (case/underscore insensitive).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== PATHS ==================
BASE    = r"/output/edgeshaper"
OUT_DIR = os.path.join(BASE, "analysis_out")
os.makedirs(OUT_DIR, exist_ok=True)

# ================== SETTINGS ==================
TASKS = ["human", "mouse", "rat"]     # tasks to process
FDR_Q = 0.10                          # FDR threshold for significance (used for picking columns)
DPI   = 220

# Heatmap size control (avoid overcrowding)
TOP_FRAGS_FOR_HEATMAP = 5             # number of fragment rows
TOP_ENDPOINTS_CONT    = 10            # number of continuous endpoints (columns)
TOP_ENDPOINTS_BIN     = 10            # number of binary endpoints (columns)

# Plot options
SHOW_SIG = False                      # draw significance markers (○) on cells (q <= FDR_Q)
VMIN_CONT, VMAX_CONT = -1.2,  1.2     # fixed color scale for continuous (Cohen's d)
VMIN_BIN,  VMAX_BIN  = -2.5,  2.5     # fixed color scale for binary (logOR)

# Fonts / layout
HEATMAP_TITLE_FS = 13                 # title font size
HEATMAP_ROT      = 35                 # x tick rotation (degrees)
HEATMAP_TICK_MIN = 8                  # min tick font size
HEATMAP_TICK_MAX = 12                 # max tick font size
CBAR_FS          = 10                 # colorbar tick font size

# Optional tag mapping (used in output file names / titles)
TASK_TAG = {"human": "HLM", "rat": "RLM", "mouse": "MLM",
            "HLM": "HLM", "RLM": "RLM", "MLM": "MLM"}


# ================== HELPERS ==================
def _auto_fontsizes(n_rows, n_cols,
                    tick_min=HEATMAP_TICK_MIN, tick_max=HEATMAP_TICK_MAX):
    """Adjust tick label fonts based on matrix size."""
    fs_y = np.clip(12 - 0.25*(n_rows-10), tick_min, tick_max)  # y-axis (fragments)
    fs_x = np.clip(11 - 0.25*(n_cols-10), tick_min, tick_max)  # x-axis (endpoints)
    return int(fs_y), int(fs_x)


def _load(task: str) -> pd.DataFrame:
    """
    Load enrichment CSV for a task and normalize expected columns.
    - Prefer OUT_DIR; fallback to BASE.
    - Robust column picking (case/underscore insensitive).
    """
    cands = [
        os.path.join(OUT_DIR, f"frag_adme_enrichment_{task}.csv"),
        os.path.join(BASE,    f"frag_adme_enrichment_{task}.csv"),
    ]
    p = next((x for x in cands if os.path.exists(x)), None)
    if p is None:
        raise FileNotFoundError(f"Missing frag_adme_enrichment_{task}.csv in {OUT_DIR} or {BASE}.")
    df = pd.read_csv(p)

    # Robust name resolver
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n.lower() in cols: return cols[n.lower()]
        for n in names:
            for c in df.columns:
                if n.lower().replace("_","") in c.lower().replace("_",""):
                    return c
        return None

    c_task  = pick("task")
    c_frag  = pick("fragment","frag","smarts","smiles")
    c_kind  = pick("kind","type")
    c_col   = pick("adme_col","adme","endpoint","column")
    c_es    = pick("es","edgeshaper")
    c_q     = pick("q","fdr","qvalue")
    c_dm    = pick("delta_median","deltamedian","median_diff")
    c_d     = pick("cohens_d","d","effect_size")
    c_logor = pick("logor","log_or","lor")
    c_n1    = pick("n_present","npos","count_present")
    c_n0    = pick("n_absent","nneg","count_absent")

    need = [c_task, c_frag, c_kind, c_col, c_es, c_q]
    if any(v is None for v in need):
        raise ValueError("Required columns not found in enrichment CSV.")

    df = df.rename(columns={
        c_task:"task", c_frag:"fragment", c_kind:"kind", c_col:"adme_col",
        c_es:"es", c_q:"q",
        **({c_dm:"delta_median"} if c_dm else {}),
        **({c_d:"cohens_d"} if c_d else {}),
        **({c_logor:"logOR"} if c_logor else {}),
        **({c_n1:"n_present"} if c_n1 else {}),
        **({c_n0:"n_absent"} if c_n0 else {}),
    })

    # Normalize "kind" values to {continuous, binary}
    df["kind"] = (
        df["kind"].astype(str).str.strip().str.lower()
          .map(lambda s: "continuous" if s in {
                "cont","continuous","num","numeric","real","float"
              } else ("binary" if s in {
                "bin","binary","bool","boolean","categorical","discrete","class","label"
              } else s))
    )

    # Force numeric dtype and clean infs
    for c in ["es","q","cohens_d","logOR","delta_median","n_present","n_absent"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "logOR" in df.columns:
        df["logOR"] = df["logOR"].replace([np.inf, -np.inf], np.nan)

    # Console diagnostics (optional)
    try:
        print(f"[{task}] kind counts:", df["kind"].value_counts().to_dict())
    except Exception:
        pass

    return df


def _select_top_for_heatmap(df: pd.DataFrame, kind: str,
                            top_rows=5, top_cols=10, fill=True):
    """
    Create a matrix for heatmap:
    - Row selection: top fragments by median |ES| (EdgeSHAPer score) across all endpoints.
    - Column selection: top endpoints by max |effect| among significant (q <= FDR_Q);
      fallback to overall max |effect| if none significant.
    - Value: median effect per (fragment, endpoint); Cohen's d for continuous,
      log(OR) for binary.
    - Also returns a boolean matrix of significance for optional markers.
    """
    sub = df[df["kind"] == kind].copy()
    if sub.empty:
        return None, None, None

    # Define effect magnitude by kind
    sub["effect_abs"] = sub["cohens_d"].abs() if kind == "continuous" else sub["logOR"].abs()

    # Rank fragments by median |ES|
    frag_order = (sub.assign(es_abs=sub["es"].abs())
                    .groupby("fragment")["es_abs"].median()
                    .sort_values(ascending=False).index.tolist())
    frag_keep = frag_order[:top_rows]

    # If fewer than requested, fill from global ES ranking in the full dataframe
    if fill and len(frag_keep) < top_rows:
        extra = (df.assign(es_abs=df["es"].abs())
                   .groupby("fragment")["es_abs"].median()
                   .sort_values(ascending=False).index.tolist())
        for f in extra:
            if f not in frag_keep:
                frag_keep.append(f)
            if len(frag_keep) == top_rows:
                break

    sub_keep = sub[sub["fragment"].isin(frag_keep)].copy()

    # Rank endpoints by max |effect| among significant cells
    eff_by_col = (sub_keep[sub_keep["q"] <= FDR_Q]
                    .groupby("adme_col")["effect_abs"].max()
                    .sort_values(ascending=False))
    if eff_by_col.empty:
        # Fallback: ignore significance if no significant cells
        eff_by_col = (sub_keep.groupby("adme_col")["effect_abs"].max()
                        .sort_values(ascending=False))
    cols_keep = eff_by_col.index.tolist()[:top_cols]
    sub_keep = sub_keep[sub_keep["adme_col"].isin(cols_keep)]

    # Build value matrix (median effect per cell)
    value_col = "cohens_d" if kind == "continuous" else "logOR"
    mat = (sub_keep.pivot_table(index="fragment", columns="adme_col",
                                values=value_col, aggfunc="median")
           .reindex(index=frag_keep, columns=cols_keep))

    # Build significance (q <= FDR_Q) matrix
    sig = (sub_keep.assign(sig=(sub_keep["q"] <= FDR_Q))
                    .pivot_table(index="fragment", columns="adme_col",
                                 values="sig", aggfunc="max")
                    .reindex(index=frag_keep, columns=cols_keep))

    return mat, sig, (frag_keep, cols_keep)


def _heatmap(ax, M: pd.DataFrame, title: str, cmap: str, vmin: float, vmax: float,
             sig: pd.DataFrame = None, show_sig: bool = False,
             hide_spines: bool = True, hide_tickmarks: bool = True):
    """Draw a heatmap with optional significance markers (open circles)."""
    ax.set_title(title, fontsize=HEATMAP_TITLE_FS, pad=8)

    if M is None or M.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return None

    im = ax.imshow(M.values, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")

    # Auto tick font sizes
    fs_y, fs_x = _auto_fontsizes(*M.shape)

    # Axis tick labels
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index, fontsize=fs_y)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, fontsize=fs_x,
                                                         rotation=HEATMAP_ROT, ha="right")

    # Optional significance markers
    if show_sig and sig is not None:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                try:
                    if bool(sig.iloc[i, j]):
                        ax.plot(j, i, marker="o", ms=4, mfc="none", mec="k", mew=0.9)
                except Exception:
                    pass

    # Clean frame/tick marks
    if hide_spines:
        for sp in ax.spines.values():
            sp.set_visible(False)
    if hide_tickmarks:
        ax.tick_params(axis="both", length=0)

    return im


def _figsize_for_matrix(M: pd.DataFrame, base=(3.8, 2.2), step=(0.55, 0.40)):
    """Pick figure size based on matrix shape: base + step * (cols/rows)."""
    if M is None or M.size == 0:
        return (8, 6)
    n_rows, n_cols = M.shape
    return (base[0] + step[0]*n_cols, base[1] + step[1]*n_rows)


# ================== MAIN ==================
if __name__ == "__main__":
    for task in TASKS:
        df = _load(task)
        tag = TASK_TAG.get(task, task)

        # ---- Continuous heatmap (Cohen's d) ----
        cont_mat, cont_sig, _ = _select_top_for_heatmap(
            df, "continuous", TOP_FRAGS_FOR_HEATMAP, TOP_ENDPOINTS_CONT
        )
        fig, ax = plt.subplots(figsize=_figsize_for_matrix(cont_mat), dpi=DPI)
        im = _heatmap(
            ax, cont_mat, f"[{tag}] Continuous endpoints (Cohen's d)",
            cmap="coolwarm", vmin=VMIN_CONT, vmax=VMAX_CONT,
            sig=cont_sig, show_sig=SHOW_SIG
        )
        if im is not None:
            cbar = plt.colorbar(im, ax=ax, shrink=0.82)
            cbar.set_label("Cohen's d")
            cbar.outline.set_visible(False)
            for sp in cbar.ax.spines.values():
                sp.set_visible(False)
            cbar.ax.tick_params(labelsize=CBAR_FS)
        fig.subplots_adjust(bottom=0.28, left=0.22, right=0.96, top=0.90)
        fig.savefig(os.path.join(OUT_DIR, f"heatmap_cont_{tag}.png"), bbox_inches="tight")
        plt.close(fig)

        # ---- Binary heatmap (logOR) ----
        bin_mat, bin_sig, _ = _select_top_for_heatmap(
            df, "binary", TOP_FRAGS_FOR_HEATMAP, TOP_ENDPOINTS_BIN
        )
        fig, ax = plt.subplots(figsize=_figsize_for_matrix(bin_mat), dpi=DPI)
        im = _heatmap(
            ax, bin_mat, f"[{tag}] Binary endpoints (logOR)",
            cmap="coolwarm", vmin=VMIN_BIN, vmax=VMAX_BIN,
            sig=bin_sig, show_sig=SHOW_SIG
        )
        if im is not None:
            cbar = plt.colorbar(im, ax=ax, shrink=0.82)
            cbar.set_label("log(OR)")
            cbar.outline.set_visible(False)
            for sp in cbar.ax.spines.values():
                sp.set_visible(False)
            cbar.ax.tick_params(labelsize=CBAR_FS)
        fig.subplots_adjust(bottom=0.28, left=0.22, right=0.96, top=0.90)
        fig.savefig(os.path.join(OUT_DIR, f"heatmap_bin_{tag}.png"), bbox_inches="tight")
        plt.close(fig)

    print("Saved heatmaps to:", OUT_DIR)
