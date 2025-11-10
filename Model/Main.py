import os
import numpy as np
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader
import torch.nn as nn 
from Dataset import build_multilabel_stratified_loader, MolDataset, seq_dict_smi
from Model import multitask_multimodal
from utile import seed_set, create_logger, EarlyStopping, printPerformance, compute_pos_weight
from collections import defaultdict
import shutil
from typing import Optional, List, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data, InMemoryDataset
import torch.nn.functional as F
from math import cos, pi
from torch.optim.lr_scheduler import LambdaLR
from Train import train, validate, test

def make_cosine_with_warmup(optimizer, max_epochs, warmup_epochs=5, min_lr=1e-6, base_lr=1e-4):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        t = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        cos_decay = 0.5 * (1 + cos(pi * t))
        return max(min_lr / base_lr, cos_decay)
    return LambdaLR(optimizer, lr_lambda)

def compute_pos_weight_per_task(loaders, num_tasks, device):
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

def _save_parquet_or_csv(df, parquet_path: str, logger=None):
    import os, pandas as pd, importlib
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
    if train_desc_cols is not None:
        return len(train_desc_cols)
    for ds in (tr_ds, test_ds):
        if len(ds) > 0 and hasattr(ds[0], "desc") and getattr(ds[0], "desc") is not None:
            d = ds[0].desc
            if d.dim() == 1:
                return 1
            return int(d.view(d.size(0), -1).size(1)) if d.dim() > 1 else int(d.size(-1))
    return 0  

def main_train(
    output_dir="output",
    tag="default",
    seed=42,
    batch_size=128,
    task_type='classification',
    metric='prc',
    base_lr=1e-4,
    n_splits=10,
    data_path='D:home/dataset/',
    patience: int = 5,
    perf_threshold: float = 0.5,
    perf_thresholds: Optional[Dict[str, float]] = None,
    perf_plot: bool = False,
    perf_printout: bool = True,
    use_cosine=True,
    max_epochs: int = 200,
):
    # ---------------------------
    # Setup
    # ---------------------------
    seed_set(seed)
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=output_dir, tag=tag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_names = {'train': 'train.csv', 'test': 'test.csv'}
    tasks = ['human', 'rat', 'mouse']

    # ---------------------------
    # Fold loaders
    # ---------------------------
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

    # ---------------------------
    # Global vocab/desc 크기
    # ---------------------------
    tr_ds = MolDataset(root=data_path, dataset=dataset_names['train'],
                       task_type=task_type, tasks=tasks, logger=logger)
    global_max_idx = compute_global_max_token_idx([tr_ds, test_dataset])
    base_vocab = len(seq_dict_smi) + 1 
    vocab_size = max(base_vocab, global_max_idx + 1)
    logger.info(f"[Vocab] base={base_vocab}, global_max_idx={global_max_idx} -> vocab_size={vocab_size}")
    _ = (len(train_desc_cols) if train_desc_cols is not None else 0)
    logger.info(f"[Desc] descriptor dim = {_}")

    # ---------------------------
    # K-fold training
    # ---------------------------
    fold_train_loss_means, fold_val_loss_means = [], []  
    fold_best_val_losses, fold_ckpt_paths = [], []

    for fold_idx, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        logger.info(f"======== Fold {fold_idx+1}/{n_splits} ========")
        model = multitask_multimodal(
            vocab_size=vocab_size,     
            device=device,
            num_tasks=len(tasks),
            desc_in_dim=(len(train_desc_cols) if train_desc_cols is not None else 0),
            fp_mode="dense",
            fp_type="morgan+maccs+rdit",
            fp_emb_dim=512,
            graph_out_dim=512,
            fusion_dim=512,
            dropout=0.3,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
        scheduler = make_cosine_with_warmup(optimizer, max_epochs=max_epochs, warmup_epochs=5, min_lr=1e-6, base_lr=base_lr)

        num_tasks = len(tasks)
        pos_w = compute_pos_weight_per_task([train_loader], num_tasks, device)
        criterion_list = [nn.BCEWithLogitsLoss(pos_weight=pos_w[i]) for i in range(num_tasks)]
        criterion = nn.BCEWithLogitsLoss() 

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
                criterion_list=criterion_list,
            )
            val_loss, val_score = validate(
                model=model, criterion=criterion, val_loader=val_loader,
                device=device, task_type=task_type, metric=metric, logger=logger, epoch=epoch,
                criterion_list=criterion_list,
            )
            fold_train_losses.append(trn_loss)
            fold_val_losses.append(val_loss)

            if use_cosine:
                scheduler.step()
            else:
                scheduler.step(val_loss)
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

    best_fold = int(np.nanargmin(np.array(fold_best_val_losses)))
    best_ckpt = fold_ckpt_paths[best_fold]
    logger.info(f"[Best Fold] #{best_fold+1} | best_val_loss={fold_best_val_losses[best_fold]:.6f} | ckpt={best_ckpt}")

    best_model = multitask_multimodal(
        vocab_size=vocab_size,
        device=device,
        num_tasks=len(tasks),
        desc_in_dim=(len(train_desc_cols) if train_desc_cols is not None else 0),
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
    from collections import defaultdict
    import pandas as pd

    test_labels, test_probs = defaultdict(list), defaultdict(list)
    best_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            B = batch.num_graphs

            # --- FP handling (shape/type normalize) ---
            fp = getattr(batch, 'smil2vec', None)
            if fp is None:
                fp = torch.zeros((B, 100), device=device, dtype=torch.long)
            else:
                fp = fp.to(device)
                # (B,1,L) -> (B,L)
                if fp.dim() == 3 and fp.size(1) == 1:
                    fp = fp.squeeze(1)
                # (B*L,) -> (B,L)
                if fp.dim() == 1 and fp.numel() % B == 0:
                    L = fp.numel() // B
                    fp = fp.view(B, L)
                if fp.dtype != torch.long:
                    fp = fp.long()

            # --- Descriptor handling (normalize to (B, D)) ---
            desc_in_dim = int(getattr(best_model, "desc_in_dim", 0) or 0)
            desc = getattr(batch, "desc", None)
            if desc_in_dim > 0:
                if desc is None:
                    desc = torch.zeros((B, desc_in_dim), device=device, dtype=torch.float32)
                else:
                    desc = desc.to(device).float()
                    if desc.dim() == 1 and desc.numel() % B == 0:
                        desc = desc.view(B, -1)
                    elif desc.dim() == 1:
                        desc = desc.unsqueeze(1)
                    elif desc.dim() > 2:
                        desc = desc.view(B, -1)
                    if desc.size(-1) != desc_in_dim:
                        raise RuntimeError(f"desc dim mismatch: expected {desc_in_dim}, got {desc.size(-1)}")
            else:
                desc = None

            # --- Forward (fp/graph/desc) ---
            _, preds = best_model({'fp': fp, 'graph': batch, 'desc': desc})   # preds: list[(B,1), ...]
            probs  = torch.cat([torch.sigmoid(p).detach().cpu() for p in preds], dim=1).numpy()
            labels = batch.y.detach().cpu().numpy()

            # --- Collect by species(task) ---
            for i, sp in enumerate(tasks):  # tasks = ['human','rat','mouse']
                lab_i = labels[:, i]
                prb_i = probs[:, i]
                valid = (lab_i != -1)
                if not np.any(valid):
                    continue
                test_labels[sp].extend(lab_i[valid].tolist())
                test_probs[sp].extend(prb_i[valid].tolist())

    # --- Per-species metric printouts ---
    for sp in tasks:
        if len(test_labels[sp]) == 0:
            logger.info(f"Performance for {sp}: (no valid labels; all were -1)")
            logger.info("-" * 40)
            continue
        thr = perf_threshold if not (perf_thresholds and sp in perf_thresholds) else float(perf_thresholds[sp])
        logger.info(f"Performance for {sp} (threshold={thr:.3f}):")
        printPerformance(
            test_labels[sp],
            test_probs[sp],
            threshold=thr,
            plot=perf_plot,
            printout=perf_printout,
        )
        logger.info("-" * 40)

    # --- Save predictions for ROC/PR plotting ---
    import pandas as pd
    save_dir = os.path.join(output_dir, "preds")
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    for sp in tasks:
        if len(test_labels[sp]) == 0:
            continue
        n = len(test_labels[sp])
        rows.append(pd.DataFrame({
            "model":   [tag]*n, 
            "species": [sp]*n, 
            "fold":    [best_fold]*n,
            "y_true":  test_labels[sp],
            "y_score": test_probs[sp],
        }))

    if rows:
        pred_df = pd.concat(rows, ignore_index=True)
        parquet_path = os.path.join(save_dir, f"{tag}_preds.parquet")
        final_path = _save_parquet_or_csv(pred_df, parquet_path, logger)
        logger.info(f"[SAVE DONE] {final_path}")

    return best_model, fold_train_loss_means, fold_val_loss_means

if __name__ == "__main__":
    model, trn_losses, val_losses = main_train(
        tag="our_model",   
        perf_threshold=0.5,
        perf_plot=True,
        perf_printout=True,
    )