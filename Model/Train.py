from typing import Optional, List
import time
import numpy as np
import torch
from collections import defaultdict
from utile import get_metric_func
def _prep_fp_tensor(fp: torch.Tensor, B: int, L: int = 100) -> torch.Tensor:
    # (B,1,L) -> (B,L)
    if fp.dim() == 3 and fp.size(1) == 1:
        fp = fp.squeeze(1)
    # (B*L,) -> (B,L)
    if fp.dim() == 1 and fp.numel() == B * L:
        fp = fp.view(B, L)
    return fp

def _forward_three_modal(model, data, device, logger=None):
    B = data.num_graphs

    # ---------- FP ----------
    fp = getattr(data, "smil2vec", None)
    if fp is None:
        fp = torch.zeros((B, 100), device=device, dtype=torch.long)
    else:
        fp = _prep_fp_tensor(fp, B=B, L=100).to(device)
        if fp.dtype != torch.long:
            fp = fp.long()

    # ---------- DESC ----------
    expected_desc = int(getattr(model, 'desc_in_dim', 0) or 0)
    desc = getattr(data, "desc", None)

    if desc is None:
        desc = torch.zeros((B, expected_desc if expected_desc > 0 else 1),
                           device=device, dtype=torch.float32)
    else:
        desc = desc.to(device).float()
        # 모양 정규화: (B,D)로
        if desc.dim() == 1:
            if desc.numel() == B:
                desc = desc.unsqueeze(1)        # (B,) -> (B,1)
            elif desc.numel() % B == 0:
                desc = desc.view(B, -1)         # (B*D,) -> (B,D)
            else:
                desc = desc.unsqueeze(1)        # 안전 기본값
        elif desc.dim() > 2:
            desc = desc.view(B, -1)

        # 기대 차원과 맞추기 (pad 또는 slice)
        if expected_desc > 0:
            D = desc.size(-1)
            if D < expected_desc:
                pad = torch.zeros((B, expected_desc - D), device=device, dtype=desc.dtype)
                desc = torch.cat([desc, pad], dim=1)
                if logger: logger.debug(f"[DESC] padded from {D} to {expected_desc}")
            elif D > expected_desc:
                desc = desc[:, :expected_desc]
                if logger: logger.debug(f"[DESC] sliced from {D} to {expected_desc}")

    # ---------- GRAPH ----------
    graph = data  # PyG Batch 전체

    pooled, task_outputs = model({'fp': fp, 'graph': graph, 'desc': desc})
    return pooled, task_outputs

def _dummy_zero_loss(pooled, task_outputs, model):
    if isinstance(pooled, torch.Tensor):
        anchor = pooled
    elif isinstance(task_outputs, (list, tuple)) and len(task_outputs) > 0 and isinstance(task_outputs[0], torch.Tensor):
        anchor = task_outputs[0]
    else:
        anchor = next(model.parameters())
    return anchor.mean() * 0.0

def train(epoch, model, criterion, train_loader, optimizer, lr_scheduler, device,
          task_type='classification', metric='auc', logger=None, max_grad_norm: float = 0.5,
          criterion_list: Optional[List[torch.nn.Module]] = None):
    model.train()
    losses = []
    num_tasks = getattr(model, 'num_tasks', 3)
    y_pred_list = {i: [] for i in range(num_tasks)}
    y_label_list = {i: [] for i in range(num_tasks)}

    for batch_idx, batch in enumerate(train_loader):
        data = batch.to(device)
        pooled, task_outputs = _forward_three_modal(model, data, device, logger)

        optimizer.zero_grad(set_to_none=True)
        loss_t = None

        y_labels = data.y
        if y_labels.dim() == 1:
            y_labels = y_labels.view(-1, num_tasks)
        assert y_labels.dim() == 2 and y_labels.size(1) == num_tasks, f"Expected y (B,{num_tasks}), got {tuple(y_labels.shape)}"

        for i in range(num_tasks):
            y_pred  = task_outputs[i].squeeze(-1)
            y_label = y_labels[:, i]
            valid_idx = (y_label != -1)

            if valid_idx.any():
                y_pred_v  = y_pred[valid_idx]
                y_label_v = y_label[valid_idx].float()

                crit = criterion_list[i] if criterion_list is not None else criterion
                task_loss = crit(y_pred_v, y_label_v)
                loss_t = task_loss if loss_t is None else (loss_t + task_loss)

                if task_type == 'classification':
                    y_pred_np = torch.sigmoid(y_pred_v).detach().cpu().numpy()
                else:
                    y_pred_np = y_pred_v.detach().cpu().numpy()

                y_pred_list[i].extend(y_pred_np)
                y_label_list[i].extend(y_label_v.cpu().numpy())

        if loss_t is None:
            loss_t = _dummy_zero_loss(pooled, task_outputs, model)

        loss_t.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        losses.append(loss_t.item())

    trn_loss = float(np.mean(losses)) if losses else float('nan')
    metric_func = get_metric_func(metric=metric)
    train_results = [metric_func(y_label_list[i], y_pred_list[i]) for i in range(num_tasks) if len(y_label_list[i])>0]
    avg_results = float(np.nanmean(train_results)) if train_results else float('nan')

    if logger: logger.info(f'[Train] Epoch {epoch} | Loss {trn_loss:.4f} | {metric}: {avg_results:.4f}')
    else:      print(f'Epoch: {epoch}, Train Loss: {trn_loss:.4f}, {metric}: {avg_results:.4f}')
    return trn_loss, avg_results

@torch.no_grad()
def validate(model, criterion, val_loader, device, task_type='classification', metric='auc',
             logger=None, eval_mode=False, epoch=None, seed=None,
             criterion_list: Optional[List[torch.nn.Module]] = None):
    model.eval()
    start = time.time()
    losses = []
    num_tasks = getattr(model, 'num_tasks', 3)
    y_pred_list = {i: [] for i in range(num_tasks)}
    y_label_list = {i: [] for i in range(num_tasks)}

    for batch_idx, batch in enumerate(val_loader):
        data = batch.to(device)
        _, task_outputs = _forward_three_modal(model, data, device, logger)

        y_labels = data.y
        if y_labels.dim() == 1:
            y_labels = y_labels.view(-1, num_tasks)
        assert y_labels.dim() == 2 and y_labels.size(1) == num_tasks

        for i in range(num_tasks):
            y_pred  = task_outputs[i].squeeze(-1)
            y_label = y_labels[:, i]
            valid_idx = (y_label != -1)
            if not valid_idx.any(): continue
            y_pred_v  = y_pred[valid_idx]
            y_label_v = y_label[valid_idx].float()
            crit = criterion_list[i] if criterion_list is not None else criterion
            loss = crit(y_pred_v, y_label_v)
            losses.append(loss.item())
            if task_type == 'classification':
                y_pred_list[i].extend(torch.sigmoid(y_pred_v).detach().cpu().numpy())
            else:
                y_pred_list[i].extend(y_pred_v.detach().cpu().numpy())
            y_label_list[i].extend(y_label_v.cpu().numpy())

    val_loss = float(np.mean(losses)) if losses else float('nan')
    metric_func = get_metric_func(metric=metric)
    val_results = [metric_func(y_label_list[i], y_pred_list[i]) for i in range(num_tasks) if len(y_label_list[i])>0]
    avg_val_results = float(np.nanmean(val_results)) if val_results else float('nan')

    duration = time.time() - start
    if logger: logger.info(f'[Valid] Epoch {epoch} | Loss {val_loss:.4f} | {metric}: {avg_val_results:.4f} | {duration:.2f}s')
    else:      print(f'Val Loss: {val_loss:.4f}, {metric}: {avg_val_results:.4f}')
    return val_loss, avg_val_results

@torch.no_grad()
def test(model, criterion, test_loader, device, task_type='classification', metric='auc',
         logger=None, drop_last=True, criterion_list: Optional[List[torch.nn.Module]] = None):
    model.eval()
    start = time.time()
    losses = []
    num_tasks = getattr(model, 'num_tasks', 3)
    y_pred_list = {i: [] for i in range(num_tasks)}
    y_label_list = {i: [] for i in range(num_tasks)}

    for batch_idx, batch in enumerate(test_loader):
        data = batch.to(device)
        _, task_outputs = _forward_three_modal(model, data, device, logger)

        y_labels = data.y
        if y_labels.dim() == 1:
            y_labels = y_labels.view(-1, num_tasks)
        assert y_labels.dim() == 2 and y_labels.size(1) == num_tasks

        for i in range(num_tasks):
            y_pred  = task_outputs[i].squeeze(-1)
            y_label = y_labels[:, i]
            valid_idx = (y_label != -1)
            if not valid_idx.any(): continue
            y_pred_v  = y_pred[valid_idx]
            y_label_v = y_label[valid_idx].float()
            crit = criterion_list[i] if criterion_list is not None else criterion
            loss = crit(y_pred_v, y_label_v)
            losses.append(loss.item())
            if task_type == 'classification':
                y_pred_list[i].extend(torch.sigmoid(y_pred_v).detach().cpu().numpy())
            else:
                y_pred_list[i].extend(y_pred_v.detach().cpu().numpy())
            y_label_list[i].extend(y_label_v.cpu().numpy())

    test_loss = float(np.mean(losses)) if losses else float('nan')
    metric_func = get_metric_func(metric=metric)
    test_results = [metric_func(y_label_list[i], y_pred_list[i]) for i in range(num_tasks) if len(y_label_list[i])>0]
    avg_test_results = float(np.nanmean(test_results)) if test_results else float('nan')

    duration = time.time() - start
    if logger: logger.info(f'[Test] Loss {test_loss:.4f} | {metric}: {avg_test_results:.4f} | {duration:.2f}s')
    else:      print(f'Test Loss: {test_loss:.4f}, {metric}: {avg_test_results:.4f}')
    return test_loss, avg_test_results
