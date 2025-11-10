# utile.py
import os
import time
import random
import math
import logging
import numpy as np
import torch
from typing import Optional, Callable, Literal

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve, auc,
    mean_squared_error, mean_absolute_error, average_precision_score,
    confusion_matrix, accuracy_score, matthews_corrcoef
)
import matplotlib.pyplot as plt

# =============================
# Reproducibility
# =============================
def seed_set(seed: int = 2024) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic for reproducibility; benchmark False recommended with deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================
# Logger
# =============================
def create_logger(output_dir: str = "output", tag: str = "default") -> logging.Logger:
    """Create a colorized console logger + file logger."""
    log_name = f"training_{tag}_{time.strftime('%Y-%m-%d')}.log"
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    fmt = '[%(asctime)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = '\033[92m[%(asctime)s]\033[0m \033[93m(%(filename)s %(lineno)d):\033[0m \033[95m%(levelname)-5s\033[0m %(message)s'

    # console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(ch)

    # file
    fh = logging.FileHandler(os.path.join(log_dir, log_name))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)

    return logger


# =============================
# Optimizer / Scheduler
# =============================
def build_optimizer(
    model: torch.nn.Module,
    optimizer_type: Literal["sgd","adam","adamw","ranger"] = "adamw",
    base_lr: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 1e-4
):
    """Return optimizer by name."""
    params = model.parameters()
    opt = optimizer_type.lower()
    if opt == 'sgd':
        return torch.optim.SGD(params, lr=base_lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    if opt == 'adam':
        return torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
    if opt == 'adamw':
        return torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    if opt == 'ranger':
        try:
            from ranger_adabelief import Ranger
        except Exception as e:
            raise ImportError("Ranger optimizer not installed. `pip install ranger-adabelief`") from e
        return Ranger(params, lr=base_lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def build_optimizer(model, optimizer_type="adamw", base_lr=0.001, momentum=0.9, weight_decay=1e-4):
    params = model.parameters()
    opt_lower = optimizer_type.lower()

    if opt_lower == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif opt_lower == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            weight_decay=weight_decay,
        )
    elif opt_lower == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=base_lr,
            weight_decay=weight_decay,
        )
    elif opt_lower == 'ranger':
        try:
            from ranger_adabelief import Ranger  # 라이브러리 설치 필요
            optimizer = Ranger(
                params,
                lr=base_lr,
                weight_decay=weight_decay,
            )
        except ImportError:
            raise ImportError("Ranger optimizer is not installed. Install it via 'pip install ranger-adabelief'.")
    else:
        raise ValueError(f"Optimizer '{optimizer_type}' is not supported. Choose from 'sgd', 'adam', 'adamw', or 'ranger'.")
    
    return optimizer

def build_scheduler(optimizer, scheduler_type="reduce", factor=0.1, patience=10, min_lr=1e-5, steps_per_epoch=None):
    if scheduler_type == "reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr
        )
    else:
        raise NotImplementedError(f"Unsupported LR Scheduler: {scheduler_type}")

    return scheduler

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.001, path='checkpoint.pt', trace_func=print, monitor='auc'):
        """
        :param monitor: 조기 종료를 위한 모니터링 메트릭 ('auc' 또는 'loss')
        :param patience: 개선되지 않더라도 얼마나 기다릴지
        :param delta: 개선을 정의할 최소 변화량
        :param path: 모델이 저장될 경로
        :param trace_func: 로그 출력 함수
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')  # 최소 검증 손실 추적
        self.monitor = monitor  # 'auc' 또는 'loss' 모니터링
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_metric, model):
        if self.monitor == 'auc':
            # AUC는 클수록 좋음
            score = val_metric
            improvement_condition = score > (self.best_score + self.delta) if self.best_score is not None else True
        else:  # 검증 손실을 모니터링할 때 (손실은 작을수록 좋음)
            score = -val_metric  # 손실이 작을수록 좋기 때문에 부호를 반대로 설정
            improvement_condition = score > (self.best_score + self.delta) if self.best_score is not None else True

        # 개선된 경우 처리
        if self.best_score is None or improvement_condition:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(f'EarlyStopping 카운터: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_metric, model):
        '''모니터링하는 메트릭이 개선되었을 때 모델을 저장합니다.'''
        if self.verbose:
            if self.monitor == 'auc':
                self.trace_func(f'검증 AUC가 개선되었습니다 ({self.best_score:.6f} --> {val_metric:.6f}). 모델을 저장합니다...')
            else:
                self.trace_func(f'검증 손실이 개선되었습니다 ({self.val_loss_min:.6f} --> {val_metric:.6f}). 모델을 저장합니다...')
        torch.save(model.state_dict(), self.path)
        if self.monitor == 'loss':
            self.val_loss_min = val_metric


# =============================
# Metrics (safe wrappers)
# =============================
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve, auc,
    average_precision_score, accuracy_score, matthews_corrcoef, confusion_matrix
)
import numpy as np
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score

def best_f1_threshold(y_true, y_prob):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    if len(thr) == 0:   # 전부 같은 레이블 등 특이 케이스
        return 0.5, 0.0
    f1 = (2*p*r) / (p + r + 1e-9)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i]), float(f1[i])

def binarize_with_thresholds(y_prob_dict, thresholds):
    y_pred_bin = {}
    for t_idx, probs in y_prob_dict.items():
        thr = thresholds[t_idx] if thresholds is not None and t_idx < len(thresholds) else 0.5
        y_pred_bin[t_idx] = (np.array(probs) >= thr).astype(int)
    return y_pred_bin

def op_metrics(y_true_dict, y_pred_bin):
    out = {}
    for t_idx in y_true_dict.keys():
        y_t = np.array(y_true_dict[t_idx]); y_p = np.array(y_pred_bin[t_idx])
        if y_t.size == 0: 
            out[t_idx] = None; continue
        prec = precision_score(y_t, y_p, zero_division=0)
        rec  = recall_score(y_t, y_p, zero_division=0)
        f1   = f1_score(y_t, y_p, zero_division=0)
        mcc  = matthews_corrcoef(y_t, y_p) if len(np.unique(y_t)) > 1 else 0.0
        acc  = accuracy_score(y_t, y_p)
        tn = np.sum((y_t == 0) & (y_p == 0))
        fp = np.sum((y_t == 0) & (y_p == 1))
        spec = tn / (tn + fp + 1e-9)
        out[t_idx] = dict(precision=prec, recall=rec, f1=f1, mcc=mcc, acc=acc, specificity=spec)
    return out


# ---- SAFE metrics ----
def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs,  dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0 or np.unique(y[m]).size < 2:
        return float('nan')
    return roc_auc_score(y[m], p[m])

def _safe_aupr(labels: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs,  dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0 or np.unique(y[m]).size < 2:
        return float('nan')
    return average_precision_score(y[m], p[m])

def prc_auc(targets, preds):
    """
    Trapezoidal PR-AUC (not AP). NaN/Inf 및 단일 클래스에 안전.
    """
    y = np.asarray(targets, dtype=float)
    p = np.asarray(preds,  dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() == 0:
        return float('nan')
    y = y[m]; p = p[m]
    if np.unique(y).size < 2:
        return float('nan')
    precision, recall, _ = precision_recall_curve(y, p)
    return auc(recall, precision)

def get_metric_func(metric: str = "auc"):
    metric = metric.lower()
    if metric == 'auc':   # ROC-AUC (safe)
        return _safe_auc
    if metric == 'prc':   # trapezoidal PR-AUC (safe)
        return prc_auc
    if metric == 'ap':    # Average Precision (safe)
        return _safe_aupr
    if metric == 'rmse':
        from math import sqrt
        from sklearn.metrics import mean_squared_error
        return lambda y, p: sqrt(mean_squared_error(y, p))
    if metric == 'mae':
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error
    raise ValueError(f'Metric "{metric}" not supported.')


def rmse(targets, preds):
    return math.sqrt(mean_squared_error(targets, preds))


def validate_loss_nan(loss: torch.Tensor, logger: logging.Logger, epoch: int) -> bool:
    """Return True if NaN loss detected and log an error."""
    if torch.isnan(loss):
        logger.error(f"NaN loss detected at epoch {epoch}.")
        return True
    return False


# =============================
# Visualization & Reporting
# =============================
def _plot_curve(x, y, xlabel, ylabel, title, legend_text):
    """ROC 및 PR 곡선 플롯"""
    plt.figure()
    plt.plot(x, y, color='darkorange', lw=2, label=legend_text)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def _safe_div(n, d):
    return n / d if d else 0.0

def _confusion_safe(labels_bin: np.ndarray, preds_bin: np.ndarray):
    # confusion_matrix needs both classes present; handle safely
    classes = np.unique(labels_bin)
    if len(classes) == 1:
        # fabricate the missing class with zero counts
        if classes[0] == 1:
            tn = fp = fn = 0; tp = int((preds_bin == 1).sum())
        else:
            tp = fp = fn = 0; tn = int((preds_bin == 0).sum())
        return tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(labels_bin, preds_bin).ravel()
    return tn, fp, fn, tp


def printPerformance(labels, probs, threshold: float = 0.5, printout: bool = True, plot: bool = True):
    labels = np.asarray(labels, dtype=float)
    probs  = np.asarray(probs,  dtype=float)

    # -1 마스크 + 유한성 마스크
    valid = (labels != -1) & np.isfinite(labels) & np.isfinite(probs)
    labels = labels[valid]; probs = probs[valid]

    if labels.size == 0:
        if printout:
            print("No valid labels to evaluate (all were -1 or non-finite).")
        return [float('nan')]*8

    preds_bin = (probs >= threshold).astype(int)

    # confusion 안전 처리
    classes = np.unique(labels)
    if len(classes) == 1:
        if classes[0] == 1:
            tn = fp = fn = 0; tp = int((preds_bin == 1).sum())
        else:
            tp = fp = fn = 0; tn = int((preds_bin == 0).sum())
    else:
        tn, fp, fn, tp = confusion_matrix(labels.astype(int), preds_bin).ravel()

    def _safe_div(n, d): return n / d if d else 0.0

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision   = _safe_div(tp, tp + fp)
    f1_score    = _safe_div(2 * precision * sensitivity, precision + sensitivity)

    acc   = accuracy_score(labels, preds_bin) if np.unique(labels).size > 0 else float('nan')
    rocA  = roc_auc_score(labels, probs)
    prA   = average_precision_score(labels, probs)  # AP
    mcc   = matthews_corrcoef(labels, preds_bin) if np.unique(labels).size > 1 else float('nan')

    metrics = [acc, rocA, prA, mcc, sensitivity, specificity, precision, f1_score]

    if printout:
        names = ['Accuracy', 'AUC-ROC', 'AP (AUPR)', 'MCC', 'Recall', 'Specificity', 'Precision', 'F1-score']
        for n, v in zip(names, metrics):
            print(f'{n}: {v:.4f}' if v == v else f'{n}: nan')

    if plot and np.unique(labels).size > 1:
        fpr, tpr, _ = roc_curve(labels, probs)
        import matplotlib.pyplot as plt
        plt.figure(); plt.plot(fpr, tpr, lw=2); plt.xlim([0,1]); plt.ylim([0,1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title(f'ROC (AUC = {rocA if rocA==rocA else float("nan"):.3f})'); plt.show()

        prec, rec, _ = precision_recall_curve(labels, probs)
        plt.figure(); plt.plot(rec, prec, lw=2); plt.xlim([0,1]); plt.ylim([0,1.05])
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR (AP = {prA if prA==prA else float("nan"):.3f})'); plt.show()

    return metrics


@torch.no_grad()
def compute_pos_weight(dataset):
    import numpy as np
    ylist = []
    for d in dataset:
        y = d.y.view(-1).cpu().numpy()  # (3,)
        ylist.append(y)
    Y = np.vstack(ylist)  # (N,3)
    mask = (Y >= 0)
    pos_w = []
    for i in range(Y.shape[1]):
        m = mask[:, i]
        if m.sum() == 0:
            pos_w.append(1.0)
        else:
            pos = Y[m, i].sum()
            neg = m.sum() - pos
            w = (neg + 1e-6) / (pos + 1e-6)
            pos_w.append(float(w))
    return torch.tensor(pos_w, dtype=torch.float)