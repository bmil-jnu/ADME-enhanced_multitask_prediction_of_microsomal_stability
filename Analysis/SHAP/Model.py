# model.py
import math
from typing import Optional, Tuple, Union, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init, Parameter

from torch_geometric.nn import (
    GCNConv,
    global_max_pool,
)


# ============================================================
# 0) Utilities
# ============================================================
def kaiming_init_(m: nn.Module):
    """Apply Kaiming init to Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _get(container: Union[Dict[str, Any], Any], key: str, default=None):
    """Safely get attribute or dict item."""
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


# ============================================================
# 1) Fingerprint Encoder (SEQ) : FingerprintEmbed
# ============================================================
class FingerprintEmbed(nn.Module):
    """
    Token-sequence fingerprint encoder.
    - Input: (B, L) LongTensor
    - Output: (B, out_dim)

    Notes
    -----
    - padding_idx=0 recommended.
    - Tokens must be in [0, vocab_size-1].
    """
    def __init__(
        self,
        vocab_size: int,
        seq_len: int = 100,
        emb_token_dim: int = 128,
        out_dim: int = 128,
        dropout: float = 0.4,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)

        self.emb = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=emb_token_dim,
            padding_idx=padding_idx,
        )

        # light conv block
        self.conv1 = nn.Conv1d(emb_token_dim, emb_token_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(emb_token_dim)
        self.conv2 = nn.Conv1d(emb_token_dim, emb_token_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(emb_token_dim)

        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(emb_token_dim, out_dim)

        # init
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, fp_idx: Tensor) -> Tensor:
        # fp_idx: (B, L)
        if fp_idx.dim() != 2:
            raise RuntimeError(f"[FingerprintEmbed] expected (B,L), got {tuple(fp_idx.shape)}")

        if torch.any(fp_idx < 0):
            mn = int(fp_idx.min())
            raise RuntimeError(f"[FingerprintEmbed] negative token id detected: min={mn}")

        # (B, L, C)
        x = self.emb(fp_idx.long())
        x = self.drop(x)

        # (B, C, L)
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.1)

        # global max pool over L -> (B, C)
        x = torch.max(x, dim=2).values
        x = self.proj(x)
        return x


# ============================================================
# 2) Fingerprint Encoder (DENSE) : MolFPEncoder
# ============================================================
class MolFPEncoder(nn.Module):
    """
    Dense fingerprint encoder.
    Expects fingerprint tensors stored in graph_data (PyG batch):

    Required fields (depending on fp_type):
      - graph_data.morgan_fp : (B, 2048)
      - graph_data.maccs_fp  : (B, 167)
      - graph_data.rdit_fp   : (B, 2048)

    fp_type examples:
      - "morgan"
      - "morgan+maccs"
      - "morgan+maccs+rdit"
    """
    def __init__(
        self,
        emb_dim: int = 128,
        drop_ratio: float = 0.4,
        fp_type: str = "morgan+maccs+rdit",
    ):
        super().__init__()
        self.fp_type = fp_type.lower()

        morgan_dim = 2048 if "morgan" in self.fp_type else 0
        maccs_dim  = 167  if "maccs"  in self.fp_type else 0
        rdit_dim   = 2048 if "rdit"   in self.fp_type else 0

        init_dim = morgan_dim + maccs_dim + rdit_dim
        if init_dim == 0:
            raise ValueError(f"[MolFPEncoder] fp_type='{fp_type}' has no valid fp.")

        self.net = nn.Sequential(
            nn.Linear(init_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=drop_ratio),
            nn.Linear(512, emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
        )

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph_data) -> Tensor:
        feats: List[Tensor] = []

        if "morgan" in self.fp_type:
            if not hasattr(graph_data, "morgan_fp"):
                raise RuntimeError("[MolFPEncoder] missing graph_data.morgan_fp")
            feats.append(graph_data.morgan_fp)

        if "maccs" in self.fp_type:
            if not hasattr(graph_data, "maccs_fp"):
                raise RuntimeError("[MolFPEncoder] missing graph_data.maccs_fp")
            feats.append(graph_data.maccs_fp)

        if "rdit" in self.fp_type:
            if not hasattr(graph_data, "rdit_fp"):
                raise RuntimeError("[MolFPEncoder] missing graph_data.rdit_fp")
            feats.append(graph_data.rdit_fp)

        fps = torch.cat(feats, dim=1).float()
        return self.net(fps)


# ============================================================
# 3) Descriptor MLP (Residual)
# ============================================================
class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.3,
        norm: str = "bn",
        act: str = "leakyrelu",
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim) if norm == "bn" else nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.LeakyReLU(0.1) if act == "leakyrelu" else nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class DescriptorMLP(nn.Module):
    """
    Descriptor -> embedding MLP (with residual blocks).
    Input:  (B, in_dim)
    Output: (B, emb_dim)
    """
    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 128,
        width: int = 128,
        depth: int = 2,
        dropout: float = 0.4,
        norm: str = "bn",
    ):
        super().__init__()
        self.in_dim = int(in_dim)

        self.input_norm = nn.BatchNorm1d(in_dim) if norm == "bn" else nn.LayerNorm(in_dim)

        self.stem = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.LeakyReLU(0.1),
        )

        self.blocks = nn.Sequential(*[
            ResidualMLPBlock(width, dropout=dropout, norm=norm, act="leakyrelu")
            for _ in range(depth)
        ])

        self.head = nn.Sequential(
            (nn.BatchNorm1d(width) if norm == "bn" else nn.LayerNorm(width)),
            nn.Linear(width, emb_dim),
        )

        self.apply(kaiming_init_)

    def forward(self, x: Tensor) -> Tensor:
        # NaN/inf guard
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.input_norm(x)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


# ============================================================
# 4) Graph Encoder (2-layer GCN + Global Max Pool)
# ============================================================
class GraphModule(nn.Module):
    """
    2-layer GCN + BN + LeakyReLU + Dropout + GlobalMaxPool.

    Expects PyG batch:
      - data.x (N_nodes, F)
      - data.edge_index (2, E)
      - data.batch (N_nodes,)
      - optional: data.edge_weight (E,)
    """
    def __init__(
        self,
        out_channels: int = 128,
        hidden: int = 128,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.conv1 = GCNConv(
            in_channels=-1,
            out_channels=hidden,
            improved=True,
            add_self_loops=True,
            normalize=True,
        )
        self.conv2 = GCNConv(
            in_channels=hidden,
            out_channels=hidden,
            improved=True,
            add_self_loops=True,
            normalize=True,
        )

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)

        self.act = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(dropout)

        self.fc_final = nn.Linear(hidden, out_channels)

    def forward(self, data) -> Tensor:
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = getattr(data, "edge_weight", None)

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = self.act(x)

        x = global_max_pool(x, batch)   # (B, hidden)
        x = self.fc_final(x)            # (B, out_channels)
        return x


# ============================================================
# 5) Fusion Modules
# ============================================================
class WeightFusion(nn.Module):
    """
    View-wise weighted sum fusion.

    inputs:  (B, V, D)
    weight:  (V, D)
      - softmax over V with temperature
      - gate g in [0,1]: g=1 -> uniform avg, g=0 -> learned weights
    """
    def __init__(
        self,
        feat_views: int,
        feat_dim: int,
        bias: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.weight = Parameter(torch.empty(feat_views, feat_dim))
        self.bias = Parameter(torch.empty(feat_dim)) if bias else None
        self.drop = nn.Dropout(p=dropout)

        self.temperature: float = 1.0
        self.gate: float = 0.0

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.weight.size(0))
            init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def set_temperature(self, t: float):
        self.temperature = float(max(t, 1e-6))

    @torch.no_grad()
    def set_gate(self, g: float):
        self.gate = float(min(max(g, 0.0), 1.0))

    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs: (B, V, D)
        returns: (B, D)
        """
        if inputs.dim() != 3:
            raise RuntimeError(f"[WeightFusion] expected (B,V,D), got {tuple(inputs.shape)}")

        B, V, D = inputs.shape
        x = self.drop(inputs)

        w = self.weight / self.temperature
        w = torch.softmax(w, dim=0)  # (V, D)

        if self.gate != 0.0:
            uniform = inputs.new_full((V, D), 1.0 / V)
            w = (1.0 - self.gate) * w + self.gate * uniform

        out = (x * w.unsqueeze(0)).sum(dim=1)  # (B, D)
        if self.bias is not None:
            out = out + self.bias
        return out


class ConcatFusion(nn.Module):
    """
    Concatenate along view dimension then (optional) project.

    inputs:  (B, V, D)
    output:  (B, V*D) or (B, out_dim)
    """
    def __init__(
        self,
        feat_views: int,
        feat_dim: int,
        out_dim: Optional[int] = None,
        norm: str = "ln",
        dropout: float = 0.5,
    ):
        super().__init__()
        self.V = int(feat_views)
        self.D = int(feat_dim)

        in_dim = self.V * self.D
        self.norm = nn.BatchNorm1d(in_dim) if norm == "bn" else nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)

        if out_dim is None or out_dim == in_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() != 3:
            raise RuntimeError(f"[ConcatFusion] expected (B,V,D), got {tuple(inputs.shape)}")

        B, V, D = inputs.shape
        if V != self.V or D != self.D:
            raise RuntimeError(
                f"[ConcatFusion] shape mismatch: got (V={V},D={D}), expected (V={self.V},D={self.D})"
            )

        x = inputs.reshape(B, V * D)
        x = self.norm(x)
        x = self.drop(x)
        return self.proj(x)


# ============================================================
# 6) ADME_Multimdal_Multitask (FP + Graph (+ Desc) -> Fusion -> Task Heads)
# ============================================================
class ADME_Multimdal_Multitask(nn.Module):
    """
    Multi-Task Multi-Modal model.

    Input (dict-like or object)
    ---------------------------
    Required:
      - data['graph'] : PyG Batch (must include x/edge_index/batch)
    Optional:
      - data['desc']  : (B, desc_in_dim) float
      - data['fp'] or data['smil2vec'] : (B, L) long  (fp_mode='seq' only)

    fp_mode
    -------
    - "dense": uses graph_data.(morgan_fp/maccs_fp/rdit_fp)
    - "seq"  : uses token sequence (fp or smil2vec)
    """
    def __init__(
        self,
        vocab_size: int,
        num_tasks: int = 3,
        desc_in_dim: int = 128,

        fp_mode: str = "dense",                 # "seq" | "dense"
        fp_type: str = "morgan+maccs+rdit",     # dense mode
        fp_emb_dim: int = 128,                  # dense encoder out
        fp_out_dim: int = 128,                  # seq encoder out
        fp_seq_len: int = 100,
        fp_token_dim: int = 128,

        graph_out_dim: int = 128,
        fusion_dim: int = 128,
        dropout: float = 0.5,

        desc_width: int = 128,
        desc_depth: int = 2,

        fusion_type: str = "concat",            # "concat" | "weight"
    ):
        super().__init__()

        self.num_tasks = int(num_tasks)
        self.desc_in_dim = int(desc_in_dim)
        self.fusion_dim = int(fusion_dim)

        self.fp_mode = fp_mode.lower()
        self.fp_type = fp_type

        # ----- FP branch -----
        if self.fp_mode == "seq":
            self.fp_encoder = FingerprintEmbed(
                vocab_size=vocab_size,
                seq_len=fp_seq_len,
                emb_token_dim=fp_token_dim,
                out_dim=fp_out_dim,
                dropout=dropout,
                padding_idx=0,
            )
            self.fp_fc = nn.Linear(fp_out_dim, fusion_dim)

        elif self.fp_mode == "dense":
            self.fp_encoder = MolFPEncoder(
                emb_dim=fp_emb_dim,
                drop_ratio=dropout,
                fp_type=self.fp_type,
            )
            self.fp_fc = nn.Linear(fp_emb_dim, fusion_dim)

        else:
            raise ValueError(f"[MTMM] Unknown fp_mode: {fp_mode}")

        # ----- Graph branch -----
        self.graph_encoder = GraphModule(
            out_channels=graph_out_dim,
            hidden=graph_out_dim,
            dropout=dropout,
        )
        self.graph_fc = nn.Linear(graph_out_dim, fusion_dim)

        # ----- Desc branch (optional) -----
        if self.desc_in_dim > 0:
            self.desc_mlp = DescriptorMLP(
                in_dim=self.desc_in_dim,
                emb_dim=fusion_dim,
                width=desc_width,
                depth=desc_depth,
                dropout=dropout,
                norm="bn",
            )
            self.desc_norm = nn.LayerNorm(fusion_dim)
        else:
            self.desc_mlp = None
            self.desc_norm = None

        # ----- Branch norms -----
        self.graph_norm = nn.LayerNorm(fusion_dim)
        self.fp_norm = nn.LayerNorm(fusion_dim)

        # ----- Fusion -----
        self.n_views = 2 + (1 if self.desc_mlp is not None else 0)

        fusion_type = fusion_type.lower()
        if fusion_type == "weight":
            self.fusion = WeightFusion(
                feat_views=self.n_views,
                feat_dim=fusion_dim,
                dropout=dropout,
            )
            self.fused_dim = fusion_dim
        elif fusion_type == "concat":
            self.fusion = ConcatFusion(
                feat_views=self.n_views,
                feat_dim=fusion_dim,
                out_dim=fusion_dim,
                norm="ln",
                dropout=dropout,
            )
            self.fused_dim = fusion_dim
        else:
            raise ValueError(f"[MTMM] Unknown fusion_type: {fusion_type}")

        # ----- Task heads -----
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fused_dim, self.fused_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
                nn.Linear(self.fused_dim, self.fused_dim // 2),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            )
            for _ in range(self.num_tasks)
        ])

        self.outputs = nn.ModuleList([
            nn.Linear(self.fused_dim // 2, 1)
            for _ in range(self.num_tasks)
        ])

    def _get_fp_like(self, data):
        """Return data['fp'] or data['smil2vec'] if available."""
        t = _get(data, "fp", None)
        if t is None:
            t = _get(data, "smil2vec", None)
        return t

    def forward(self, data):
        # ----- graph -----
        graph_data = _get(data, "graph", None)
        if graph_data is None:
            raise RuntimeError("[MTMM] 'graph' is required in data.")

        # Graph features: (B, graph_out_dim) -> (B, fusion_dim)
        gfeat = self.graph_encoder(graph_data)
        graph_feat = self.graph_norm(self.graph_fc(gfeat))

        # ----- fp -----
        if self.fp_mode == "seq":
            fp_idx = self._get_fp_like(data)
            if fp_idx is None:
                raise RuntimeError("[MTMM] fp_mode='seq' needs data['fp'] or data['smil2vec']")
            ffeat = self.fp_encoder(fp_idx.long())
        else:
            # dense mode uses fingerprints stored in graph_data
            ffeat = self.fp_encoder(graph_data)

        fp_feat = self.fp_norm(self.fp_fc(ffeat))

        # ----- desc (optional) -----
        desc_feat = None
        if self.desc_mlp is not None:
            desc = _get(data, "desc", None)
            if desc is None:
                # allow missing -> zeros (keeps fusion shape)
                desc_feat = torch.zeros_like(graph_feat)
            else:
                desc = desc.float()
                if desc.dim() == 1:
                    desc = desc.unsqueeze(1)
                elif desc.dim() > 2:
                    desc = desc.view(desc.size(0), -1)

                if desc.size(-1) != self.desc_in_dim:
                    raise RuntimeError(
                        f"[MTMM] desc dim mismatch: expected {self.desc_in_dim}, got {desc.size(-1)}"
                    )

                desc_feat = self.desc_mlp(desc)
                desc_feat = self.desc_norm(desc_feat)

        # ----- align batch size (safety) -----
        sizes = [graph_feat.size(0), fp_feat.size(0)]
        if desc_feat is not None:
            sizes.append(desc_feat.size(0))
        B = int(min(sizes))

        graph_feat = graph_feat[:B]
        fp_feat = fp_feat[:B]
        if desc_feat is not None:
            desc_feat = desc_feat[:B]

        # ----- fusion -----
        views = [graph_feat, fp_feat]
        if desc_feat is not None:
            views.append(desc_feat)

        fusion_in = torch.stack(views, dim=1)      # (B, V, D)
        fused = self.fusion(fusion_in)             # (B, fused_dim)

        # ----- heads -----
        outs = []
        for head, out_lin in zip(self.task_heads, self.outputs):
            h = head(fused)
            outs.append(out_lin(h).view(-1, 1))

        return fused, tuple(outs)

    
class MTMMTaskWrapper(nn.Module):
    def __init__(self, base_model, task_idx, fp_vec=None, desc_vec=None, device=None, template_graph=None):
        super().__init__()
        self.base_model = base_model
        self.task_idx = int(task_idx)
        self.fp_vec = fp_vec
        self.desc_vec = desc_vec
        self.device = device if device is not None else next(base_model.parameters()).device
        self.template_graph = template_graph

    @torch.no_grad()
    def forward(self, x, edge_index, batch=None, edge_weight=None, **kwargs):
        # ← 텐서에 or 사용하지 말고 명시적으로 선택
        src_batch = batch if (batch is not None) else getattr(self.template_graph, 'batch', None)
        batch_vec = self._ensure_batch(x, src_batch)

        graph = self._rebuild_graph(x, edge_index, edge_weight, batch_vec)

        _, outs = self.base_model({
            'fp': self.fp_vec,
            'graph': graph,
            'desc': self.desc_vec
        })

        z = outs[self.task_idx].view(-1).mean()
        logits2 = torch.stack([-z, z], dim=0).unsqueeze(0)  # (1,2)
        return logits2

    def _ensure_batch(self, x, src_batch=None):
        if src_batch is not None and int(src_batch.numel()) == int(x.size(0)):
            return src_batch.to(self.device)
        return torch.zeros(x.size(0), dtype=torch.long, device=self.device)

    def _rebuild_graph(self, x, edge_index, edge_weight, batch_vec):
        from torch_geometric.data import Data
        tg = self.template_graph

        # edge_attr 정렬/길이 맞추기 (생략: 기존 코드 유지)
        edge_attr = getattr(tg, 'edge_attr', None) if tg is not None else None
        if edge_attr is not None and edge_attr.size(0) != edge_index.size(1):
            idxs = self._match_edge_order(getattr(tg, 'edge_index', None), edge_index) if tg is not None else None
            if idxs is not None: edge_attr = edge_attr[idxs]
            else: edge_attr = edge_attr[:edge_index.size(1)]

        g = Data(
            x=x.to(self.device),
            edge_index=edge_index.to(self.device),
            edge_attr=edge_attr.to(self.device) if edge_attr is not None else None,
        )
        g.batch = batch_vec
        g.num_nodes = int(x.size(0))
        if edge_weight is not None:
            g.edge_weight = edge_weight.to(self.device)

        # ✅ 템플릿에서 분자 관련/유틸 필드 복사 (있으면)
        if tg is not None:
            for k in [
                'ring_mask','ring_index','nf_node','nf_ring','num_rings','n_nodes',
                'n_nfs','smiles','edge_to_bond','edge_uv'
            ]:
                if hasattr(tg, k):
                    setattr(g, k, getattr(tg, k))

            # ✅ FP 필드 복사 (dense 모드용)
            for k in ['morgan_fp', 'maccs_fp', 'rdit_fp']:
                if hasattr(tg, k):
                    v = getattr(tg, k)
                    v = v.to(self.device)
                    if v.dim() == 1:           # 단일 그래프일 때 (D,) → (1, D)
                        v = v.unsqueeze(0)
                    setattr(g, k, v)

            # (선택) seq 모드 대비: 필요 시 smil2vec/fp도 복사
            for k in ['fp', 'smil2vec']:
                if hasattr(tg, k):
                    setattr(g, k, getattr(tg, k).to(self.device))

        return g.to(self.device)

    def _match_edge_order(self, E0, E_new):
        """템플릿 E0에서 E_new 각 (u,v) 위치 찾기(간단 매칭). 실패 시 None."""
        if E0 is None: 
            return None
        try:
            import collections, torch
            buckets = collections.defaultdict(list)
            for i in range(E0.size(1)):
                u, v = int(E0[0, i]), int(E0[1, i])
                buckets[(u, v)].append(i)
            idxs = []
            for j in range(E_new.size(1)):
                u, v = int(E_new[0, j]), int(E_new[1, j])
                cand = buckets.get((u, v)) or buckets.get((v, u))
                if not cand:
                    return None
                idxs.append(cand[0])
            return torch.tensor(idxs, dtype=torch.long, device=E_new.device)
        except Exception:
            return None
