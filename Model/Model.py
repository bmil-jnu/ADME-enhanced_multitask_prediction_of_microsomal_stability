import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init, Parameter
from typing import Optional, Tuple
from torch_geometric.nn import GCNConv, global_max_pool

class MolFPEncoder(nn.Module):
    def __init__(self, emb_dim=256, drop_ratio=0.3, fp_type="morgan+maccs+rdit", device=None):
        super().__init__()
        self.fp_type = fp_type.lower()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        morgan_dim = 2048 if 'morgan' in self.fp_type else 0
        maccs_dim  = 167  if 'maccs'  in self.fp_type else 0
        rdit_dim   = 2048 if 'rdit'   in self.fp_type else 0   # RDKit hashed

        init_dim = morgan_dim + maccs_dim + rdit_dim
        if init_dim == 0:
            raise ValueError(f"[MolFPEncoder] fp_type='{fp_type}'가 비어 있습니다.")

        self.net = nn.Sequential(
            nn.Linear(init_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=drop_ratio),
            nn.Linear(512, emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim),
        ).to(self.device)

        # Xavier로 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        feats = []
        if 'morgan' in self.fp_type:
            feats.append(data.morgan_fp.to(self.device))
        if 'maccs' in self.fp_type:
            feats.append(data.maccs_fp.to(self.device))
        if 'rdit' in self.fp_type:
            feats.append(data.rdit_fp.to(self.device))

        fps = torch.cat(feats, dim=1).float()
        return self.net(fps)  # (B, emb_dim)

def kaiming_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.2, norm='bn', act='leakyrelu'):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim) if norm=='bn' else nn.LayerNorm(dim)
        self.fc1   = nn.Linear(dim, dim)
        self.act   = nn.LeakyReLU(0.1) if act=='leakyrelu' else nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.fc2   = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.norm1(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h  # residual

class DescriptorMLP(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 256, width: int = 256,
                 depth: int = 3, dropout: float = 0.3, norm: str = 'bn'):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(in_dim) if norm=='bn' else nn.LayerNorm(in_dim)
        self.stem = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.LeakyReLU(0.1),
        )
        self.blocks = nn.Sequential(*[
            ResidualMLPBlock(width, dropout=dropout, norm=norm, act='leakyrelu')
            for _ in range(depth)
        ])
        self.head = nn.Sequential(
            nn.BatchNorm1d(width) if norm=='bn' else nn.LayerNorm(width),
            nn.Linear(width, emb_dim)
        )
        self.apply(kaiming_init_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.input_norm(x)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class GraphModule(nn.Module):
    def __init__(self, out_channels=256, hidden=512, dropout=0.3, device=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = GCNConv(-1, hidden, improved=True, add_self_loops=True, normalize=True).to(self.device)
        self.conv2 = GCNConv(hidden, hidden, improved=True, add_self_loops=True, normalize=True).to(self.device)

        self.bn1 = nn.BatchNorm1d(hidden).to(self.device)
        self.bn2 = nn.BatchNorm1d(hidden).to(self.device)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_final = nn.Linear(hidden, out_channels).to(self.device)

    def forward(self, data):
        x, edge_index, batch = data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)
        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = self.relu(x)

        x = global_max_pool(x, batch)   # (B, hidden)
        x = self.fc_final(x)            # (B, out_channels)
        return x

class WeightFusion(nn.Module):
    def __init__(self, feat_views: int, feat_dim: int, bias: bool = True, dropout: float = 0.3):
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
            bound = 1 / math.sqrt(self.weight.size(0))
            init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def set_temperature(self, t: float):
        self.temperature = float(max(t, 1e-6))

    @torch.no_grad()
    def set_gate(self, g: float):
        # 0.0 ~ 1.0
        self.gate = float(min(max(g, 0.0), 1.0))

    def forward(self, inputs: Tensor) -> Tensor:
        B, V, D = inputs.shape
        x = self.drop(inputs)
        w = self.weight / self.temperature         
        w = torch.softmax(w, dim=0)                 

        if self.gate != 0.0:
            uni = inputs.new_full((V, D), 1.0 / V)
            w = (1.0 - self.gate) * w + self.gate * uni

        out = (x * w.unsqueeze(0)).sum(dim=1)      
        if self.bias is not None:
            out = out + self.bias
        return out

class multitask_multimodal(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        device=None,
        num_tasks: int = 3,
        desc_in_dim: int = 0,
        fp_mode: str = "dense",                  
        fp_type: str = "morgan+maccs+rdit",  
        fp_emb_dim: int = 256,             
        # 기존
        fp_out_dim: int = 512,       
        graph_out_dim: int = 256,
        fusion_dim: int = 512,
        dropout: float = 0.3,
        desc_width: int = 256,
        desc_depth: int = 3,
    ):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_tasks = int(num_tasks)
        self.desc_in_dim = int(desc_in_dim)
        self.fusion_dim = int(fusion_dim)
        self.fp_mode = fp_mode.lower()
        self.fp_type = fp_type
        

        if self.fp_mode == "seq":
            self.fp_encoder = FingerprintEmbed(
                vocab_size=vocab_size, seq_len=100, emb_token_dim=128, out_dim=fp_out_dim, dropout=dropout
            ).to(self.device)
            self.fp_fc = nn.Linear(fp_out_dim, fusion_dim).to(self.device)
        elif self.fp_mode == "dense":
            self.fp_encoder = MolFPEncoder(
                emb_dim=fp_emb_dim, drop_ratio=dropout, fp_type=self.fp_type, device=self.device
            ).to(self.device)
            self.fp_fc = nn.Linear(fp_emb_dim, fusion_dim).to(self.device)
        else:
            raise ValueError(f"[model] Unknown fp_mode: {fp_mode}")

        self.graph_encoder = GraphModule(out_channels=graph_out_dim, dropout=dropout, device=self.device)
        self.graph_fc      = nn.Linear(graph_out_dim, fusion_dim).to(self.device)

        if self.desc_in_dim > 0:
            self.desc_mlp = DescriptorMLP(
                in_dim=self.desc_in_dim, emb_dim=fusion_dim,
                width=desc_width, depth=desc_depth, dropout=dropout, norm='bn'
            ).to(self.device)
        else:
            self.desc_mlp = None
        self.n_views = 2 + (1 if self.desc_in_dim > 0 else 0)
        self.fusion = ConcatFusion(
            feat_views=self.n_views,
            feat_dim=fusion_dim,
            out_dim=fusion_dim,
            norm='ln', 
            dropout=dropout
        )

        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
            ) for _ in range(self.num_tasks)
        ])
        self.outputs = nn.ModuleList([nn.Linear(fusion_dim // 2, 1) for _ in range(self.num_tasks)])

    def _get_tensor(self, container, key):
        if isinstance(container, dict):
            return container.get(key, None)
        return getattr(container, key, None)

    def _get_fp_like(self, data):
        t = self._get_tensor(data, 'fp')
        if t is None:
            t = self._get_tensor(data, 'smil2vec')
        return t

    def forward(self, data):

        graph_data = self._get_tensor(data, 'graph')
        if graph_data is None:
            raise RuntimeError("[model] 'graph'가 필요합니다.")
        gfeat = self.graph_encoder(graph_data) 
        graph_feat = self.graph_fc(gfeat)

        if self.fp_mode == "seq":
            fp_idx = self._get_fp_like(data)      # 'fp' 또는 'smil2vec'
            if fp_idx is None:
                raise RuntimeError("[model] fp_mode='seq'에서는 data['fp'] 또는 data['smil2vec'](LongTensor)가 필요합니다.")
            fp_idx = fp_idx.to(self.device).long()
            if torch.any(fp_idx < 0):
                raise RuntimeError(f"[model] fp_idx contains negative values: min={int(fp_idx.min())}")
            ffeat = self.fp_encoder(fp_idx)   
        else:  # dense
            ffeat = self.fp_encoder(graph_data)  
        fp_feat = self.fp_fc(ffeat)             

        desc_feat = None
        if (self.desc_mlp is not None):
            desc = self._get_tensor(data, 'desc')
            if desc is not None:
                desc = desc.to(self.device).float()
                if desc.dim() == 1:   desc = desc.unsqueeze(1)
                elif desc.dim() > 2:  desc = desc.view(desc.size(0), -1)
                if desc.size(-1) != self.desc_in_dim:
                    raise RuntimeError(f"[model] desc_in_dim mismatch: expected {self.desc_in_dim}, got {desc.size(-1)}")
                desc_feat = self.desc_mlp(desc)   

        sizes = [t.size(0) for t in (graph_feat, fp_feat, desc_feat) if t is not None]
        if len(sizes) == 0:
            raise RuntimeError("[model] No modality provided (graph/fp/desc are all None).")
        B = int(min(sizes))

        def ensure_B(x):
            if x is None:
                return torch.zeros(B, self.fusion_dim, device=self.device, dtype=torch.float32)
            return x[:B]

        graph_feat = ensure_B(graph_feat)
        fp_feat    = ensure_B(fp_feat)
        desc_feat  = ensure_B(desc_feat)

        views = [graph_feat, fp_feat]
        if self.desc_mlp is not None:
            views.append(desc_feat)
        # Fuse
        fusion_in = torch.stack([graph_feat, fp_feat], dim=1)  # (B, 2, D)
        fused = self.fusion(fusion_in)

        # Heads
        outs = []
        for head, out_lin in zip(self.task_heads, self.outputs):
            h = head(fused)
            outs.append(out_lin(h).view(-1, 1))
        return fused, tuple(outs)
