
# =============================
# Imports & Globals
# =============================
import os, json
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem.MolStandardize import rdMolStandardize
from torch_geometric.utils import from_networkx
from tqdm import tqdm
from rdkit import Chem
import networkx as nx
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import DataStructs
# =============================
# Tokenizer (단일 정의)
# =============================
smi_to_seq = "(.02468@BDFHLNPRTVZ/bdfhlnprt#*%)+-/13579=ACEGIKMOSUWY[]acegimosuy\\"
seq_dict_smi = {ch: (i + 1) for i, ch in enumerate(smi_to_seq)}  # PAD/UNK=0
MAX_SEQ_SMI_LEN = 100  # 모델과 일치

def seq_smi(smile: str, max_seq_smi_len: int = MAX_SEQ_SMI_LEN):
    idx = np.array([seq_dict_smi.get(ch, 0) for ch in smile[:max_seq_smi_len]], dtype=int)
    if len(idx) < max_seq_smi_len:
        idx = np.pad(idx, (0, max_seq_smi_len - len(idx)), 'constant', constant_values=0)
    return idx

# =============================
# Descriptor 설정
# =============================
USE_EXPLICIT_COLS = True

PHYS_COLS = [
    'MW','TPSA','iLOGP','XLOGP3','WLOGP','MLOGP',
    '#Heavy atoms','#Aromatic heavy atoms','Fraction Csp3',
    '#Rotatable bonds','#H-bond acceptors','#H-bond donors',
    'MR','log Kp (cm/s)'
]
CONT_COLS = [
    'Lipinski #violations','Ghose #violations','Veber #violations',
    'Egan #violations','Muegge #violations','Bioavailability Score',
    'PAINS #alerts','Brenk #alerts','Leadlikeness #violations',
    'Synthetic Accessibility'
]
CAT_COLS = [
    'GI absorption','BBB permeant','Pgp substrate',
    'CYP1A2 inhibitor','CYP2C19 inhibitor',
    'CYP2C9 inhibitor','CYP2D6 inhibitor','CYP3A4 inhibitor'
]

def _coerce_numeric_series(s: pd.Series):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")

def build_desc_df(df: pd.DataFrame, tasks, logger=None, fixed_cols: list = None):
    exclude = set(['Cano_Smile','SMILES','PUBCHEM_EXT_DATASOURCE_SMILES'] + list(tasks))

    def _make_desc_df(df_):
        if USE_EXPLICIT_COLS:
            phys_in = [c for c in PHYS_COLS if c in df_.columns and c not in exclude]
            cont_in = [c for c in CONT_COLS if c in df_.columns and c not in exclude]
            cat_in  = [c for c in CAT_COLS  if c in df_.columns and c not in exclude]

            # numeric
            if phys_in or cont_in:
                num_df = df_[phys_in + cont_in].copy()
                for c in num_df.columns:
                    num_df[c] = _coerce_numeric_series(num_df[c])
                num_df = num_df.fillna(num_df.mean(numeric_only=True))
            else:
                num_df = pd.DataFrame(index=df_.index)

            # categorical (원-핫)
            if cat_in:
                cat_src = df_[cat_in].astype(str).apply(lambda s: s.str.strip())
                cat_df = pd.get_dummies(cat_src.astype('category'), dummy_na=False).fillna(0)
            else:
                cat_df = pd.DataFrame(index=df_.index)

            desc_df = pd.concat([num_df, cat_df], axis=1)

        else:
            candidate_numeric_cols = []
            for c in df_.columns:
                if c in exclude: 
                    continue
                coerced = _coerce_numeric_series(df_[c])
                if coerced.notna().mean() > 0.9:
                    candidate_numeric_cols.append(c)
            if candidate_numeric_cols:
                desc_df = df_[candidate_numeric_cols].copy()
                for c in desc_df.columns:
                    desc_df[c] = _coerce_numeric_series(desc_df[c])
                desc_df = desc_df.fillna(desc_df.mean(numeric_only=True))
            else:
                desc_df = pd.DataFrame(index=df_.index)

        if desc_df.shape[1] == 0:
            desc_df = pd.DataFrame({"__desc_dummy__": np.zeros(len(df_), dtype=float)}, index=df_.index)
        return desc_df

    if fixed_cols is None:
        # (A) 스키마 자동 생성 (train에서만)
        desc_df = _make_desc_df(df)
        cols = list(desc_df.columns)
    else:
        # (B) 고정 스키마 적용 (val/test에서)
        # 1) 전체 desc_df 생성
        temp = _make_desc_df(df)
        # 2) fixed에 없는 열은 버리고, fixed에 있는데 temp에 없는 열은 0으로 생성
        add_missing = [c for c in fixed_cols if c not in temp.columns]
        for c in add_missing:
            temp[c] = 0.0
        desc_df = temp.reindex(columns=fixed_cols)  # 순서 강제
        cols = list(desc_df.columns)

    if logger:
        logger.info(f"[MolDataset] Using {desc_df.shape[1]} descriptor cols")
    return desc_df, cols


# =============================
# 원자 특징
# =============================
from rdkit.Chem import rdchem
def one_of_k_encoding_unk(x, allowable_set):
    default_value = allowable_set[-1]
    if x not in allowable_set:
        x = default_value
    return [x == s for s in allowable_set]

def atom_features(atom):
    features = (
        one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
                                                 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
                                                 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                                                 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                                 'Pt', 'Hg', 'Pb', 'Nd', 'Ru', 'W', 'Unknown', 'Mo', 'Sr',
                                                 'Bi', 'S', 'Ba', 'Be', 'Ba', 'Dy']) +
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) +
        one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5]) +
        one_of_k_encoding_unk(atom.GetHybridization(), [rdchem.HybridizationType.SP, 
                                                        rdchem.HybridizationType.SP2, 
                                                        rdchem.HybridizationType.SP3, 
                                                        rdchem.HybridizationType.SP3D,
                                                        rdchem.HybridizationType.SP3D2]) +
        [atom.GetIsAromatic()])
    return np.array(features, dtype=float)


def filter_valid_smiles(smiles_list):
    """Filter valid SMILES from a list."""
    valid_smiles = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(smi)
        except Exception as e:
            print(f"Invalid SMILES: {smi} | Error: {e}")
    return valid_smiles

def mol2graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print(f"Warning: Molecule for SMILES {smile} could not be parsed.")
        return None

    features = np.array([atom_features(atom) for atom in mol.GetAtoms()], dtype=float)
    edges = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]

    g = nx.Graph(edges)
    if g.number_of_edges() == 0:
        print(f"Warning: Graph for SMILES {smile} has no edges.")
        return None

    data = from_networkx(g)
    data.x = torch.tensor(features, dtype=torch.float)
    data.smiles = smile
    return data
# =============================
# Dataset
# =============================
class MolDataset(InMemoryDataset):
    def __init__(self, root, dataset, task_type, tasks, logger=None,
                 transform=None, pre_transform=None, pre_filter=None,
                 desc_cols: list = None):  # ← 추가
        self.tasks = tasks
        self.dataset = dataset
        self.task_type = task_type
        self.logger = logger
        self.fixed_desc_cols = desc_cols  # ← train=None, test=train스키마
        super(MolDataset, self).__init__(root, transform, pre_transform, pre_filter)

        loaded = torch.load(self.processed_paths[0], map_location='cpu')
        # 3-tuple 과 4-tuple 모두 지원
        if isinstance(loaded, (list, tuple)) and len(loaded) == 4:
            self.data, self.slices, self.smiles_list, self.desc_cols_ = loaded
        else:
            self.data, self.slices, self.smiles_list = loaded
            self.desc_cols_ = None  # 구버전 호환

    @property
    def raw_file_names(self):
        return [self.dataset]

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.dataset)]

    def download(self):
        pass

    def process(self):
        dataset_path = os.path.join(self.root, self.dataset)
        df = pd.read_csv(dataset_path)

        # SMILES 컬럼 탐지/정규화
        SMI_CANDIDATES = ['Cano_Smile', 'SMILES', 'PUBCHEM_EXT_DATASOURCE_SMILES']
        smi_col = next((c for c in SMI_CANDIDATES if c in df.columns), None)
        if smi_col is None:
            raise ValueError(f"Input CSV must contain one of {SMI_CANDIDATES}")
        if smi_col != 'Cano_Smile':
            df = df.rename(columns={smi_col: 'Cano_Smile'})
            smi_col = 'Cano_Smile'

        # 라벨 보정
        for t in self.tasks:
            if t not in df.columns:
                df[t] = -1
            df[t] = _coerce_numeric_series(df[t]).fillna(-1).astype(np.float32)

        df = df.reset_index(drop=True)

        # (핵심) 스키마 생성/적용
        desc_df, used_cols = build_desc_df(
            df, tasks=self.tasks, logger=self.logger, fixed_cols=self.fixed_desc_cols
        )
        self.desc_cols_ = list(used_cols)  # 저장

        data_list, smiles_attr = [], []
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing SMILES"):
            smi = str(row[smi_col]).strip()
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            # 그래프 구성 (원래 코드 동일)
            feats = np.array([atom_features(atom) for atom in mol.GetAtoms()], dtype=float)
            edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
            n_atoms = mol.GetNumAtoms()
            g = nx.Graph()
            g.add_nodes_from(range(n_atoms))
            g.add_edges_from(edges)
            if g.number_of_edges() == 0:
                continue
            data = from_networkx(g)
            data.x = torch.tensor(feats, dtype=torch.float)
            bv_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr_morgan = np.zeros((2048,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv_morgan, arr_morgan)
            data.morgan_fp = torch.from_numpy(arr_morgan.astype(np.float32)).unsqueeze(0)

            # MACCS keys (167)
            bv_maccs = MACCSkeys.GenMACCSKeys(mol)
            arr_maccs = np.zeros((167,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv_maccs, arr_maccs)
            data.maccs_fp = torch.from_numpy(arr_maccs.astype(np.float32)).unsqueeze(0)

            # RDKit hashed FP (2048)
            bv_rdk = Chem.RDKFingerprint(mol, fpSize=2048)
            arr_rdk = np.zeros((2048,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv_rdk, arr_rdk)
            data.rdit_fp = torch.from_numpy(arr_rdk.astype(np.float32)).unsqueeze(0)

            num_nodes = data.x.size(0)
            if data.edge_index.numel() > 0:
                max_idx = int(data.edge_index.max().item())
                assert max_idx < num_nodes, f"edge_index.max()={max_idx} >= num_nodes={num_nodes} (SMILES={smi})"
            assert num_nodes == g.number_of_nodes(), f"x.rows={num_nodes} != g.nodes={g.number_of_nodes()} (SMILES={smi})"

            # 라벨
            y = np.array([row[t] for t in self.tasks], dtype=np.float32)
            data.y = torch.tensor(y, dtype=torch.float).unsqueeze(0)

            # FP
            smi_idx = seq_smi(smi, max_seq_smi_len=MAX_SEQ_SMI_LEN)
            data.smil2vec = torch.LongTensor(smi_idx).unsqueeze(0)

            # 설명자 (고정 스키마 순서로 보장됨)
            desc_vec = desc_df.iloc[i].values.astype(np.float32)
            data.desc = torch.tensor(desc_vec, dtype=torch.float).unsqueeze(0)
            data.smiles = smi
            data_list.append(data)
            smiles_attr.append(smi)

        data, slices = self.collate(data_list)
        # desc_cols_까지 함께 저장(신규 형식)
        torch.save((data, slices, smiles_attr, self.desc_cols_), self.processed_paths[0])

# =============================
# Fold Builder (라벨 (N,3) 보장)
# =============================
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from collections import Counter

def build_multilabel_stratified_loader(data_path, dataset_name, task_type, batch_size, tasks, logger, n_splits=5):
    full_dataset = MolDataset(root=data_path, dataset=dataset_name, task_type=task_type, tasks=tasks, logger=logger)

    # ← train 전체의 desc 스키마 확보
    train_desc_cols = getattr(full_dataset, "desc_cols_", None)

    ys = []
    for d in full_dataset:
        y = d.y
        if y.dim() == 1:
            y = y.unsqueeze(0)
        elif y.dim() > 2:
            y = y.view(1, -1)
        ys.append(y)
    _all_labels_t = torch.cat(ys, dim=0).cpu()          # torch.Tensor (CPU)
    all_labels = np.asarray(_all_labels_t.tolist()) 

    all_labels = np.where(all_labels < 0, 0, all_labels).astype(int)

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    fold_data = list(mskf.split(range(len(all_labels)), all_labels))

    train_loaders, val_loaders = [], []
    for fold_idx, (train_index, val_index) in enumerate(fold_data):
        train_data_list = [full_dataset[i] for i in train_index]
        val_data_list   = [full_dataset[i] for i in val_index]

        train_loader = GeometricDataLoader(
            train_data_list, batch_size=batch_size, shuffle=True,
            num_workers=0, drop_last=True     # 학습은 True (OK)
        )
        val_loader   = GeometricDataLoader(
            val_data_list, batch_size=batch_size, shuffle=False,
            num_workers=0, drop_last=False    # ✅ 평가에서는 False
        )

        if logger:
            from collections import Counter
            trc = Counter(); vlc = Counter()
            for d in train_data_list:
                yrow = d.y.view(-1)
                for i, t in enumerate(tasks):
                    if yrow[i].item() != -1: trc[t]+=1
            for d in val_data_list:
                yrow = d.y.view(-1)
                for i, t in enumerate(tasks):
                    if yrow[i].item() != -1: vlc[t]+=1
            logger.info(f"[Fold {fold_idx+1}] Train size={len(train_data_list)}, Val size={len(val_data_list)}")
            logger.info(f"[Fold {fold_idx+1}] Train label counts={dict(trc)}")
            logger.info(f"[Fold {fold_idx+1}] Val   label counts={dict(vlc)}")

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    # ← 스키마를 함께 반환
    return train_loaders, val_loaders, train_desc_cols



def build_loader(data_path, dataset_names, task_type, batch_size, tasks, logger):
    train_loader = val_loader = test_loader = None

    if 'train' in dataset_names:
        train_dataset = MolDataset(root=data_path, dataset=dataset_names['train'], task_type=task_type, tasks=tasks, logger=logger)
        train_loader  = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=True)

    if 'val' in dataset_names:
        val_dataset = MolDataset(root=data_path, dataset=dataset_names['val'], task_type=task_type, tasks=tasks, logger=logger)
        val_loader   = GeometricDataLoader(val_dataset, batch_size=batch_size,   shuffle=False, num_workers=0, drop_last=False)

    if 'test' in dataset_names:
        test_dataset = MolDataset(root=data_path, dataset=dataset_names['test'], task_type=task_type, tasks=tasks, logger=logger)
        test_loader  = GeometricDataLoader(test_dataset, batch_size=batch_size,  shuffle=False, num_workers=0, drop_last=False)

    return train_loader, val_loader, test_loader
