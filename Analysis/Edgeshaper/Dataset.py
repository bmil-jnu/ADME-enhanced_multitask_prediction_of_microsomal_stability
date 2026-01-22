# dataset_scaffold.py

import os
import random
from collections import defaultdict, Counter
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader as GeometricDataLoader

from sklearn.preprocessing import StandardScaler

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator


# ============================================================
# 1) SMILES -> sequence indices (useful for fp_mode='seq')
# ============================================================
SMI_VOCAB = "(.02468@BDFHLNPRTVZ/bdfhlnprt#*%)+-/13579=ACEGIKMOSUWY[]acegimosuy\\"
SEQ_DICT_SMI = {ch: (i + 1) for i, ch in enumerate(SMI_VOCAB)}  # PAD/UNK = 0
MAX_SEQ_SMI_LEN = 100  # Must match the model setting


def seq_smi(smile: str, max_len: int = MAX_SEQ_SMI_LEN) -> np.ndarray:
    """
    Convert a SMILES string to a fixed-length integer sequence.
    Unknown characters are mapped to 0. Output is padded/truncated to max_len.
    """
    idx = np.array([SEQ_DICT_SMI.get(ch, 0) for ch in smile[:max_len]], dtype=np.int64)
    if idx.size < max_len:
        idx = np.pad(idx, (0, max_len - idx.size), mode="constant", constant_values=0)
    return idx


# ============================================================
# 2) Fingerprint generators (global singletons)
# ============================================================
MORGAN_GEN = GetMorganGenerator(radius=2, fpSize=2048)
RDK_GEN = GetRDKitFPGenerator(fpSize=2048)


# ============================================================
# 3) Descriptor configuration (lean ADME subset)
# ============================================================
USE_EXPLICIT_COLS = True

PHYS_COLS = [
    "MW",
    "TPSA",
    "iLOGP",
    "XLOGP3",
    "WLOGP",
    "MLOGP",
    "#Heavy atoms",
    "#Aromatic heavy atoms",
    "Fraction Csp3",
    "#Rotatable bonds",
    "#H-bond acceptors",
    "#H-bond donors",
    "MR",
    "log Kp (cm/s)",
]
CONT_COLS = [
    "Lipinski #violations",
    "Ghose #violations",
    "Veber #violations",
    "Egan #violations",
    "Muegge #violations",
    "Bioavailability Score",
    "PAINS #alerts",
    "Brenk #alerts",
    "Leadlikeness #violations",
    "Synthetic Accessibility",
]
CAT_COLS = [
    "GI absorption",
    "BBB permeant",
    "Pgp substrate",
    "CYP1A2 inhibitor",
    "CYP2C19 inhibitor",
    "CYP2C9 inhibitor",
    "CYP2D6 inhibitor",
    "CYP3A4 inhibitor",
]


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """
    Coerce a column into numeric values safely.
    - Removes commas
    - Strips whitespace
    - Non-parsable entries become NaN
    """
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def _make_desc_df_raw(df_: pd.DataFrame, tasks: List[str]) -> pd.DataFrame:
    """
    Build a raw descriptor DataFrame:
    - Selects descriptor subsets (lean ADME subset) if USE_EXPLICIT_COLS=True
    - Converts numeric-like columns to float
    - Encodes categorical descriptors:
        * binary -> single 0/1 column (positive=lexicographically last)
        * multi-class -> one-hot (dummy_na=False)
    This function DOES NOT:
    - Replace +/-Inf
    - Impute NaNs
    - Scale values
    """
    exclude = set(["Cano_Smile", "SMILES", "PUBCHEM_EXT_DATASOURCE_SMILES"] + list(tasks))

    if USE_EXPLICIT_COLS:
        phys_in = [c for c in PHYS_COLS if c in df_.columns and c not in exclude]
        cont_in = [c for c in CONT_COLS if c in df_.columns and c not in exclude]
        cat_in = [c for c in CAT_COLS if c in df_.columns and c not in exclude]

        # --- numeric subset ---
        if phys_in or cont_in:
            num_df = df_[phys_in + cont_in].copy()
            for c in num_df.columns:
                num_df[c] = _coerce_numeric_series(num_df[c])
        else:
            num_df = pd.DataFrame(index=df_.index)

        # --- categorical subset ---
        if cat_in:
            cat_blocks = []
            for c in cat_in:
                col = df_[c].copy()
                col = col.astype(str).str.strip()
                col = col.replace("nan", np.nan)

                uniq = sorted([u for u in col.dropna().unique()])

                # Binary: use a single column, "positive" is lexicographically last
                if 0 < len(uniq) <= 2:
                    pos = uniq[-1]
                    s = (col == pos).astype(float).fillna(0.0)
                    s.name = f"{c}__{pos}"
                    cat_blocks.append(s)
                else:
                    d = pd.get_dummies(col, prefix=c, dummy_na=False)
                    cat_blocks.append(d)

            cat_df = pd.concat(cat_blocks, axis=1) if cat_blocks else pd.DataFrame(index=df_.index)
        else:
            cat_df = pd.DataFrame(index=df_.index)

        desc_df = pd.concat([num_df, cat_df], axis=1)

    else:
        # Fallback: automatically detect numeric-like columns (>=90% numeric after coercion)
        cand = []
        for c in df_.columns:
            if c in exclude:
                continue
            coerced = _coerce_numeric_series(df_[c])
            if coerced.notna().mean() > 0.9:
                cand.append(c)

        if cand:
            desc_df = df_[cand].copy()
            for c in desc_df.columns:
                desc_df[c] = _coerce_numeric_series(desc_df[c])
        else:
            desc_df = pd.DataFrame(index=df_.index)

    # Prevent zero-dimension descriptor vectors
    if desc_df.shape[1] == 0:
        desc_df = pd.DataFrame({"__desc_dummy__": np.zeros(len(df_), dtype=float)}, index=df_.index)

    return desc_df


def build_desc_df_scaled(
    df: pd.DataFrame,
    tasks: List[str],
    logger=None,
    fixed_cols: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, List[str], StandardScaler]:
    """
    Build a scaled descriptor DataFrame:
      raw descriptors -> +/-Inf to NaN -> median imputation -> StandardScaler

    Parameters
    ----------
    fixed_cols:
      - None: (train) build schema from df
      - list: (val/test) enforce train schema
          * missing columns are created with zeros
          * extra columns are dropped
    scaler:
      - None: (train) fit a new StandardScaler
      - scaler: (val/test) transform using the provided scaler

    Returns
    -------
    desc_df_scaled: pd.DataFrame
    cols: List[str]
    scaler: StandardScaler
    """
    desc_df_raw = _make_desc_df_raw(df, tasks)

    # Align schema
    if fixed_cols is None:
        cols = list(desc_df_raw.columns)
    else:
        missing = [c for c in fixed_cols if c not in desc_df_raw.columns]
        for c in missing:
            desc_df_raw[c] = 0.0
        desc_df_raw = desc_df_raw.reindex(columns=fixed_cols)
        cols = list(fixed_cols)

    # Replace +/-Inf with NaN
    desc_df = desc_df_raw.replace([np.inf, -np.inf], np.nan)

    # Median imputation per column
    for c in desc_df.columns:
        if desc_df[c].isna().any():
            med = desc_df[c].median()
            if pd.isna(med):
                med = 0.0
            desc_df[c] = desc_df[c].fillna(med)

    X = desc_df.values.astype("float32")

    # Scale
    if scaler is None:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = scaler.transform(X)

    out = pd.DataFrame(Xs, columns=cols, index=df.index)

    if logger:
        logger.info(f"[MolDataset] Using {out.shape[1]} descriptor columns (scaled).")

    return out, cols, scaler


# ============================================================
# 4) Graph featurization
# ============================================================
def one_of_k_encoding_unk(x, allowable_set):
    """One-hot encode x; if x not in allowable_set, map it to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom) -> np.ndarray:
    """
    Atom-level features:
      - element symbol
      - degree
      - formal charge
      - explicit valence
      - total Hs
      - hybridization
      - aromaticity
      - chirality tag
    """
    hyb_S = getattr(Chem.rdchem.HybridizationType, "S", Chem.rdchem.HybridizationType.UNSPECIFIED)

    feats = (
        one_of_k_encoding_unk(atom.GetSymbol(), [
            "C","N","O","S","F","Si","P","Cl","Br","Mg","Na","Ca","Fe","As","Al","I","B","V","K",
            "Tl","Yb","Sb","Sn","Ag","Pd","Co","Se","Ti","Zn","H","Li","Ge","Cu","Au","Ni","Cd",
            "In","Mn","Zr","Cr","Pt","Hg","Pb","Nd","Ru","W","Unknown","Mo","Sr","Bi","Ba","Be","Dy"
        ]) +
        one_of_k_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5,6, "UNK"]) +
        one_of_k_encoding_unk(atom.GetFormalCharge(), [-1,0,1, "UNK"]) +
        one_of_k_encoding_unk(atom.GetExplicitValence(), [0,1,2,3,4,5,6, "UNK"]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4,5, "UNK"]) +
        one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            hyb_S,
            Chem.rdchem.HybridizationType.UNSPECIFIED
        ]) +
        [atom.GetIsAromatic()] +
        one_of_k_encoding_unk(atom.GetChiralTag(), [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ])
    )
    return np.asarray(feats, dtype=np.float32)


def bond_features(bond) -> np.ndarray:
    """
    Bond-level features:
      - bond type (single/double/triple/aromatic)
      - conjugation
      - ring membership
    """
    bt = bond.GetBondType()
    feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    return np.asarray(feats, dtype=np.float32)


# ============================================================
# 5) MolDataset: CSV -> PyG graphs -> cached processed file
# ============================================================
class MolDataset(InMemoryDataset):
    """
    CSV -> PyG Data list -> saved as:
        (data, slices, smiles_list, desc_cols, desc_scaler)

    Each Data object contains:
      - x, edge_index, edge_attr
      - smiles: canonical SMILES string
      - morgan_fp, rdit_fp, maccs_fp: (1, D) float32 (batched into (B, D))
      - smil2vec: (1, L) long (batched into (B, L))
      - y: (1, n_tasks) float32, -1 indicates missing label
      - desc: (1, n_desc) scaled float32
    """

    def __init__(
        self,
        root: str,
        dataset: str,
        task_type: str,
        tasks: List[str],
        logger=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        desc_cols: Optional[List[str]] = None,
        desc_scaler: Optional[StandardScaler] = None,
    ):
        self.dataset = dataset
        self.task_type = task_type
        self.tasks = list(tasks)
        self.logger = logger

        # Enforce schema/scaler from train when building val/test sets
        self.fixed_desc_cols = desc_cols
        self.fixed_desc_scaler = desc_scaler

        super().__init__(root, transform, pre_transform, pre_filter)

        loaded = torch.load(self.processed_paths[0], map_location="cpu")
        if isinstance(loaded, tuple) and len(loaded) == 5:
            self.data, self.slices, self.smiles_list, self.desc_cols_, self.desc_scaler_ = loaded
        else:
            # Backward compatibility for old cache formats
            self.data, self.slices, self.smiles_list = loaded[:3]
            self.desc_cols_, self.desc_scaler_ = None, None

    @property
    def raw_file_names(self) -> List[str]:
        return [self.dataset]

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self.dataset}.pt"]

    def process(self):
        path = os.path.join(self.root, self.dataset)
        df = pd.read_csv(path)

        # 1) Identify SMILES column
        smi_candidates = ["Cano_Smile", "SMILES", "PUBCHEM_EXT_DATASOURCE_SMILES"]
        smi_col = next((c for c in smi_candidates if c in df.columns), None)
        if smi_col is None:
            raise ValueError(f"Input CSV must contain one of {smi_candidates}")

        # 2) Label processing: missing labels are -1
        for t in self.tasks:
            if t not in df.columns:
                df[t] = -1
            df[t] = _coerce_numeric_series(df[t]).fillna(-1).astype(np.float32)

        label_mat = df[self.tasks].values
        valid_mask = (label_mat != -1).any(axis=1)
        df = df.loc[valid_mask].reset_index(drop=True)

        if self.logger:
            self.logger.info(f"[{self.dataset}] kept rows: {len(df)} (at least one label available)")

        # 3) Build scaled descriptors
        desc_df, used_cols, scaler = build_desc_df_scaled(
            df,
            tasks=self.tasks,
            logger=self.logger,
            fixed_cols=self.fixed_desc_cols,
            scaler=self.fixed_desc_scaler,
        )
        self.desc_cols_ = list(used_cols)
        self.desc_scaler_ = scaler

        data_list: List[Data] = []
        smiles_attr: List[str] = []

        n_invalid = 0
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Graph Conversion"):
            smi = str(row[smi_col]).strip()
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_invalid += 1
                continue

            canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=True)

            # Node features
            x = torch.tensor(np.stack([atom_features(a) for a in mol.GetAtoms()], axis=0), dtype=torch.float32)

            # Edge index and edge attributes (undirected graph: add both directions)
            edge_indices, edge_attrs = [], []
            for b in mol.GetBonds():
                s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                bf = bond_features(b)
                edge_indices.append([s, e])
                edge_attrs.append(bf)
                edge_indices.append([e, s])
                edge_attrs.append(bf)

            if len(edge_indices) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 6), dtype=torch.float32)
            else:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(np.stack(edge_attrs, axis=0), dtype=torch.float32)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data.smiles = canonical_smi

            # Fingerprints saved as (1, D) so that batching yields (B, D)
            def _bv_to_tensor(bv) -> torch.Tensor:
                arr = np.zeros((bv.GetNumBits(),), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(bv, arr)
                return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)

            data.morgan_fp = _bv_to_tensor(MORGAN_GEN.GetFingerprint(mol))
            data.rdit_fp = _bv_to_tensor(RDK_GEN.GetFingerprint(mol))
            data.maccs_fp = _bv_to_tensor(MACCSkeys.GenMACCSKeys(mol))

            # Labels / descriptors / SMILES sequence
            y = row[self.tasks].values.astype(np.float32)
            data.y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

            data.desc = torch.tensor(desc_df.iloc[i].values.astype(np.float32), dtype=torch.float32).unsqueeze(0)
            data.smil2vec = torch.tensor(seq_smi(canonical_smi), dtype=torch.long).unsqueeze(0)

            data_list.append(data)
            smiles_attr.append(canonical_smi)

        if self.logger:
            self.logger.info(f"[{self.dataset}] invalid SMILES skipped: {n_invalid}")

        data, slices = self.collate(data_list)
        torch.save((data, slices, smiles_attr, self.desc_cols_, self.desc_scaler_), self.processed_paths[0])


# ============================================================
# 6) Scaffold split utilities
# ============================================================
def _murcko_scaffold(smi: str, include_chirality: bool = False) -> str:
    """Return Bemis–Murcko scaffold SMILES (empty string if failure)."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        return scaf or ""
    except Exception:
        return ""


def _group_indices_by_scaffold(smiles_list: List[str], include_chirality: bool = False) -> Dict[str, List[int]]:
    """Group sample indices by scaffold."""
    buckets = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaf = _murcko_scaffold(smi, include_chirality=include_chirality)
        key = scaf if scaf else smi  # fallback to SMILES when scaffold is empty
        buckets[key].append(i)
    return buckets


def _greedy_pack_scaffolds_to_folds(scaffold_buckets: Dict[str, List[int]], n_splits: int) -> List[List[int]]:
    """
    Greedy packing:
      - Sort scaffold groups by decreasing size (and key for tie-break)
      - Assign each group to the fold with currently smallest size
    """
    groups = list(scaffold_buckets.items())
    groups.sort(key=lambda kv: (-len(kv[1]), kv[0]))

    fold_bins = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    for _, idx_list in groups:
        k = min(range(n_splits), key=lambda f: fold_sizes[f])
        fold_bins[k].extend(idx_list)
        fold_sizes[k] += len(idx_list)

    return fold_bins


def _log_fold_stats(data_list: List[Data], tasks: List[str], logger, tag: str = ""):
    """Log how many non-missing labels exist per task in a given split."""
    if not logger:
        return
    cnt = Counter()
    for d in data_list:
        yrow = d.y.view(-1)
        for i, t in enumerate(tasks):
            if float(yrow[i].item()) != -1.0:
                cnt[t] += 1
    logger.info(f"{tag} size={len(data_list)} | available labels per task={dict(cnt)}")


# ============================================================
# 7) Loader builders
# ============================================================
def build_scaffold_kfold_loader(
    data_path: str,
    dataset_name: str,
    task_type: str,
    batch_size: int,
    tasks: List[str],
    logger=None,
    n_splits: int = 10,
    include_chirality: bool = False,
    seed: int = 42,
) -> Tuple[List[GeometricDataLoader], List[GeometricDataLoader], Optional[List[str]], Optional[StandardScaler]]:
    """
    Build scaffold-based K-fold loaders.

    Note (same caveat as the original pipeline):
    - If MolDataset is processed once on the full dataset, the descriptor scaler can be fit using
      all samples (including those later assigned to validation folds). This can introduce mild leakage
      in strict cross-validation settings.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    full_dataset = MolDataset(
        root=data_path,
        dataset=dataset_name,
        task_type=task_type,
        tasks=tasks,
        logger=logger,
    )

    train_desc_cols = getattr(full_dataset, "desc_cols_", None)
    train_desc_scaler = getattr(full_dataset, "desc_scaler_", None)

    smiles_list = getattr(full_dataset, "smiles_list", None)
    if smiles_list is None:
        smiles_list = [getattr(full_dataset[i], "smiles", "") for i in range(len(full_dataset))]

    buckets = _group_indices_by_scaffold(smiles_list, include_chirality=include_chirality)
    fold_bins = _greedy_pack_scaffolds_to_folds(buckets, n_splits=n_splits)

    train_loaders, val_loaders = [], []
    for fold_idx in range(n_splits):
        val_index = sorted(fold_bins[fold_idx])
        train_index = sorted([i for k in range(n_splits) if k != fold_idx for i in fold_bins[k]])

        train_data_list = [full_dataset[i] for i in train_index]
        val_data_list = [full_dataset[i] for i in val_index]

        train_loader = GeometricDataLoader(
            train_data_list,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        val_loader = GeometricDataLoader(
            val_data_list,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        if logger:
            logger.info(f"[Scaffold Fold {fold_idx+1}/{n_splits}] Train={len(train_data_list)}, Val={len(val_data_list)}")
            _log_fold_stats(train_data_list, tasks, logger, tag="  Train")
            _log_fold_stats(val_data_list, tasks, logger, tag="  Val  ")

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders, train_desc_cols, train_desc_scaler


def build_loader(
    data_path: str,
    dataset_names: Dict[str, str],  # e.g., {'train':..., 'val':..., 'test':...}
    task_type: str,
    batch_size: int,
    tasks: List[str],
    logger=None,
):
    """
    Build standard train/val/test loaders from separate CSV files:
      - Train dataset fits descriptor schema & scaler.
      - Val/Test enforce train schema and use train scaler for transform.

    Returns:
      train_loader, val_loader, test_loader, train_desc_cols, train_desc_scaler
    """
    train_loader = val_loader = test_loader = None
    train_desc_cols = None
    train_desc_scaler = None

    if "train" in dataset_names:
        train_dataset = MolDataset(
            root=data_path,
            dataset=dataset_names["train"],
            task_type=task_type,
            tasks=tasks,
            logger=logger,
            desc_cols=None,
            desc_scaler=None,
        )
        train_desc_cols = getattr(train_dataset, "desc_cols_", None)
        train_desc_scaler = getattr(train_dataset, "desc_scaler_", None)

        train_loader = GeometricDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

    if "val" in dataset_names:
        val_dataset = MolDataset(
            root=data_path,
            dataset=dataset_names["val"],
            task_type=task_type,
            tasks=tasks,
            logger=logger,
            desc_cols=train_desc_cols,
            desc_scaler=train_desc_scaler,
        )
        val_loader = GeometricDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

    if "test" in dataset_names:
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

    return train_loader, val_loader, test_loader, train_desc_cols, train_desc_scaler