import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from dotenv import load_dotenv
load_dotenv() 

data_dir = os.getenv('DATA_DIR')
base_data_dir = os.path.join(project_root, 'data')
downstream_data_path = os.path.join(base_data_dir, 'downstream')

PATIENT_VOCAB = {
    "gender":         2,
    "insurance":      5,
    "marital_status": 5,
    "race":          30,
    "language":      25,
}

ADMISSION_VOCAB = {
    "admission_type":     9,
    "admission_location": 11,
    "drg_type":           2,
    "drg_severity":       4,   # ordinal 1-4, embedded not scalar
    "drg_mortality":      4,   # ordinal 1-4, embedded not scalar
}
 
EMBED_DIM  = 8    # per categorical field
N_SCALARS  = 1    # age only
OUTPUT_DIM = 64

class PatientStaticEncoder(nn.Module):
    """
    Encodes static patient demographics into a 64-dim vector.
 
    Args:
        vocab_sizes (dict): field → n_unique. Defaults to PATIENT_VOCAB.
        dropout (float):    applied in projection MLP.
 
    Forward inputs:
        cat_inputs    (dict[str, LongTensor]): {field: (B,)} index tensors
        scalar_inputs (FloatTensor):           (B, 1)  age, pre-normalised
 
    Forward output:
        (B, 64)
    """
 
    def __init__(self, vocab_sizes: dict = None, dropout: float = 0.1):
        super().__init__()
 
        if vocab_sizes is None:
            vocab_sizes = PATIENT_VOCAB
 
        self.fields = list(vocab_sizes.keys())
 
        # +1 on each vocab for explicit UNK at index 0
        self.embeddings = nn.ModuleDict({
            field: nn.Embedding(vocab_sizes[field] + 1, EMBED_DIM, padding_idx=0)
            for field in self.fields
        })
 
        cat_dim   = len(self.fields) * EMBED_DIM   # 5 × 8 = 40
        total_dim = cat_dim + N_SCALARS             # 40 + 1 = 41
 
        self.proj = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, OUTPUT_DIM),
        )
 
    def forward(
        self,
        cat_inputs: dict,             # {field: LongTensor (B,)}
        scalar_inputs: torch.Tensor,  # (B, 1)  — age normalised
    ) -> torch.Tensor:
        """Returns (B, 64)."""
 
        embedded = [self.embeddings[f](cat_inputs[f]) for f in self.fields]
        cat_vec  = torch.cat(embedded, dim=-1)                  # (B, 40)
        x        = torch.cat([cat_vec, scalar_inputs], dim=-1)  # (B, 41)
 
        return self.proj(x)    # (B, 64)

class AdmissionStaticEncoder(nn.Module):
    """
    Encodes static admission features into a 64-dim vector.
 
    Args:
        vocab_sizes (dict): field → n_unique. Defaults to ADMISSION_VOCAB.
        dropout (float):    applied in projection MLP.
 
    Forward inputs:
        cat_inputs (dict[str, LongTensor]): {field: (B,)} index tensors
                                            drg_severity/mortality passed as
                                            raw int values (1-4), shifted to
                                            0-based index inside forward.
 
    Forward output:
        (B, 64)
    """
 
    def __init__(self, vocab_sizes: dict = None, dropout: float = 0.1):
        super().__init__()
 
        if vocab_sizes is None:
            vocab_sizes = ADMISSION_VOCAB
 
        self.fields = list(vocab_sizes.keys())
 
        # Ordinal fields (drg_severity, drg_mortality) are 1-indexed in data
        # We handle the shift in encode_admission_row, so +1 here for UNK=0
        self.embeddings = nn.ModuleDict({
            field: nn.Embedding(vocab_sizes[field] + 1, EMBED_DIM, padding_idx=0)
            for field in self.fields
        })
 
        total_dim = len(self.fields) * EMBED_DIM   # 5 × 8 = 40
 
        self.proj = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, OUTPUT_DIM),
        )
 
    def forward(self, cat_inputs: dict) -> torch.Tensor:
        """
        Args:
            cat_inputs: {field: LongTensor (B,)}
                        all indices already 1-based (0 = UNK)
        Returns:
            (B, 64)
        """
        embedded = [self.embeddings[f](cat_inputs[f]) for f in self.fields]
        x = torch.cat(embedded, dim=-1)   # (B, 40)
        return self.proj(x)               # (B, 64)

## Helper Functions

def build_category_maps(df, fields=None):
    """
    Build index maps from a DataFrame of patient records.
 
    Returns:
        maps  : {field: {value_str → int (1-based, 0=UNK)}}
        sizes : {field: n_unique}
 
    Usage:
        maps, sizes = build_category_maps(patients_df)
        encoder = PatientStaticEncoder(vocab_sizes=sizes)
    """
    if fields is None:
        fields = list(PATIENT_VOCAB.keys())
 
    maps, sizes = {}, {}
    for field in fields:
        unique_vals  = sorted(df[field].dropna().astype(str).unique())
        maps[field]  = {v: i + 1 for i, v in enumerate(unique_vals)}
        sizes[field] = len(unique_vals)
 
    return maps, sizes

def build_admission_category_maps(df, fields=None):
    """
    Build index maps from a DataFrame of admission records.
    drg_severity and drg_mortality are integer 1-4 — mapped as strings
    so UNK handling is consistent.
 
    Returns:
        maps  : {field: {value_str → int (1-based, 0=UNK)}}
        sizes : {field: n_unique}
    """
    if fields is None:
        fields = ["admission_type", "admission_location", "drg_type",
                  "drg_severity", "drg_mortality"]
 
    maps, sizes = {}, {}
    for field in fields:
        unique_vals  = sorted(df[field].dropna().astype(str).unique())
        maps[field]  = {v: i + 1 for i, v in enumerate(unique_vals)}
        sizes[field] = len(unique_vals)
 
    return maps, sizes

def encode_patient_row(row, cat_maps, age_mean, age_std):
    """
    Convert one patient dict into model-ready tensors.
 
    Args:
        row      : dict with patient fields
        cat_maps : output of build_category_maps
        age_mean : float — fit on train patients only
        age_std  : float — fit on train patients only
 
    Returns:
        cat_tensors   : {field: LongTensor (1,)}
        scalar_tensor : FloatTensor (1, 1)
    """
    cat_tensors = {}
    for field, mapping in cat_maps.items():
        val = str(row.get(field, ""))
        cat_tensors[field] = torch.tensor([mapping.get(val, 0)], dtype=torch.long)
 
    age      = float(row.get("age", 0.0) or 0.0)
    age_norm = (age - age_mean) / max(age_std, 1e-6)
    scalar_tensor = torch.tensor([[age_norm]], dtype=torch.float32)
 
    return cat_tensors, scalar_tensor

def encode_admission_row(row, cat_maps):
    """
    Convert one admission dict into model-ready tensors.
 
    Args:
        row      : dict with admission fields (from admission_nodes.json
                   or a Neo4j query result)
        cat_maps : output of build_admission_category_maps
 
    Returns:
        cat_tensors: {field: LongTensor (1,)}
    """
    cat_tensors = {}
    for field, mapping in cat_maps.items():
        val = str(row.get(field, ""))
        cat_tensors[field] = torch.tensor([mapping.get(val, 0)], dtype=torch.long)
 
    return cat_tensors

def precompute_all_patient_vectors(patients_df, encoder, cat_maps, age_mean, age_std, device):
    """
    Precompute and cache 64-dim vectors for all patients.
    Call once before training — reuse the dict every forward pass.
 
    Returns:
        dict {patient_id: Tensor (64,)}
    """
    encoder.eval()
    cache = {}
    with torch.no_grad():
        for _, row in patients_df.iterrows():
            pid = row["id"]
            cat_t, scalar_t = encode_patient_row(row, cat_maps, age_mean, age_std)
            cat_t    = {f: v.to(device) for f, v in cat_t.items()}
            scalar_t = scalar_t.to(device)
            cache[pid] = encoder(cat_t, scalar_t).squeeze(0)   # (64,)
    return cache

def precompute_all_admission_vectors(admissions_df, encoder, cat_maps, device):
    """
    Precompute and cache 64-dim vectors for all admissions.
    Call once before training — look up by adm_id at DISCHARGE slice time.
 
    Returns:
        dict {adm_id (str): Tensor (64,)}
    """
    encoder.eval()
    cache = {}
    with torch.no_grad():
        for _, row in admissions_df.iterrows():
            adm_id = str(row["id"])
            cat_t  = encode_admission_row(row, cat_maps)
            cat_t  = {f: v.to(device) for f, v in cat_t.items()}
            cache[adm_id] = encoder(cat_t).squeeze(0)   # (64,)
    return cache

# if __name__ == "__main__":
 
