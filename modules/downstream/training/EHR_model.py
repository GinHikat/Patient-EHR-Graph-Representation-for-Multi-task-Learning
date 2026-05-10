import pandas as pd
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import sys, os
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

if current_dir not in sys.path:
    sys.path.append(current_dir)

# from shared_functions.global_functions import *

base_data_dir = os.path.join(project_root, 'data')
base_data_path = os.path.join(base_data_dir, 'Timeline')

N_DIAGNOSES = 200   # top-200 diagnosis 
N_DRUGS     = 50    # top-50 drugs

pos_weight_mortality   = torch.tensor([63.93])   
pos_weight_los         = torch.tensor([4.39])    
pos_weight_readmission = torch.tensor([2.88])    
pos_weight_progression = torch.from_numpy(np.load(os.path.join(base_data_path, 'progression_pos_weights.npy'))).float()  # 200-dim

class EHRDataset(Dataset):
    """
    Args:
        admissions_df   : DataFrame with columns:
                          id, patient_id, inhospital_dead, los_log, los_7d,
                          readmission_30d
        timeline_dir    : path to Timelines/ folder
        admission_nodes : dict {adm_id (str): {'diagnoses': [...], 'drugs': [...]}}
        diag_to_idx     : dict {diagnosis_name (str, lower): int 0-199}
        drug_to_idx     : dict {drug_name (str, lower): int 0-49}
        patient_cache   : dict {patient_id: Tensor (64,)}  — precomputed
        admission_cache : dict {adm_id (str): Tensor (64,)} — precomputed
    """

    def __init__(
        self,
        admissions_df,
        timeline_dir,
        admission_nodes,
        diag_to_idx,
        drug_to_idx,                  
        patient_cache,
        admission_cache,
        max_len=None,
    ):
        self.timeline_dir    = Path(timeline_dir)
        self.admission_nodes = admission_nodes
        self.diag_to_idx     = diag_to_idx
        self.drug_to_idx     = drug_to_idx   
        self.patient_cache   = patient_cache
        self.admission_cache = admission_cache
        self.max_len         = max_len

        # One row per admission — drop rows with missing critical labels
        df = admissions_df.copy()
        df = df[df['inhospital_dead'].notna()]
        df = df[df['los_log'].notna()]
        df['id']         = df['id'].astype(float).apply(lambda x: str(int(x)) if not pd.isna(x) else x)
        df['patient_id'] = df['patient_id'].astype(float).apply(lambda x: str(int(x)) if not pd.isna(x) else x)
        self.df = df.reset_index(drop=True)

        self._meta_cache = {}   # Cache meta per patient
        self._cache_keys = []   # Track keys for FIFO eviction
        self.MAX_CACHE   = 5000 # Cap cache at 5,000 patients

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        adm_id = str(row['id'])
        pid    = str(row['patient_id'])

        if pid not in self._meta_cache:
            meta_path = self.timeline_dir / f'{pid}_meta.json'
            if not meta_path.exists():
                return None
            with open(meta_path) as f:
                self._meta_cache[pid] = json.load(f)
                self._cache_keys.append(pid)

            # Evict oldest if cache too large
            if len(self._cache_keys) > self.MAX_CACHE:
                oldest = self._cache_keys.pop(0)
                self._meta_cache.pop(oldest, None)

        meta = self._meta_cache[pid]

        discharge_pos = None
        for i, entry in enumerate(meta):
            if entry['type'] == 'DISCHARGE' and str(entry.get('adm_id')) == adm_id:
                discharge_pos = i
                break

        if discharge_pos is None:
            return None

        # Load and slice timeline causally up to DISCHARGE 
        emb_path = self.timeline_dir / f'{pid}_emb.npy'
        dt_path  = self.timeline_dir / f'{pid}_dt.npy'

        if not emb_path.exists() or not dt_path.exists():
            return None

        emb = np.load(emb_path, mmap_mode='r')   # (T_full, 128)
        dt  = np.load(dt_path, mmap_mode='r')    # (T_full,)

        if np.isnan(emb).any():
            return None

        # Slice and copy to ensure we aren't holding mmap handles in memory
        emb = emb[:discharge_pos + 1]
        dt  = dt[:discharge_pos + 1]

        # Causal capping: keep the MOST RECENT max_len notes before discharge
        if self.max_len is not None and len(emb) > self.max_len:
            emb = emb[-self.max_len:]
            dt  = dt[-self.max_len:]

        emb = emb.copy()
        dt  = dt.copy()

        # Static vectors from precomputed cache 
        patient_vec = self.patient_cache.get(pid)
        if patient_vec is None and pid.isdigit():
            patient_vec = self.patient_cache.get(int(pid))

        admission_vec = self.admission_cache.get(adm_id)
        if admission_vec is None and adm_id.isdigit():
            admission_vec = self.admission_cache.get(int(adm_id))

        if patient_vec is None or admission_vec is None:
            return None

        if torch.isnan(patient_vec).any() or torch.isnan(admission_vec).any():
            return None

        # Build progression multilabel vector (200,)
        progression = np.zeros(N_DIAGNOSES, dtype=np.float32)
        adm_data    = self.admission_nodes.get(adm_id, {})
        for diag in adm_data.get('diagnoses', []):
            i = self.diag_to_idx.get(diag.lower())
            if i is not None:
                progression[i] = 1.0

        # Build drug_rec multilabel vector (50,)
        drug_rec = np.zeros(N_DRUGS, dtype=np.float32)
        for drug in adm_data.get('drugs', []):
            i = self.drug_to_idx.get(drug.lower())
            if i is not None:
                drug_rec[i] = 1.0

        # Labels
        mortality   = float(row['inhospital_dead'])
        los_log     = float(row['los_log'])
        los_7d      = float(row['los_7d'])
        readmission = float(row['readmission_30d']) if not np.isnan(row['readmission_30d']) else -1.0

        return {
            'emb':           torch.tensor(emb,          dtype=torch.float32),  # (T, 128)
            'dt':            torch.tensor(dt,            dtype=torch.float32),  # (T,)
            'patient_vec':   patient_vec,                                        # (64,)
            'admission_vec': admission_vec,                                      # (64,)
            'mortality':     torch.tensor(mortality,     dtype=torch.float32),
            'los_log':       torch.tensor(los_log,       dtype=torch.float32),
            'los_7d':        torch.tensor(los_7d,        dtype=torch.float32),
            'readmission':   torch.tensor(readmission,   dtype=torch.float32),
            'progression':   torch.tensor(progression,   dtype=torch.float32),  # (200,)
            'drug_rec':      torch.tensor(drug_rec,      dtype=torch.float32),  # (50,)
            'adm_id':        adm_id,
            'pid':           pid,
        }

## Helper functions

def ehr_collate_fn(batch):
    """
    Pads variable-length timelines to max T in batch.
    Filters out None samples.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    max_T = max(b['emb'].shape[0] for b in batch)

    emb_padded, dt_padded, lengths = [], [], []
    for b in batch:
        T   = b['emb'].shape[0]
        pad = max_T - T
        emb_padded.append(torch.nn.functional.pad(b['emb'], (0, 0, 0, pad)))
        dt_padded.append(torch.nn.functional.pad(b['dt'],   (0, pad)))
        lengths.append(T)

    return {
        'emb':           torch.stack(emb_padded),                               # (B, max_T, 128)
        'dt':            torch.stack(dt_padded),                                # (B, max_T)
        'lengths':       torch.tensor(lengths, dtype=torch.long),               # (B,)
        'patient_vec':   torch.stack([b['patient_vec']   for b in batch]),      # (B, 64)
        'admission_vec': torch.stack([b['admission_vec'] for b in batch]),      # (B, 64)
        'mortality':     torch.stack([b['mortality']     for b in batch]),      # (B,)
        'los_log':       torch.stack([b['los_log']       for b in batch]),      # (B,)
        'los_7d':        torch.stack([b['los_7d']        for b in batch]),      # (B,)
        'readmission':   torch.stack([b['readmission']   for b in batch]),      # (B,)
        'progression':   torch.stack([b['progression']   for b in batch]),      # (B, 200)
        'drug_rec':      torch.stack([b['drug_rec']      for b in batch]),      # (B, 50)
        'adm_ids':       [b['adm_id'] for b in batch],
        'pids':          [b['pid']    for b in batch],
    }
# Models
EMBED_DIM   = 128
HIDDEN_SIZE = 256
STATIC_DIM  = 64    # patient_vec and admission_vec each
PROJ_DIM    = 128
N_DIAGNOSES = 200

class TimeEncoding(nn.Module):
    """
    Learns a continuous embedding for time deltas (dt).
    Uses a combination of linear and periodic components.
    """
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, d_model - 1)
        
    def forward(self, dt):
        # dt shape: (B, T)
        dt = dt.unsqueeze(-1) # (B, T, 1)
        
        v1 = self.linear(dt) # Linear component
        v2 = torch.sin(self.periodic(dt)) # Periodic components
        
        return torch.cat([v1, v2], dim=-1) # (B, T, d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EHRTransformer(nn.Module):
    """
    Upgraded Transformer-based sequence modeling for EHR data.
    Injects static vectors as tokens and encodes relative time deltas (dt).
    """
    def __init__(
        self,
        n_diagnoses: int = N_DIAGNOSES,
        n_drugs: int     = N_DRUGS,
        dropout: float   = 0.1,
        lambda_init: float = 0.1,
    ):
        super().__init__()

        self.log_lambda = nn.Parameter(torch.tensor(lambda_init).log())

        # Projections
        self.input_proj = nn.Linear(EMBED_DIM, HIDDEN_SIZE)
        self.static_proj = nn.Linear(STATIC_DIM, HIDDEN_SIZE) 
        
        self.time_encoder = TimeEncoding(HIDDEN_SIZE)
        self.pos_encoder = PositionalEncoding(HIDDEN_SIZE)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_SIZE,
            nhead=8,
            dim_feedforward=HIDDEN_SIZE * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.use_gradient_checkpointing = False

        self.proj = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, PROJ_DIM),
            nn.LayerNorm(PROJ_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head_mortality   = nn.Linear(PROJ_DIM, 1)
        self.head_los         = nn.Linear(PROJ_DIM, 1)
        self.head_readmission = nn.Linear(PROJ_DIM, 1)
        self.head_progression = nn.Linear(PROJ_DIM, n_diagnoses)
        self.head_drug_rec    = nn.Linear(PROJ_DIM, n_drugs)

    def forward(self, batch):
        emb           = batch['emb']            # (B, T, 128)
        dt            = batch['dt']             # (B, T)
        lengths       = batch['lengths']        # (B,)
        patient_vec   = batch['patient_vec']    # (B, 64)
        admission_vec = batch['admission_vec']  # (B, 64)

        # 1. Temporal Decay (Learnable exponential decay)
        lam = torch.nn.functional.softplus(self.log_lambda)
        decay = torch.exp(-lam * dt).unsqueeze(-1)
        emb   = emb * decay

        # 2. Project clinical embeddings + add Time Embeddings
        x = self.input_proj(emb)                # (B, T, 256)
        x = x + self.time_encoder(dt)           # Inject continuous time information
        
        # 3. Project Static Contexts
        p_tok = self.static_proj(patient_vec).unsqueeze(1)    # (B, 1, 256)
        a_tok = self.static_proj(admission_vec).unsqueeze(1)  # (B, 1, 256)

        # 4. Positional Encoding (Absolute sequence position)
        x = self.pos_encoder(x)

        # 5. Prepend Static Context Tokens
        x_full = torch.cat([p_tok, a_tok, x], dim=1) # (B, T+2, 256)
        
        # 5. Masking
        B, T_full, _ = x_full.shape
        # Adjust mask to account for 2 prepended tokens (which are never masked)
        mask = torch.zeros((B, T_full), dtype=torch.bool, device=x.device)
        for i, length in enumerate(lengths):
            mask[i, length+2:] = True
        
        # 6. Transformer Pass
        trans_out = self.transformer(x_full, src_key_padding_mask=mask)

        # 7. Hybrid Global Representation
        # Use the Patient_Token (index 0) as it has now attended to everything
        # Plus the last clinical token (discharge)
        idx_discharge = (lengths + 1).clamp(min=1) # +1 because of 2 prepended tokens
        idx_expanded = idx_discharge.view(-1, 1, 1).expand(-1, 1, HIDDEN_SIZE)
        h_discharge = trans_out.gather(1, idx_expanded).squeeze(1)
        
        h_global = trans_out[:, 0] # The evolved Patient Token
        
        # Combine global context and final state
        shared_repr = (h_global + h_discharge) / 2.0
        
        shared = self.proj(shared_repr)

        return {
            'mortality':   self.head_mortality(shared),
            'los_7d':      self.head_los(shared),
            'readmission': self.head_readmission(shared),
            'progression': self.head_progression(shared),
            'drug_rec':    self.head_drug_rec(shared),
        }

class EHRModel(nn.Module):
    """
    Args:
        n_diagnoses (int) : size of progression label space (default 200)
        n_drugs (int)     : size of drug recommendation label space (default 50)
        dropout (float)   : dropout rate in projection and heads
        lambda_init (float): initial value for Δt decay parameter λ
                             learned during training
    """

    def __init__(
        self,
        n_diagnoses: int   = N_DIAGNOSES,
        n_drugs: int       = N_DRUGS,       # ← NEW
        dropout: float     = 0.1,
        lambda_init: float = 0.1,
    ):
        super().__init__()

        # Δt decay parameter — learned scalar
        # emb_t = emb_t * exp(-λ * Δt)
        # λ > 0 enforced via softplus in forward
        self.log_lambda = nn.Parameter(torch.tensor(lambda_init).log())

        # lstm
        self.lstm = nn.LSTM(
            input_size  = EMBED_DIM,    # 128
            hidden_size = HIDDEN_SIZE,  # 256
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.5,          # dropout handled outside for single layer
        )

        self.use_gradient_checkpointing = False

        # Projection: concat → shared repr
        concat_dim = HIDDEN_SIZE + STATIC_DIM + STATIC_DIM  # 256 + 64 (patient) + 64 (admission) = 384
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, PROJ_DIM),
            nn.LayerNorm(PROJ_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Prediction heads — all output raw logits, sigmoid/BCE applied in loss
        self.head_mortality   = nn.Linear(PROJ_DIM, 1)
        self.head_los         = nn.Linear(PROJ_DIM, 1)
        self.head_readmission = nn.Linear(PROJ_DIM, 1)
        self.head_progression = nn.Linear(PROJ_DIM, n_diagnoses)
        self.head_drug_rec    = nn.Linear(PROJ_DIM, n_drugs)   # ← NEW

    def forward(self, batch):
        """
        Args:
            batch (dict) from ehr_collate_fn:
                emb           : (B, T, 128)  padded timeline embeddings
                dt            : (B, T)       Δt values in days
                lengths       : (B,)         true sequence lengths
                patient_vec   : (B, 64)
                admission_vec : (B, 64)

        Returns:
            dict of logits:
                mortality   : (B, 1)
                los_7d      : (B, 1)
                readmission : (B, 1)
                progression : (B, 200)
                drug_rec    : (B, 50)
        """
        emb           = batch['emb']            # (B, T, 128)
        dt            = batch['dt']             # (B, T)
        lengths       = batch['lengths']        # (B,)
        patient_vec   = batch['patient_vec']    # (B, 64)
        admission_vec = batch['admission_vec']  # (B, 64)

        # Apply exponential Δt decay
        # λ = softplus(log_lambda) ensures λ > 0
        lam = torch.nn.functional.softplus(self.log_lambda)

        # decay shape: (B, T, 1) → broadcast over 128 dims
        decay = torch.exp(-lam * dt).unsqueeze(-1)   # (B, T, 1)
        emb   = emb * decay                           # (B, T, 128)

        # lstm with packed sequence (ignores padding)
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # lstm pass
        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            def lstm_forward(x):
                out, _ = self.lstm(x)
                return out
            lstm_out_packed = checkpoint(lstm_forward, packed, use_reentrant=False)
        else:
            lstm_out_packed, _ = self.lstm(packed)

        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out_packed, batch_first=True
        )                                            # (B, T, 256)

        # Slice hidden state at last real token (DISCHARGE position)
        # lengths[i] - 1 is the index of the DISCHARGE token since we slice
        # the timeline up to and including DISCHARGE in EHRDataset
        idx          = (lengths - 1).clamp(min=0)             # (B,)
        idx_expanded = idx.view(-1, 1, 1).expand(-1, 1, HIDDEN_SIZE)
        h_discharge  = lstm_out.gather(1, idx_expanded).squeeze(1)  # (B, 256)

        # Concat static vectors + project
        combined = torch.cat([h_discharge, patient_vec, admission_vec], dim=-1)
        # (B, 384)
        shared = self.proj(combined)   # (B, 128)

        # DEBUG: Check for NaNs
        if torch.isnan(shared).any():
            print(f"\n[DEBUG] NaN detected in model output!")
            print(f"  - emb max/min: {emb.max().item():.2f}/{emb.min().item():.2f}")
            print(f"  - dt max/min: {dt.max().item():.2f}/{dt.min().item():.2f}")
            print(f"  - lam: {lam.item():.4f}")
            print(f"  - h_discharge max/min: {h_discharge.max().item():.2f}/{h_discharge.min().item():.2f}")
            print(f"  - patient_vec max/min: {patient_vec.max().item():.2f}/{patient_vec.min().item():.2f}")
            print(f"  - admission_vec max/min: {admission_vec.max().item():.2f}/{admission_vec.min().item():.2f}")

        # 5 prediction heads
        return {
            'mortality':   self.head_mortality(shared),    # (B, 1)
            'los_7d':      self.head_los(shared),          # (B, 1)
            'readmission': self.head_readmission(shared),  # (B, 1)
            'progression': self.head_progression(shared),  # (B, 200)
            'drug_rec':    self.head_drug_rec(shared),     # (B, 50)   ← NEW
        }

# Loss function
class EHRLoss(nn.Module):
    """
    Multi-task BCE loss with per-task pos_weight and label masking.

    Masking rules:
        readmission : label == -1.0 → excluded (last admission / patient died)
        progression : all-zero vector → excluded (no top-200 diagnoses)
        drug_rec    : all-zero vector → excluded (no top-50 drugs)

    Task weights allow tuning relative importance during training.
    Default: all tasks weighted equally.
    """

    def __init__(
        self,
        pos_weight_mortality:   torch.Tensor,   # (1,)
        pos_weight_los:         torch.Tensor,   # (1,)
        pos_weight_readmission: torch.Tensor,   # (1,)
        pos_weight_progression: torch.Tensor,   # (200,)
        pos_weight_drug_rec:    torch.Tensor,   # (50,)   ← NEW
        w_mortality:   float = 1.0,
        w_los:         float = 1.0,
        w_readmission: float = 1.0,
        w_progression: float = 1.0,
        w_drug_rec:    float = 1.0,             # ← NEW
    ):
        super().__init__()

        self.w_mortality   = w_mortality
        self.w_los         = w_los
        self.w_readmission = w_readmission
        self.w_progression = w_progression
        self.w_drug_rec    = w_drug_rec         # ← NEW

        # Scalar tasks — standard weighted BCE
        self.crit_mortality   = nn.BCEWithLogitsLoss(pos_weight=pos_weight_mortality)
        self.crit_los         = nn.BCEWithLogitsLoss(pos_weight=pos_weight_los)
        self.crit_readmission = nn.BCEWithLogitsLoss(pos_weight=pos_weight_readmission)

        # Multilabel tasks — reduction='none' so we can mask per-sample
        # pos_weight broadcasts over the label dimension automatically
        self.crit_progression = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_progression, reduction='none'
        )
        self.crit_drug_rec = nn.BCEWithLogitsLoss(   # ← NEW
            pos_weight=pos_weight_drug_rec, reduction='none'
        )

    def _masked_multilabel_loss(self, criterion, logits, labels):
        """
        Compute mean multilabel BCE loss over samples that have at least
        one positive label. Samples with all-zero labels are excluded.

        Args:
            criterion : BCEWithLogitsLoss(reduction='none')
            logits    : (B, C)
            labels    : (B, C)

        Returns:
            scalar loss or 0.0 if no valid samples
        """
        mask = (labels.sum(dim=-1) > 0)   # (B,) — True if sample has ≥1 positive
        if mask.any():
            loss_unreduced = criterion(logits[mask], labels[mask])  # (n_valid, C)
            return loss_unreduced.mean()
        return torch.tensor(0.0, device=logits.device)

    def forward(self, logits: dict, batch: dict):
        """
        Returns:
            total_loss : scalar
            loss_dict  : {task: scalar} for logging
        """
        labels_mort  = batch['mortality'].unsqueeze(1)    # (B, 1)
        labels_los   = batch['los_7d'].unsqueeze(1)       # (B, 1)
        labels_readm = batch['readmission'].unsqueeze(1)  # (B, 1)
        labels_prog  = batch['progression']               # (B, 200)
        labels_drug  = batch['drug_rec']                  # (B, 50)   ← NEW

        # Mortality
        loss_mort = self.crit_mortality(logits['mortality'], labels_mort)

        # LOS
        loss_los = self.crit_los(logits['los_7d'], labels_los)

        # Readmission — mask missing labels (-1.0)
        readm_mask = (labels_readm >= 0)   # (B, 1)
        if readm_mask.any():
            loss_readm = self.crit_readmission(
                logits['readmission'][readm_mask],
                labels_readm[readm_mask]
            )
        else:
            loss_readm = torch.tensor(0.0, device=logits['readmission'].device)

        # Progression — mask admissions with no top-200 diagnoses
        loss_prog = self._masked_multilabel_loss(
            self.crit_progression, logits['progression'], labels_prog
        )

        # Drug rec — mask admissions with no top-50 drugs
        loss_drug = self._masked_multilabel_loss(
            self.crit_drug_rec, logits['drug_rec'], labels_drug
        )

        # Weighted total
        total = (
            self.w_mortality   * loss_mort  +
            self.w_los         * loss_los   +
            self.w_readmission * loss_readm +
            self.w_progression * loss_prog  +
            self.w_drug_rec    * loss_drug    # ← NEW
        )

        loss_dict = {
            'total':       total.item(),
            'mortality':   loss_mort.item(),
            'los_7d':      loss_los.item(),
            'readmission': loss_readm.item(),
            'progression': loss_prog.item(),
            'drug_rec':    loss_drug.item(),  # ← NEW
        }

        return total, loss_dict


# if __name__ == '__main__':
#     B, T = 4, 52   # batch of 4, max timeline length 52

#     # Fake batch
#     batch = {
#         'emb':           torch.randn(B, T, 128),
#         'dt':            torch.rand(B, T) * 30,       # 0-30 days
#         'lengths':       torch.tensor([52, 40, 30, 20]),
#         'patient_vec':   torch.randn(B, 64),
#         'admission_vec': torch.randn(B, 64),
#         'mortality':     torch.tensor([0., 1., 0., 0.]),
#         'los_7d':        torch.tensor([1., 0., 0., 1.]),
#         'readmission':   torch.tensor([1., -1., 0., 1.]),   # -1 = missing
#         'progression':   torch.zeros(B, 200),
#     }
#     batch['progression'][0, [3, 17, 42]] = 1.0   # some positives

#     model = EHRModel()
#     print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

#     logits = model(batch)
#     print("Output shapes:")
#     for k, v in logits.items():
#         print(f"  {k}: {tuple(v.shape)}")

#     # Test loss
#     criterion = EHRLoss(
#         pos_weight_mortality   = torch.tensor([63.93]),
#         pos_weight_los         = torch.tensor([4.39]),
#         pos_weight_readmission = torch.tensor([2.88]),
#         pos_weight_progression = torch.ones(200),
#     )
#     total_loss, loss_dict = criterion(logits, batch)
#     print(f"\nLoss breakdown:")
#     for k, v in loss_dict.items():
#         print(f"  {k}: {v:.4f}")

#     assert not torch.isnan(total_loss), "NaN in loss!"
#     print("\nEHRModel OK ✓")
