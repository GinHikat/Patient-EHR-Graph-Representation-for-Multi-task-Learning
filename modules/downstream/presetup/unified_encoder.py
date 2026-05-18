import numpy as np
import torch
import torch.nn as nn

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

class LabPanelEncoder(nn.Module):
    def __init__(self, vocab_size=165, hidden_dim=256, output_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size * 2, hidden_dim),  # [val(165) + mask(165)] = 330-dim
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),    # 330->128
        )

    def forward(self, values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        x = torch.cat([values, masks], dim=-1)        
        return self.encoder(x)                        

class OMREncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 64),   # [values(4) || masks(4)] = 8-dim
            nn.ReLU(),
            nn.Linear(64, output_dim)        # 64 → 128
        )

    def forward(self, values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        x = torch.cat([values, masks], dim=-1)  
        return self.encoder(x)                    # (B, 128)                      

class AdmissionEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        # Attention scoring — how important is each node?
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Project concatenated drug+diagnosis pooled vector
        self.project = nn.Linear(embed_dim, embed_dim)
        self.norm    = nn.LayerNorm(embed_dim)

    def forward(self, node_embeddings: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            node_embeddings : (B, N, 128) — B admissions, N nodes each (Diagnosis + Drug combined)
            mask            : (B, N) bool — True where node exists, False where padding
        Returns:
            admission_emb   : (B, 128)
        """
        # Compute attention scores
        scores = self.attention(node_embeddings)     # (B, N, 1)
        scores = scores.squeeze(-1)                  # (B, N)

        # Mask out padding positions before softmax
        if mask is not None:
            # Prevent all-False rows from producing NaN in softmax
            all_false_rows = ~mask.any(dim=-1, keepdim=True)
            safe_mask = mask.clone()
            safe_mask[:, 0] = safe_mask[:, 0] | all_false_rows.squeeze(-1)
            scores = scores.masked_fill(~safe_mask, float('-inf'))

        weights = torch.softmax(scores, dim=-1)      # (B, N)
        weights = weights.unsqueeze(-1)              # (B, N, 1)

        # Weighted sum
        pooled = (weights * node_embeddings).sum(dim=1)  # (B, 128)

        out = self.norm(self.project(pooled))        # (B, 128)
        return out

class OutpatientEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        # Attention scoring — how important is each diagnosis GAT node?
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Project pooled vector to output 128-dim
        self.project = nn.Linear(embed_dim, embed_dim)
        self.norm    = nn.LayerNorm(embed_dim)

    def forward(self, node_embeddings: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            node_embeddings : (B, N, 128) — B outpatient note events, N flat ICD codes each
            mask            : (B, N) bool — True where node exists, False where padding
        Returns:
            outpatient_emb  : (B, 128)
        """
        # Compute attention scores
        scores = self.attention(node_embeddings)     # (B, N, 1)
        scores = scores.squeeze(-1)                  # (B, N)

        # Mask out padding positions before softmax
        if mask is not None:
            # Prevent all-False rows from producing NaN in softmax
            all_false_rows = ~mask.any(dim=-1, keepdim=True)
            safe_mask = mask.clone()
            safe_mask[:, 0] = safe_mask[:, 0] | all_false_rows.squeeze(-1)
            scores = scores.masked_fill(~safe_mask, float('-inf'))

        weights = torch.softmax(scores, dim=-1)      # (B, N)
        weights = weights.unsqueeze(-1)              # (B, N, 1)

        # Weighted sum
        pooled = (weights * node_embeddings).sum(dim=1)  # (B, 128)

        out = self.norm(self.project(pooled))        # (B, 128)
        return out

SPECIAL_TOKENS = {
    'ADMIT':     0,
    'DISCHARGE': 1,
    'CLS':       2,
    'PAD':       3,
}

class SpecialTokenEncoder(nn.Module):
    def __init__(self, n_tokens=4, output_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, output_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)              

class ICUEncoder(nn.Module):
    def __init__(self, num_units=15, output_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_units + 1, output_dim, padding_idx=0)

    def forward(self, unit_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            unit_ids: LongTensor (B,) of ICU unit vocabulary indices
        Returns:
            (B, output_dim) ICU event embeddings
        """
        return self.embedding(unit_ids)

class TransferEncoder(nn.Module):
    def __init__(self, num_care_units=30, num_types=10, output_dim=128):
        super().__init__()
        self.care_unit_emb = nn.Embedding(num_care_units + 1, 64, padding_idx=0)
        self.type_emb      = nn.Embedding(num_types + 1, 64, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

    def forward(self, care_unit_ids: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            care_unit_ids : LongTensor (B,) of care unit vocabulary indices
            type_ids      : LongTensor (B,) of transfer type vocabulary indices
        Returns:
            (B, output_dim) Transfer event embeddings
        """
        c_emb = self.care_unit_emb(care_unit_ids)  # (B, 64)
        t_emb = self.type_emb(type_ids)            # (B, 64)
        x = torch.cat([c_emb, t_emb], dim=-1)      # (B, 128)
        return self.proj(x)

class EventEncoder(nn.Module):
    def __init__(self, vocab_size=165, omr_dim=4, n_special=4, embed_dim=128, num_icu_units=15, num_care_units=30, num_transfer_types=10):
        super().__init__()
        self.lab_encoder        = LabPanelEncoder(vocab_size=vocab_size, output_dim=embed_dim)
        self.omr_encoder        = OMREncoder(input_dim=omr_dim, output_dim=embed_dim)
        self.special_encoder    = SpecialTokenEncoder(n_tokens=n_special, output_dim=embed_dim)
        self.icu_encoder        = ICUEncoder(num_units=num_icu_units, output_dim=embed_dim)
        self.transfer_encoder   = TransferEncoder(num_care_units=num_care_units, num_types=num_transfer_types, output_dim=embed_dim)
        self.outpatient_encoder = OutpatientEncoder(embed_dim=embed_dim)
        self.embed_dim          = embed_dim

    def encode_lab(self, values, masks):
        return self.lab_encoder(values, masks)        # (B, 128)

    def encode_omr(self, values, masks):
        return self.omr_encoder(values, masks)        # (B, 128)

    def encode_special(self, token_ids):
        return self.special_encoder(token_ids)        # (B, 128)

    def encode_kg_node(self, node_embeddings):
        return node_embeddings                        # (B, 128)

    def encode_icu(self, unit_ids):
        return self.icu_encoder(unit_ids)             # (B, 128)

    def encode_transfer(self, care_unit_ids, type_ids):
        return self.transfer_encoder(care_unit_ids, type_ids)  # (B, 128)

    def encode_outpatient(self, node_embeddings, mask):
        return self.outpatient_encoder(node_embeddings, mask)  # (B, 128)