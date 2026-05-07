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
            scores = scores.masked_fill(~mask, float('-inf'))

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

class EventEncoder(nn.Module):
    def __init__(self, vocab_size=170, omr_dim=4, n_special=4, embed_dim=128):
        super().__init__()
        self.lab_encoder     = LabPanelEncoder(vocab_size=vocab_size, output_dim=embed_dim)
        self.omr_encoder     = OMREncoder(input_dim=omr_dim, output_dim=embed_dim)
        self.special_encoder = SpecialTokenEncoder(n_tokens=n_special, output_dim=embed_dim)
        self.embed_dim       = embed_dim

    def encode_lab(self, values, masks):
        return self.lab_encoder(values, masks)        # (B, 128)

    def encode_omr(self, values, masks):
        return self.omr_encoder(values, masks)        # (B, 128)

    def encode_special(self, token_ids):
        return self.special_encoder(token_ids)        # (B, 128)

    def encode_kg_node(self, node_embeddings):
        return node_embeddings                        # (B, 128)