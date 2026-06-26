import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional

class LAATLayer(nn.Module):
    """
    Label Attention Layer (LAAT) as described in the PLM-ICD paper.
    Computes label-specific document representations using an attention mechanism.
    """
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        # Label query matrix
        self.U = nn.Parameter(torch.zeros(num_labels, hidden_size))
        # Final weights for classification
        self.W = nn.Parameter(torch.zeros(num_labels, hidden_size))
        # Final bias
        self.bias = nn.Parameter(torch.zeros(num_labels))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.W)

    def forward(self, H, attention_mask=None):
        # H: [batch_size, seq_len, hidden_size]
        
        # Calculate attention scores in float32 for stability
        scores = torch.matmul(H.float(), self.U.transpose(0, 1).float())
        
        if attention_mask is not None:
            # attention_mask is [B, seq_len] with 1s for tokens and 0s for padding
            # Convert to [B, seq_len, 1] and apply large negative value to padding tokens
            mask = (1.0 - attention_mask.unsqueeze(-1).float()) * -1e9
            scores = scores + mask
            
        # Softmax over sequence length L
        A = torch.softmax(scores, dim=1) # [B, L, N]
        
        # Label-specific document representation V = A^T H -> [B, N, D]
        V = torch.matmul(A.transpose(1, 2), H.float())
        
        # Final classification logits [B, N]
        logits = (V * self.W.float()).sum(dim=-1) + self.bias.float() 
        
        return logits.to(H.dtype)

MODEL_CHOICES = {
    "vihealthbert": "demdecuong/vihealthbert-base-syllable",
    "phobert": "vinai/phobert-base-v2",
    "sapbert": "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
}

class FocalLossMultiLabel(nn.Module):
    """
    Focal Loss for Multi-Label Classification.
    Down-weights the massive number of easy negative examples so the model 
    can focus its gradients on learning the rare positive diseases.
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Calculate standard BCE loss without reducing
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # pt is the predicted probability for the TRUE class
        # (if target=1, pt=p; if target=0, pt=1-p)
        pt = torch.exp(-bce_loss)
        
        # Apply the focal loss scaling factor (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()

class PLMICDModel(nn.Module):
    """
    PLM-ICD Model compatible with standard HuggingFace models 
    (ViHealthBERT, PhoBERT, SapBERT, etc.)
    """
    def __init__(self, num_labels: int, model_name: str = "vihealthbert"):
        super().__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        
        # Map simple name to full HuggingFace path (fallback to original string if not found)
        self.model_path = MODEL_CHOICES.get(model_name.lower(), model_name)
        
        # Initialize standard encoder (will work for BERT, RoBERTa, DeBERTa variants)
        self.encoder = AutoModel.from_pretrained(self.model_path)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Initialize LAAT Head
        self.laat = LAATLayer(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Extract features using the backbone
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state
        
        # Pass hidden states through Label Attention with mask!
        logits = self.laat(H, attention_mask=attention_mask)
        
        loss = None
        if labels is not None:
            # Swap standard BCE for Focal Loss to conquer extreme sparsity
            loss_fct = FocalLossMultiLabel(gamma=2.0)
            # Labels must be float
            loss = loss_fct(logits, labels.float())
            
            # Fix for Multi-GPU DataParallel warning: 
            # DataParallel expects 1D tensors to gather across GPUs, not 0D scalars.
            loss = loss.unsqueeze(0)
            
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()

    def save_pretrained(self, save_directory):
        """Allow HuggingFace Trainer to save the model."""
        self.encoder.save_pretrained(save_directory)
        self.encoder.config.save_pretrained(save_directory)
        
        # Save LAAT weights
        torch.save(self.laat.state_dict(), f"{save_directory}/laat_head.pt")
