import os, sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pickle
import pandas as pd
import json as json_lib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

if current_dir not in sys.path:
    sys.path.append(current_dir)

from EHR_model import EHRDataset, ehr_collate_fn, EHRModel, EHRLoss

data_dir = os.getenv('DATA_DIR')
base_data_dir = os.path.join(project_root, 'data')
downstream_data_path = os.path.join(base_data_dir, 'downstream')

TIMELINE_DIR         = os.path.join(data_dir, 'Timelines')
ADMISSION_NODES_PATH = os.path.join(downstream_data_path, 'admission_nodes.json')
DIAG_VOCAB_PATH      = os.path.join(downstream_data_path, 'top200_diag_vocab.json')
PROG_WEIGHTS_PATH    = os.path.join(downstream_data_path, 'progression_pos_weights.npy')
TRAIN_DF_PATH        = os.path.join(downstream_data_path, 'models', 'train_df.csv')
VAL_DF_PATH          = os.path.join(downstream_data_path, 'models', 'val_df.csv')
PATIENT_CACHE_PATH   = 'patient_cache.pt'
ADMISSION_CACHE_PATH = 'admission_cache.pt'

CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training hyperparameters
BATCH_SIZE    = 64
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
N_EPOCHS      = 50
PATIENCE      = 7        # early stopping patience
GRAD_CLIP     = 1.0      # max gradient norm
NUM_WORKERS   = 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# Evaluation — AUROC per task
def evaluate(model, loader, criterion, device):
    """
    Run one pass over the loader and compute:
        - Per-task AUROC
        - Per-task loss
        - Mean AUROC across all 4 tasks (primary metric for early stopping)

    Returns:
        metrics (dict): {task: auroc, 'mean_auroc': float, 'loss_dict': dict}
    """
    model.eval()

    # Accumulators
    all_logits = {t: [] for t in ['mortality', 'los_7d', 'readmission', 'progression']}
    all_labels = {t: [] for t in ['mortality', 'los_7d', 'readmission', 'progression']}
    total_loss  = {t: 0.0 for t in ['total', 'mortality', 'los_7d', 'readmission', 'progression']}
    n_batches   = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            if batch is None:
                continue

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            logits = model(batch)
            loss, loss_dict = criterion(logits, batch)

            for k, v in loss_dict.items():
                total_loss[k] += v
            n_batches += 1

            # Collect predictions and labels for AUROC
            # Mortality
            all_logits['mortality'].append(
                torch.sigmoid(logits['mortality']).squeeze(1).cpu().numpy()
            )
            all_labels['mortality'].append(batch['mortality'].cpu().numpy())

            # LOS
            all_logits['los_7d'].append(
                torch.sigmoid(logits['los_7d']).squeeze(1).cpu().numpy()
            )
            all_labels['los_7d'].append(batch['los_7d'].cpu().numpy())

            # Readmission — exclude masked (-1) labels
            readm_mask = batch['readmission'] >= 0
            if readm_mask.any():
                all_logits['readmission'].append(
                    torch.sigmoid(logits['readmission']).squeeze(1)[readm_mask].cpu().numpy()
                )
                all_labels['readmission'].append(
                    batch['readmission'][readm_mask].cpu().numpy()
                )

            # Progression — collect per-admission, compute macro AUROC later
            prog_mask = batch['progression'].sum(dim=-1) > 0
            if prog_mask.any():
                all_logits['progression'].append(
                    torch.sigmoid(logits['progression'])[prog_mask].cpu().numpy()
                )
                all_labels['progression'].append(
                    batch['progression'][prog_mask].cpu().numpy()
                )

    # Compute AUROC per task
    metrics = {}

    for task in ['mortality', 'los_7d', 'readmission']:
        if len(all_labels[task]) == 0:
            metrics[task] = 0.0
            continue
        y_true = np.concatenate(all_labels[task])
        y_pred = np.concatenate(all_logits[task])
        try:
            metrics[task] = roc_auc_score(y_true, y_pred)
        except ValueError:
            # Only one class in batch — skip
            metrics[task] = 0.0

    # Progression — macro AUROC across 200 diagnoses
    if len(all_labels['progression']) > 0:
        y_true_prog = np.concatenate(all_labels['progression'])   # (N, 200)
        y_pred_prog = np.concatenate(all_logits['progression'])   # (N, 200)
        aurocs = []
        for i in range(y_true_prog.shape[1]):
            if y_true_prog[:, i].sum() > 0:   # skip if no positives
                try:
                    aurocs.append(roc_auc_score(y_true_prog[:, i], y_pred_prog[:, i]))
                except ValueError:
                    pass
        metrics['progression'] = float(np.mean(aurocs)) if aurocs else 0.0
    else:
        metrics['progression'] = 0.0

    # Mean AUROC — primary metric for early stopping and model selection
    metrics['mean_auroc'] = float(np.mean([
        metrics['mortality'],
        metrics['los_7d'],
        metrics['readmission'],
        metrics['progression'],
    ]))

    # Average losses
    if n_batches > 0:
        metrics['loss_dict'] = {k: v / n_batches for k, v in total_loss.items()}
    else:
        metrics['loss_dict'] = total_loss

    return metrics

# Training loop
def train(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    n_epochs=N_EPOCHS, patience=PATIENCE, device=DEVICE
):
    best_mean_auroc = 0.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = {t: 0.0 for t in ['total', 'mortality', 'los_7d', 'readmission', 'progression']}
        n_batches  = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{n_epochs}')
        for batch in pbar:
            if batch is None:
                continue

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            optimizer.zero_grad()
            logits = model(batch)
            loss, loss_dict = criterion(logits, batch)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            for k, v in loss_dict.items():
                train_loss[k] += v
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.3f}",
                'mort': f"{loss_dict['mortality']:.3f}",
                'los':  f"{loss_dict['los_7d']:.3f}",
            })

        # Average train losses
        train_loss_avg = {k: v / max(n_batches, 1) for k, v in train_loss.items()}

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_metrics['mean_auroc'])

        # Log
        print(
            f"\nEpoch {epoch:>3} | "
            f"Train loss: {train_loss_avg['total']:.4f} | "
            f"Val mean AUROC: {val_metrics['mean_auroc']:.4f}\n"
            f"  Mortality   AUROC: {val_metrics['mortality']:.4f}  "
            f"  LOS         AUROC: {val_metrics['los_7d']:.4f}\n"
            f"  Readmission AUROC: {val_metrics['readmission']:.4f}  "
            f"  Progression AUROC: {val_metrics['progression']:.4f}\n"
            f"  Val losses → "
            f"mort: {val_metrics['loss_dict']['mortality']:.3f}  "
            f"los: {val_metrics['loss_dict']['los_7d']:.3f}  "
            f"readm: {val_metrics['loss_dict']['readmission']:.3f}  "
            f"prog: {val_metrics['loss_dict']['progression']:.3f}"
        )

        # Save history
        history.append({
            'epoch':          epoch,
            'train_loss':     train_loss_avg,
            'val_metrics':    val_metrics,
        })
        with open(os.path.join(CHECKPOINT_DIR, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        # Save latest checkpoint
        torch.save({
            'epoch':      epoch,
            'model':      model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'history':    history,
        }, os.path.join(CHECKPOINT_DIR, 'latest.pt'))

        # Save best model
        if val_metrics['mean_auroc'] > best_mean_auroc:
            best_mean_auroc = val_metrics['mean_auroc']
            epochs_no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
            print(f"  ✓ New best mean AUROC: {best_mean_auroc:.4f} — model saved")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"Best mean AUROC: {best_mean_auroc:.4f}")
            break

    return history

if __name__ == '__main__':

    print('Loading data...')
    train_df = pd.read_csv(TRAIN_DF_PATH)
    val_df   = pd.read_csv(VAL_DF_PATH)

    # Add los_7d if not already there
    if 'los_7d' not in train_df.columns:
        train_df['los_7d'] = (train_df['length_of_stay'] >= 7).astype(float)
        val_df['los_7d']   = (val_df['length_of_stay']   >= 7).astype(float)

    with open(ADMISSION_NODES_PATH) as f:
        admission_nodes = json_lib.load(f)
    with open(DIAG_VOCAB_PATH) as f:
        diag_to_idx = json_lib.load(f)

    patient_cache   = torch.load(PATIENT_CACHE_PATH)
    admission_cache = torch.load(ADMISSION_CACHE_PATH)

    print('Building datasets...')
    train_dataset = EHRDataset(
        admissions_df   = train_df,
        timeline_dir    = TIMELINE_DIR,
        admission_nodes = admission_nodes,
        diag_to_idx     = diag_to_idx,
        patient_cache   = patient_cache,
        admission_cache = admission_cache,
    )
    val_dataset = EHRDataset(
        admissions_df   = val_df,
        timeline_dir    = TIMELINE_DIR,
        admission_nodes = admission_nodes,
        diag_to_idx     = diag_to_idx,
        patient_cache   = patient_cache,
        admission_cache = admission_cache,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=ehr_collate_fn, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=ehr_collate_fn, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == 'cuda'),
    )

    # Pos weights
    prog_weights = torch.tensor(
        np.load(PROG_WEIGHTS_PATH), dtype=torch.float32
    ).to(DEVICE)

    # Compute mortality and readmission pos_weight from train_df
    n_train     = len(train_df)
    pw_mort     = (train_df['inhospital_dead'] == 0).sum() / (train_df['inhospital_dead'] == 1).sum()
    pw_los      = (train_df['los_7d'] == 0).sum() / (train_df['los_7d'] == 1).sum()
    pw_readm    = (train_df['readmission_30d'] == 0).sum() / (train_df['readmission_30d'] == 1).sum()

    print(f'pos_weight — mortality: {pw_mort:.2f}, los: {pw_los:.2f}, readmission: {pw_readm:.2f}')

    # Model
    model = EHRModel().to(DEVICE)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Loss
    criterion = EHRLoss(
        pos_weight_mortality   = torch.tensor([pw_mort],  dtype=torch.float32).to(DEVICE),
        pos_weight_los         = torch.tensor([pw_los],   dtype=torch.float32).to(DEVICE),
        pos_weight_readmission = torch.tensor([pw_readm], dtype=torch.float32).to(DEVICE),
        pos_weight_progression = prog_weights,
    ).to(DEVICE)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # Train
    history = train(model, train_loader, val_loader, criterion, optimizer, scheduler)

    print('\nTraining complete.')
    print(f'Best model saved to {CHECKPOINT_DIR}/best_model.pt')
    print(f'Training history saved to {CHECKPOINT_DIR}/history.json')