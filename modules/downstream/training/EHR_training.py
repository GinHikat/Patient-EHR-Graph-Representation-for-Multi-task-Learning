import os, sys
import json
import numpy as np
import torch
import gc
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, 
    accuracy_score, average_precision_score
)
from tqdm import tqdm
import pickle
import pandas as pd
import json as json_lib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

if current_dir not in sys.path:
    sys.path.append(current_dir)

from EHR_model import EHRDataset, ehr_collate_fn, EHRModel, EHRLoss

base_data_dir = os.path.join(project_root, 'data')
data_path = os.path.join(base_data_dir, 'Timeline')

TIMELINE_DIR         = os.path.join(base_data_dir, 'Timelines')
ADMISSION_NODES_PATH = os.path.join(data_path, 'admission_nodes.json')
DIAG_VOCAB_PATH      = os.path.join(data_path, 'top200_diag_vocab.json')
PROG_WEIGHTS_PATH    = os.path.join(data_path, 'progression_pos_weights.npy')
TRAIN_DF_PATH        = os.path.join(data_path, 'models', 'train_df.csv')
VAL_DF_PATH          = os.path.join(data_path, 'models', 'val_df.csv')
TEST_DF_PATH         = os.path.join(data_path, 'models', 'test_df.csv')
PATIENT_CACHE_PATH   = os.path.join(data_path, 'setup', 'patient_cache.pt')
ADMISSION_CACHE_PATH = os.path.join(data_path, 'setup', 'admission_cache.pt')

CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training hyperparameters
BATCH_SIZE    = 64
LR            = 1e-4
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

    # Compute metrics per task
    metrics = {}
    
    # Binary tasks: Mortality, LOS, Readmission
    for task in ['mortality', 'los_7d', 'readmission']:
        if len(all_labels[task]) == 0:
            metrics[task] = 0.0
            metrics[f'{task}_f1'] = 0.0
            metrics[f'{task}_mAP'] = 0.0
            continue
        
        y_true = np.concatenate(all_labels[task])
        y_pred_prob = np.concatenate(all_logits[task])
        y_pred = (y_pred_prob >= 0.5).astype(float)
        
        # AUROC
        try:
            metrics[task] = float(roc_auc_score(y_true, y_pred_prob))
        except ValueError:
            metrics[task] = 0.0
            
        # Others
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        metrics[f'{task}_precision'] = float(p)
        metrics[f'{task}_recall'] = float(r)
        metrics[f'{task}_f1'] = float(f1)
        metrics[f'{task}_accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics[f'{task}_mAP'] = float(average_precision_score(y_true, y_pred_prob))

    # Multilabel task: Progression
    if len(all_labels['progression']) > 0:
        y_true_prog = np.concatenate(all_labels['progression'])
        y_pred_prog = np.concatenate(all_logits['progression'])
        
        # Macro AUROC
        aurocs = []
        for i in range(y_true_prog.shape[1]):
            if y_true_prog[:, i].sum() > 0:
                try:
                    aurocs.append(roc_auc_score(y_true_prog[:, i], y_pred_prog[:, i]))
                except ValueError: pass
        metrics['progression'] = float(np.mean(aurocs)) if aurocs else 0.0
        
        # mAP (macro)
        metrics['progression_mAP'] = float(average_precision_score(y_true_prog, y_pred_prog, average='macro'))
    else:
        metrics['progression'] = 0.0
        metrics['progression_mAP'] = 0.0

    # Mean AUROC — primary metric for early stopping
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
    n_epochs=N_EPOCHS, patience=PATIENCE, device=DEVICE, checkpoint_path=None):
    
    best_mean_auroc = 0.0
    epochs_no_improve = 0
    start_epoch = 1
    start_step = 0
    history = []

    # Resume from checkpoint if it exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch']
        start_step = checkpoint.get('step', 0)
        history = checkpoint.get('history', [])
        best_mean_auroc = checkpoint.get('best_mean_auroc', 0.0)
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        print(f"Resuming from Epoch {start_epoch}, Step {start_step}")

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        train_loss = {t: 0.0 for t in ['total', 'mortality', 'los_7d', 'readmission', 'progression']}
        n_batches  = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{n_epochs}')
        for i, batch in enumerate(pbar):
            # Skip steps if resuming
            if epoch == start_epoch and i < start_step:
                continue
            if batch is None:
                continue

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            optimizer.zero_grad()
            logits = model(batch)
            loss, loss_dict = criterion(logits, batch)

            if torch.isnan(loss):
                print(f"\n[ERROR] NaN Loss detected at Epoch {epoch}, Batch {n_batches}. Skipping.")
                continue

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

            # Periodic checkpointing (e.g., every 500 batches)
            if (i + 1) % 500 == 0:
                torch.save({
                    'epoch': epoch,
                    'step': i + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'history': history,
                    'best_mean_auroc': best_mean_auroc,
                    'epochs_no_improve': epochs_no_improve,
                }, os.path.join(CHECKPOINT_DIR, 'checkpoint.pt'))

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

        # Force garbage collection to free memory
        gc.collect()
        torch.cuda.empty_cache()

        # Write to log file in a descriptive format
        log_file = os.path.join(CHECKPOINT_DIR, 'metrics.txt')
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*40}\n")
            f.write(f"EPOCH {epoch} SUMMARY\n")
            f.write(f"{'='*40}\n")
            f.write(f"Mean AUROC: {val_metrics['mean_auroc']:.4f}\n")
            f.write(f"Train Loss: {train_loss_avg['total']:.4f}\n\n")
            
            f.write(f"Mortality:   AUROC: {val_metrics['mortality']:.4f}, F1: {val_metrics['mortality_f1']:.4f}, mAP: {val_metrics['mortality_mAP']:.4f}\n")
            f.write(f"             Acc: {val_metrics['mortality_accuracy']:.4f}, Prec: {val_metrics['mortality_precision']:.4f}, Recall: {val_metrics['mortality_recall']:.4f}\n\n")
            
            f.write(f"LOS:         AUROC: {val_metrics['los_7d']:.4f}, F1: {val_metrics['los_7d_f1']:.4f}, mAP: {val_metrics['los_7d_mAP']:.4f}\n")
            f.write(f"             Acc: {val_metrics['los_7d_accuracy']:.4f}, Prec: {val_metrics['los_7d_precision']:.4f}, Recall: {val_metrics['los_7d_recall']:.4f}\n\n")
            
            f.write(f"Readmission: AUROC: {val_metrics['readmission']:.4f}, F1: {val_metrics['readmission_f1']:.4f}, mAP: {val_metrics['readmission_mAP']:.4f}\n")
            f.write(f"             Acc: {val_metrics['readmission_accuracy']:.4f}, Prec: {val_metrics['readmission_precision']:.4f}, Recall: {val_metrics['readmission_recall']:.4f}\n\n")
            
            f.write(f"Progression: AUROC: {val_metrics['progression']:.4f}, mAP: {val_metrics['progression_mAP']:.4f}\n")
            f.write(f"{'-'*40}\n")

        # Save history
        history.append({
            'epoch':          epoch,
            'train_loss':     train_loss_avg,
            'val_metrics':    val_metrics,
        })
        # Save latest checkpoint
        torch.save({
            'epoch':      epoch + 1,
            'step':       0,
            'model_state_dict':      model.state_dict(),
            'optimizer_state_dict':  optimizer.state_dict(),
            'scheduler_state_dict':  scheduler.state_dict() if scheduler else None,
            'history':    history,
            'best_mean_auroc': best_mean_auroc,
            'epochs_no_improve': epochs_no_improve,
        }, os.path.join(CHECKPOINT_DIR, 'checkpoint.pt'))

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

    try:
        patient_cache   = torch.load(PATIENT_CACHE_PATH, map_location='cpu', mmap=True)
        admission_cache = torch.load(ADMISSION_CACHE_PATH, map_location='cpu', mmap=True)
    except Exception:
        patient_cache   = torch.load(PATIENT_CACHE_PATH, map_location='cpu')
        admission_cache = torch.load(ADMISSION_CACHE_PATH, map_location='cpu')

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
    model.use_gradient_checkpointing = False
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Gradient checkpointing: {model.use_gradient_checkpointing}')

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
        optimizer, mode='max', factor=0.5, patience=3
    )

    # Train
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'checkpoint.pt')
    history = train(model, train_loader, val_loader, criterion, optimizer, scheduler, checkpoint_path=ckpt_path)

    print('\nTraining complete.')
    print(f'Best model saved to {CHECKPOINT_DIR}/best_model.pt')
    
    # Final Evaluation on Test Set
    print('\n' + '='*30)
    print('RUNNING FINAL TEST EVALUATION')
    print('='*30)
    
    # Load best model weights
    best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        print(f"Loaded best model weights from {best_model_path}")
    
    # Load test data
    test_df = pd.read_csv(TEST_DF_PATH)
    test_df['patient_id'] = test_df['patient_id'].astype(str)
    
    test_dataset = EHRDataset(
        admissions_df   = test_df,
        timeline_dir    = TIMELINE_DIR,
        admission_nodes = admission_nodes,
        diag_to_idx     = diag_to_idx,
        patient_cache   = patient_cache,
        admission_cache = admission_cache,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=ehr_collate_fn, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == 'cuda')
    )
    
    test_metrics = evaluate(model, test_loader, criterion, DEVICE)
    
    print(f"FINAL TEST Mean AUROC: {test_metrics['mean_auroc']:.4f}")
    print(f"  Mortality   AUROC: {test_metrics['mortality']:.4f}  F1: {test_metrics['mortality_f1']:.4f}  mAP: {test_metrics['mortality_mAP']:.4f}")
    print(f"  LOS         AUROC: {test_metrics['los_7d']:.4f}  F1: {test_metrics['los_7d_f1']:.4f}")
    print(f"  Readmission AUROC: {test_metrics['readmission']:.4f}  F1: {test_metrics['readmission_f1']:.4f}")
    print(f"  Progression AUROC: {test_metrics['progression']:.4f}  mAP: {test_metrics['progression_mAP']:.4f}")

    # Log final test results to file
    with open(os.path.join(CHECKPOINT_DIR, 'metrics.txt'), 'a') as f:
        f.write("\n" + "="*20 + " FINAL TEST RESULTS " + "="*20 + "\n")
        f.write(f"Mean AUROC: {test_metrics['mean_auroc']:.4f}\n")
        f.write(json.dumps(test_metrics, indent=2))
    print(f'Training history saved to {CHECKPOINT_DIR}/history.json')