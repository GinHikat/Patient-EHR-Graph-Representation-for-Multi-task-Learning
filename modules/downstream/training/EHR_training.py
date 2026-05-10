import os, sys
import time
import datetime
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
import argparse

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

if current_dir not in sys.path:
    sys.path.append(current_dir)

from EHR_model import EHRDataset, ehr_collate_fn, EHRModel, EHRTransformer, EHRLoss

def parse_args():
    parser = argparse.ArgumentParser(description='EHR Training')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'transformer'], help='Model type: lstm or transformer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--threshold', type=float, default=0.6, help='Classification threshold')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to best_model.pt to resume from')
    parser.add_argument('--start_epoch', type=int, default=1, help='Epoch to start from')
    return parser.parse_args()

args = parse_args()

base_data_dir = os.path.join(project_root, 'data')
downstream_data_path = os.path.join(base_data_dir, 'Timeline')

TIMELINE_DIR         = os.path.join(base_data_dir, 'Timelines')
ADMISSION_NODES_PATH = os.path.join(downstream_data_path, 'admission_nodes.json')
DIAG_VOCAB_PATH      = os.path.join(downstream_data_path, 'top200_diag_vocab.json')
PROG_WEIGHTS_PATH    = os.path.join(downstream_data_path, 'progression_pos_weights.npy')
TRAIN_DF_PATH        = os.path.join(downstream_data_path, 'models', 'train_df.csv')
VAL_DF_PATH          = os.path.join(downstream_data_path, 'models', 'val_df.csv')
TEST_DF_PATH         = os.path.join(downstream_data_path, 'models', 'test_df.csv')
PATIENT_CACHE_PATH   = os.path.join(downstream_data_path, 'setup', 'patient_cache.pt')
ADMISSION_CACHE_PATH = os.path.join(downstream_data_path, 'setup', 'admission_cache.pt')
DRUG_VOCAB_PATH      = os.path.join(downstream_data_path, 'top50_drug_vocab.json')
DRUG_WEIGHTS_PATH    = os.path.join(downstream_data_path, 'drug_rec_pos_weights.npy')

run_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
config_str = f"{args.model_type}_bs{args.batch_size}_lr{args.lr}_ep{args.epochs}"
CHECKPOINT_DIR = os.path.join('checkpoints', f"{config_str}_{run_time_str}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training hyperparameters from args
BATCH_SIZE    = args.batch_size
LR            = args.lr
WEIGHT_DECAY  = args.weight_decay
N_EPOCHS      = args.epochs
PATIENCE      = args.patience
GRAD_CLIP     = args.grad_clip
NUM_WORKERS   = args.num_workers
THRESHOLD     = args.threshold
MAX_LEN       = 512 # Cap long timelines for Transformer O(T^2) efficiency

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')
print(f'Project root: {project_root}')
# print(f'DATA_DIR: {data_dir}')
print(f'TIMELINE_DIR: {TIMELINE_DIR}')
if not os.path.exists(TIMELINE_DIR):
    print(f"ERROR: TIMELINE_DIR '{TIMELINE_DIR}' does not exist! Check your DATA_DIR in .env or environment variables.")
    sys.exit(1)
print(f'TRAIN_DF_PATH: {TRAIN_DF_PATH}')

# Evaluation — AUROC per task
def evaluate(model, loader, criterion, device):
    model.eval()

    tasks      = ['mortality', 'los_7d', 'readmission', 'progression', 'drug_rec']
    all_logits = {t: [] for t in tasks}
    all_labels = {t: [] for t in tasks}
    total_loss = {t: 0.0 for t in ['total', 'mortality', 'los_7d', 'readmission', 'progression', 'drug_rec']}
    n_batches  = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            if batch is None: continue
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch)
            loss, loss_dict = criterion(logits, batch)
            for k, v in loss_dict.items(): total_loss[k] += v
            n_batches += 1

            # Mortality
            all_logits['mortality'].append(torch.sigmoid(logits['mortality']).squeeze(1).cpu().numpy())
            all_labels['mortality'].append(batch['mortality'].cpu().numpy())

            # LOS
            all_logits['los_7d'].append(torch.sigmoid(logits['los_7d']).squeeze(1).cpu().numpy())
            all_labels['los_7d'].append(batch['los_7d'].cpu().numpy())

            # Readmission (Mask missing -1 labels)
            readm_mask = batch['readmission'] >= 0
            if readm_mask.any():
                all_logits['readmission'].append(torch.sigmoid(logits['readmission']).squeeze(1)[readm_mask].cpu().numpy())
                all_labels['readmission'].append(batch['readmission'][readm_mask].cpu().numpy())

            # Progression (Mask empty samples)
            prog_mask = batch['progression'].sum(dim=-1) > 0
            if prog_mask.any():
                all_logits['progression'].append(torch.sigmoid(logits['progression'])[prog_mask].cpu().numpy())
                all_labels['progression'].append(batch['progression'][prog_mask].cpu().numpy())

            # Drug rec (Mask empty samples)
            drug_mask = batch['drug_rec'].sum(dim=-1) > 0
            if drug_mask.any():
                all_logits['drug_rec'].append(torch.sigmoid(logits['drug_rec'])[drug_mask].cpu().numpy())
                all_labels['drug_rec'].append(batch['drug_rec'][drug_mask].cpu().numpy())

    metrics = {}

    # ── Binary tasks (Mortality, LOS, Readmission) ──────────────────
    for task in ['mortality', 'los_7d', 'readmission']:
        if len(all_labels[task]) == 0:
            metrics[task] = metrics[f'{task}_aupr'] = metrics[f'{task}_mAP'] = 0.0
            metrics[f'{task}_f1'] = metrics[f'{task}_accuracy'] = 0.0
            metrics[f'{task}_precision'] = metrics[f'{task}_recall'] = 0.0
            continue

        y_true = np.concatenate(all_labels[task])
        y_prob = np.concatenate(all_logits[task])
        y_pred = (y_prob >= THRESHOLD).astype(float)

        try:
            metrics[task] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics[task] = 0.0
        
        ap = float(average_precision_score(y_true, y_prob))
        metrics[f'{task}_aupr'] = metrics[f'{task}_mAP'] = ap
        
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        metrics[f'{task}_accuracy']  = float(accuracy_score(y_true, y_pred))
        metrics[f'{task}_precision'] = float(p)
        metrics[f'{task}_recall']    = float(r)
        metrics[f'{task}_f1']        = float(f1)

    # ── Multilabel helper (Progression, Drug Rec) ───────────────────
    def multilabel_metrics(labels_list, logits_list, prefix):
        if len(labels_list) == 0:
            metrics[prefix] = metrics[f'{prefix}_aupr'] = metrics[f'{prefix}_mAP'] = metrics[f'{prefix}_f1'] = 0.0
            return

        y_true = np.concatenate(labels_list)
        y_prob = np.concatenate(logits_list)
        y_pred = (y_prob >= THRESHOLD).astype(float)

        # Macro AUROC
        aurocs = []
        for i in range(y_true.shape[1]):
            if y_true[:, i].sum() > 0:
                try: aurocs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
                except ValueError: pass
        metrics[prefix] = float(np.mean(aurocs)) if aurocs else 0.0

        # Macro AUPR (mAP)
        ap = float(average_precision_score(y_true, y_prob, average='macro'))
        metrics[f'{prefix}_aupr'] = metrics[f'{prefix}_mAP'] = ap

        # Micro F1
        _, _, f1, _ = precision_recall_fscore_support(y_true.ravel(), y_pred.ravel(), average='binary', zero_division=0)
        metrics[f'{prefix}_f1'] = float(f1)

    multilabel_metrics(all_labels['progression'], all_logits['progression'], 'progression')
    multilabel_metrics(all_labels['drug_rec'], all_logits['drug_rec'], 'drug_rec')

    # Mean AUROC for early stopping
    metrics['mean_auroc'] = float(np.mean([metrics[t] for t in tasks]))
    metrics['loss_dict'] = {k: v / max(n_batches, 1) for k, v in total_loss.items()}

    return metrics

# Training loop
def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    start_epoch=1,
    n_epochs=N_EPOCHS,
    patience=PATIENCE,
    device=DEVICE):

    best_mean_auroc   = 0.0
    epochs_no_improve = 0
    history           = []

    # Initialize/Clear metrics log file only if starting from scratch
    log_mode = 'w' if start_epoch == 1 else 'a'
    with open(os.path.join(CHECKPOINT_DIR, 'metrics.txt'), log_mode) as f:
        if start_epoch == 1:
            f.write(f"=== TRAINING LOG START ===\n")
            f.write(f"Device: {device}\n")
            f.write(f"Batch size: {BATCH_SIZE}, LR: {LR}, Epochs: {n_epochs}\n")
            f.write("="*40 + "\n")
        else:
            f.write(f"\n=== RESUMING FROM EPOCH {start_epoch} ===\n")

    for epoch in range(start_epoch, n_epochs + 1):

        model.train()

        train_loss = {
            t: 0.0
            for t in ['total', 'mortality', 'los_7d', 'readmission', 'progression', 'drug_rec'] 
        }

        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{n_epochs}')

        for batch in pbar:

            if batch is None:
                continue

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

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

            # ── Linear LR Warmup ──────────────────────────────────────
            if epoch <= 5: # Warmup over first 5 epochs
                total_steps = 5 * len(train_loader)
                current_step = (epoch - 1) * len(train_loader) + n_batches
                lr_scale = min(1., float(current_step) / total_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = LR * lr_scale
            # ──────────────────────────────────────────────────────────


            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.3f}",
                'mort': f"{loss_dict['mortality']:.3f}",
                'los':  f"{loss_dict['los_7d']:.3f}",
                'drug': f"{loss_dict['drug_rec']:.3f}",  
            })

        # Average train losses
        train_loss_avg = {
            k: v / max(n_batches, 1)
            for k, v in train_loss.items()
        }

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_metrics['mean_auroc'])

        # Console log & File log
        metrics_msg = (
            f"\nEpoch {epoch:>3} | "
            f"Train loss: {train_loss_avg['total']:.4f} | "
            f"Val mean AUROC: {val_metrics['mean_auroc']:.4f}\n"

            f"\n[MORTALITY]\n"
            f"AUROC: {val_metrics['mortality']:.4f} | "
            f"AUPR: {val_metrics['mortality_aupr']:.4f} | "
            f"F1: {val_metrics['mortality_f1']:.4f}\n"

            f"\n[LOS > 7 DAYS]\n"
            f"AUROC: {val_metrics['los_7d']:.4f} | "
            f"AUPR: {val_metrics['los_7d_aupr']:.4f} | "
            f"F1: {val_metrics['los_7d_f1']:.4f}\n"

            f"\n[READMISSION]\n"
            f"AUROC: {val_metrics['readmission']:.4f} | "
            f"AUPR: {val_metrics['readmission_aupr']:.4f} | "
            f"F1: {val_metrics['readmission_f1']:.4f}\n"

            f"\n[PROGRESSION]\n"
            f"AUROC: {val_metrics['progression']:.4f} | "
            f"AUPR: {val_metrics['progression_aupr']:.4f} | "
            f"F1: {val_metrics['progression_f1']:.4f}\n"

            f"\n[DRUG RECOMMENDATION]\n"
            f"AUROC: {val_metrics['drug_rec']:.4f} | "
            f"AUPR: {val_metrics['drug_rec_aupr']:.4f} | "
            f"F1: {val_metrics['drug_rec_f1']:.4f}\n"

            f"\n[VALIDATION LOSSES]\n"
            f"mort: {val_metrics['loss_dict']['mortality']:.3f} | "
            f"los: {val_metrics['loss_dict']['los_7d']:.3f} | "
            f"readm: {val_metrics['loss_dict']['readmission']:.3f} | "
            f"prog: {val_metrics['loss_dict']['progression']:.3f} | "
            f"drug: {val_metrics['loss_dict']['drug_rec']:.3f}\n"
            f"{'='*40}\n"
        )
        print(metrics_msg)
        with open(os.path.join(CHECKPOINT_DIR, 'metrics.txt'), 'a') as f:
            f.write(metrics_msg)

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # Save history
        history.append({
            'epoch':      epoch,
            'train_loss': train_loss_avg,
            'val_metrics': val_metrics,
        })

        # Save best model
        if val_metrics['mean_auroc'] > best_mean_auroc:

            best_mean_auroc   = val_metrics['mean_auroc']
            epochs_no_improve = 0

            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, 'best_model.pt')
            )

            print(f"✓ New best mean AUROC: {best_mean_auroc:.4f}")

        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs")

        # Early stopping
        if epochs_no_improve >= patience:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best mean AUROC: {best_mean_auroc:.4f}"
            )
            break

    return history

if __name__ == '__main__':

    print('Loading data...')
    train_df = pd.read_csv(TRAIN_DF_PATH, dtype={'id': str, 'patient_id': str})
    val_df   = pd.read_csv(VAL_DF_PATH, dtype={'id': str, 'patient_id': str})

    # Add los_7d if not already there
    if 'los_7d' not in train_df.columns:
        train_df['los_7d'] = (train_df['length_of_stay'] >= 7).astype(float)
        val_df['los_7d']   = (val_df['length_of_stay']   >= 7).astype(float)

    with open(ADMISSION_NODES_PATH) as f:
        admission_nodes = json_lib.load(f)
    with open(DIAG_VOCAB_PATH) as f:
        diag_to_idx = json_lib.load(f)
    with open(DRUG_VOCAB_PATH) as f:
        drug_to_idx = json_lib.load(f)

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
        drug_to_idx     = drug_to_idx,
        patient_cache   = patient_cache,
        admission_cache = admission_cache,
        max_len         = MAX_LEN,
    )
    val_dataset = EHRDataset(
        admissions_df   = val_df,
        timeline_dir    = TIMELINE_DIR,
        admission_nodes = admission_nodes,
        diag_to_idx     = diag_to_idx,
        drug_to_idx     = drug_to_idx,
        patient_cache   = patient_cache,
        admission_cache = admission_cache,
        max_len         = MAX_LEN,
    )

    MAX_LEN = 512 # Cap long timelines for Transformer O(T^2) efficiency
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=ehr_collate_fn, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=ehr_collate_fn, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0)
    )

    # Pos weights
    prog_weights = torch.tensor(
        np.load(PROG_WEIGHTS_PATH), dtype=torch.float32
    ).to(DEVICE)
    drug_weights = torch.tensor(
        np.load(DRUG_WEIGHTS_PATH), dtype=torch.float32
    ).to(DEVICE)

    # Compute mortality and readmission pos_weight from train_df
    n_train     = len(train_df)
    pw_mort     = (train_df['inhospital_dead'] == 0).sum() / (train_df['inhospital_dead'] == 1).sum()
    pw_los      = (train_df['los_7d'] == 0).sum() / (train_df['los_7d'] == 1).sum()
    pw_readm    = (train_df['readmission_30d'] == 0).sum() / (train_df['readmission_30d'] == 1).sum()

    print(f'pos_weight — mortality: {pw_mort:.2f}, los: {pw_los:.2f}, readmission: {pw_readm:.2f}')

    # Model
    # Model selection
    if args.model_type == 'transformer':
        model = EHRTransformer().to(DEVICE)
    else:
        model = EHRModel().to(DEVICE)
    
    model.use_gradient_checkpointing = False
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Gradient checkpointing: {model.use_gradient_checkpointing}')

    # Resume from checkpoint if provided
    if args.resume_from:
        # Check if it's a full path or just a folder name in checkpoints/
        ckpt_path = args.resume_from
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join('checkpoints', args.resume_from, 'best_model.pt')
        
        if os.path.exists(ckpt_path):
            print(f"Resuming from checkpoint: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
        else:
            print(f"ERROR: Checkpoint file '{ckpt_path}' not found!")
            sys.exit(1)

    # Loss
    criterion = EHRLoss(
        pos_weight_mortality   = torch.tensor([pw_mort],  dtype=torch.float32).to(DEVICE),
        pos_weight_los         = torch.tensor([pw_los],   dtype=torch.float32).to(DEVICE),
        pos_weight_readmission = torch.tensor([pw_readm], dtype=torch.float32).to(DEVICE),
        pos_weight_progression = prog_weights,
        pos_weight_drug_rec    = drug_weights,
    ).to(DEVICE)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # Train
    start_time = time.time()
    history = train(model, train_loader, val_loader, criterion, optimizer, scheduler, start_epoch=args.start_epoch)
    end_time = time.time()
    
    run_model_time = end_time - start_time
    print(f'\nTraining complete in {run_model_time/60:.2f} minutes.')
    print(f'Best model saved to {CHECKPOINT_DIR}/best_model.pt')
    
    # Log run time
    with open(os.path.join(CHECKPOINT_DIR, 'metrics.txt'), 'a') as f:
        f.write(f"\nTotal Run Time: {run_model_time/60:.2f} minutes\n")
    
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
    test_df = pd.read_csv(TEST_DF_PATH, dtype={'id': str, 'patient_id': str})
    # patient_id is already str due to dtype above
    
    test_dataset = EHRDataset(
        admissions_df   = test_df,
        timeline_dir    = TIMELINE_DIR,
        admission_nodes = admission_nodes,
        diag_to_idx     = diag_to_idx,
        drug_to_idx     = drug_to_idx,
        patient_cache   = patient_cache,
        admission_cache = admission_cache,
        max_len         = MAX_LEN,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=ehr_collate_fn, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == 'cuda')
    )
    
    test_metrics = evaluate(model, test_loader, criterion, DEVICE)
    
    print(
        f"\nFINAL TEST Mean AUROC: {test_metrics['mean_auroc']:.4f}\n"

        f"\n[MORTALITY]\n"
        f"AUROC: {test_metrics['mortality']:.4f} | "
        f"AUPR: {test_metrics['mortality_aupr']:.4f} | "
        f"F1: {test_metrics['mortality_f1']:.4f}\n"

        f"\n[LOS > 7 DAYS]\n"
        f"AUROC: {test_metrics['los_7d']:.4f} | "
        f"AUPR: {test_metrics['los_7d_aupr']:.4f} | "
        f"F1: {test_metrics['los_7d_f1']:.4f}\n"

        f"\n[READMISSION]\n"
        f"AUROC: {test_metrics['readmission']:.4f} | "
        f"AUPR: {test_metrics['readmission_aupr']:.4f} | "
        f"F1: {test_metrics['readmission_f1']:.4f}\n"

        f"\n[PROGRESSION]\n"
        f"AUROC: {test_metrics['progression']:.4f} | "
        f"AUPR: {test_metrics['progression_aupr']:.4f} | "
        f"F1: {test_metrics['progression_f1']:.4f}\n"

        f"\n[DRUG RECOMMENDATION]\n"
        f"AUROC: {test_metrics['drug_rec']:.4f} | "
        f"AUPR: {test_metrics['drug_rec_aupr']:.4f} | "
        f"F1: {test_metrics['drug_rec_f1']:.4f}\n"
    )

    # Log final test results to file
    with open(os.path.join(CHECKPOINT_DIR, 'metrics.txt'), 'a') as f:
        f.write("\n" + "="*20 + " FINAL TEST RESULTS " + "="*20 + "\n")
        f.write(f"Mean AUROC: {test_metrics['mean_auroc']:.4f}\n")
        f.write(json.dumps(test_metrics, indent=2))

    with open(os.path.join(CHECKPOINT_DIR, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Training history saved to {CHECKPOINT_DIR}/history.json')