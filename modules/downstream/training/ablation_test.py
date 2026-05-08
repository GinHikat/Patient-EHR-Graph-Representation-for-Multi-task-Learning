import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from copy import deepcopy
import json as json_lib
from EHR_model import EHRDataset, ehr_collate_fn
from EHR_model import EHRLoss

import pandas as pd

TIMELINE_DIR         = 'Timelines'
ADMISSION_NODES_PATH = 'admission_nodes.json'
DIAG_VOCAB_PATH      = 'top200_diag_vocab.json'
PROG_WEIGHTS_PATH    = 'progression_pos_weights.npy'
TRAIN_DF_PATH        = 'train_df.csv'
VAL_DF_PATH          = 'val_df.csv'
TEST_DF_PATH         = 'test_df.csv'
PATIENT_CACHE_PATH   = 'patient_cache.pt'
ADMISSION_CACHE_PATH = 'admission_cache.pt'
ABLATION_DIR         = 'ablations'

BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
N_EPOCHS     = 100
PATIENCE     = 15
GRAD_CLIP    = 1.0
NUM_WORKERS  = 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ABLATIONS = {
    # Group 1: Static features
    'no_patient_static': {
        'group':           'static_features',
        'description':     'Zero out patient static vector (64-dim → zeros)',
        'zero_patient':    True,
        'zero_admission':  False,
        'use_dt_decay':    True,
        'use_gru':         True,
        'zero_labs':       False,
        'zero_omr':        False,
        'zero_kg':         False,
        'tasks':           ['mortality', 'los_7d', 'readmission', 'progression'],
        'w_mortality':     1.0,
        'w_los':           1.0,
        'w_readmission':   1.0,
        'w_progression':   1.0,
    },
    'no_admission_static': {
        'group':           'static_features',
        'description':     'Zero out admission static vector (64-dim → zeros)',
        'zero_patient':    False,
        'zero_admission':  True,
        'use_dt_decay':    True,
        'use_gru':         True,
        'zero_labs':       False,
        'zero_omr':        False,
        'zero_kg':         False,
        'tasks':           ['mortality', 'los_7d', 'readmission', 'progression'],
        'w_mortality':     1.0,
        'w_los':           1.0,
        'w_readmission':   1.0,
        'w_progression':   1.0,
    },
    'no_static_at_all': {
        'group':           'static_features',
        'description':     'Zero out both patient and admission static vectors',
        'zero_patient':    True,
        'zero_admission':  True,
        'use_dt_decay':    True,
        'use_gru':         True,
        'zero_labs':       False,
        'zero_omr':        False,
        'zero_kg':         False,
        'tasks':           ['mortality', 'los_7d', 'readmission', 'progression'],
        'w_mortality':     1.0,
        'w_los':           1.0,
        'w_readmission':   1.0,
        'w_progression':   1.0,
    },

    # Group 2: Temporal modeling
    'no_dt_decay': {
        'group':           'temporal',
        'description':     'Remove exponential Δt decay (λ=0)',
        'zero_patient':    False,
        'zero_admission':  False,
        'use_dt_decay':    False,
        'use_gru':         True,
        'zero_labs':       False,
        'zero_omr':        False,
        'zero_kg':         False,
        'tasks':           ['mortality', 'los_7d', 'readmission', 'progression'],
        'w_mortality':     1.0,
        'w_los':           1.0,
        'w_readmission':   1.0,
        'w_progression':   1.0,
    },
    'no_history': {
        'group':           'temporal',
        'description':     'Skip GRU — use only static features (strong baseline)',
        'zero_patient':    False,
        'zero_admission':  False,
        'use_dt_decay':    False,
        'use_gru':         False,
        'zero_labs':       False,
        'zero_omr':        False,
        'zero_kg':         False,
        'tasks':           ['mortality', 'los_7d', 'readmission', 'progression'],
        'w_mortality':     1.0,
        'w_los':           1.0,
        'w_readmission':   1.0,
        'w_progression':   1.0,
    },

    # Group 3: Single task
    'single_mortality': {
        'group':           'single_task',
        'description':     'Train on mortality only',
        'zero_patient':    False,
        'zero_admission':  False,
        'use_dt_decay':    True,
        'use_gru':         True,
        'zero_labs':       False,
        'zero_omr':        False,
        'zero_kg':         False,
        'tasks':           ['mortality'],
        'w_mortality':     1.0,
        'w_los':           0.0,
        'w_readmission':   0.0,
        'w_progression':   0.0,
    },
    'single_los': {
        'group':           'single_task',
        'description':     'Train on LOS only',
        'zero_patient':    False,
        'zero_admission':  False,
        'use_dt_decay':    True,
        'use_gru':         True,
        'zero_labs':       False,
        'zero_omr':        False,
        'zero_kg':         False,
        'tasks':           ['los_7d'],
        'w_mortality':     0.0,
        'w_los':           1.0,
        'w_readmission':   0.0,
        'w_progression':   0.0,
    },
    'single_readmission': {
        'group':           'single_task',
        'description':     'Train on readmission only',
        'zero_patient':    False,
        'zero_admission':  False,
        'use_dt_decay':    True,
        'use_gru':         True,
        'zero_labs':       False,
        'zero_omr':        False,
        'zero_kg':         False,
        'tasks':           ['readmission'],
        'w_mortality':     0.0,
        'w_los':           0.0,
        'w_readmission':   1.0,
        'w_progression':   0.0,
    },
    'single_progression': {
        'group':           'single_task',
        'description':     'Train on progression only',
        'zero_patient':    False,
        'zero_admission':  False,
        'use_dt_decay':    True,
        'use_gru':         True,
        'zero_labs':       False,
        'zero_omr':        False,
        'zero_kg':         False,
        'tasks':           ['progression'],
        'w_mortality':     0.0,
        'w_los':           0.0,
        'w_readmission':   0.0,
        'w_progression':   1.0,
    },

    # Group 4: Loss weighting
    'equal_weights': {
        'group':           'loss_weighting',
        'description':     'All task weights = 1.0 (no tuning)',
        'zero_patient':    False,
        'zero_admission':  False,
        'use_dt_decay':    True,
        'use_gru':         True,
        'zero_labs':       False,
        'zero_omr':        False,
        'zero_kg':         False,
        'tasks':           ['mortality', 'los_7d', 'readmission', 'progression'],
        'w_mortality':     1.0,
        'w_los':           1.0,
        'w_readmission':   1.0,
        'w_progression':   1.0,
    },

    # Group 5: Input modalities
    'no_labs': {
        'group':           'input_modality',
        'description':     'Zero out lab panel embeddings in timeline',
        'zero_patient':    False,
        'zero_admission':  False,
        'use_dt_decay':    True,
        'use_gru':         True,
        'zero_labs':       True,
        'zero_omr':        False,
        'zero_kg':         False,
        'tasks':           ['mortality', 'los_7d', 'readmission', 'progression'],
        'w_mortality':     1.0,
        'w_los':           1.0,
        'w_readmission':   1.0,
        'w_progression':   1.0,
    },
    'no_omr': {
        'group':           'input_modality',
        'description':     'Zero out OMR vitals embeddings in timeline',
        'zero_patient':    False,
        'zero_admission':  False,
        'use_dt_decay':    True,
        'use_gru':         True,
        'zero_labs':       False,
        'zero_omr':        True,
        'zero_kg':         False,
        'tasks':           ['mortality', 'los_7d', 'readmission', 'progression'],
        'w_mortality':     1.0,
        'w_los':           1.0,
        'w_readmission':   1.0,
        'w_progression':   1.0,
    },
    'no_kg': {
        'group':           'input_modality',
        'description':     'Zero out KG-GAT admission embeddings in timeline',
        'zero_patient':    False,
        'zero_admission':  False,
        'use_dt_decay':    True,
        'use_gru':         True,
        'zero_labs':       False,
        'zero_omr':        False,
        'zero_kg':         True,
        'tasks':           ['mortality', 'los_7d', 'readmission', 'progression'],
        'w_mortality':     1.0,
        'w_los':           1.0,
        'w_readmission':   1.0,
        'w_progression':   1.0,
    },
}

# Base config — full model
BASE_CONFIG = {
    'group':           'base',
    'description':     'Full model — all components enabled',
    'zero_patient':    False,
    'zero_admission':  False,
    'use_dt_decay':    True,
    'use_gru':         True,
    'zero_labs':       False,
    'zero_omr':        False,
    'zero_kg':         False,
    'tasks':           ['mortality', 'los_7d', 'readmission', 'progression'],
    'w_mortality':     1.0,
    'w_los':           1.0,
    'w_readmission':   1.0,
    'w_progression':   1.0,
}


# Ablation-aware EHRModel
class AblationEHRModel(nn.Module):
    """
    EHRModel with ablation flags controlling which components are active.
    """

    EMBED_DIM   = 128
    HIDDEN_SIZE = 256
    STATIC_DIM  = 64
    PROJ_DIM    = 128
    N_DIAGNOSES = 200

    def __init__(self, config: dict, lambda_init: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.config = config

        # Δt decay
        self.log_lambda = nn.Parameter(torch.tensor(lambda_init).log())

        # GRU (always instantiated, conditionally used)
        self.gru = nn.GRU(
            input_size  = self.EMBED_DIM,
            hidden_size = self.HIDDEN_SIZE,
            num_layers  = 1,
            batch_first = True,
        )

        # Projection input dim depends on what's enabled
        if config['use_gru']:
            temporal_dim = self.HIDDEN_SIZE   # 256
        else:
            temporal_dim = 0                  # no GRU output

        concat_dim = temporal_dim + self.STATIC_DIM + self.STATIC_DIM  # up to 384

        self.proj = nn.Sequential(
            nn.Linear(concat_dim, self.PROJ_DIM),
            nn.LayerNorm(self.PROJ_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 4 heads — always present, masked in loss if task not active
        self.head_mortality   = nn.Linear(self.PROJ_DIM, 1)
        self.head_los         = nn.Linear(self.PROJ_DIM, 1)
        self.head_readmission = nn.Linear(self.PROJ_DIM, 1)
        self.head_progression = nn.Linear(self.PROJ_DIM, self.N_DIAGNOSES)

    def forward(self, batch):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        emb           = batch['emb'].clone()      # (B, T, 128)
        dt            = batch['dt']               # (B, T)
        lengths       = batch['lengths']          # (B,)
        patient_vec   = batch['patient_vec'].clone()    # (B, 64)
        admission_vec = batch['admission_vec'].clone()  # (B, 64)

        # Input modality ablations
        # Zero out specific event types using meta type info
        if self.config['zero_labs']:
            lab_mask = batch.get('lab_mask')    # (B, T) bool
            if lab_mask is not None:
                emb[lab_mask] = 0.0

        if self.config['zero_omr']:
            omr_mask = batch.get('omr_mask')
            if omr_mask is not None:
                emb[omr_mask] = 0.0

        if self.config['zero_kg']:
            kg_mask = batch.get('kg_mask')
            if kg_mask is not None:
                emb[kg_mask] = 0.0

        # Static ablations
        if self.config['zero_patient']:
            patient_vec = torch.zeros_like(patient_vec)

        if self.config['zero_admission']:
            admission_vec = torch.zeros_like(admission_vec)

        # Temporal: GRU with optional Δt decay
        if self.config['use_gru']:
            if self.config['use_dt_decay']:
                lam   = torch.nn.functional.softplus(self.log_lambda)
                decay = torch.exp(-lam * dt).unsqueeze(-1)
                emb   = emb * decay

            packed = pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            gru_out, _    = pad_packed_sequence(packed_out, batch_first=True)

            idx          = (lengths - 1).clamp(min=0)
            idx_expanded = idx.view(-1, 1, 1).expand(-1, 1, self.HIDDEN_SIZE)
            h_discharge  = gru_out.gather(1, idx_expanded).squeeze(1)  # (B, 256)

            combined = torch.cat([h_discharge, patient_vec, admission_vec], dim=-1)
        else:
            # No GRU — static features only
            combined = torch.cat([patient_vec, admission_vec], dim=-1)  # (B, 128)

        shared = self.proj(combined)   # (B, 128)

        return {
            'mortality':   self.head_mortality(shared),
            'los_7d':      self.head_los(shared),
            'readmission': self.head_readmission(shared),
            'progression': self.head_progression(shared),
        }


# Run one ablation
def run_ablation(name, config, data, pos_weights):
    print(f'\n{"="*60}')
    print(f'ABLATION: {name}')
    print(f'Description: {config["description"]}')
    print(f'{"="*60}')

    out_dir = Path(ABLATION_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / 'config.json', 'w') as f:
        json.dump({k: v for k, v in config.items()}, f, indent=2)

    # Build model
    model = AblationEHRModel(config).to(DEVICE)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Build loss with task weights from config
    criterion = EHRLoss(
        pos_weight_mortality   = pos_weights['mortality'].to(DEVICE),
        pos_weight_los         = pos_weights['los'].to(DEVICE),
        pos_weight_readmission = pos_weights['readmission'].to(DEVICE),
        pos_weight_progression = pos_weights['progression'].to(DEVICE),
        w_mortality   = config['w_mortality'],
        w_los         = config['w_los'],
        w_readmission = config['w_readmission'],
        w_progression = config['w_progression'],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=False
    )

    # Train
    history = train(
        model         = model,
        train_loader  = data['train_loader'],
        val_loader    = data['val_loader'],
        criterion     = criterion,
        optimizer     = optimizer,
        scheduler     = scheduler,
        n_epochs      = N_EPOCHS,
        patience      = PATIENCE,
        device        = DEVICE,
        checkpoint_dir= str(out_dir),
    )

    # Load best model and evaluate on test set
    best_path = out_dir / 'best_model.pt'
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_metrics = evaluate(model, data['test_loader'], criterion, DEVICE)

    # Save test results
    with open(out_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f'\n{name} TEST RESULTS:')
    print(f'  Mean AUROC:  {test_metrics["mean_auroc"]:.4f}')
    print(f'  Mortality:   {test_metrics["mortality"]:.4f}')
    print(f'  LOS:         {test_metrics["los_7d"]:.4f}')
    print(f'  Readmission: {test_metrics["readmission"]:.4f}')
    print(f'  Progression: {test_metrics["progression"]:.4f}')

    return test_metrics


# Comparison table
def print_comparison_table(all_results):
    """Print a formatted comparison table across all ablations."""

    print('\n' + '='*90)
    print('ABLATION COMPARISON TABLE')
    print('='*90)
    print(f'{"Ablation":<25} {"Group":<18} {"Mean":>6} {"Mort":>6} {"LOS":>6} {"Readm":>6} {"Prog":>6}')
    print('-'*90)

    # Sort by mean AUROC descending
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1].get('mean_auroc', 0),
        reverse=True
    )

    for name, metrics in sorted_results:
        group = ABLATIONS.get(name, BASE_CONFIG).get('group', 'base')
        print(
            f'{name:<25} {group:<18} '
            f'{metrics.get("mean_auroc", 0):.4f} '
            f'{metrics.get("mortality", 0):.4f} '
            f'{metrics.get("los_7d", 0):.4f} '
            f'{metrics.get("readmission", 0):.4f} '
            f'{metrics.get("progression", 0):.4f}'
        )

    print('='*90)

    # Save table to CSV
    rows = []
    for name, metrics in sorted_results:
        rows.append({
            'ablation':    name,
            'group':       ABLATIONS.get(name, BASE_CONFIG).get('group', 'base'),
            'mean_auroc':  metrics.get('mean_auroc', 0),
            'mortality':   metrics.get('mortality', 0),
            'los_7d':      metrics.get('los_7d', 0),
            'readmission': metrics.get('readmission', 0),
            'progression': metrics.get('progression', 0),
        })

    pd.DataFrame(rows).to_csv(
        Path(ABLATION_DIR) / 'ablation_results.csv', index=False
    )
    print(f'\nResults saved → {ABLATION_DIR}/ablation_results.csv')


# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', type=str, default=None,
                        help='Run specific ablation by name')
    parser.add_argument('--group', type=str, default=None,
                        help='Run all ablations in a group')
    parser.add_argument('--skip_base', action='store_true',
                        help='Skip base model (use if already trained)')
    args = parser.parse_args()

    Path(ABLATION_DIR).mkdir(exist_ok=True)

    # Load data
    print('Loading data...')
    import json as json_lib

    train_df = pd.read_csv(TRAIN_DF_PATH)
    val_df   = pd.read_csv(VAL_DF_PATH)
    test_df  = pd.read_csv(TEST_DF_PATH)

    for df in [train_df, val_df, test_df]:
        if 'los_7d' not in df.columns:
            df['los_7d'] = (df['length_of_stay'] >= 7).astype(float)

    with open(ADMISSION_NODES_PATH) as f:
        admission_nodes = json_lib.load(f)
    with open(DIAG_VOCAB_PATH) as f:
        diag_to_idx = json_lib.load(f)

    patient_cache   = torch.load(PATIENT_CACHE_PATH)
    admission_cache = torch.load(ADMISSION_CACHE_PATH)

    def make_dataset(df):
        return EHRDataset(
            admissions_df   = df,
            timeline_dir    = TIMELINE_DIR,
            admission_nodes = admission_nodes,
            diag_to_idx     = diag_to_idx,
            patient_cache   = patient_cache,
            admission_cache = admission_cache,
        )

    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
            collate_fn=ehr_collate_fn, num_workers=NUM_WORKERS,
            pin_memory=(DEVICE.type == 'cuda'),
        )

    data = {
        'train_loader': make_loader(make_dataset(train_df), shuffle=True),
        'val_loader':   make_loader(make_dataset(val_df),   shuffle=False),
        'test_loader':  make_loader(make_dataset(test_df),  shuffle=False),
    }

    # Pos weights
    pw_mort  = (train_df['inhospital_dead'] == 0).sum() / (train_df['inhospital_dead'] == 1).sum()
    pw_los   = (train_df['los_7d'] == 0).sum() / (train_df['los_7d'] == 1).sum()
    pw_readm = (train_df['readmission_30d'] == 0).sum() / (train_df['readmission_30d'] == 1).sum()
    prog_w   = torch.tensor(np.load(PROG_WEIGHTS_PATH), dtype=torch.float32)

    pos_weights = {
        'mortality':   torch.tensor([pw_mort],  dtype=torch.float32),
        'los':         torch.tensor([pw_los],   dtype=torch.float32),
        'readmission': torch.tensor([pw_readm], dtype=torch.float32),
        'progression': prog_w,
    }
    print(f'pos_weights — mort: {pw_mort:.2f}, los: {pw_los:.2f}, readm: {pw_readm:.2f}')

    # Select ablations to run
    if args.ablation:
        # Single ablation
        configs_to_run = {args.ablation: ABLATIONS[args.ablation]}
    elif args.group:
        # All ablations in a group
        configs_to_run = {
            k: v for k, v in ABLATIONS.items() if v['group'] == args.group
        }
    else:
        # All ablations
        configs_to_run = ABLATIONS.copy()

    # Always run base first unless skipped
    all_results = {}

    if not args.skip_base:
        base_result_path = Path(ABLATION_DIR) / 'base' / 'test_results.json'
        if base_result_path.exists():
            print('Base model results already exist, loading...')
            with open(base_result_path) as f:
                all_results['base'] = json_lib.load(f)
        else:
            all_results['base'] = run_ablation('base', BASE_CONFIG, data, pos_weights)

    # Run ablations
    for name, config in configs_to_run.items():
        result_path = Path(ABLATION_DIR) / name / 'test_results.json'

        # Skip if already done
        if result_path.exists():
            print(f'Skipping {name} — results already exist')
            with open(result_path) as f:
                all_results[name] = json_lib.load(f)
            continue

        all_results[name] = run_ablation(name, config, data, pos_weights)

    # Final comparison table
    print_comparison_table(all_results)