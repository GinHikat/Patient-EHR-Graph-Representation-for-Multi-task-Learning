import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
import os, sys
load_dotenv() 

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

data_dir = os.getenv('DATA_DIR')
base_data_dir = os.path.join(project_root, 'Thesis', 'data')
downstream_data_path = os.path.join(base_data_dir, 'downstream')

ADMISSIONS_PATH = os.path.join(downstream_data_path, "admission.parquet")
TRAIN_PIDS_OUT  = os.path.join(downstream_data_path, 'models', "split_train_pids.txt")
VAL_PIDS_OUT    = os.path.join(downstream_data_path, 'models', "split_val_pids.txt")
TEST_PIDS_OUT   = os.path.join(downstream_data_path, 'models', "split_test_pids.txt")
TRAIN_DF_OUT    = os.path.join(downstream_data_path, 'models', "train_df.csv")
VAL_DF_OUT      = os.path.join(downstream_data_path, 'models', "val_df.csv")
TEST_DF_OUT     = os.path.join(downstream_data_path, 'models', "test_df.csv")

RANDOM_STATE = 42
VAL_SIZE     = 0.10
TEST_SIZE    = 0.10

def split_dataset(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """
    Split admission dataframe into train/val/test at the patient level,
    stratified on whether the patient ever had an in-hospital death.

    Args:
        df           : full admissions dataframe with columns:
                       patient_id, inhospital_dead, length_of_stay,
                       readmission_30d, ...
        random_state : for reproducibility

    Returns:
        train_df, val_df, test_df : admission-level dataframes
        train_pids, val_pids, test_pids : patient ID arrays
    """

    n_before = len(df)
    df = df[df['length_of_stay'] > 0].copy()
    print(f"Dropped {n_before - len(df)} rows with negative/zero LOS")
    print(f"Remaining admissions: {len(df):,}")

    # Log-transform LOS
    df['los_log'] = np.log1p(df['length_of_stay'])

    # Patient-level stratification label
    # Did this patient EVER have an in-hospital death?
    patient_ever_dead = (
        df.groupby('patient_id')['inhospital_dead']
        .max()
        .reset_index()
        .rename(columns={'inhospital_dead': 'ever_dead'})
    )
    patient_ever_dead['ever_dead'] = patient_ever_dead['ever_dead'].fillna(0).astype(int)

    all_pids    = patient_ever_dead['patient_id'].values
    strat_label = patient_ever_dead['ever_dead'].values

    print(f"\nTotal patients: {len(all_pids):,}")
    print(f"Patients with ≥1 death: {strat_label.sum():,} "
          f"({100*strat_label.mean():.2f}%)")

    # First split: train vs temp (val + test)
    train_pids, temp_pids, train_strat, temp_strat = train_test_split(
        all_pids,
        strat_label,
        test_size=(VAL_SIZE + TEST_SIZE),
        stratify=strat_label,
        random_state=random_state,
    )

    # Second split: val vs test
    relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    val_pids, test_pids, _, _ = train_test_split(
        temp_pids,
        temp_strat,
        test_size=relative_test_size,
        stratify=temp_strat,
        random_state=random_state,
    )

    # Filter admission df by split
    train_pid_set = set(train_pids)
    val_pid_set   = set(val_pids)
    test_pid_set  = set(test_pids)

    train_df = df[df['patient_id'].isin(train_pid_set)].copy()
    val_df   = df[df['patient_id'].isin(val_pid_set)].copy()
    test_df  = df[df['patient_id'].isin(test_pid_set)].copy()

    return train_df, val_df, test_df, train_pids, val_pids, test_pids

def print_split_stats(train_df, val_df, test_df):
    """Print label distribution across splits to verify stratification."""

    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)

    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_pat = split_df['patient_id'].nunique()
        n_adm = len(split_df)
        mort  = split_df['inhospital_dead'].mean() * 100
        readm = split_df['readmission_30d'].mean() * 100
        los_m = split_df['length_of_stay'].median()

        print(f"\n{name}:")
        print(f"  Patients   : {n_pat:>8,}")
        print(f"  Admissions : {n_adm:>8,}")
        print(f"  Mortality  : {mort:>7.2f}%")
        print(f"  Readm 30d  : {readm:>7.2f}%")
        print(f"  LOS median : {los_m:>7.2f} days")

        # Absolute positive counts
        n_mort  = int(split_df['inhospital_dead'].sum())
        n_readm = int(split_df['readmission_30d'].sum())
        print(f"  Mortality positives  : {n_mort:,}")
        print(f"  Readmission positives: {n_readm:,}")

    print("\n" + "="*60)

def compute_pos_weights(train_df):
    """
    Compute pos_weight for BCEWithLogitsLoss per task.
    pos_weight = n_negative / n_positive

    Returns dict of FloatTensors ready to pass to nn.BCEWithLogitsLoss.
    """
    import torch

    n_total = len(train_df)

    # Mortality
    n_pos_mort  = int(train_df['inhospital_dead'].sum())
    n_neg_mort  = n_total - n_pos_mort
    pw_mort     = n_neg_mort / n_pos_mort

    # Readmission
    n_pos_readm = int(train_df['readmission_30d'].sum())
    n_neg_readm = n_total - n_pos_readm
    pw_readm    = n_neg_readm / n_pos_readm

    pos_weights = {
        'mortality':   torch.tensor([pw_mort],  dtype=torch.float32),
        'readmission': torch.tensor([pw_readm], dtype=torch.float32),
    }

    print("\nPos weights for BCEWithLogitsLoss:")
    print(f"  Mortality   pos_weight: {pw_mort:.2f}  "
          f"({n_pos_mort:,} pos / {n_neg_mort:,} neg)")
    print(f"  Readmission pos_weight: {pw_readm:.2f}  "
          f"({n_pos_readm:,} pos / {n_neg_readm:,} neg)")

    return pos_weights

def save_splits(train_df, val_df, test_df, train_pids, val_pids, test_pids):
    """Save pid lists and admission dataframes to disk."""

    # Patient ID files (one per line)
    np.savetxt(TRAIN_PIDS_OUT, train_pids, fmt='%s')
    np.savetxt(VAL_PIDS_OUT,   val_pids,   fmt='%s')
    np.savetxt(TEST_PIDS_OUT,  test_pids,  fmt='%s')
    print(f"\nSaved pid files:")
    print(f"  {TRAIN_PIDS_OUT}  ({len(train_pids):,} patients)")
    print(f"  {VAL_PIDS_OUT}    ({len(val_pids):,} patients)")
    print(f"  {TEST_PIDS_OUT}   ({len(test_pids):,} patients)")

    # Admission dataframes
    train_df.to_csv(TRAIN_DF_OUT, index=False)
    val_df.to_csv(VAL_DF_OUT,     index=False)
    test_df.to_csv(TEST_DF_OUT,   index=False)
    print(f"\nSaved admission dataframes:")
    print(f"  {TRAIN_DF_OUT}  ({len(train_df):,} admissions)")
    print(f"  {VAL_DF_OUT}    ({len(val_df):,} admissions)")
    print(f"  {TEST_DF_OUT}   ({len(test_df):,} admissions)")

if __name__ == "__main__":
    os.makedirs(os.path.join(downstream_data_path, 'models'), exist_ok=True)

    print(f"Loading admissions dataframe from {ADMISSIONS_PATH}...")
    df = pd.read_parquet(ADMISSIONS_PATH)
    print(f"Loaded {len(df):,} admissions, {df['patient_id'].nunique():,} patients")

    # Run split
    train_df, val_df, test_df, train_pids, val_pids, test_pids = split_dataset(df)

    # Print stats to verify stratification worked
    print_split_stats(train_df, val_df, test_df)

    # Compute pos_weights for training
    pos_weights = compute_pos_weights(train_df)

    # Save everything
    save_splits(train_df, val_df, test_df, train_pids, val_pids, test_pids)

    print("\nDone. Use these files in EHRDataset:")
    print("  train_pids = np.loadtxt('split_train_pids.txt', dtype=str)")
    print("  val_pids   = np.loadtxt('split_val_pids.txt',   dtype=str)")
    print("  test_pids  = np.loadtxt('split_test_pids.txt',  dtype=str)")