import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description='EHR Cluster Validation')
    parser.add_argument('--assignment_path', type=str, required=True, help='Path to cluster_assignments.csv')
    parser.add_argument('--train_df_path', type=str, required=True, help='Path to train_df.csv')
    parser.add_argument('--data_dir', type=str, default='data/Timeline', help='Path to data/Timeline directory')
    return parser.parse_args()

def calculate_purity(y_true, y_pred):
    # Use pandas crosstab as a robust replacement for contingency_matrix
    contingency = pd.crosstab(y_true, y_pred)
    return contingency.max().sum() / contingency.sum().sum()

def main():
    args = parse_args()
    output_dir = os.path.dirname(args.assignment_path)
    
    # Load Data
    print("Loading data for validation...")
    # Support both old 'adm_id' format and new 'patient_id' format
    assignments = pd.read_csv(args.assignment_path, dtype={'adm_id': str, 'pid': str, 'patient_id': str})
    train_df = pd.read_csv(args.train_df_path, dtype={'id': str, 'patient_id': str})
    
    # Determine Level
    is_patient_level = 'patient_id' in assignments.columns
    
    if is_patient_level:
        print("Detected patient-level clusters. Aggregating patient utilization profiles...")
        # Aggregating training data to patient level
        train_df_patient = train_df.groupby('patient_id').agg({
            'id': 'count',              # Number of admissions for this patient
            'length_of_stay': 'mean',   # Average stay length across their visits
            'inhospital_dead': 'max',   # Necessary for KM plotting
            'drg_severity': 'max'       # Max severity recorded
        }).rename(columns={'id': 'num_admissions'}).reset_index()
        df = assignments.merge(train_df_patient, on='patient_id')
        id_col = 'patient_id'
        
        # Load patient age demographics from patient.parquet
        patient_meta_path = os.path.join(args.data_dir, 'patient.parquet')
        if os.path.exists(patient_meta_path):
            patient_meta = pd.read_parquet(patient_meta_path)
            patient_meta['id'] = patient_meta['id'].astype(str)
            df = df.merge(patient_meta[['id', 'age']], left_on='patient_id', right_on='id', how='left')
        
        # New Summary Aggregation (Clean report for USER)
        summary = df.groupby('cluster').agg({
            id_col: 'count',
            'num_admissions': 'mean',
            'length_of_stay': 'mean',
            'age': 'mean' if 'age' in df.columns else id_col
        }).rename(columns={
            id_col: 'N_Patients', 
            'num_admissions': 'Avg_Admissions', 
            'length_of_stay': 'Mean_LOS',
            'age': 'Avg_Age'
        })
        if 'id' in summary.columns: # fallback if age not in df
            summary = summary.drop(columns=['id'])
    else:
        print("Detected admission-level clusters.")
        df = assignments.merge(train_df, left_on='adm_id', right_on='id')
        id_col = 'id'
        
        # Load patient age demographics from patient.parquet
        patient_meta_path = os.path.join(args.data_dir, 'patient.parquet')
        if os.path.exists(patient_meta_path):
            patient_meta = pd.read_parquet(patient_meta_path)
            patient_meta['id'] = patient_meta['id'].astype(str)
            df = df.merge(patient_meta[['id', 'age']], left_on='patient_id', right_on='id', how='left')
            
        summary = df.groupby('cluster').agg({
            id_col: 'count',
            'inhospital_dead': 'mean',
            'length_of_stay': 'mean',
            'readmission_30d': 'mean',
            'age': 'mean' if 'age' in df.columns else id_col
        }).rename(columns={
            id_col: 'N_Admissions', 
            'inhospital_dead': 'Mortality%', 
            'readmission_30d': 'Readm%',
            'age': 'Avg_Age'
        })
        summary['Mortality%'] *= 100
        summary['Readm%'] *= 100
        if 'id' in summary.columns: # fallback if age not in df
            summary = summary.drop(columns=['id'])

    # Outcome Distribution (Section 4.2)
    print("\n" + "="*30)
    print(f"CLUSTER OUTCOME DISTRIBUTIONS ({'PATIENT' if is_patient_level else 'ADMISSION'} LEVEL)")
    print("="*30)
    
    print(summary)
    summary.to_csv(os.path.join(output_dir, 'cluster_outcome_summary.csv'))
    
    # Kaplan-Meier Survival Analysis
    print("\nRunning Kaplan-Meier survival analysis...")
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(12, 8))
    
    for cluster in sorted(df['cluster'].unique()):
        c_df = df[df['cluster'] == cluster]
        kmf.fit(c_df['length_of_stay'], event_observed=c_df['inhospital_dead'], label=f'Cluster {cluster}')
        kmf.plot_survival_function()
        
    plt.title('Kaplan-Meier Survival Curves per Cluster')
    plt.xlabel('Days in Hospital')
    plt.ylabel('Survival Probability')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'km_survival_clusters.png'))
    
    # Log-rank test
    results = multivariate_logrank_test(df['length_of_stay'], df['cluster'], df['inhospital_dead'])
    print(f"\nMultivariate Log-rank Test p-value: {results.p_value:.6f}")

    # --- NEW: Clinical Interpretation (Diagnoses & Drugs) ---
    print("\nExtracting common clinical features per cluster...")
    with open(os.path.join(args.data_dir, 'admission_nodes.json')) as f:
        nodes = json.load(f)
    with open(os.path.join(args.data_dir, 'top200_diag_vocab.json')) as f:
        diag_vocab = {v: k for k, v in json.load(f).items()}
    with open(os.path.join(args.data_dir, 'top50_drug_vocab.json')) as f:
        drug_vocab = {v: k for k, v in json.load(f).items()}

    cluster_interpret = {}
    for cluster in sorted(df['cluster'].unique()):
        if is_patient_level:
            # Find all admissions for these patients
            c_pids = df[df['cluster'] == cluster]['patient_id'].tolist()
            c_adm_ids = train_df[train_df['patient_id'].isin(c_pids)]['id'].tolist()
        else:
            c_adm_ids = df[df['cluster'] == cluster]['adm_id'].tolist()
        
        all_diags = []
        all_drugs = []
        for aid in c_adm_ids:
            if aid in nodes:
                all_diags.extend([diag_vocab.get(i, f"D{i}") for i in nodes[aid].get('diagnoses', [])])
                all_drugs.extend([drug_vocab.get(i, f"Rx{i}") for i in nodes[aid].get('drugs', [])])
        
        top_diags = pd.Series(all_diags).value_counts().head(10).index.tolist()
        top_drugs = pd.Series(all_drugs).value_counts().head(10).index.tolist()
        cluster_interpret[cluster] = {'diags': top_diags, 'drugs': top_drugs}

    # --- NEW: Clustering Metrics ---
    # Using 'drg_severity' as the ground truth label for external validation
    print("\nCalculating clustering metrics (Ground Truth = DRG Severity)...")
    valid_mask = df['drg_severity'].notna()
    y_true = df[valid_mask]['drg_severity']
    y_pred = df[valid_mask]['cluster']
    
    purity = calculate_purity(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    print(f"Purity: {purity:.4f}")
    print(f"NMI:    {nmi:.4f}")
    print(f"ARI:    {ari:.4f}")

    # --- Save All Stats ---
    with open(os.path.join(output_dir, 'validation_stats.txt'), 'w') as f:
        f.write(f"=== CLUSTERING VALIDATION RESULTS ===\n")
        f.write(f"Log-rank p-value: {results.p_value:.10f}\n")
        f.write(f"Purity (vs DRG Severity): {purity:.4f}\n")
        f.write(f"NMI: {nmi:.4f}\n")
        f.write(f"ARI: {ari:.4f}\n\n")
        
        f.write("=== CLUSTER CLINICAL PROFILES ===\n")
        for c, data in cluster_interpret.items():
            f.write(f"\nCluster {c}:\n")
            f.write(f"  Top Diagnoses: {', '.join(data['diags'])}\n")
            f.write(f"  Top Drugs:     {', '.join(data['drugs'])}\n")
            
        f.write("\n\n=== OUTCOME SUMMARY ===\n")
        f.write(summary.to_string())

    print(f"\nValidation complete. Detailed results saved in {output_dir}")

if __name__ == "__main__":
    main()
