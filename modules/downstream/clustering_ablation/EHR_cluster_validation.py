import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='EHR Cluster Validation')
    parser.add_argument('--assignment_path', type=str, required=True, help='Path to cluster_assignments.csv')
    parser.add_argument('--train_df_path', type=str, required=True, help='Path to train_df.csv')
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = os.path.dirname(args.assignment_path)
    
    # Load Data
    print("Loading data for validation...")
    assignments = pd.read_csv(args.assignment_path)
    train_df = pd.read_csv(args.train_df_path, dtype={'id': str, 'patient_id': str})
    
    # Merge assignments with labels
    # Note: Using 'id' as adm_id
    df = assignments.merge(train_df, left_on='adm_id', right_on='id')
    
    # Outcome Distribution (Section 4.2)
    print("\n" + "="*30)
    print("CLUSTER OUTCOME DISTRIBUTIONS")
    print("="*30)
    
    summary = df.groupby('cluster').agg({
        'id': 'count',
        'inhospital_dead': 'mean',
        'length_of_stay': 'median',
        'readmission_30d': 'mean'
    }).rename(columns={'id': 'N', 'inhospital_dead': 'Mortality%', 'readmission_30d': 'Readm%'})
    
    summary['Mortality%'] *= 100
    summary['Readm%'] *= 100
    
    print(summary)
    summary.to_csv(os.path.join(output_dir, 'cluster_outcome_summary.csv'))
    
    # Kaplan-Meier Survival Analysis (Section 4.3)
    print("\nRunning Kaplan-Meier survival analysis...")
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 7))
    
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
    
    with open(os.path.join(output_dir, 'validation_stats.txt'), 'w') as f:
        f.write(f"Log-rank p-value: {results.p_value:.10f}\n")
        f.write("\nCluster Summary:\n")
        f.write(summary.to_string())

    print(f"\nValidation complete. Files saved in {output_dir}")

if __name__ == "__main__":
    main()
