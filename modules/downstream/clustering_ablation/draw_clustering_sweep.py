import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description='Draw Inertia and Silhouette Score on a Unified Plot')
    parser.add_argument('--summary_path', type=str, required=True, help='Path to the clustering_sweep_xxx_summary.csv')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.summary_path):
        print(f"ERROR: File not found: {args.summary_path}")
        return
        
    print(f"Reading sweep summary from {args.summary_path}...")
    df = pd.read_csv(args.summary_path)
    
    # Check columns
    required_cols = {'K', 'Inertia', 'Silhouette'}
    if not required_cols.issubset(df.columns):
        print(f"ERROR: CSV must contain columns: {required_cols}. Found: {df.columns.tolist()}")
        return

    # Use a professional style
    sns.set_theme(style="white")
    
    fig, ax1 = plt.subplots(figsize=(10, 6.5))
    
    # Colors
    color_inertia = '#1f77b4'      # Sleek blue
    color_silhouette = '#2ca02c'   # Sleek green for contrast
    
    # 1. Plot Inertia (Left axis)
    line1 = ax1.plot(df['K'], df['Inertia'], marker='o', markersize=8, color=color_inertia, 
                     linewidth=2.5, label='Inertia (Elbow)')
    ax1.set_xlabel('Number of Clusters (K)', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Inertia', color=color_inertia, fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_inertia, labelsize=11)
    ax1.set_xticks(df['K'])
    ax1.tick_params(axis='x', labelsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Plot Silhouette Score (Right axis)
    ax2 = ax1.twinx()
    line2 = ax2.plot(df['K'], df['Silhouette'], marker='s', markersize=8, color=color_silhouette, 
                     linewidth=2.5, label='Silhouette Score')
    ax2.set_ylabel('Silhouette Score', color=color_silhouette, fontsize=13, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_silhouette, labelsize=11)
    
    # Highlight max silhouette score
    max_idx = df['Silhouette'].idxmax()
    max_k = df.loc[max_idx, 'K']
    max_sil = df.loc[max_idx, 'Silhouette']
    line3 = ax2.plot(max_k, max_sil, marker='*', markersize=16, color='gold', 
                     markeredgecolor='darkorange', markeredgewidth=1.5,
                     linestyle='None', label=f'Max Silhouette ({max_k}: {max_sil:.4f})')
    
    # Combine legends cleanly
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, shadow=True, fontsize=11)
    
    plt.title('Clustering Sweep Analysis\n(Elbow Method & Silhouette Score Curves)', 
              fontsize=15, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    # Save path
    output_dir = os.path.dirname(args.summary_path)
    base_name = os.path.basename(args.summary_path).replace('.csv', '')
    output_path = os.path.join(output_dir, f"{base_name}_unified_plot.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Unified plot successfully generated and saved to: {output_path}")

if __name__ == '__main__':
    main()
