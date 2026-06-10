import pandas as pd
import random
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

def build_dataset():
    csv_path = r"d:\Study\Education\Projects\Thesis\data\viettel\combine\diagnosis_10.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # 1. Parse the ICD hierarchy
    # The first 3 characters represent the Parent Category (e.g., A00)
    df['parent_id'] = df['id'].astype(str).str[:3]
    
    # Separate Parents from Children
    parents = df[df['id'] == df['parent_id']].copy()
    children = df[df['id'] != df['parent_id']].copy()
    
    # Create a fast lookup dictionary for parents: { 'A00': 'Bệnh tả', ... }
    parent_dict = dict(zip(parents['parent_id'], parents['name_vi']))
    
    # Group children by parent_id
    children_grouped = children.groupby('parent_id')
    
    positives = []
    
    # 2. Generate all possible Positives
    for parent_id, group in children_grouped:
        if parent_id in parent_dict:
            parent_name = parent_dict[parent_id]
            for _, child_row in group.iterrows():
                child_name = child_row['name_vi']
                
                # Ensure we don't add trivial exact matches
                if str(parent_name).strip().lower() != str(child_name).strip().lower():
                    positives.append({
                        "query_term": parent_name,
                        "matching_term": child_name,
                        "label": 1
                    })
                
    # We want exactly 2500 positives for the 5000 limit
    random.seed(42)
    random.shuffle(positives)
    positives = positives[:2500]
    
    negatives = []
    
    # List of all children records for negative sampling
    all_children = children.to_dict('records')
    
    # 3. Generate exactly 2500 Negatives
    for pos in positives:
        parent_name = pos['query_term']
        
        # Find the original parent_id so we don't accidentally pick a true child
        original_parent_id = None
        for pid, pname in parent_dict.items():
            if pname == parent_name:
                original_parent_id = pid
                break
                
        # Randomly sample until we find a child from a DIFFERENT parent category
        while True:
            rand_child = random.choice(all_children)
            if rand_child['parent_id'] != original_parent_id:
                
                # Ensure no trivial exact matches just in case
                if str(parent_name).strip().lower() != str(rand_child['name_vi']).strip().lower():
                    negatives.append({
                        "query_term": parent_name,
                        "matching_term": rand_child['name_vi'],
                        "label": 0
                    })
                    break
                
    # Combine and shuffle
    dataset = positives + negatives
    random.shuffle(dataset)
    
    df_out = pd.DataFrame(dataset)
    
    out_path = r"d:\Study\Education\Projects\Thesis\data\viettel\combine\icd_pairwise_dataset.csv"
    df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
    
    print("="*50)
    print("ICD DATASET GENERATION COMPLETE")
    print("="*50)
    print(f"Total pairs generated: {len(df_out)}\n")
    print("Label Breakdown:")
    print(df_out['label'].value_counts().to_string())
    print(f"\nSaved to: {out_path}")

if __name__ == "__main__":
    build_dataset()
