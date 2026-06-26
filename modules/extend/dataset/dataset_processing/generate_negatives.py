import pandas as pd
import json
import re
import itertools

def get_indices(text, entity):
    if not entity or pd.isna(entity): return None
    # Try word boundary first
    matches = list(re.finditer(r'\b' + re.escape(str(entity)) + r'\b', text))
    if not matches:
        matches = list(re.finditer(re.escape(str(entity)), text))
    if matches:
        # Just return the first match for simplicity
        return [matches[0].start(), matches[0].end()]
    return None

import os

def create_negative_samples():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..', '..', '..'))
    csv_path = os.path.join(project_root, 'data', 'viettel', 'vietnamese_ner', 'training', 'english', 'unified_re.csv')
    
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    # Clean and lowercase
    df.columns = [col.lower() for col in df.columns]
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()

    # Group by input to avoid O(N^2) dataframe filtering in the loop
    grouped = df.groupby('input')
    
    enhanced_data = []
    
    print("Generating negative samples...")
    from tqdm import tqdm
    for text, sentence_df in tqdm(grouped, desc="Processing Sentences"):
        
        # Get all unique heads and targets in this sentence
        heads = sentence_df['head'].unique().tolist()
        targets = sentence_df['target'].unique().tolist()
        
        # Store positive pairs to check against
        positive_pairs = set()
        for _, row in sentence_df.iterrows():
            positive_pairs.add((row['head'], row['target']))
            # Add the positive row to our new data
            enhanced_data.append({
                'input': text,
                'head': row['head'],
                'target': row['target'],
                'rel': row['rel']
            })
            
        # Generate all possible combinations (permutations) of the entities in this sentence
        for h, t in itertools.product(heads, targets):
            if (h, t) not in positive_pairs:
                # This pair was NOT annotated with a relation, so it's a Negative Sample!
                enhanced_data.append({
                    'input': text,
                    'head': h,
                    'target': t,
                    'rel': 'None'  # Our new negative class
                })

    enhanced_df = pd.DataFrame(enhanced_data)
    print(f"Total rows before: {len(df)}")
    print(f"Total rows after (with negatives): {len(enhanced_df)}")
    
    # Now, we aggregate it into the index format you need for training
    print("Processing indices (this might take a minute)...")
    from tqdm import tqdm
    tqdm.pandas(desc="Finding character indices")
    
    def process_row(row):
        h_idx = get_indices(row['input'], row['head'])
        t_idx = get_indices(row['input'], row['target'])
        if h_idx and t_idx:
            return [h_idx, t_idx, row['rel']]
        return None

    enhanced_df['rel_data'] = enhanced_df.progress_apply(process_row, axis=1)
    enhanced_df = enhanced_df.dropna(subset=['rel_data'])

    print("Aggregating by input...")
    aggregated = enhanced_df.groupby('input')['rel_data'].apply(list).reset_index()
    aggregated['relations_json'] = aggregated['rel_data'].apply(json.dumps)
    aggregated = aggregated[['input', 'relations_json']]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..', '..', '..'))
    output_path = os.path.join(project_root, 'data', 'viettel', 'vietnamese_ner', 'training', 'english', 'unified_re_indices_with_negatives.csv')
    
    aggregated.to_csv(output_path, index=False)
    print(f"Saved new dataset to: {output_path}")

if __name__ == "__main__":
    create_negative_samples()
