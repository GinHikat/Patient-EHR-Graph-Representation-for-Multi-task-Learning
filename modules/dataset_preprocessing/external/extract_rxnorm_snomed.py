import pandas as pd
import os
import json

data_dir = "F:/Din/Study/Education/Projects/Thesis/data/UML"
optimized_path = os.path.join(data_dir, "MRCONSO_optimized.parquet")
mrrel_path = os.path.join(data_dir, "META", "MRREL.RRF")

print("Loading MRCONSO_optimized.parquet...")
df = pd.read_parquet(optimized_path)

# RXNORM: Extract ID and Name
print("Extracting RxNorm...")
rxnorm_df = df[df['SAB'] == 'RXNORM'][['CODE', 'STR']].drop_duplicates(subset=['CODE'])
rxnorm_df.rename(columns={'CODE': 'rxnorm_id', 'STR': 'name'}, inplace=True)
rxnorm_df.to_csv("rxnorm_terms.csv", index=False)
print(f"Saved {len(rxnorm_df)} RxNorm terms to rxnorm_terms.csv")


# SNOMED CT: Extract ID, Name, Aliases, Hierarchy
print("\nExtracting SNOMED CT...")
snomed = df[df['SAB'] == 'SNOMEDCT_US']

# Group by CODE to get the primary name and aliases
print("Grouping aliases...")
snomed_grouped = snomed.groupby('CODE')['STR'].apply(list).reset_index()

# We'll treat the first string as the primary 'name' and the rest as 'aliases'
snomed_grouped['name'] = snomed_grouped['STR'].apply(lambda x: x[0])
snomed_grouped['aliases'] = snomed_grouped['STR'].apply(lambda x: x[1:])
snomed_grouped = snomed_grouped.drop(columns=['STR'])

# Create a CUI -> SNOMED CODE mapping for the hierarchy lookup
cui_to_snomed = dict(zip(snomed['CUI'], snomed['CODE']))

# Process MRREL.RRF line by line to prevent Out-Of-Memory errors
print("Extracting Hierarchy from MRREL.RRF (this may take a few minutes)...")
parent_mapping = {code: [] for code in snomed_grouped['CODE']}

if os.path.exists(mrrel_path):
    with open(mrrel_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) > 10 and parts[10] == 'SNOMEDCT_US':
                rel = parts[3]
                if rel == 'PAR':  # Parent relationship
                    cui1 = parts[0] # The child concept
                    cui2 = parts[4] # The parent concept
                    
                    if cui1 in cui_to_snomed and cui2 in cui_to_snomed:
                        child_code = cui_to_snomed[cui1]
                        parent_code = cui_to_snomed[cui2]
                        
                        if parent_code not in parent_mapping[child_code]:
                            parent_mapping[child_code].append(parent_code)
else:
    print(f"Warning: MRREL.RRF not found at {mrrel_path}. Hierarchy will be empty.")

# Add hierarchy to our dataframe
snomed_grouped['parents'] = snomed_grouped['CODE'].map(parent_mapping)

snomed_grouped.rename(columns={'CODE': 'snomed_id'}, inplace=True)
snomed_grouped.to_csv("snomed_terms_full.csv", index=False)
print(f"Saved {len(snomed_grouped)} SNOMED CT terms with aliases and hierarchy to snomed_terms_full.csv")

print("Done!")
