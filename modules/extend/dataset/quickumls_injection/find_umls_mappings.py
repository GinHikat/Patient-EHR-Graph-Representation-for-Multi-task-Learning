import pandas as pd
import os
from tqdm import tqdm
import dotenv
dotenv.load_dotenv()

data_dir = os.getenv('DATA_DIR')

# Absolute Paths
umls_parquet = os.path.join(data_dir, 'UML', 'MRCONSO_optimized.parquet')
diag_csv = r'data\viettel\combine\diagnosis_10.csv'
proc_csv = r'data\viettel\combine\procedure_9.csv'
output_txt = r'data\viettel\combine\umls_match_report.txt'

def main():
    print("Loading CSV files to extract target codes...")
    diag_df = pd.read_csv(diag_csv, dtype=str)
    proc_df = pd.read_csv(proc_csv, dtype=str)
    
    import re
    # Clean and collect target codes (strip dots, daggers, asterisks, etc.)
    target_diag_codes = set(diag_df['id'].dropna().str.replace(r'[^a-zA-Z0-9]', '', regex=True))
    target_proc_codes = set(proc_df['id'].dropna().str.replace(r'[^a-zA-Z0-9]', '', regex=True))
    all_target_codes = target_diag_codes.union(target_proc_codes)
    
    print(f"Total unique ICD/Procedure codes to find: {len(all_target_codes)}")
    
    # We only care about sources that are ICD related
    valid_sabs = {'ICD10', 'ICD10CM', 'ICD10AM', 'ICD9CM'}
    
    print("Loading optimized MRCONSO parquet file (this is incredibly fast!)...")
    mrconso_df = pd.read_parquet(umls_parquet, engine='pyarrow')
    
    print("Filtering and mapping codes...")
    # Filter for ICD SABs
    icd_df = mrconso_df[mrconso_df['SAB'].isin(valid_sabs)].copy()
    
    # Clean the codes in the database (e.g. A00.9 -> A009)
    icd_df['CLEAN_CODE'] = icd_df['CODE'].astype(str).str.replace('.', '', regex=False)
    
    # Create the mapping dictionary
    # Dropping duplicates keeps the first CUI encountered for each code
    mapping_df = icd_df.drop_duplicates(subset=['CLEAN_CODE'])
    code_to_cui = dict(zip(mapping_df['CLEAN_CODE'], mapping_df['CUI']))

    # Now verify the matches
    print("\nVerifying Matches...")
    
    diag_matched = 0
    proc_matched = 0
    
    with open(output_txt, 'w', encoding='utf-8') as out_f:
        out_f.write("=== UMLS MATCHING REPORT ===\n")
        
        # Diagnosis
        out_f.write("\n[DIAGNOSIS_10]\n")
        for _, row in tqdm(diag_df.iterrows(), total=len(diag_df), desc="Diagnoses"):
            raw_code = str(row['id']) if pd.notna(row['id']) else ""
            code = re.sub(r'[^a-zA-Z0-9]', '', raw_code)
            
            if code in code_to_cui:
                diag_matched += 1
                out_f.write(f"MATCH: {code} -> {code_to_cui[code]} | {row['name_vi']}\n")
            else:
                out_f.write(f"MISSING: {code} | {row.get('name_vi', '')}\n")
                
        # Procedures
        out_f.write("\n[PROCEDURE_9]\n")
        for _, row in tqdm(proc_df.iterrows(), total=len(proc_df), desc="Procedures"):
            raw_code = str(row['id']) if pd.notna(row['id']) else ""
            code = re.sub(r'[^a-zA-Z0-9]', '', raw_code)
            
            if code in code_to_cui:
                proc_matched += 1
                out_f.write(f"MATCH: {code} -> {code_to_cui[code]} | {row['term_vi']}\n")
            else:
                out_f.write(f"MISSING: {code} | {row.get('term_vi', '')}\n")
        
        summary = f"""
=== SUMMARY ===
Diagnosis Matched: {diag_matched} / {len(diag_df)} ({(diag_matched/len(diag_df)*100) if len(diag_df) > 0 else 0:.2f}%)
Procedures Matched: {proc_matched} / {len(proc_df)} ({(proc_matched/len(proc_df)*100) if len(proc_df) > 0 else 0:.2f}%)
Total Unique Codes Mapped: {len(code_to_cui)}
================
"""
        out_f.write(summary)
        print(summary)
        print(f"Detailed results saved to: {output_txt}")

if __name__ == '__main__':
    main()
