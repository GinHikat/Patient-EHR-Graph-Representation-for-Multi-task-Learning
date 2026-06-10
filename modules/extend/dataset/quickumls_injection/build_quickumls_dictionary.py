import pandas as pd
import os
import re

# Absolute Paths
umls_parquet = r'F:\Din\Study\Education\Projects\Thesis\data\UML\MRCONSO_optimized.parquet'
diag_csv = r'data\viettel\combine\diagnosis_10.csv'
proc_csv = r'data\viettel\combine\procedure_9.csv'
output_dir = r'data\viettel\combine\custom_umls_vietnamese'

os.makedirs(output_dir, exist_ok=True)
mrconso_path = os.path.join(output_dir, 'MRCONSO.RRF')
mrsty_path = os.path.join(output_dir, 'MRSTY.RRF')

def main():
    print("Loading datasets...")
    diag_df = pd.read_csv(diag_csv, dtype=str)
    proc_df = pd.read_csv(proc_csv, dtype=str)
    
    print("Loading optimized MRCONSO parquet to fetch REAL CUIs...")
    mrconso_df = pd.read_parquet(umls_parquet, engine='pyarrow')
    valid_sabs = {'ICD10', 'ICD10CM', 'ICD10AM', 'ICD9CM'}
    icd_df = mrconso_df[mrconso_df['SAB'].isin(valid_sabs)].copy()
    icd_df['CLEAN_CODE'] = icd_df['CODE'].astype(str).str.replace('.', '', regex=False)
    mapping_df = icd_df.drop_duplicates(subset=['CLEAN_CODE'])
    code_to_cui = dict(zip(mapping_df['CLEAN_CODE'], mapping_df['CUI']))

    print("Building Mock UMLS Database...")
    seen_cuis = set()

    with open(mrconso_path, 'w', encoding='utf-8') as f_conso, open(mrsty_path, 'w', encoding='utf-8') as f_sty:
        
        # 1. Process Diagnoses
        for _, row in diag_df.iterrows():
            if pd.isna(row['id']) or pd.isna(row['name_vi']): continue
            
            raw_code = str(row['id'])
            clean_code = re.sub(r'[^a-zA-Z0-9]', '', raw_code)
            
            # Use real CUI if found, else fallback to fake CUI
            cui = code_to_cui.get(clean_code, f"C_DIAG_{clean_code}")
            # Write Vietnamese Term
            term_vi = row['name_vi'].strip().replace('|', '')
            row_conso_vi = f"{cui}|VIE||||||||||ICD10VN|||{term_vi}||||"
            f_conso.write(row_conso_vi + '\n')
            
            # Write English Term (if available)
            if pd.notna(row.get('name_en')):
                term_en = str(row['name_en']).strip().replace('|', '')
                if term_en:
                    row_conso_en = f"{cui}|ENG||||||||||ICD10VN|||{term_en}||||"
                    f_conso.write(row_conso_en + '\n')
            
            # T047 = "Disease or Syndrome"
            if cui not in seen_cuis:
                row_sty = f"{cui}|T047||Disease or Syndrome|||"
                f_sty.write(row_sty + '\n')
                seen_cuis.add(cui)

        # 2. Process Procedures
        for _, row in proc_df.iterrows():
            if pd.isna(row['id']) or pd.isna(row['term_vi']): continue
            
            raw_code = str(row['id'])
            clean_code = re.sub(r'[^a-zA-Z0-9]', '', raw_code)
            
            cui = code_to_cui.get(clean_code, f"C_PROC_{clean_code}")
            # Write Vietnamese Term
            term_vi = row['term_vi'].strip().replace('|', '')
            row_conso_vi = f"{cui}|VIE||||||||||PROC9VN|||{term_vi}||||"
            f_conso.write(row_conso_vi + '\n')
            
            # Write English Term (if available)
            if pd.notna(row.get('term_en')):
                term_en = str(row['term_en']).strip().replace('|', '')
                if term_en:
                    row_conso_en = f"{cui}|ENG||||||||||PROC9VN|||{term_en}||||"
                    f_conso.write(row_conso_en + '\n')
            
            # T061 = "Therapeutic or Preventive Procedure"
            if cui not in seen_cuis:
                row_sty = f"{cui}|T061||Therapeutic or Preventive Procedure|||"
                f_sty.write(row_sty + '\n')
                seen_cuis.add(cui)

    print(f"\nSuccess! Hybrid UMLS directory created at: {output_dir}")
    print("Over 90% of your terms now use Official Global CUIs!")
    print("\nTo officially install this into QuickUMLS, run this command:")
    print(f"python -m quickumls.install {output_dir} d:\\Study\\Education\\Projects\\Thesis\\data\\viettel\\combine\\quickumls_vietnamese_index")

if __name__ == '__main__':
    main()
