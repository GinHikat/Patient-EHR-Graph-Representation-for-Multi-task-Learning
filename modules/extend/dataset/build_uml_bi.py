import pandas as pd
import os
import re
import shutil
import time

# Paths
original_uml_dir = r'F:\Din\Study\Education\Projects\Thesis\data\UML'
new_uml_bi_dir = r'F:\Din\Study\Education\Projects\Thesis\data\UML_bi'

umls_parquet = os.path.join(original_uml_dir, 'MRCONSO_optimized.parquet')
diag_csv = r'd:\Study\Education\Projects\Thesis\data\viettel\combine\diagnosis_10.csv'
proc_csv = r'd:\Study\Education\Projects\Thesis\data\viettel\combine\procedure_9.csv'

def copy_with_progress(src, dst):
    """Copies a massive file and prints progress based on file size."""
    total_size = os.path.getsize(src)
    copied = 0
    start_time = time.time()
    
    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
        while True:
            # Copy in 100MB chunks
            buf = fsrc.read(1024 * 1024 * 100)
            if not buf:
                break
            fdst.write(buf)
            copied += len(buf)
            
            # Print progress every loop
            percent = (copied / total_size) * 100
            elapsed = time.time() - start_time
            speed = (copied / (1024*1024)) / elapsed if elapsed > 0 else 0
            print(f"\rCopying {os.path.basename(src)}: {percent:.1f}% ({copied/(1024*1024*1024):.2f} GB / {total_size/(1024*1024*1024):.2f} GB) - {speed:.1f} MB/s", end='')
    print("\nCopy complete!")

def main():
    os.makedirs(new_uml_bi_dir, exist_ok=True)
    
    orig_conso = os.path.join(original_uml_dir, 'MRCONSO.RRF')
    orig_sty = os.path.join(original_uml_dir, 'META', 'MRSTY.RRF')
    
    bi_conso = os.path.join(new_uml_bi_dir, 'MRCONSO.RRF')
    bi_sty = os.path.join(new_uml_bi_dir, 'MRSTY.RRF')
    
    print("--- STEP 1: Creating UML_bi Workspace ---")
    if not os.path.exists(bi_conso):
        print("Copying MRCONSO.RRF (This may take 5-10 minutes depending on your SSD)...")
        copy_with_progress(orig_conso, bi_conso)
    else:
        print("MRCONSO.RRF already exists in UML_bi, skipping copy.")
        
    if not os.path.exists(bi_sty):
        print("Copying MRSTY.RRF...")
        shutil.copy2(orig_sty, bi_sty)
        print("MRSTY.RRF copied!")
    else:
        print("MRSTY.RRF already exists in UML_bi, skipping copy.")

    print("\n--- STEP 2: Loading Data for Injection ---")
    diag_df = pd.read_csv(diag_csv, dtype=str)
    proc_df = pd.read_csv(proc_csv, dtype=str)
    
    mrconso_df = pd.read_parquet(umls_parquet, engine='pyarrow')
    valid_sabs = {'ICD10', 'ICD10CM', 'ICD10AM', 'ICD9CM'}
    icd_df = mrconso_df[mrconso_df['SAB'].isin(valid_sabs)].copy()
    icd_df['CLEAN_CODE'] = icd_df['CODE'].astype(str).str.replace('.', '', regex=False)
    mapping_df = icd_df.drop_duplicates(subset=['CLEAN_CODE'])
    code_to_cui = dict(zip(mapping_df['CLEAN_CODE'], mapping_df['CUI']))

    print("\n--- STEP 3: Injecting Vietnamese Terms ---")
    seen_cuis = set()
    injected_count = 0
    fallback_count = 0

    # Open the newly copied files in APPEND mode ('a')
    with open(bi_conso, 'a', encoding='utf-8') as f_conso, open(bi_sty, 'a', encoding='utf-8') as f_sty:
        
        # 1. Process Diagnoses
        for _, row in diag_df.iterrows():
            if pd.isna(row['id']) or pd.isna(row['name_vi']): continue
            
            clean_code = re.sub(r'[^a-zA-Z0-9]', '', str(row['id']))
            term_vi = str(row['name_vi']).strip().replace('|', '')
            
            # If real CUI exists, we ONLY append to MRCONSO
            if clean_code in code_to_cui:
                cui = code_to_cui[clean_code]
                f_conso.write(f"{cui}|ENG||||||||||ICD10VN|||{term_vi}||||\n")
                injected_count += 1
            else:
                # If missing, we must generate a fake CUI and append to BOTH files
                cui = f"C_DIAG_{clean_code}"
                f_conso.write(f"{cui}|ENG||||||||||ICD10VN|||{term_vi}||||\n")
                if cui not in seen_cuis:
                    f_sty.write(f"{cui}|T047||Disease or Syndrome|||\n")
                    seen_cuis.add(cui)
                fallback_count += 1

        # 2. Process Procedures
        for _, row in proc_df.iterrows():
            if pd.isna(row['id']) or pd.isna(row['term_vi']): continue
            
            clean_code = re.sub(r'[^a-zA-Z0-9]', '', str(row['id']))
            term_vi = str(row['term_vi']).strip().replace('|', '')
            
            if clean_code in code_to_cui:
                cui = code_to_cui[clean_code]
                f_conso.write(f"{cui}|ENG||||||||||PROC9VN|||{term_vi}||||\n")
                injected_count += 1
            else:
                cui = f"C_PROC_{clean_code}"
                f_conso.write(f"{cui}|ENG||||||||||PROC9VN|||{term_vi}||||\n")
                if cui not in seen_cuis:
                    f_sty.write(f"{cui}|T061||Therapeutic or Preventive Procedure|||\n")
                    seen_cuis.add(cui)
                fallback_count += 1

    print(f"\nSUCCESS! Bilingual UMLS Database created safely at: {new_uml_bi_dir}")
    print(f"Injected {injected_count} true global CUIs and {fallback_count} local fallback CUIs.")
    print("\n--- STEP 4: Build the Final QuickUMLS Index ---")
    print("Run this exact command to compile your massive unified database:")
    print(f"python -m quickumls.install {new_uml_bi_dir} {new_uml_bi_dir}\\quickumls_index")

if __name__ == '__main__':
    main()
