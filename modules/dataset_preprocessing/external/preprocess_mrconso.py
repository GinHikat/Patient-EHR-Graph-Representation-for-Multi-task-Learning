import os
import pandas as pd
import dotenv

dotenv.load_dotenv()

# Determine project root and data dir
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.getenv('DATA_DIR')
if not data_dir:
    data_dir = os.path.join(project_root, 'data')

def preprocess_mrconso():
    print("Starting MRCONSO preprocessing...")
    
    columns = [
        "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF",
        "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY",
        "CODE", "STR", "SRL", "SUPPRESS", "CVF"
    ]
    
    mrconso_path = os.path.join(data_dir, 'UML', "MRCONSO.RRF")
    output_path = os.path.join(data_dir, 'UML', "MRCONSO_optimized.parquet")
    
    if not os.path.exists(mrconso_path):
        print(f"Error: {mrconso_path} not found.")
        return

    print(f"Reading raw file from {mrconso_path}...")
    df_uml = pd.read_csv(
        mrconso_path,
        sep="|",
        header=None,
        names=columns,
        dtype=str,        # IMPORTANT: keep everything as string
        index_col=False,
        quoting=3         # avoid issues with quotes
    )
    
    print("Parsing and filtering data...")
    # Drop the last empty column if it exists (common in RRF files)
    if df_uml.columns[-1] == 'CVF' and df_uml['CVF'].isna().all():
        pass  
    # Sometimes there's an extra unnamed column due to trailing "|"
    if df_uml.shape[1] > len(columns):
        df_uml = df_uml.iloc[:, :len(columns)]
        
    df_uml = df_uml[df_uml['LAT'] == 'ENG']
    df_uml = df_uml[['CUI', 'SAB', 'CODE', 'STR']]
    
    list_sab = ['MSH', 'RXNORM', 'SNOMEDCT_US', 'ATC', 'DRUGBANK', 'OMIM', 'ICD10', 'ICD10CM', 'HPO', 'ICD9CM', 'CCSR_ICD10PCS', 'ICD10AE', 'CCSR_ICD10CM', 'ICD10AMAE', 'ICD10PCS']
    df_uml = df_uml[df_uml['SAB'].isin(list_sab)].dropna(subset=['CODE'])
    df_uml['STR'] = df_uml['STR'].str.title()
    
    diag_mappings = {
        'ICD10CM': 'ICD10',
        'ICD10AE': 'ICD10',
        'ICD10AMAE': 'ICD10'
    }
    df_uml['SAB'] = df_uml['SAB'].replace(diag_mappings)
    
    print(f"Saving optimized parquet to {output_path}...")
    df_uml.to_parquet(output_path, index=False)
    print("Optimization complete!")

if __name__ == "__main__":
    preprocess_mrconso()
