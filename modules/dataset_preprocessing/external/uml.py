import pandas as pd
import numpy as np
from quickumls import QuickUMLS
import duckdb
import dotenv
from tqdm import tqdm
dotenv.load_dotenv()

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

quick_umls_path = os.getenv('QUICKUMLS_PATH')
data_dir = os.getenv('DATA_DIR')
base_data_dir = os.path.join(project_root, 'Thesis', 'data')

# Initial load from MRCONSO.RRF
columns = [
    "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF",
    "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY",
    "CODE", "STR", "SRL", "SUPPRESS", "CVF"
]

# Load MRCONSO.RRF
uml = pd.read_csv(
    os.path.join(data_dir, 'UML', "MRCONSO.RRF"),
    sep="|",
    header=None,
    names=columns,
    dtype=str,        # IMPORTANT: keep everything as string
    index_col=False,
    quoting=3         # avoid issues with quotes
)

# Drop the last empty column if it exists (common in RRF files)
if uml.columns[-1] == 'CVF' and uml['CVF'].isna().all():
    pass  
# Sometimes there's an extra unnamed column due to trailing "|"
if uml.shape[1] > len(columns):
    uml = uml.iloc[:, :len(columns)]

uml = uml[uml['LAT'] == 'ENG']
uml = uml[['CUI', 'SAB', 'CODE', 'STR']]

list_sab = ['MSH', 'RXNORM', 'SNOMEDCT_US', 'ATC', 'DRUGBANK', 'OMIM', 'ICD10', 'ICD10CM', 'HPO', 'ICD9CM', 'CCSR_ICD10PCS', 'ICD10AE', 'CCSR_ICD10CM', 'ICD10AMAE', 'ICD10PCS']

uml = uml[uml['SAB'].isin(list_sab)].dropna(subset = 'CODE')
uml['STR'] = uml['STR'].str.title()

diag_mappings = {
    'ICD10CM': 'ICD10',
    'ICD10AE': 'ICD10',
    'ICD10AMAE': 'ICD10'
}
uml['SAB'] = uml['SAB'].replace(diag_mappings)

# Start working with Index

# nlp = spacy.load("en_ner_bc5cdr_md")
# Initialize matcher with lower threshold for better recall
matcher = QuickUMLS(quick_umls_path, window=5)
tui_mapping = pd.read_parquet(os.path.join(data_dir, "UML", "META", 'tui_mapping.parquet'))
if 'tui' in tui_mapping.columns:
    tui_mapping = tui_mapping.set_index('tui')

def spacy_quickumls(text):
    
    # SciSpacy
    # doc = nlp(text)
    # data = [(ent.text, ent.label_) for ent in doc.ents]
    # df_ent = pd.DataFrame(data, columns=["term", "label"])
    
    # QuickUMLS on FULL TEXT
    results = matcher.match(text)

    flat_data = [item for sublist in results for item in sublist]
    
    if not flat_data:
        return pd.DataFrame(columns=['text', 'term', 'cui', 'similarity', 'type'])

    df = pd.DataFrame(flat_data)
    
    # Return empty DF with expected columns if no matches found
    if df.empty:
        return pd.DataFrame(columns=['text', 'term', 'cui', 'similarity', 'type'])

    def get_semantic_type(sem_set):
        if not isinstance(sem_set, (set, list)):
            return None
        # Return the first matching semantic type name
        for tui in sem_set:
            if tui in tui_mapping.index:
                res = tui_mapping.loc[tui, 'sty']
                return res.iloc[0] if isinstance(res, pd.Series) else res
        return None

    # Gracefully handle cases where 'semtypes' might be missing
    if 'semtypes' in df.columns:
        df['type'] = df['semtypes'].apply(get_semantic_type)
    else:
        df['type'] = None

    # Ensure all expected columns exist before sub-selecting
    for col in ['ngram', 'term', 'cui', 'similarity', 'type']:
        if col not in df.columns:
            df[col] = None

    df = df[['ngram', 'term', 'cui', 'similarity', 'type']]
    df.columns = ['text', 'term', 'cui', 'similarity', 'type']

    return df

# Process MRSTY for CUI-TUI mapping
def cui_tui_mapping():
    mrsty_path = os.path.join(data_dir, "UML", "META", "MRSTY.RRF")

    df_sty = pd.read_csv(
        mrsty_path, 
        sep='|', 
        header=None, 
        names=['cui', 'tui', 'stn', 'sty', 'atui', 'cvf', 'trailing'],
        index_col=False
    )

    df_sty = df_sty[['cui', 'tui', 'sty']]

## Use QuickUML to map between DrugBank Indication to Diagnosis (drug_df is the drugbank dataset)
def indication_to_diagnosis_uml():

    checkpoint = pd.read_csv(os.path.join(base_data_dir, 'checkpoint.csv'), quoting = 1, on_bad_lines='skip')
    done = checkpoint['id'].values
    print(f'{len(done)} done')

    diagnosis_types = [
        'Disease or Syndrome', 
        'Injury or Poisoning', 
        'Finding', 
        'Sign or Symptom',
        'Neoplastic Process',            # Cancers/Tumors
        'Mental or Behavioral Dysfunction', # Psychiatric conditions
        'Pathologic Function',           # Disease mechanisms
        'Congenital Abnormality',        # Genetic/birth defects
        'Anatomical Abnormality'         # Structural disease phenotypes
    ]

    noise_words = ['indicated', 'major', 'various', 'today']

    for i in tqdm(drug_df.index, desc='Extracting diagnosis from Indication:'):

        if drug_df.loc[i, 'id'] in done:
            continue
        else: 
            text = drug_df.loc[i, 'indication']

            df_diag = spacy_quickumls(text)
            df_diag = df_diag[~df_diag['term'].str.lower().isin(noise_words)]
            df_diag = df_diag[df_diag['type'].isin(diagnosis_types)]
            df_diag = df_diag.sort_values(['term', 'similarity'])
            df_diag = df_diag.drop_duplicates('term', keep='first')

            drug_df.loc[i, 'related_diagnosis'] = ', '.join(df_diag['cui'])

            # Append completed row to checkpoint
            checkpoint_path = os.path.join(base_data_dir, 'checkpoint.csv')
            write_header = not os.path.exists(checkpoint_path)
            drug_df.loc[[i]].to_csv(checkpoint_path, mode='a', header=write_header, index=True)

# if __name__ == '__main__':
#     indication_to_diagnosis_uml()