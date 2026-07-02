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

_uml = None
_matcher = None
_tui_mapping = None
_engine_loaded = False

def is_engine_loaded():
    global _engine_loaded
    return _engine_loaded

def load_engine():
    global _uml, _matcher, _tui_mapping, _engine_loaded
    if _engine_loaded:
        return True
    
    print("Loading Clinical NLP Engine...")
    
    # Load tui_mapping
    try:
        tui_mapping_path = os.path.join(data_dir, "UML", "META", 'tui_mapping.parquet')
        df_tui = pd.read_parquet(tui_mapping_path)
        if 'tui' in df_tui.columns:
            df_tui = df_tui.set_index('tui')
        _tui_mapping = df_tui
    except Exception as e:
        print(f"Error loading tui_mapping: {e}")
        raise e
        
    # Load QuickUMLS matcher
    try:
        _matcher = QuickUMLS(
            quick_umls_path, 
            window=5, 
            threshold=0.8, 
            similarity_name='jaccard'
        )
    except Exception as e:
        print(f"Error initializing QuickUMLS matcher: {e}")
        raise e

    # Load MRCONSO.RRF (uml)
    try:
        optimized_path = os.path.join(data_dir, 'UML', "MRCONSO_optimized.parquet")
        if os.path.exists(optimized_path):
            print("Loading optimized MRCONSO parquet...")
            _uml = pd.read_parquet(optimized_path)
        else:
            print("Optimized parquet not found. Loading raw MRCONSO.RRF...")
            columns = [
                "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF",
                "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY",
                "CODE", "STR", "SRL", "SUPPRESS", "CVF"
            ]
            mrconso_path = os.path.join(data_dir, 'UML', "MRCONSO.RRF")
            df_uml = pd.read_csv(
                mrconso_path,
                sep="|",
                header=None,
                names=columns,
                dtype=str,        # IMPORTANT: keep everything as string
                index_col=False,
                quoting=3         # avoid issues with quotes
            )
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
            _uml = df_uml
    except Exception as e:
        print(f"Error loading MRCONSO.RRF: {e}")
        raise e
        
    _engine_loaded = True
    print("Clinical NLP Engine loaded successfully!")
    return True

def get_matcher():
    if not _engine_loaded:
        load_engine()
    return _matcher

def get_tui_mapping():
    if not _engine_loaded:
        load_engine()
    return _tui_mapping

def get_uml():
    if not _engine_loaded:
        load_engine()
    return _uml

def __getattr__(name):
    if name == 'uml':
        return get_uml()
    elif name == 'matcher':
        return get_matcher()
    elif name == 'tui_mapping':
        return get_tui_mapping()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def spacy_quickumls(text):
    
    # QuickUMLS on FULL TEXT
    results = get_matcher().match(text)

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
        tui_map = get_tui_mapping()
        for tui in sem_set:
            if tui in tui_map.index:
                res = tui_map.loc[tui, 'sty']
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

### Map CUI to related IDs from other database
def map_cui_db():
    target_sabs = [
        'OMIM', 'HPO', 'ICD9CM',
        'ICD10', 'CCSR_ICD10CM', 'MSH', 'ATC'
    ]

    df_uml = get_uml()
    uml_filtered = df_uml[df_uml['SAB'].isin(target_sabs)]

    cui_map = (
        uml_filtered
        .groupby('CUI')[['SAB', 'CODE']]
        .apply(lambda x: list(set(list(zip(x['SAB'], x['CODE'])))))
        .to_dict()
    )

    # Parse to column-wise format
    records = []
    for cui, mappings in cui_map.items():
        for sab, code in mappings:
            records.append({'CUI': cui, 'SAB': sab, 'CODE': code})

    df_map = pd.DataFrame(records)

    # Pivot to create the columns
    cui_df = df_map.pivot_table(
        index='CUI', 
        columns='SAB', 
        values='CODE', 
        aggfunc=lambda x: ', '.join(x)
    )

    cui_df = cui_df.reset_index().rename_axis(None, axis=1)

    ## Start Connecting between 2 dataframes
    target_cols = [
        'ATC', 'CCSR_ICD10CM', 'HPO', 'ICD10',
        'ICD9CM', 'MSH', 'OMIM'
    ]

    # Split related_diagnosis into list of CUIs
    extract['related_diagnosis'] = extract['related_diagnosis'].fillna('')
    extract['CUI_list'] = extract['related_diagnosis'].apply(
        lambda x: [c.strip() for c in x.split(',')] if x else []
    )

    # Explode to long format
    df_long = extract[['id', 'CUI_list']].explode('CUI_list')
    df_long = df_long.rename(columns={'CUI_list': 'CUI'})

    # Merge with map on CUI
    df_merged = df_long.merge(
        map[['CUI'] + target_cols],
        on='CUI',
        how='left'
    )

    # Aggregate back (collect unique values into lists)
    def agg_list(series):
        return list(set(series.dropna()))

    df_grouped = df_merged.groupby('id')[target_cols].agg(agg_list).reset_index()

    # Merge back into original extract
    extract = extract.merge(df_grouped, on='id', how='left')

    extract = extract.drop('related_diagnosis', axis = 1)

    extract['HPO'] = extract['HPO'].apply(lambda x: [i.replace('HP:', '') for i in x])

