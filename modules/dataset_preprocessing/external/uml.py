import pandas as pd
import numpy as np
from quickumls import QuickUMLS
import spacy
import dotenv
dotenv.load_dotenv()

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

quick_umls_path = os.getenv('QUICKUMLS_PATH')
data_dir = os.getenv('DATA_DIR')

nlp = spacy.load("en_ner_bc5cdr_md")
matcher = QuickUMLS(quick_umls_path)
tui_mapping = pd.read_parquet(os.path.join(data_dir, "UML", "META", 'tui_mapping.parquet'))
if 'tui' in tui_mapping.columns:
    tui_mapping = tui_mapping.set_index('tui')

def spacy_quickumls(text):
    
    # SciSpacy
    doc = nlp(text)
    data = [(ent.text, ent.label_) for ent in doc.ents]
    entities = [ent.text for ent in doc.ents]
    df_ent = pd.DataFrame(data, columns=["term", "label"])
    
    # QuickUMLS on FULL TEXT
    results = matcher.match(text)

    flat_data = [item for sublist in results for item in sublist]

    df = pd.DataFrame(flat_data)
    
    # return list(dict.fromkeys(entities)), df_ent

    def get_semantic_type(sem_set):
        if not isinstance(sem_set, (set, list)):
            return None
        # Return the first matching semantic type name
        for tui in sem_set:
            if tui in tui_mapping.index:
                res = tui_mapping.loc[tui, 'sty']
                return res.iloc[0] if isinstance(res, pd.Series) else res
        return None

    df['type'] = df['semtypes'].apply(get_semantic_type)

    df = df[['ngram', 'term', 'cui', 'similarity', 'type']]
    df.columns = ['text', 'term', 'cui', 'similarity', 'type']

    return df, df_ent

# Process MRSTY for CUI-TUI mapping
mrsty_path = os.path.join(data_dir, "UML", "META", "MRSTY.RRF")

df_sty = pd.read_csv(
    mrsty_path, 
    sep='|', 
    header=None, 
    names=['cui', 'tui', 'stn', 'sty', 'atui', 'cvf', 'trailing'],
    index_col=False
)

df_sty = df_sty[['cui', 'tui', 'sty']]