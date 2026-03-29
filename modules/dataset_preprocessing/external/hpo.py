import pandas as pd
import numpy as np

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

def parse_hpo_obo(file_path):

    terms = []
    current_term = None

    with open(file_path, "r", encoding="utf8") as f:

        for line in f:
            line = line.strip()

            if line == "[Term]":
                if current_term:
                    terms.append(current_term)

                current_term = {
                    "id": None,
                    "name": None,
                    "definition": None,
                    "synonyms": [],
                    "parents": [],
                    "alt_ids": [],
                    "xrefs": []
                }

            elif line.startswith("id:"):
                current_term["id"] = line.replace("id:", "").strip()

            elif line.startswith("name:"):
                current_term["name"] = line.replace("name:", "").strip()

            elif line.startswith("def:"):
                current_term["definition"] = line.replace("def:", "").strip()

            elif line.startswith("synonym:"):
                synonym = line.split('"')[1]
                current_term["synonyms"].append(synonym)

            elif line.startswith("is_a:"):
                parent = line.split("!")[0].replace("is_a:", "").strip()
                current_term["parents"].append(parent)

            elif line.startswith("alt_id:"):
                alt_id = line.replace("alt_id:", "").strip()
                current_term["alt_ids"].append(alt_id)

            elif line.startswith("xref:"):
                xref = line.replace("xref:", "").strip()
                current_term["xrefs"].append(xref)

        if current_term:
            terms.append(current_term)

    df = pd.DataFrame(terms)

    return df

df = parse_hpo_obo(os.path.join(hpo_path, 'hp.obo'))

df.columns = ['id', 'phenotype', 'description', 'alternative_name', 'Is_a', 'drop', 'xrefs']

df = df.drop('drop', axis = 1)
df = df.drop(19943)

df_x = df.explode("xrefs").reset_index(drop=True)

df_x[['xref_source','xref_id']] = df_x['xrefs'].str.split(':', n=1, expand=True)

xref_pivot = (
    df_x.pivot_table(
        index='id',
        columns='xref_source',
        values='xref_id',
        aggfunc=lambda x: list(x)
    )
    .reset_index()
)

xref_pivot = xref_pivot.rename(columns={
    'UMLS': 'umls_id',
    'SNOMEDCT_US': 'snomed_id'
})

df_final = df.merge(xref_pivot, on='id', how='left')

df = df_final[['id', 'phenotype', 'description', 'alternative_name', 'Is_a', 'snomed_id', 'umls_id']]