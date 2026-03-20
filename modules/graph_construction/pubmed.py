# drug is drugbank dataset, merge is the merge_lookup from bcd5cdr
additional_drug = pd.merge(drug, merge, on = 'name', how = 'right')
additional_drug = additional_drug.drop('drugbank_id', axis = 1)

query_disease = """
UNWIND $rows AS row
MERGE (n:Disease:PubMed:Test {id: row.id})
SET n.name = row.name
"""

query_chemical = """
UNWIND $rows AS row
MERGE (n:Drug:PubMed:Test {id: row.id})
SET n.name = row.name
"""

BATCH_SIZE = 500

# Split by type
diseases  = additional_drug[additional_drug['type'] == 'Disease']
chemicals = additional_drug[additional_drug['type'] == 'Chemical']

def build_rows(df):
    return [
        {"id": row["id"], "name": row["name"]}
        for _, row in df.iterrows()
    ]

# Insert Disease nodes
disease_rows = build_rows(diseases)
for i in tqdm(range(0, len(disease_rows), BATCH_SIZE), desc="Disease nodes"):
    dml_ddl_neo4j(query_disease, progress=False, rows=disease_rows[i:i+BATCH_SIZE])

# Insert Chemical/Drug nodes
chemical_rows = build_rows(chemicals)
for i in tqdm(range(0, len(chemical_rows), BATCH_SIZE), desc="Chemical nodes"):
    dml_ddl_neo4j(query_chemical, progress=False, rows=chemical_rows[i:i+BATCH_SIZE])

### Treat multiple mentions in PubMed by introducing Alias

# Get current name (shortest) per id
current_names_disease = (disease_combine
                 .loc[disease_combine.groupby('id_y')['name']
                 .apply(lambda x: x.str.len().idxmin())]
                 [['id_y', 'name']])

# Group all names per id
all_names_disease = disease_combine.groupby('id_y')['name'].apply(list).reset_index()
all_names_disease.columns = ['id', 'all_names']

# Merge and compute aliases
alias_disease_df = all_names_disease.merge(
    current_names_disease.rename(columns={'id_y': 'id'}), on='id'
)
alias_disease_df['aliases'] = alias_disease_df.apply(
    lambda r: [n for n in r['all_names'] if n != r['name']], axis=1
)

# Only nodes that have aliases
alias_disease_df = alias_disease_df[alias_disease_df['aliases'].apply(len) > 0]
print(f"Nodes with aliases: {len(alias_disease_df)}")