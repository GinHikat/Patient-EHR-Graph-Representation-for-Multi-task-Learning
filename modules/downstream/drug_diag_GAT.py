import pandas as pd
import numpy as np
from tqdm import tqdm

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import *

from dotenv import load_dotenv
load_dotenv() 

def extract_nodes_edges():
    '''
    Extract all Drug, Diagnosis, Disease and Phenotype nodes from the graph database and their corresponding relationships
    
    Returns
    -------
    diagnosis_df : pd.DataFrame
        DataFrame containing diagnosis nodes.
    drug_df : pd.DataFrame
        DataFrame containing drug nodes.
    disease_df : pd.DataFrame
        DataFrame containing disease nodes.
    phenotype_df : pd.DataFrame
        DataFrame containing phenotype nodes.
    '''
    # Diagnosis (MIMIC ICD nodes)
    diagnosis_nodes = query_neo4j('''
    MATCH (n:Diagnosis:ICD:MIMIC)
    RETURN n.id AS id, n.name AS name
    ''')
    diagnosis_df = pd.DataFrame(diagnosis_nodes)
    print(f'Diagnosis nodes: {len(diagnosis_df)}')

    # Drug (all External drug sources)
    drug_nodes = query_neo4j('''
    MATCH (n:Drug:External)
    RETURN n.id AS id, n.name AS name
    ''')
    drug_df = pd.DataFrame(drug_nodes)
    print(f'Drug nodes: {len(drug_df)}')

    # Disease (context nodes for GAT)
    disease_nodes = query_neo4j('''
    MATCH (n:Disease:External)
    RETURN n.id AS id, n.name AS name
    ''')
    disease_df = pd.DataFrame(disease_nodes)
    print(f'Disease nodes: {len(disease_df)}')

    # Phenotype (context nodes for GAT)
    phenotype_nodes = query_neo4j('''
    MATCH (n:Phenotype:External)
    RETURN n.id AS id, n.name AS name
    ''')
    phenotype_df = pd.DataFrame(phenotype_nodes)
    print(f'Phenotype nodes: {len(phenotype_df)}')

    # Tag each group with its type
    diagnosis_df['node_type'] = 'diagnosis'
    drug_df['node_type']      = 'drug'
    disease_df['node_type']   = 'disease'
    phenotype_df['node_type'] = 'phenotype'

    # Merge all into one table
    all_nodes = pd.concat([diagnosis_df, drug_df, disease_df, phenotype_df], ignore_index=True)

    # Drop duplicates (some nodes may appear in multiple sources)
    all_nodes = all_nodes.drop_duplicates(subset='id').reset_index(drop=True)

    # Assign integer index for GAT
    all_nodes['node_idx'] = all_nodes.index

    # Build lookup dicts
    id_to_idx  = dict(zip(all_nodes['id'],       all_nodes['node_idx']))
    id_to_name = dict(zip(all_nodes['id'],       all_nodes['name']))
    idx_to_id  = dict(zip(all_nodes['node_idx'], all_nodes['id']))

    all_nodes.to_csv('kg_nodes.csv', index=False)

    # Extract edges
    edges = query_neo4j('''
    MATCH (a)-[r]->(b)
    WHERE type(r) IN ['CHILD_OF','HAS_PHENOTYPE','CAUSE','INTERACTS_WITH','TREAT','EQUIVALENT_TO']
    AND a.id IS NOT NULL AND b.id IS NOT NULL
    RETURN a.id AS src, b.id AS dst, type(r) AS relation
    ''')

    edges_df = pd.DataFrame(edges)

    # Map node IDs to integer indices, drop edges where either node not in vocab
    edges_df['src_idx'] = edges_df['src'].map(id_to_idx)
    edges_df['dst_idx'] = edges_df['dst'].map(id_to_idx)
    edges_df = edges_df.dropna(subset=['src_idx', 'dst_idx'])
    edges_df['src_idx'] = edges_df['src_idx'].astype(int)
    edges_df['dst_idx'] = edges_df['dst_idx'].astype(int)

    # Encode relation type
    RELATIONS = ['CHILD_OF','HAS_PHENOTYPE','CAUSE','INTERACTS_WITH','TREAT','EQUIVALENT_TO']
    rel_to_idx = {r: i for i, r in enumerate(RELATIONS)}
    edges_df['rel_idx'] = edges_df['relation'].map(rel_to_idx)

    edges_df.to_csv('kg_edges.csv', index=False)

