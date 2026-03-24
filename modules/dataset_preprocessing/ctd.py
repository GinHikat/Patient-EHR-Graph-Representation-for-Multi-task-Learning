import pandas as pd 
import sys, os
from tqdm import tqdm as tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import *

chem_dis = os.path.join(ctd_path, 'chem_dis.csv')
chem = os.path.join(ctd_path, 'chemicals.csv')
dis = os.path.join(ctd_path, 'diseases.csv')

# rel = pd.read_csv(chem_dis, comment='#', low_memory=False)
chem_df = pd.read_csv(chem)
dis_df = pd.read_csv(dis)

def extract_alt_ids(alt_id_list):
    doid = []
    omim = []
    for item in alt_id_list:
        if item.startswith('DO:DOID:'):
            doid.append(item.replace('DO:DOID:', ''))
        elif item.startswith('OMIM:'):
            omim.append(item.replace('OMIM:', ''))
    return doid, omim

def string2list(df, col):
    def convert(x):
        if isinstance(x, list):
            return x  # already a list, skip
        if pd.isna(x) or x == '':
            return []
        return str(x).split('|')
    
    df[col] = df[col].apply(convert)

def removeMesh(df, col):
    df[col] = df[col].fillna('').apply(lambda x: x.replace('MESH:', ''))