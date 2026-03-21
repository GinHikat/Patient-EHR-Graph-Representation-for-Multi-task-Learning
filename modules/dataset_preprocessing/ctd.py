
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