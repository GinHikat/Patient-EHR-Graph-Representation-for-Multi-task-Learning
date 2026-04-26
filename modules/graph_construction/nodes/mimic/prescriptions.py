import pandas as pd
import numpy as np
import duckdb
import re

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import *
from modules.dataset_preprocessing.utils import *

from dotenv import load_dotenv
load_dotenv() 

prescription = read_full('hosp', 'prescription_clean')

def normalize_dose_unit(unit):
    if pd.isna(unit):
        return unit
    
    unit = str(unit).strip()
    
    # mcg first (before mg, to avoid partial match)
    if re.search(r'mcg|nanogram', unit, re.IGNORECASE):
        return 'mcg'
    
    # mg variants
    if re.search(r'mg', unit, re.IGNORECASE):
        return 'mg'
    
    # g variants (gm, gtt excluded)
    if re.search(r'\bgm\b|\bg\b', unit, re.IGNORECASE):
        return 'g'
    
    return unit  # leave others as-is

prescription['dose_unit_normalized'] = prescription['dose_unit'].apply(normalize_dose_unit)

# g -> mg
mask = prescription['dose_unit_normalized'] == 'g'
prescription.loc[mask, 'dose_value'] = prescription.loc[mask, 'dose_value'] * 1000
prescription.loc[mask, 'dose_unit_normalized'] = 'mg'

# Remove drugs that have rare unusual units (occurence < 50)
counts = prescription['dose_unit_normalized'].value_counts()
valid_units = counts[counts >= 50].index
prescription = prescription[prescription['dose_unit_normalized'].isin(valid_units)]

prescription = prescription.drop(['dose_unit', 'ndc'], axis = 1)
prescription = prescription.rename(columns = {'dose_unit_normalized':'unit'})

