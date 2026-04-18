import pandas as pd
import numpy as np
import dotenv
dotenv.load_dotenv()

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.dataset_preprocessing.utils import *

prescription = read_full('hosp', 'prescriptions')

prescription = prescription[['subject_id', 'hadm_id', 'drug', 'ndc', 'dose_val_rx', 'dose_unit_rx']]
prescription['ndc'] = prescription['ndc'].astype('Int64').astype(str)
prescription = prescription.dropna(subset = 'drug')
prescription['drug'] = prescription['drug'].apply(lambda x: x.title().strip())
prescription.columns = ['subject_id', 'hadm_id', 'drug', 'ndc', 'dose_value', 'dose_unit']