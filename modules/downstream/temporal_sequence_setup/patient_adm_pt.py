import torch
import pandas as pd
import json
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

if current_dir not in sys.path:
    sys.path.append(current_dir)

from modules.downstream.temporal_tasks import (
    PatientStaticEncoder, build_category_maps, precompute_all_patient_vectors,
    AdmissionStaticEncoder, build_admission_category_maps, precompute_all_admission_vectors
)
from shared_functions.global_functions import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Query patients from Neo4j
print('Querying patients...')
patients = query_neo4j('''
    MATCH (p:Patient)
    RETURN p.id AS id, p.gender AS gender, p.race AS race,
           p.language AS language, p.insurance AS insurance,
           p.marital_status AS marital_status, p.age AS age
''')
patients_df = pd.DataFrame(patients)
print(f'Loaded {len(patients_df):,} patients')

# Query admissions from Neo4j
print('Querying admissions...')
admissions = query_neo4j('''
    MATCH (a:Admission)
    RETURN a.id AS id, a.admission_type AS admission_type,
           a.admission_location AS admission_location,
           a.drg_type AS drg_type,
           a.drg_severity AS drg_severity,
           a.drg_mortality AS drg_mortality
''')
admissions_df = pd.DataFrame(admissions)
print(f'Loaded {len(admissions_df):,} admissions')

# Build PatientStaticEncoder + cache
cat_maps_pat, vocab_sizes_pat = build_category_maps(patients_df)

age_mean = patients_df['age'].mean()
age_std  = patients_df['age'].std()

patient_encoder = PatientStaticEncoder(vocab_sizes=vocab_sizes_pat).to(DEVICE)

patient_cache = precompute_all_patient_vectors(
    patients_df, patient_encoder, cat_maps_pat, age_mean, age_std, DEVICE
)
torch.save(patient_cache, 'patient_cache.pt')
print(f'patient_cache.pt saved — {len(patient_cache):,} patients')

# Build AdmissionStaticEncoder + cache
cat_maps_adm, vocab_sizes_adm = build_admission_category_maps(admissions_df)

admission_encoder = AdmissionStaticEncoder(vocab_sizes=vocab_sizes_adm).to(DEVICE)

admission_cache = precompute_all_admission_vectors(
    admissions_df, admission_encoder, cat_maps_adm, DEVICE
)
torch.save(admission_cache, 'admission_cache.pt')
print(f'admission_cache.pt saved — {len(admission_cache):,} admissions')

# Save encoder weights and maps for later use
torch.save(patient_encoder.state_dict(),   'patient_encoder.pt')
torch.save(admission_encoder.state_dict(), 'admission_encoder.pt')

import pickle
with open('cat_maps_patient.pkl', 'wb') as f:
    pickle.dump({'cat_maps': cat_maps_pat, 'age_mean': age_mean, 'age_std': age_std,
                 'vocab_sizes': vocab_sizes_pat}, f)
with open('cat_maps_admission.pkl', 'wb') as f:
    pickle.dump({'cat_maps': cat_maps_adm, 'vocab_sizes': vocab_sizes_adm}, f)

print('Done. Files saved:')
print('  patient_cache.pt, admission_cache.pt')
print('  patient_encoder.pt, admission_encoder.pt')
print('  cat_maps_patient.pkl, cat_maps_admission.pkl')