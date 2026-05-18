import os
import sys
import numpy as np
import pandas as pd
import json

# Setup project root path
project_root = "d:/Study/Education/Projects/Thesis"
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.downstream.temporal_sequence_setup.temporal_modeling import (
    all_patient_ids, admission_nodes, kg_embeddings, name_to_idx,
    LabPanelEncoder, OMREncoder, SpecialTokenEncoder, AdmissionEncoder,
    ICUEncoder, TransferEncoder, OutpatientEncoder,
    build_patient_timeline, cat_to_icds, id_to_idx
)

def main():
    pid = 10000032
    print(f"Verifying event ordering at discharge for PID: {pid}")
    
    # Initialize encoders
    lab_encoder        = LabPanelEncoder()
    omr_encoder        = OMREncoder()
    special_encoder    = SpecialTokenEncoder()
    admission_encoder  = AdmissionEncoder()
    icu_encoder        = ICUEncoder()
    transfer_encoder   = TransferEncoder()
    outpatient_encoder = OutpatientEncoder()
    
    embeddings, times, meta = build_patient_timeline(
        pid, admission_nodes, kg_embeddings, name_to_idx,
        lab_encoder, omr_encoder, special_encoder, admission_encoder,
        icu_encoder, transfer_encoder, outpatient_encoder,
        cat_to_icds=cat_to_icds,
        id_to_idx=id_to_idx
    )
    
    print("\nListing all ADMIT, DISCHARGE, and admission_emb events in chronological order:")
    for idx, (t, m) in enumerate(zip(times, meta)):
        if m['type'] in ['ADMIT', 'DISCHARGE', 'admission_emb']:
            print(f"Index {idx:02d} | Time: {t.isoformat()} | Event Type: {m['type']} | adm_id: {m['adm_id']}")

if __name__ == "__main__":
    main()
