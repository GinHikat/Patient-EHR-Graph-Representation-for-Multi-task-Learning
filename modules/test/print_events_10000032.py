import os
import sys
import torch

# Setup project root path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.downstream.temporal_sequence_setup.temporal_modeling import build_patient_timeline, admission_nodes, kg_embeddings, name_to_idx
from modules.downstream.presetup.unified_encoder import LabPanelEncoder, OMREncoder, SpecialTokenEncoder, AdmissionEncoder, ICUEncoder, TransferEncoder

def main():
    print("Initializing encoders...")
    lab_encoder       = LabPanelEncoder()
    omr_encoder       = OMREncoder()
    special_encoder   = SpecialTokenEncoder()
    admission_encoder = AdmissionEncoder()
    icu_encoder       = ICUEncoder()
    transfer_encoder  = TransferEncoder()

    pid = 10000032
    print(f"Parsing timeline events for patient: {pid}")

    embeddings, times, meta = build_patient_timeline(
        pid,
        admission_nodes,
        kg_embeddings,
        name_to_idx,
        lab_encoder,
        omr_encoder,
        special_encoder,
        admission_encoder,
        icu_encoder,
        transfer_encoder,
        device=torch.device('cpu')
    )

    if embeddings is None:
        print("Timeline is empty.")
        return

    print("\n" + "="*80)
    print(f" CHRONOLOGICAL TIMELINE EVENTS FOR PATIENT {pid} (Total: {len(times)})")
    print("="*80)
    print(f"{'No.':<4} | {'Timestamp':<19} | {'Event Type':<15} | Details")
    print("-"*80)

    for idx, (t, m) in enumerate(zip(times, meta)):
        time_str = t.strftime('%Y-%m-%d %H:%M:%S')
        e_type = m['type']
        details = ''
        if e_type == 'ICU':
            details = f"\033[91mUnit: {m['unit']}\033[0m"
        elif e_type == 'Transfer':
            details = f"\033[94mCare Unit: {m['care_unit']} | Transfer Type: {m['transfer_type']}\033[0m"
        elif e_type in ('ADMIT', 'DISCHARGE', 'admission_emb'):
            details = f"\033[92mAdm ID: {m['adm_id']}\033[0m"
        elif e_type == 'lab':
            details = "Lab Panel Measurement"
        elif e_type == 'omr':
            details = "OMR Event"
        
        print(f"{idx+1:<3} | {time_str} | {e_type:<15} | {details}")
    print("="*80)

if __name__ == "__main__":
    main()
