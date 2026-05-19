import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def shift_datetime_back(dt_obj, years):
    try:
        return dt_obj.replace(year=dt_obj.year - years)
    except ValueError:
        # Handle February 29 in leap years
        return dt_obj.replace(year=dt_obj.year - years, day=28)

def process_patient(pid, timeline_dir):
    meta_path = timeline_dir / f"{pid}_meta.json"
    emb_path = timeline_dir / f"{pid}_emb.npy"
    dt_path = timeline_dir / f"{pid}_dt.npy"
    
    if not (meta_path.exists() and emb_path.exists() and dt_path.exists()):
        return False
        
    # 1. Load data
    with open(meta_path) as f:
        meta = json.load(f)
        
    emb = np.load(emb_path)
    
    # 2. Find non-outnote and outpatient note years
    non_outnote_years = [
        datetime.fromisoformat(entry['time']).year 
        for entry in meta 
        if entry.get('type', '').upper() != 'OUTNOTE'
    ]
    outnote_years = [
        datetime.fromisoformat(entry['time']).year 
        for entry in meta 
        if entry.get('type', '').upper() == 'OUTNOTE'
    ]
    
    if not outnote_years or not non_outnote_years:
        # No alignment needed if there are no outpatient notes
        return False
        
    # Calculate year shift offset
    offset_years = int(round(np.median(outnote_years) - np.median(non_outnote_years)))
    
    if offset_years == 0:
        return False
        
    # 3. Apply year shift to outpatient notes in metadata
    parsed_dates = []
    for entry in meta:
        dt_obj = datetime.fromisoformat(entry['time'])
        if entry.get('type', '').upper() == 'OUTNOTE':
            aligned_dt = shift_datetime_back(dt_obj, offset_years)
            entry['time'] = aligned_dt.isoformat()
            parsed_dates.append(aligned_dt)
        else:
            parsed_dates.append(dt_obj)
            
    # 4. Sort timeline chronologically
    sorted_indices = np.argsort(parsed_dates)
    
    # Re-order metadata and embeddings
    sorted_meta = [meta[idx] for idx in sorted_indices]
    sorted_emb = emb[sorted_indices]
    
    # 5. Recalculate time-to-end delta (dt) based on new chronological alignment
    sorted_dates = [parsed_dates[idx] for idx in sorted_indices]
    max_date = sorted_dates[-1]
    sorted_dt = np.array([
        (max_date - d).total_seconds() / 86400.0 
        for d in sorted_dates
    ], dtype=np.float32)
    
    # 6. Save back to disk
    with open(meta_path, 'w') as f:
        json.dump(sorted_meta, f, indent=2)
        
    np.save(emb_path, sorted_emb)
    np.save(dt_path, sorted_dt)
    return True

if __name__ == "__main__":
    timeline_dir = Path("F:/Din/Study/Education/Projects/Thesis/data/Timelines")
    
    # Test on patient 10000032 first
    test_pid = "10000032"
    print(f"Testing on patient {test_pid}...")
    success = process_patient(test_pid, timeline_dir)
    
    if success:
        print(f"✓ Test on {test_pid} succeeded!")
        # Let's inspect test dt and metadata
        with open(timeline_dir / f"{test_pid}_meta.json") as f:
            test_meta = json.load(f)
        test_dt = np.load(timeline_dir / f"{test_pid}_dt.npy")
        
        print("\nFirst 5 events after alignment:")
        for entry, dt_val in zip(test_meta[:5], test_dt[:5]):
            print(f"  {entry['time']} | {entry['type']:<15} | dt = {dt_val:.2f} days")
            
        print("\nLast 5 events after alignment:")
        for entry, dt_val in zip(test_meta[-5:], test_dt[-5:]):
            print(f"  {entry['time']} | {entry['type']:<15} | dt = {dt_val:.2f} days")
            
        # Run on all patients
        print("\nProcessing all patients in data/Timeline_new...")
        all_meta_files = list(timeline_dir.glob("*_meta.json"))
        updated_count = 0
        
        for mf in tqdm(all_meta_files):
            pid = mf.name.split("_")[0]
            if pid == test_pid:
                updated_count += 1
                continue
            if process_patient(pid, timeline_dir):
                updated_count += 1
                
        print(f"\n✓ Complete! Aligned outpatient notes for {updated_count} / {len(all_meta_files)} patients.")
    else:
        print("✗ Test patient failed or already processed.")
