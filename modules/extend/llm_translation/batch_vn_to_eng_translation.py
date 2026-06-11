import os
import time
import pandas as pd
from tqdm import tqdm

from modules.extend.model.translation import ClinicalTranslator

INPUT_CSV = "data/viettel/mapping/aggregated_entities.csv"
OUTPUT_CSV = "data/viettel/mapping/translated_entities.csv"

CHUNK_SIZE = 50 

def main():
    print("Initializing Translators...")
    hf_trans = ClinicalTranslator(backend="vinai")
    gg_trans = ClinicalTranslator(backend="google")
    
    print(f"Loading input dataset: {INPUT_CSV}")
    df_input = pd.read_csv(INPUT_CSV)
    
    if os.path.exists(OUTPUT_CSV):
        df_output = pd.read_csv(OUTPUT_CSV)
        start_idx = len(df_output)
        print(f"Checkpoint found. Resuming from index {start_idx} out of {len(df_input)}")
    else:
        start_idx = 0
        print("No checkpoint found. Starting from the beginning.")
        headers = list(df_input.columns) + ["vinai_en", "google_en"]
        pd.DataFrame(columns=headers).to_csv(OUTPUT_CSV, index=False)
        
    if start_idx >= len(df_input):
        print("Translation is already finished.")
        return
        
    print("\nStarting translation pipeline...")
    pbar = tqdm(total=len(df_input), initial=start_idx, desc="Total Progress")
    
    try:
        for start in range(start_idx, len(df_input), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(df_input))
            chunk_df = df_input.iloc[start:end].copy()
            
            entities = chunk_df['entity'].tolist()
            
            hf_results = hf_trans.translate_batch(entities, batch_size=32, show_progress=False)
            gg_results = gg_trans.translate_batch(entities, show_progress=False)
            
            chunk_df['vinai_en'] = hf_results
            chunk_df['google_en'] = gg_results
            
            chunk_df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
            
            pbar.update(len(entities))
            
    except KeyboardInterrupt:
        print("\nScript paused by user. Progress saved to disk.")
    except Exception as e:
        print(f"\nA critical error occurred: {e}")
    finally:
        pbar.close()

if __name__ == "__main__":
    main()
