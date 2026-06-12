import pandas as pd
import argparse
import requests
import os
import random
import difflib
from tqdm import tqdm

DIAGNOSIS_PATH = "../VN-Clinical-Text/trans/diagnosis_10.csv" 
INPUT_TERMS_PATH = "../VN-Clinical-Text/trans/translated_qwen72b.csv" 
ENTITIES_PATH = "../VN-Clinical-Text/trans/aggregated_entities.csv"

# Load diagnosis dictionary for few-shot examples
print("Loading diagnosis dictionary for few-shot examples...")
few_shot_pool = []
try:
    df_diag = pd.read_csv(DIAGNOSIS_PATH)
    # Extract pairs from all 3 levels
    for col_en, col_vi in [('group_name_en', 'group_name_vi'), ('category_en', 'category_vi'), ('name_en', 'name_vi')]:
        if col_en in df_diag.columns and col_vi in df_diag.columns:
            pairs = df_diag[[col_en, col_vi]].dropna().drop_duplicates()
            for _, row in pairs.iterrows():
                few_shot_pool.append({"en": row[col_en], "vi": row[col_vi]})
except FileNotFoundError:
    print(f"Warning: Could not find {DIAGNOSIS_PATH}. Few-shot prompting will be disabled.")

# Load aggregated entities for matching
print("Loading aggregated entities for post-processing...")
try:
    df_entities = pd.read_csv(ENTITIES_PATH)
    valid_entities = df_entities['entity'].dropna().astype(str).tolist()
except FileNotFoundError:
    print(f"Warning: Could not find {ENTITIES_PATH}. Entity matching will be disabled.")
    valid_entities = []

def translate_with_llm(target_term: str, num_shots: int = 10) -> str:
    """
    Calls your local vLLM server to translate a single term from English to Vietnamese.dd
    Uses zero-shot or few-shot depending on available data.
    """
    if not target_term or not str(target_term).strip():
        return ""
      
    # We MUST use a fixed set of samples instead of randomizing them!
    # If the prefix is different every time, vLLM's automatic prefix caching will fail!
    if few_shot_pool:
        # Sort or just take the first `num_shots` so the prefix is identical every time
        samples = few_shot_pool[:num_shots]
    else:
        samples = []
        
    # Inject hardcoded anatomical vocabulary that the 7B model struggles with
    samples.extend([
        {"en": "Myocardial infarction", "vi": "Nhồi máu cơ tim"},
        {"en": "Epiphysis", "vi": "Đầu xương"},
        {"en": "Phalanx", "vi": "Đốt ngón"},
        {"en": "Proximal Phalanx", "vi": "Đốt ngón gần"},
        {"en": "Middle Phalanx", "vi": "Đốt ngón giữa"},
        {"en": "Distal Phalanx", "vi": "Đốt ngón xa"},
    ])
  
    # Build the strict System Prompt for JSON output
    system_prompt = (
        "You are an expert clinical medical translator. Translate the English medical term into its exact, standardized Vietnamese equivalent.\n"
        "CRITICAL RULES:\n"
        "1. You MUST output ONLY a valid JSON object.\n"
        "2. The JSON object must have exactly one key: 'translation'.\n"
        "3. The value must be the pure Vietnamese translation.\n"
        "4. MUST NOT contain Chinese characters or conversational text. If you are unsure, provide a direct literal Vietnamese translation instead.\n"
    )
    
    user_prompt = "Translate the following English medical terms to Vietnamese in JSON format:\n\n"
    
    # Inject the few-shot examples as JSON strings
    for pair in samples:
        user_prompt += f'{{"term": "{pair["en"]}"}}\n'
        user_prompt += f'{{"translation": "{pair["vi"]}"}}\n\n'
        
    user_prompt += f"Now translate this specific term:\n"
    user_prompt += f'{{"term": "{target_term}"}}\n'
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
  
    # Call the Local vLLM Server
    url = "http://localhost:8080/v1/chat/completions"
    payload = {
        "model": "google/medgemma-27b-it",
        "messages": messages,
        "stream": False,
        "temperature": 0.0, # Strict mode,
        'max_tokens': 100,
        "response_format": {"type": "json_object"}
    }
  
    try:
        response = requests.post(url, json=payload).json()
        raw_translation = response['choices'][0]['message']['content'].strip()
        
        # Parse the JSON
        import json
        try:
            parsed = json.loads(raw_translation)
            raw_translation = parsed.get("translation", raw_translation)
        except json.JSONDecodeError:
            pass
        
        # Post-processing in Python to remove underscores as requested
        raw_translation = raw_translation.replace('_', ' ').strip()
        
        # Fuzzy Entity Matching Post-Processing
        if valid_entities and raw_translation:
            # Revert cutoff to 0.85 to prevent dangerous false positives like 'Trầm cảm' -> 'Đông máu'
            matches = difflib.get_close_matches(raw_translation.lower(), valid_entities, n=1, cutoff=0.85)
            if matches:
                return matches[0]
                
        return raw_translation
    except Exception as e:
        # Silently fail on mass translation so the loop doesn't crash
        return ""

def main():
    parser = argparse.ArgumentParser(description="Translate medical terms.")
    parser.add_argument("--mode", choices=["continue", "fresh"], default="continue",
                        help="Choose 'continue' to stream translations into an existing file or 'fresh' to start from the parquet file and append to a new CSV.")
    args = parser.parse_args()

    if args.mode == "fresh":
        input_path = "../VN-Clinical-Text/external_kg.parquet"
        output_path = "../VN-Clinical-Text/trans/translated_medgemma27b.csv"
        
        try:
            df_terms = pd.read_parquet(input_path)
        except FileNotFoundError:
            print(f"Error: Could not find {input_path}. Please provide the file with terms to translate.")
            return
            
        if os.path.exists(output_path):
            processed_df = pd.read_csv(output_path)
            start_index = len(processed_df)
            print(f"Resuming fresh mode from index {start_index}...")
        else:
            start_index = 0
            pd.DataFrame(columns=["english_term", "vietnamese_translation"]).to_csv(output_path, index=False)
            
        term_column = "name"
        if term_column not in df_terms.columns:
            print(f"Error: Column '{term_column}' not found in {input_path}")
            return
            
        if start_index >= len(df_terms):
            print("All terms have been translated!")
            return
            
        terms_to_translate = df_terms.iloc[start_index:][term_column].tolist()
        if 'labels' in df_terms.columns:
            labels_list = df_terms.iloc[start_index:]['labels'].tolist()
        else:
            labels_list = [""] * len(terms_to_translate)
            
    else:
        # Continue mode
        input_path = "../VN-Clinical-Text/trans/translated_qwen72b.csv"
        try:
            df_terms = pd.read_csv(input_path)
        except FileNotFoundError:
            print(f"Error: Could not find {input_path}")
            return
            
        if "medgemma_trans" not in df_terms.columns:
            df_terms["medgemma_trans"] = ""
            
        missing_mask = df_terms["medgemma_trans"].isna() | (df_terms["medgemma_trans"] == "")
        if not missing_mask.any():
            print("All terms have been translated!")
            return
            
        indices_to_process = df_terms[missing_mask].index.tolist()
        terms_to_translate = df_terms.loc[indices_to_process, "english_term"].tolist()
        
        # Load labels from parquet so we can skip "Drug" in continue mode
        try:
            df_kg = pd.read_parquet("../VN-Clinical-Text/external_kg.parquet")
            if 'name' in df_kg.columns and 'labels' in df_kg.columns:
                label_mapping = dict(zip(df_kg['name'], df_kg['labels']))
                labels_list = [label_mapping.get(term, "") for term in terms_to_translate]
            else:
                labels_list = [""] * len(terms_to_translate)
        except Exception:
            labels_list = [""] * len(terms_to_translate)

    print(f"Translating {len(terms_to_translate)} remaining terms using high-concurrency batching...")
    
    import sys
    from concurrent.futures import ThreadPoolExecutor
    
    # Send up to 64 requests simultaneously to vLLM to saturate the GPU
    CONCURRENT_REQUESTS = 64
    CHUNK_SIZE = 250  # Save to disk every 250 terms
    
    def process_term(term, label):
        if str(label).strip() == "Drug":
            return term  # Drugs don't need translation
        return translate_with_llm(term)
    
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        # Use tqdm to track terms (not chunks) so the ETA is perfectly accurate
        pbar = tqdm(total=len(terms_to_translate), desc="Translating terms", file=sys.stdout, dynamic_ncols=True, unit="term")
        
        for i in range(0, len(terms_to_translate), CHUNK_SIZE):
            chunk_terms = terms_to_translate[i : i + CHUNK_SIZE]
            chunk_labels = labels_list[i : i + CHUNK_SIZE]
            
            if args.mode == "continue":
                chunk_indices = indices_to_process[i : i + CHUNK_SIZE]
            
            # executor.map automatically maintains the order of the inputs
            chunk_results = list(executor.map(process_term, chunk_terms, chunk_labels))
            
            if args.mode == "fresh":
                # Combine into rows
                rows = []
                for en_term, vi_term in zip(chunk_terms, chunk_results):
                    rows.append({
                        "english_term": en_term,
                        "vietnamese_translation": vi_term
                    })
                # Save the chunk to CSV immediately
                pd.DataFrame(rows).to_csv(output_path, mode='a', header=False, index=False)
            else:
                # Update dataframe
                df_terms.loc[chunk_indices, "medgemma_trans"] = chunk_results
                # Save the chunk to CSV safely by writing to a temp file and replacing
                temp_file = input_path + ".tmp"
                df_terms.to_csv(temp_file, index=False)
                os.replace(temp_file, input_path)
            
            pbar.update(len(chunk_terms))
            
        pbar.close()

if __name__ == "__main__":
    main()
