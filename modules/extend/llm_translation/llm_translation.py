import pandas as pd
import requests
import os
import random
import difflib
from tqdm import tqdm

DIAGNOSIS_PATH = "../VN-Clinical-Text/diagnosis_10.csv" 
INPUT_TERMS_PATH = "../VN-Clinical-Text/external_kg.parquet" 
ENTITIES_PATH = "../VN-Clinical-Text/aggregated_entities.csv"
OUTPUT_FILE = "translated_silver_standard.csv"

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

def translate_with_llm(target_term: str, num_shots: int = 0) -> str:
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
  
    # Call the Local vLLM Server (OpenAI Compatible)
    url = "http://localhost:8080/v1/chat/completions"
    payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": messages,
        "stream": False,
        "temperature": 0.0, # Strict mode
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
    # Load the terms to translate
    try:
        df_terms = pd.read_parquet(INPUT_TERMS_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_TERMS_PATH}. Please provide the file with terms to translate.")
        return

    # Check if we already started processing so we don't start from 0
    if os.path.exists(OUTPUT_FILE):
        processed_df = pd.read_csv(OUTPUT_FILE)
        start_index = len(processed_df)
        print(f"Resuming from index {start_index}...")
    else:
        start_index = 0
        # Create empty file with headers
        pd.DataFrame(columns=["english_term", "vietnamese_translation"]).to_csv(OUTPUT_FILE, index=False)

    term_column = "name"
    if term_column not in df_terms.columns:
        print(f"Error: Column '{term_column}' not found in {INPUT_TERMS_PATH}")
        return

    if start_index >= len(df_terms):
        print("All terms have been translated!")
        return

    terms_to_translate = df_terms.iloc[start_index:][term_column].tolist()
    if 'labels' in df_terms.columns:
        labels_list = df_terms.iloc[start_index:]['labels'].tolist()
    else:
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
            
            # executor.map automatically maintains the order of the inputs
            chunk_results = list(executor.map(process_term, chunk_terms, chunk_labels))
            
            # Combine into rows
            rows = []
            for en_term, vi_term in zip(chunk_terms, chunk_results):
                rows.append({
                    "english_term": en_term,
                    "vietnamese_translation": vi_term
                })
                
            # Save the chunk to CSV immediately
            pd.DataFrame(rows).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            pbar.update(len(chunk_terms))
            
        pbar.close()

if __name__ == "__main__":
    main()
