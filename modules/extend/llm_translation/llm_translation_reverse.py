import pandas as pd
import requests
import os
from tqdm import tqdm

DIAGNOSIS_PATH = "../VN-Clinical-Text/trans/diagnosis_10.csv" 
INPUT_TERMS_PATH = "../VN-Clinical-Text/trans/translated_entities.csv"
OUTPUT_FILE = "../VN-Clinical-Text/trans/translated_entities_medgemma.csv"

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
                # Store reverse mapping for VI -> EN
                few_shot_pool.append({"vi": row[col_vi], "en": row[col_en]})
except FileNotFoundError:
    print(f"Warning: Could not find {DIAGNOSIS_PATH}. Few-shot prompting will be disabled.")

def translate_with_llm(target_term: str, num_shots: int = 20) -> str:
    """
    Calls your local vLLM server to translate a single term from Vietnamese to English.
    Uses zero-shot or few-shot depending on available data.
    """
    if pd.isna(target_term) or not str(target_term).strip():
        return ""
      
    # We MUST use a fixed set of samples instead of randomizing them!
    # If the prefix is different every time, vLLM's automatic prefix caching will fail!
    if few_shot_pool:
        # Sort or just take the first `num_shots` so the prefix is identical every time
        samples = few_shot_pool[:num_shots]
    else:
        samples = []
        
    # Inject hardcoded anatomical vocabulary in reverse
    samples.extend([
        {"vi": "Nhồi máu cơ tim", "en": "Myocardial infarction"},
        {"vi": "Đầu xương", "en": "Epiphysis"},
        {"vi": "Đốt ngón", "en": "Phalanx"},
        {"vi": "Đốt ngón gần", "en": "Proximal Phalanx"},
        {"vi": "Đốt ngón giữa", "en": "Middle Phalanx"},
        {"vi": "Đốt ngón xa", "en": "Distal Phalanx"},
    ])
  
    # Build the strict System Prompt for JSON output
    system_prompt = (
        "You are an expert clinical medical translator. Translate the Vietnamese medical term into its exact, standardized English equivalent.\n"
        "CRITICAL RULES:\n"
        "1. You MUST output ONLY a valid JSON object.\n"
        "2. The JSON object must have exactly one key: 'translation'.\n"
        "3. The value must be the pure English translation.\n"
        "4. MUST NOT contain conversational text. If you are unsure, provide a direct literal English translation instead.\n"
    )
    
    user_prompt = "Translate the following Vietnamese medical terms to English in JSON format:\n\n"
    
    # Inject the few-shot examples as JSON strings
    for pair in samples:
        user_prompt += f'{{"term": "{pair["vi"]}"}}\n'
        user_prompt += f'{{"translation": "{pair["en"]}"}}\n\n'
        
    user_prompt += f"Now translate this specific term:\n"
    user_prompt += f'{{"term": "{target_term}"}}\n'
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
  
    # Call the Local vLLM Server
    url = "http://localhost:8080/v1/chat/completions"
    payload = {
        "model": "google/medgemma-4b-it",
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
        
        return raw_translation
    except Exception as e:
        # Silently fail on mass translation so the loop doesn't crash
        return ""

def main():
    # Load the terms to translate
    try:
        df_terms = pd.read_csv(INPUT_TERMS_PATH)
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
        # Create empty file with headers matching the input plus medgemma_en
        pd.DataFrame(columns=["entity", "type", "vinai_en", "google_en", "medgemma_en"]).to_csv(OUTPUT_FILE, index=False)

    term_column = "entity"
    if term_column not in df_terms.columns:
        print(f"Error: Column '{term_column}' not found in {INPUT_TERMS_PATH}")
        return

    if start_index >= len(df_terms):
        print("All terms have been translated!")
        return

    df_to_process = df_terms.iloc[start_index:]
    terms_to_translate = df_to_process[term_column].tolist()
    
    types_list = df_to_process['type'].tolist() if 'type' in df_to_process.columns else [""] * len(terms_to_translate)
    vinai_en_list = df_to_process['vinai_en'].tolist() if 'vinai_en' in df_to_process.columns else [""] * len(terms_to_translate)
    google_en_list = df_to_process['google_en'].tolist() if 'google_en' in df_to_process.columns else [""] * len(terms_to_translate)
        
    print(f"Translating {len(terms_to_translate)} remaining terms using high-concurrency batching...")
    
    import sys
    from concurrent.futures import ThreadPoolExecutor
    
    # Send up to 64 requests simultaneously to vLLM to saturate the GPU
    CONCURRENT_REQUESTS = 64
    CHUNK_SIZE = 250  # Save to disk every 250 terms
    
    def process_term(term, term_type):
        if str(term_type).strip() == "Drug":
            # Just translating everything, but you can skip Drugs if preferred. 
            pass
        return translate_with_llm(term)
    
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        # Use tqdm to track terms (not chunks) so the ETA is perfectly accurate
        pbar = tqdm(total=len(terms_to_translate), desc="Translating terms", file=sys.stdout, dynamic_ncols=True, unit="term")
        
        for i in range(0, len(terms_to_translate), CHUNK_SIZE):
            chunk_terms = terms_to_translate[i : i + CHUNK_SIZE]
            chunk_types = types_list[i : i + CHUNK_SIZE]
            chunk_vinai = vinai_en_list[i : i + CHUNK_SIZE]
            chunk_google = google_en_list[i : i + CHUNK_SIZE]
            
            # executor.map automatically maintains the order of the inputs
            chunk_results = list(executor.map(process_term, chunk_terms, chunk_types))
            
            # Combine into rows
            rows = []
            for e, t, v, g, m in zip(chunk_terms, chunk_types, chunk_vinai, chunk_google, chunk_results):
                rows.append({
                    "entity": e,
                    "type": t,
                    "vinai_en": v,
                    "google_en": g,
                    "medgemma_en": m
                })
                
            # Save the chunk to CSV immediately
            pd.DataFrame(rows).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            pbar.update(len(chunk_terms))
            
        pbar.close()

if __name__ == "__main__":
    main()
