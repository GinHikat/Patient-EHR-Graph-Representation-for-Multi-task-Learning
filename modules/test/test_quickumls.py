import os
import sys
import unicodedata

# Ensure the App backend module can be found
project_root = r"d:\Study\Education\Projects\Thesis"
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the exact engine used by the API routes!
from modules.dataset_preprocessing.external.uml import spacy_quickumls
from modules.extend.ner_engine import get_cui_vocab_codes

def test_backend_quickumls():
    print("Asking the App Backend to process our sentences...")
    print("(If the server isn't already running, it will boot the engine and take ~3 mins to load)")
    
    test_sentences = [
        "18h ngày 11/4, Bộ Y tế ghi nhận thêm một ca dương tính nCoV.",
        "Bệnh nhân có biểu hiện sốt cao và đau đầu, được chẩn đoán mắc bệnh tả.",
        "Tiến hành siêu âm điều trị cho bệnh nhân suy tim."
    ]
    
    for text in test_sentences:
        text = unicodedata.normalize('NFC', text)
        print(f"\n[INPUT TEXT]: {text}")
        
        # We simply pass the string to your custom backend function!
        df_results = spacy_quickumls(text)
        
        if df_results.empty:
            print("  => No medical entities found.")
            continue
            
        # The backend elegantly returns a Pandas DataFrame instead of messy raw objects
        for _, row in df_results.iterrows():
            other_ids = get_cui_vocab_codes(row['cui'])
            print(f"  => [ENTITY EXTRACTED]: '{row['text']}'")
            print(f"       - CUI: {row['cui']}")
            print(f"       - Standard Name: {row['term']}")
            print(f"       - Category: {row['type']}")
            print(f"       - Similarity: {row['similarity']:.2f}")
            print(f"       - Other DB Codes: {other_ids}")

if __name__ == "__main__":
    test_backend_quickumls()
