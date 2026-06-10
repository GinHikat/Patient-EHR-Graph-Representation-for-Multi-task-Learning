import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report
import warnings

# Suppress some HuggingFace warnings for clean output
warnings.filterwarnings("ignore")

# Define the exact labels we expect in our CoNLL files
LABEL_LIST = [
    "O",
    "B-Disease/Symptom",
    "I-Disease/Symptom",
    "B-Procedure/Treatment",
    "I-Procedure/Treatment",
    "B-Drug",
    "I-Drug"
]
# Map index to label string
LABEL_MAP = {i: label for i, label in enumerate(LABEL_LIST)}

def load_conll(file_path):
    sentences = []
    labels = []
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return sentences, labels
        
    with open(file_path, 'r', encoding='utf-8') as f:
        current_words = []
        current_labels = []
        for line in f:
            line = line.strip()
            if not line:
                if current_words:
                    sentences.append(current_words)
                    labels.append(current_labels)
                    current_words = []
                    current_labels = []
                continue
            parts = line.split('\t') if '\t' in line else line.split(' ')
            if len(parts) >= 2:
                current_words.append(parts[0])
                current_labels.append(parts[-1])
        if current_words:
            sentences.append(current_words)
            labels.append(current_labels)
    return sentences, labels

def evaluate_model(model_name, test_file):
    print(f"\n{'='*50}")
    print(f"EVALUATING UN-FINETUNED MODEL: {model_name}")
    print(f"{'='*50}")
    
    sentences, true_labels = load_conll(test_file)
    print(f"Loaded {len(sentences)} test sentences.\n")
    if not sentences: return
    
    print(f"Loading Tokenizer & Model from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, 
        num_labels=len(LABEL_LIST)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    pred_labels = []
    
    print("Running Inference on Test Set...")
    for words in sentences:
        inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()
        
        if isinstance(predictions, int):
            predictions = [predictions]
            
        aligned_preds = [LABEL_MAP[p] for p in predictions]
        
        if len(aligned_preds) >= 2:
            aligned_preds = aligned_preds[1:-1]
            
        # Ensure exact length match (truncate or pad if tokenizer truncated the sequence)
        if len(aligned_preds) > len(words):
            aligned_preds = aligned_preds[:len(words)]
        elif len(aligned_preds) < len(words):
            aligned_preds.extend(["O"] * (len(words) - len(aligned_preds)))
            
        pred_labels.append(aligned_preds)
        
    print("\n" + "="*50)
    print("EVALUATION RESULTS (via seqeval):")
    print("="*50)
    try:
        report = classification_report(true_labels, pred_labels, digits=4)
        print(report)
    except Exception as e:
        print(f"Seqeval encountered an error (likely due to random invalid BIO tags): {e}")

if __name__ == "__main__":
    test_path = r"data\viettel\vietnamese_ner\training\vietnamese\ner_train\ner_test.conll"
    
    # Test ViPubmed-DeBERTa Zero-Shot (Un-finetuned MLM)
    evaluate_model("manhtt-079/vipubmed-deberta-base", test_path)
