import json
import torch
import os
import sys

# Dynamically add the project root to sys.path so python can find the 'modules' package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.models.models import EmbeddingModels

import argparse
from modules.extend.model.inference_ner import NER

def main():
    parser = argparse.ArgumentParser(description="End-to-End Evaluation against Gold Standard SapBERT CUIs")
    parser.add_argument("--extractor", type=str, choices=["qwen", "ner"], default="ner", help="Which model type to use for extraction.")
    parser.add_argument("--qwen_model", type=str, default="PeterPaker123/Qwen2.5-7B-Vietnamese-Medical-NER-GRPO", help="HuggingFace Hub ID for Qwen model")
    parser.add_argument("--ner_model", type=str, default="vihealthbert", help="Local NER model folder name (e.g. vihealthbert, vipubmed-deberta)")
    args = parser.parse_args()

    input_file = "data/viettel/vietnamese_ner/gold_standard_benchmark.jsonl"
    output_file = "data/viettel/vietnamese_ner/ner_predictions.jsonl"
    THRESHOLD = 0.80
    
    # Check if we are running in the correct directory
    if not os.path.exists(input_file):
        print(f"Error: Could not find '{input_file}'.")
        print("Please run this script from the root of the 'Patient-EHR-Graph-Representation-for-Multi-task-Learning' directory.")
        return

    if args.extractor == "qwen":
        print(f"Loading tokenizer and Qwen model from {args.qwen_model}...")
        tokenizer_id = "PeterPaker123/Qwen2.5-7B-Vietnamese-Medical-NER"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        
        model = AutoModelForCausalLM.from_pretrained(
            args.qwen_model, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )
    else:
        print(f"Loading local NER model '{args.ner_model}'...")
        ner_extractor = NER(args.ner_model)
    
    # Load SapBERT for entity linking
    print("Loading SapBERT model for entity linking...")
    sapbert_model_name = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
    embedder = EmbeddingModels(model_choice=sapbert_model_name)
    
    # Load mapped entities with their SapBERT embeddings for quick cosine similarity matching
    print("Loading pre-computed SapBERT embeddings database...")
    base_df = pd.read_pickle("data/viettel/mapping/mapped_entities_embedded.pkl")
    # We only care about rows that have a valid CUI
    base_df = base_df.dropna(subset=['mapped_cui']).reset_index(drop=True)
    base_embeddings = np.vstack(base_df['embedding'].values)
    
    # Label mapping dictionary (from Qwen NER output) - expanded to catch variations
    label_mapping = {
        "SYMPTOM_AND_DISEASE": "Disease/Symptom",
        "SYMPTOM": "Disease/Symptom",
        "DISEASE": "Disease/Symptom",
        "CONDITION": "Disease/Symptom",
        "DRUG": "Drug",
        "MEDICATION": "Drug",
        "MEDICAL_PROCEDURE": "Procedure/Treatment",
        "PROCEDURE": "Procedure/Treatment",
        "TREATMENT": "Procedure/Treatment",
        "TEST": "Procedure/Treatment",
        "LAB": "Procedure/Treatment",
        # "DIAGNOSTIC": "Procedure/Treatment"
    }
    
    # UMLS Semantic Type to Macro Class Mapping
    umls_to_three_classes = {
        "Clinical Attribute": "Disease/Symptom",
        "Disease or Syndrome": "Disease/Symptom",
        "Finding": "Disease/Symptom",
        "Injury or Poisoning": "Disease/Symptom",
        "Mental or Behavioral Dysfunction": "Disease/Symptom",
        "Neoplastic Process": "Disease/Symptom",
        "Pathologic Function": "Disease/Symptom",
        "Physiologic Function": "Disease/Symptom",
        "Sign or Symptom": "Disease/Symptom",
        "Amino Acid, Peptide, or Protein": "Drug",
        "Biologically Active Substance": "Drug",
        "Clinical Drug": "Drug",
        "Immunologic Factor": "Drug",
        "Indicator, Reagent, or Diagnostic Aid": "Drug",
        "Nucleic Acid, Nucleoside, or Nucleotide": "Drug",
        "Organic Chemical": "Drug",
        "Pharmacologic Substance": "Drug",
        "Vitamin": "Drug",
        "Diagnostic Procedure": "Procedure/Treatment",
        "Health Care Activity": "Procedure/Treatment",
        "Laboratory Procedure": "Procedure/Treatment",
        "Laboratory or Test Result": "Procedure/Treatment",
        "Medical Device": "Procedure/Treatment",
        "Therapeutic or Preventive Procedure": "Procedure/Treatment",
    }
    
    base_df['macro_type'] = base_df['mapped_type'].map(umls_to_three_classes)
    
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    print(f"Loaded {len(lines)} examples from {input_file}")
    
    # Global tracking for TP, FP, FN
    metrics = {
        "Disease/Symptom": {"tp": 0, "fp": 0, "fn": 0},
        "Procedure/Treatment": {"tp": 0, "fp": 0, "fn": 0},
        "Drug": {"tp": 0, "fp": 0, "fn": 0}
    }
    
    # System prompt if using Qwen for NER
    system_prompt = (
        "You are an expert clinical annotator. Your task is to extract medical entities from Vietnamese clinical text.\n"
        "The entity types to extract are STRICTLY limited to the following three:\n"
        "- SYMPTOM_AND_DISEASE: Clinical diagnoses, symptoms, signs of illness, diseases, or medical conditions (e.g., 'viêm phổi', 'đau đầu', 'tăng huyết áp').\n"
        "- MEDICAL_PROCEDURE: Medical procedures, surgeries, therapies, tests, imaging, or diagnostic methods (e.g., 'chụp X-quang', 'xét nghiệm máu', 'nội soi', 'phẫu thuật').\n"
        "- DRUG: Names of medicines, drugs, active ingredients, vitamins, or supplements (e.g., 'paracetamol', 'vitamin C', 'thuốc kháng sinh').\n"
        "\nCRITICAL RULES FOR EXTRACTION:\n"
        "1. PREVENT OVER-EXTRACTION: Extract only the most complete, primary medical concept. Do NOT extract granular attributes, causes, or locations independently.\n"
        "2. NO OVERLAPPING: Do NOT extract nested overlapping entities. If 'viêm kết mạc cấp' is extracted, do not also extract 'viêm kết mạc'. Keep only the longest, most complete one.\n"
        "3. NO ATTRIBUTES: Avoid extracting descriptive words like 'do vi khuẩn' (due to bacteria) or 'mức độ nhẹ' (mild) as standalone entities.\n"
        "4. EXACT TEXT: Extract precise, concise clinical terms exactly as they appear in the original text.\n"
        "5. EXAMPLE: For 'bệnh nhân bị viêm kết mạc cấp do vi khuẩn', extract ONLY 'viêm kết mạc cấp'. Do NOT extract 'viêm kết mạc do vi khuẩn' or 'do vi khuẩn'.\n"
        "\nOUTPUT FORMAT:\n"
        "Return the result STRICTLY as a JSON list containing dictionaries with \"entity\" and \"type\" keys. "
        "Ensure the \"type\" is exactly one of: SYMPTOM_AND_DISEASE, MEDICAL_PROCEDURE, DRUG.\n"
        "If no relevant entities are found, return an empty list []"
    )
    
    results = []
    
    for line in tqdm(lines, desc="Extracting Entities"):
        data = json.loads(line)
        text = data.get("text", "")
        
        # Extract entities using Qwen
        if args.extractor == "qwen":
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text: {text}"}
            ]
            
            inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True, return_dict=True).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=256, 
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None
                )
                
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            
            predicted_entities = []
            try:
                cleaned_response = response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]
                elif cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                    
                predicted_entities = json.loads(cleaned_response.strip())
            except json.JSONDecodeError:
                pass
        else:
            # NER extraction
            ner_results = ner_extractor.extract_entities(text)
            predicted_entities = [{"entity": e["term"], "type": e["label"]} for e in ner_results]
            
        # Perform Entity Linking
        linked_entities = {
            "Disease/Symptom": [],
            "Procedure/Treatment": [],
            "Drug": []
        }
        
        if predicted_entities:
            # Batch embed all extracted entities
            extracted_texts = [ent.get("entity", "") for ent in predicted_entities]
            extracted_embeddings = embedder.encode_text(extracted_texts, batch_size=32, show_progress=False)
            
            for idx, ent in enumerate(predicted_entities):
                raw_type = str(ent.get("type", "")).strip().upper()
                # If we used NER, the label is already the macro class "Disease/Symptom" etc.
                if args.extractor == "ner":
                    mapped_type = ent.get("type", "")
                else:
                    mapped_type = label_mapping.get(raw_type)
                
                # Skip unmapped or demographic entities
                if not mapped_type:
                    continue
                    
                emb = extracted_embeddings[idx:idx+1]
                
                # Filter base_df by the macro_type (derived from mapped_type)
                mask = base_df['macro_type'] == mapped_type
                if not mask.any():
                    continue
                    
                type_df = base_df[mask]
                type_embeddings = np.vstack(type_df['embedding'].values)
                
                # Calculate cosine similarity
                sims = cosine_similarity(emb, type_embeddings)[0]
                max_idx = np.argmax(sims)
                max_sim = sims[max_idx]
                
                if max_sim > THRESHOLD:
                    cui = type_df.iloc[max_idx]['mapped_cui']
                    if pd.notna(cui) and cui != "":
                        linked_entities[mapped_type].append(str(cui))
                        
        # Deduplicate the CUIs for each type
        linked_entities["Disease/Symptom"] = list(set(linked_entities["Disease/Symptom"]))
        linked_entities["Procedure/Treatment"] = list(set(linked_entities["Procedure/Treatment"]))
        linked_entities["Drug"] = list(set(linked_entities["Drug"]))
            
        # Compare Sets and Update Metrics
        gold_entities = data.get('gold_entities', {})
        gold_sets = {
            "Disease/Symptom": set(gold_entities.get("Disease/Symptom", [])),
            "Procedure/Treatment": set(gold_entities.get("Procedure/Treatment", [])),
            "Drug": set(gold_entities.get("Drug", []))
        }
        
        pred_sets = {
            "Disease/Symptom": set(linked_entities["Disease/Symptom"]),
            "Procedure/Treatment": set(linked_entities["Procedure/Treatment"]),
            "Drug": set(linked_entities["Drug"])
        }
        
        for cls in ["Disease/Symptom", "Procedure/Treatment", "Drug"]:
            gold = gold_sets[cls]
            pred = pred_sets[cls]
            
            tp = len(gold.intersection(pred))
            fp = len(pred - gold)
            fn = len(gold - pred)
            
            metrics[cls]["tp"] += tp
            metrics[cls]["fp"] += fp
            metrics[cls]["fn"] += fn
            
        data["predicted_entities"] = predicted_entities
        data["linked_entities"] = linked_entities
        results.append(data)

    print("\n" + "="*50)
    print("QWEN2.5 + SAPBERT EVALUATION RESULTS")
    print("="*50)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for cls in ["Disease/Symptom", "Procedure/Treatment", "Drug"]:
        tp = metrics[cls]["tp"]
        fp = metrics[cls]["fp"]
        fn = metrics[cls]["fn"]
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\n[{cls}]")
        print(f"  Precision: {precision:.4f}  (TP: {tp}, FP: {fp})")
        print(f"  Recall:    {recall:.4f}  (FN: {fn})")
        print(f"  F1-Score:  {f1:.4f}")

    print("\n" + "-"*50)
    macro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    macro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    macro_f1 = 2 * (macro_p * macro_r) / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0
    
    print(f"OVERALL MACRO (All Classes):")
    print(f"  Precision: {macro_p:.4f}")
    print(f"  Recall:    {macro_r:.4f}")
    print(f"  F1-Score:  {macro_f1:.4f}")
    print("="*50 + "\n")

    print(f"Saving predictions to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print("Done!")

if __name__ == "__main__":
    main()
