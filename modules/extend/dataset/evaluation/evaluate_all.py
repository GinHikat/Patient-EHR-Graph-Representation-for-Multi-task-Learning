import os
import sys
import json
import argparse
import warnings
import unicodedata
import re

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, average_precision_score

from transformers import AutoModelForCausalLM, AutoTokenizer, EvalPrediction, Trainer, TrainingArguments
from safetensors.torch import load_file
from datasets import Dataset

# Dynamically add the project root to sys.path so python can find the 'modules' package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.models.models import EmbeddingModels
from modules.extend.model.inference_ner import NER
from modules.dataset_preprocessing.external.uml import spacy_quickumls
from modules.extend.model.plmicd_model import PLMICDModel

def get_words(text):
    import re
    import unicodedata
    normalized = unicodedata.normalize('NFC', str(text).lower())
    return re.findall(r'\w+', normalized)

def jaccard_similarity(str1, str2):
    set1 = set(get_words(str1))
    set2 = set(get_words(str2))
    if not set1 or not set2: return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0

def calculate_wer(ref, hyp):
    ref_words = get_words(ref)
    hyp_words = get_words(hyp)
    if not ref_words: return 1.0 if hyp_words else 0.0
    
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1): d[i][0] = i
    for j in range(len(hyp_words) + 1): d[0][j] = j
        
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
            
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)

def match_sets_jaccard_wer(pred_set, gold_set, threshold=0.5):
    matched_preds, matched_golds = set(), set()
    total_wer, matched_count = 0.0, 0
    
    pairs = []
    for p in pred_set:
        for g in gold_set:
            score = jaccard_similarity(p, g)
            if score >= threshold:
                pairs.append((score, p, g))
                
    pairs.sort(key=lambda x: x[0], reverse=True)
    for score, p, g in pairs:
        if p not in matched_preds and g not in matched_golds:
            matched_preds.add(p)
            matched_golds.add(g)
            total_wer += calculate_wer(g, p)
            matched_count += 1
            
    tp = len(matched_preds)
    fp = len(pred_set) - tp
    fn = len(gold_set) - tp
    return tp, fp, fn, total_wer, matched_count

def print_macro_metrics(metrics, title="EVALUATION RESULTS"):
    """
    Unified function to print Precision, Recall, and F1-score for evaluation metrics.
    Expects `metrics` dictionary mapping class names to {"tp": X, "fp": Y, "fn": Z}
    """
    print("\n" + "="*50)
    print(title)
    print("="*50)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for cls in ["Disease/Symptom", "Procedure/Treatment", "Drug"]:
        if cls not in metrics: continue
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
        if "wer_sum" in metrics[cls] and metrics[cls]["matched_count"] > 0:
            avg_wer = metrics[cls]["wer_sum"] / metrics[cls]["matched_count"]
            print(f"  Avg WER:   {avg_wer:.4f}  (over {metrics[cls]['matched_count']} matched terms)")

    print("\n" + "-"*50)
    macro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    macro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    macro_f1 = 2 * (macro_p * macro_r) / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0
    
    print(f"OVERALL MACRO (All Classes):")
    print(f"  Precision: {macro_p:.4f}")
    print(f"  Recall:    {macro_r:.4f}")
    print(f"  F1-Score:  {macro_f1:.4f}")
    
    total_wer_sum = sum(metrics[cls].get("wer_sum", 0) for cls in metrics)
    total_matched = sum(metrics[cls].get("matched_count", 0) for cls in metrics)
    if total_matched > 0:
        print(f"  Avg WER:   {(total_wer_sum / total_matched):.4f}")
        
    print("="*50 + "\n")

# QUICKUMLS EVALUATION

def evaluate_quickumls(args):
    benchmark_path = args.data_file
    
    metrics = {
        "Disease/Symptom": {"tp": 0, "fp": 0, "fn": 0, "wer_sum": 0.0, "matched_count": 0},
        "Procedure/Treatment": {"tp": 0, "fp": 0, "fn": 0, "wer_sum": 0.0, "matched_count": 0},
        "Drug": {"tp": 0, "fp": 0, "fn": 0, "wer_sum": 0.0, "matched_count": 0}
    }
    
    semtype_mapping = {
        "Disease or Syndrome": "Disease/Symptom",
        "Sign or Symptom": "Disease/Symptom",
        "Pathologic Function": "Disease/Symptom",
        "Neoplastic Process": "Disease/Symptom",
        "Mental or Behavioral Dysfunction": "Disease/Symptom",
        "Finding": "Disease/Symptom",
        "Therapeutic or Preventive Procedure": "Procedure/Treatment",
        "Diagnostic Procedure": "Procedure/Treatment",
        "Health Care Activity": "Procedure/Treatment",
        "Laboratory Procedure": "Procedure/Treatment",
        "Medical Device": "Procedure/Treatment",
        "Pharmacologic Substance": "Drug",
        "Organic Chemical": "Drug",
        "Clinical Drug": "Drug",
        "Antibiotic": "Drug",
        "Biomedical or Dental Material": "Drug"
    }

    print("Loading Gold Standard Benchmark...")
    sentences = []
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        for line in f:
            sentences.append(json.loads(line))
            
    print(f"Loaded {len(sentences)} sentences for evaluation.")
    print("Running QuickUMLS Inference (This will take a minute or two)...\n")
    
    for record in tqdm(sentences, desc="Evaluating QuickUMLS"):
        text = record['text']
        gold_entities = record['gold_entities']
        
        if args.eval_mode == "term":
            gold_sets = {
                "Disease/Symptom": set(str(x).strip().lower() for x in gold_entities.get("Disease/Symptom", [])),
                "Procedure/Treatment": set(str(x).strip().lower() for x in gold_entities.get("Procedure/Treatment", [])),
                "Drug": set(str(x).strip().lower() for x in gold_entities.get("Drug", []))
            }
        else:
            gold_sets = {
                "Disease/Symptom": set(gold_entities.get("Disease/Symptom", [])),
                "Procedure/Treatment": set(gold_entities.get("Procedure/Treatment", [])),
                "Drug": set(gold_entities.get("Drug", []))
            }
        pred_sets = {"Disease/Symptom": set(), "Procedure/Treatment": set(), "Drug": set()}
        
        clean_text = unicodedata.normalize('NFC', text)
        df_results = spacy_quickumls(clean_text)
        
        if not df_results.empty:
            df_results = df_results.sort_values(by=['similarity'], ascending=[False])
            df_results = df_results.drop_duplicates(subset=['text'], keep='first')
            
            for _, row in df_results.iterrows():
                macro_class = semtype_mapping.get(row['type'])
                if macro_class:
                    if args.eval_mode == "term":
                        term = row['term']
                        if pd.notna(term):
                            pred_sets[macro_class].add(str(term).strip().lower())
                    else:
                        cui = row['cui']
                        if pd.notna(cui):
                            pred_sets[macro_class].add(cui)
                    
        for cls in ["Disease/Symptom", "Procedure/Treatment", "Drug"]:
            gold, pred = gold_sets[cls], pred_sets[cls]
            if args.eval_mode == "term":
                tp, fp, fn, wer_sum, matched = match_sets_jaccard_wer(pred, gold, threshold=0.5)
                metrics[cls]["tp"] += tp
                metrics[cls]["fp"] += fp
                metrics[cls]["fn"] += fn
                metrics[cls]["wer_sum"] += wer_sum
                metrics[cls]["matched_count"] += matched
            else:
                metrics[cls]["tp"] += len(gold.intersection(pred))
                metrics[cls]["fp"] += len(pred - gold)
                metrics[cls]["fn"] += len(gold - pred)

    print_macro_metrics(metrics, "QUICKUMLS BASELINE EVALUATION RESULTS")

# NER + RETRIEVAL EVALUATION

def evaluate_ner_retrieve(args):
    input_file = args.data_file
    output_file = "data/viettel/vietnamese_ner/ner_predictions.jsonl"
    THRESHOLD = 0.80
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find '{input_file}'.")
        return

    if args.extractor == "qwen":
        tokenizer_id = args.base_model if args.base_model else args.qwen_model
        print(f"Loading tokenizer from {tokenizer_id}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        
        print(f"Loading model {args.qwen_model}...")
        if args.base_model:
            print(f"Loading base model {args.base_model} and applying LoRA...")
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", torch_dtype=torch.bfloat16)
            model = PeftModel.from_pretrained(base_model, args.qwen_model)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.qwen_model, device_map="auto", torch_dtype=torch.bfloat16)
    else:
        print(f"Loading local NER model '{args.ner_model}'...")
        ner_extractor = NER(args.ner_model)
    
    print("Loading SapBERT model for entity linking...")
    embedder = EmbeddingModels(model_choice="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR")
    
    print("Loading pre-computed SapBERT embeddings database...")
    import pickle
    class RenameUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)
            
    with open("data/viettel/mapping/mapped_entities_embedded.pkl", "rb") as f:
        base_df = RenameUnpickler(f).load()
    base_df = base_df.dropna(subset=['mapped_cui']).reset_index(drop=True)
    base_embeddings = np.vstack(base_df['embedding'].values)
    
    label_mapping = {
        "SYMPTOM_AND_DISEASE": "Disease/Symptom", "SYMPTOM": "Disease/Symptom", "DISEASE": "Disease/Symptom", "CONDITION": "Disease/Symptom",
        "DRUG": "Drug", "MEDICATION": "Drug",
        "MEDICAL_PROCEDURE": "Procedure/Treatment", "PROCEDURE": "Procedure/Treatment", "TREATMENT": "Procedure/Treatment", "TEST": "Procedure/Treatment", "LAB": "Procedure/Treatment"
    }
    
    umls_to_three_classes = {
        "Clinical Attribute": "Disease/Symptom", "Disease or Syndrome": "Disease/Symptom", "Finding": "Disease/Symptom", "Injury or Poisoning": "Disease/Symptom", "Mental or Behavioral Dysfunction": "Disease/Symptom", "Neoplastic Process": "Disease/Symptom", "Pathologic Function": "Disease/Symptom", "Physiologic Function": "Disease/Symptom", "Sign or Symptom": "Disease/Symptom",
        "Amino Acid, Peptide, or Protein": "Drug", "Biologically Active Substance": "Drug", "Clinical Drug": "Drug", "Immunologic Factor": "Drug", "Indicator, Reagent, or Diagnostic Aid": "Drug", "Nucleic Acid, Nucleoside, or Nucleotide": "Drug", "Organic Chemical": "Drug", "Pharmacologic Substance": "Drug", "Vitamin": "Drug",
        "Diagnostic Procedure": "Procedure/Treatment", "Health Care Activity": "Procedure/Treatment", "Laboratory Procedure": "Procedure/Treatment", "Laboratory or Test Result": "Procedure/Treatment", "Medical Device": "Procedure/Treatment", "Therapeutic or Preventive Procedure": "Procedure/Treatment",
    }
    base_df['macro_type'] = base_df['mapped_type'].map(umls_to_three_classes)
    
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    metrics = {
        "Disease/Symptom": {"tp": 0, "fp": 0, "fn": 0, "wer_sum": 0.0, "matched_count": 0},
        "Procedure/Treatment": {"tp": 0, "fp": 0, "fn": 0, "wer_sum": 0.0, "matched_count": 0},
        "Drug": {"tp": 0, "fp": 0, "fn": 0, "wer_sum": 0.0, "matched_count": 0}
    }
    
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
        
        if args.extractor == "qwen":
            prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Text: {text}"}]
            inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True, return_dict=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            
            predicted_entities = []
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
                if json_match:
                    predicted_entities = json.loads(json_match.group(1).strip())
                else:
                    predicted_entities = json.loads(response.strip())
            except json.JSONDecodeError:
                pass
        else:
            ner_results = ner_extractor.extract_entities(text)
            predicted_entities = [{"entity": e["term"], "type": e["label"]} for e in ner_results]
            
        linked_entities = {"Disease/Symptom": [], "Procedure/Treatment": [], "Drug": []}
        
        if predicted_entities:
            extracted_texts = [ent.get("entity", "") for ent in predicted_entities]
            extracted_embeddings = embedder.encode_text(extracted_texts, batch_size=32, show_progress=False)
            
            for idx, ent in enumerate(predicted_entities):
                raw_type = str(ent.get("type", "")).strip().upper()
                mapped_type = ent.get("type", "") if args.extractor == "ner" else label_mapping.get(raw_type)
                
                if not mapped_type: continue
                emb = extracted_embeddings[idx:idx+1]
                mask = base_df['macro_type'] == mapped_type
                if not mask.any(): continue
                    
                type_df = base_df[mask]
                type_embeddings = np.vstack(type_df['embedding'].values)
                sims = cosine_similarity(emb, type_embeddings)[0]
                max_idx = np.argmax(sims)
                if sims[max_idx] > THRESHOLD:
                    if args.eval_mode == "term":
                        retrieved_term = type_df.iloc[max_idx]['entity']
                        if pd.notna(retrieved_term) and retrieved_term != "":
                            linked_entities[mapped_type].append(str(retrieved_term))
                    else:
                        cui = type_df.iloc[max_idx]['mapped_cui']
                        if pd.notna(cui) and cui != "":
                            linked_entities[mapped_type].append(str(cui))
                        
        linked_entities = {k: list(set(v)) for k, v in linked_entities.items()}
            
        gold_entities = data.get('gold_entities', {})
        if args.eval_mode == "term":
            gold_sets = {
                "Disease/Symptom": set(str(x).strip().lower() for x in gold_entities.get("Disease/Symptom", [])),
                "Procedure/Treatment": set(str(x).strip().lower() for x in gold_entities.get("Procedure/Treatment", [])),
                "Drug": set(str(x).strip().lower() for x in gold_entities.get("Drug", []))
            }
            pred_sets = {k: set(str(x).strip().lower() for x in v) for k, v in linked_entities.items()}
        else:
            gold_sets = {
                "Disease/Symptom": set(gold_entities.get("Disease/Symptom", [])),
                "Procedure/Treatment": set(gold_entities.get("Procedure/Treatment", [])),
                "Drug": set(gold_entities.get("Drug", []))
            }
            pred_sets = {k: set(v) for k, v in linked_entities.items()}
        
        for cls in ["Disease/Symptom", "Procedure/Treatment", "Drug"]:
            gold, pred = gold_sets[cls], pred_sets[cls]
            if args.eval_mode == "term":
                tp, fp, fn, wer_sum, matched = match_sets_jaccard_wer(pred, gold, threshold=0.5)
                metrics[cls]["tp"] += tp
                metrics[cls]["fp"] += fp
                metrics[cls]["fn"] += fn
                metrics[cls]["wer_sum"] += wer_sum
                metrics[cls]["matched_count"] += matched
            else:
                metrics[cls]["tp"] += len(gold.intersection(pred))
                metrics[cls]["fp"] += len(pred - gold)
                metrics[cls]["fn"] += len(gold - pred)
            
        data["predicted_entities"] = predicted_entities
        data["linked_entities"] = linked_entities
        results.append(data)

    print_macro_metrics(metrics, "QWEN2.5 + SAPBERT EVALUATION RESULTS")

    print(f"Saving predictions to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
# DOCUMENT CLASSIFICATION EVALUATION

def _cls_compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = torch.sigmoid(torch.tensor(preds)).numpy()
    y_pred = (preds >= 0.5).astype(int)
    y_true = p.label_ids

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            mAP = average_precision_score(y_true=y_true, y_score=preds, average='macro')
        except:
            mAP = 0
    return {"mAP": mAP, "accuracy": accuracy}

def evaluate_classification(args):
    model_name = args.ner_model
    model_dir = args.model_dir
    data_file = args.data_file
    classes_file = os.path.join(project_root, "modules", "extend", "training", "results", "classes.json")
    
    print(f"--- Starting Local Evaluation for {model_name} ---")
    if not os.path.exists(model_dir): raise FileNotFoundError(f"Cannot find model directory at {model_dir}.")
    if not os.path.exists(classes_file): raise FileNotFoundError(f"Cannot find classes.json at {classes_file}.")

    with open(classes_file, "r", encoding="utf-8") as f:
        classes = json.load(f)
    
    mlb = MultiLabelBinarizer()
    mlb.fit([classes])
    num_labels = len(classes)
    print(f"Total unique CUI classes: {num_labels}")
    
    # Load Data (Only Disease/Symptom)
    texts, labels = [], []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            texts.append(data.get("text", ""))
            gold_ent = data.get("gold_entities", [])
            if isinstance(gold_ent, dict):
                labels.append(list(set(gold_ent.get("Disease/Symptom", []))))
            else:
                labels.append(gold_ent)
                
    if args.eval_mode == "term":
        y_true = np.zeros((len(texts), num_labels))
    else:
        y_true = mlb.transform(labels)
        
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    def tokenize_function(texts, labels):
        encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=256)
        items = []
        for i in range(len(texts)):
            item = {key: val[i] for key, val in encodings.items()}
            if args.eval_mode == "term":
                item['labels'] = [0.0] * num_labels
            else:
                item['labels'] = labels[i].tolist()
            items.append(item)
        return items
        
    eval_dataset = Dataset.from_list(tokenize_function(texts, y_true))
    
    model = PLMICDModel(num_labels=num_labels, model_name=model_name)
    weights_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(weights_path):
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
        
    clean_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    eval_args = TrainingArguments(output_dir="models/doc_classifier/eval_temp", per_device_eval_batch_size=16, report_to="none", fp16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=eval_args, eval_dataset=eval_dataset, compute_metrics=_cls_compute_metrics)
    
    pred_output = trainer.predict(eval_dataset)
    metrics = pred_output.metrics
    
    preds = torch.sigmoid(torch.tensor(pred_output.predictions)).numpy()
    y_pred = (preds >= 0.5).astype(int)
    
    if args.eval_mode == "term":
        mapping_df = pd.read_csv("data/viettel/mapping/mapped_entities.csv")
        cui_to_term = mapping_df.dropna(subset=['mapped_cui', 'mapped_term']).set_index('mapped_cui')['mapped_term'].to_dict()
        
    total_tp, total_fp, total_fn, total_wer_sum, total_matched = 0, 0, 0, 0.0, 0
    for i in range(len(y_pred)):
        pred_cuis = [classes[idx] for idx in np.where(y_pred[i] == 1)[0]]
        
        if args.eval_mode == "term":
            pred_set = set()
            for cui in pred_cuis:
                term = cui_to_term.get(cui)
                if term:
                    pred_set.add(str(term).strip().lower())
            gold_set = set(str(x).strip().lower() for x in labels[i])
            tp, fp, fn, wer_sum, matched = match_sets_jaccard_wer(pred_set, gold_set, threshold=0.5)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_wer_sum += wer_sum
            total_matched += matched
        else:
            pred_set = set(pred_cuis)
            gold_set = set(labels[i])
            total_tp += len(gold_set.intersection(pred_set))
            total_fp += len(pred_set - gold_set)
            total_fn += len(gold_set - pred_set)
        
    if args.eval_mode != "term":
        print(f"\nValidation Loss : {metrics.get('test_loss', 0):.4f}")
        print(f"mAP             : {metrics.get('test_mAP', 0):.4f}")
        print(f"Accuracy (EMR)  : {metrics.get('test_accuracy', 0):.4f}")
    
    cls_metrics = {"Disease/Symptom": {"tp": total_tp, "fp": total_fp, "fn": total_fn, "wer_sum": total_wer_sum, "matched_count": total_matched}}
    print_macro_metrics(cls_metrics, "DOCUMENT CLASSIFICATION EVALUATION RESULTS")

def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation Script for QuickUMLS, NER+Retrieve, and Doc Class")
    parser.add_argument("--mode", type=str, required=True, choices=["quickumls", "ner_retrieve", "classification"], help="Which evaluation approach to run")
    parser.add_argument("--eval_mode", type=str, choices=["cui", "term"], default="cui", help="Evaluate by CUI (default) or by lexical term matching")
    
    parser.add_argument("--data_file", type=str, default=None, help="Path to JSONL dataset. Defaults to gold_standard_benchmark for CUI and gold_standard_terms_vi for term.")
    
    # Arguments for NER+Retrieve
    parser.add_argument("--extractor", type=str, choices=["qwen", "ner"], default="ner", help="[ner_retrieve] Extractor type")
    parser.add_argument("--qwen_model", type=str, default="PeterPaker123/Qwen2.5-7B-Vietnamese-Medical-NER", help="[ner_retrieve] Qwen Model")
    parser.add_argument("--base_model", type=str, default="", help="[ner_retrieve] Base model for LoRA")
    
    # Arguments for Classification / NER
    parser.add_argument("--ner_model", type=str, default="vihealthbert", help="[ner_retrieve/classification] NER model folder / Backbone model")
    parser.add_argument("--model_dir", type=str, default=os.path.join(project_root, "modules", "extend", "training", "results", "final_model"), help="[classification] Path to trained classification model directory")
    
    args = parser.parse_args()
    
    if args.data_file is None:
        if args.eval_mode == "term":
            args.data_file = "data/viettel/vietnamese_ner/gold_standard_terms_vi.jsonl"
        else:
            args.data_file = "data/viettel/vietnamese_ner/gold_standard_benchmark.jsonl"
            
    if args.mode == "quickumls":
        evaluate_quickumls(args)
    elif args.mode == "ner_retrieve":
        evaluate_ner_retrieve(args)
    elif args.mode == "classification":
        evaluate_classification(args)

if __name__ == "__main__":
    main()
