import pandas as pd
import numpy as np
from tqdm import tqdm
import unicodedata
import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.dataset_preprocessing.external.uml import *
from modules.models.models import EmbeddingModels

df = pd.read_csv(r'data\viettel\mapping\ontology_mapping_groundtruth.csv')

def normalize_text(text):
    if pd.isna(text): return ""
    return unicodedata.normalize("NFC", str(text).strip().lower())

def evaluate_quickumls(df):
    print("Loading UMLS DB for cross-referencing...")
    df_uml = get_uml()
    
    print("Building CUI to SAB mapping dictionary...")
    cui_to_codes = {}
    for row in tqdm(df_uml.itertuples(index=False), total=len(df_uml), desc="Mapping DB"):
        cui = row.CUI
        sab = row.SAB
        code = row.CODE
        if cui not in cui_to_codes:
            cui_to_codes[cui] = {}
        if sab not in cui_to_codes[cui]:
            cui_to_codes[cui][sab] = set()
        cui_to_codes[cui][sab].add(str(code))
        
    def check_match(gt_val, db_set):
        if pd.isna(gt_val) or gt_val in ['nan', '<NA>', '']:
            return False
        vals = [v.strip() for v in str(gt_val).split(',')]
        for v in vals:
            if v in db_set:
                return True
        return False
    tp, tn, fp, fn = 0, 0, 0, 0
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Entities"):
        # NORMALIZE the input term before passing to QuickUMLS
        raw_term = str(row['term'])
        term = normalize_text(raw_term)
        
        # Parse and normalize ground truths
        gt_uml = str(row.get('uml_id', '')).strip()
        gt_hpo = str(row.get('hpo_id', '')).strip()
        gt_mesh = str(row.get('mesh_id', '')).strip()
        gt_omim = str(row.get('omim_id', '')).strip()
        gt_drugbank = str(row.get('drugbank_id', '')).strip()
        
        # NORMALIZE the Vietnamese ground truth terms
        gt_qwen = normalize_text(row.get('truth_qwen', ''))
        gt_gemma = normalize_text(row.get('truth_gemma', ''))
        
        gt_uml = gt_uml if gt_uml not in ['nan', '<NA>'] else ''
        gt_hpo = gt_hpo if gt_hpo not in ['nan', '<NA>'] else ''
        gt_mesh = gt_mesh if gt_mesh not in ['nan', '<NA>'] else ''
        gt_omim = gt_omim if gt_omim not in ['nan', '<NA>'] else ''
        gt_drugbank = gt_drugbank if gt_drugbank not in ['nan', '<NA>'] else ''
        
        has_gt = any([gt_uml, gt_hpo, gt_mesh, gt_omim, gt_drugbank, gt_qwen, gt_gemma])
        
        try:
            preds_df = spacy_quickumls(term)
        except Exception:
            preds_df = pd.DataFrame()
            
        predicted_cuis = set(preds_df['cui'].dropna().tolist()) if not preds_df.empty else set()
        
        # NORMALIZE the predicted Vietnamese terms returned by QuickUMLS
        predicted_terms = set(normalize_text(t) for t in preds_df['term'].dropna().tolist()) if not preds_df.empty else set()
        
        fp_count = len(predicted_cuis)
        
        if not has_gt:
            if fp_count == 0:
                tn += 1
                results.append({'term': raw_term, 'match': True, 'reason': 'TN (Correctly predicted NIL)'})
            else:
                fp += fp_count
                results.append({'term': raw_term, 'match': False, 'reason': 'FP (Predicted on NIL)'})
            continue
            
        if fp_count == 0:
            fn += 1
            results.append({'term': raw_term, 'match': False, 'reason': 'FN (No prediction)'})
            continue
            
        matched = False
        match_reason = ""
        
        # Match by CUI, top priority
        if gt_uml and any(u in predicted_cuis for u in [x.strip() for x in str(gt_uml).split(',')]):
            matched = True
            match_reason = "CUI Match"
            
        # Match by other databases, second priority if CUI not match (target not have uml_id)
        if not matched:
            for pred_cui in predicted_cuis:
                db_mappings = cui_to_codes.get(pred_cui, {})
                if check_match(gt_hpo, db_mappings.get('HPO', set())): matched = True; match_reason = "HPO Match"; break
                if check_match(gt_mesh, db_mappings.get('MSH', set())): matched = True; match_reason = "MeSH Match"; break
                if check_match(gt_omim, db_mappings.get('OMIM', set())): matched = True; match_reason = "OMIM Match"; break
                if check_match(gt_drugbank, db_mappings.get('DRUGBANK', set())): matched = True; match_reason = "DrugBank Match"; break
                    
        # Match by String Term, least priority if other not match
        if not matched:
            for p_term in predicted_terms:
                if (gt_qwen and p_term == gt_qwen) or (gt_gemma and p_term == gt_gemma):
                    matched = True
                    match_reason = "String Match"
                    break
                    
        if matched:
            tp += 1
            fp += (fp_count - 1)
            results.append({'term': raw_term, 'match': True, 'reason': match_reason})
        else:
            fn += 1
            fp += fp_count
            results.append({'term': raw_term, 'match': False, 'reason': 'FP/FN (Wrong prediction)'})
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    row_accuracy = sum([1 for r in results if r['match']]) / len(df) if len(df) > 0 else 0
    print("\n" + "="*30)
    print("=== Evaluation Results ===")
    print("="*30)
    print(f"Total Entities Tested: {len(df)}")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print("-" * 30)
    print(f"Row Accuracy: {row_accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print("="*30)
    
    return pd.DataFrame(results)

def evaluate_fuzzy_matching(df):
    print(f"Original dataframe size: {len(df)}")
    df = df.copy()
    df['norm_term'] = df['term'].apply(normalize_text)
    
    df = df.drop_duplicates(subset=['norm_term']).reset_index(drop=True)
    print(f"Size after deduplicating unique input terms: {len(df)}")

    print("Building Fuzzy Matching KB from truth_qwen and truth_gemma...")
    kb = {}
    
    for idx, row in df.iterrows():
        gt_uml = str(row.get('uml_id', '')).strip()
        gt_hpo = str(row.get('hpo_id', '')).strip()
        gt_mesh = str(row.get('mesh_id', '')).strip()
        gt_omim = str(row.get('omim_id', '')).strip()
        gt_drugbank = str(row.get('drugbank_id', '')).strip()
        
        gt_uml = gt_uml if gt_uml not in ['nan', '<NA>'] else ''
        gt_hpo = gt_hpo if gt_hpo not in ['nan', '<NA>'] else ''
        gt_mesh = gt_mesh if gt_mesh not in ['nan', '<NA>'] else ''
        gt_omim = gt_omim if gt_omim not in ['nan', '<NA>'] else ''
        gt_drugbank = gt_drugbank if gt_drugbank not in ['nan', '<NA>'] else ''
        
        ids = {
            'uml_id': gt_uml,
            'hpo_id': gt_hpo,
            'mesh_id': gt_mesh,
            'omim_id': gt_omim,
            'drugbank_id': gt_drugbank
        }
        
        qwen = normalize_text(row.get('truth_qwen', ''))
        gemma = normalize_text(row.get('truth_gemma', ''))
        
        if qwen:
            if qwen not in kb: kb[qwen] = []
            kb[qwen].append(ids)
        if gemma:
            if gemma not in kb: kb[gemma] = []
            kb[gemma].append(ids)

    choices = list(kb.keys())
    print(f"Total unique truth keys in KB: {len(choices)}")
    
    def check_id_match(true_ids, pred_ids_list):
        for pred_ids in pred_ids_list:
            if true_ids['uml_id'] and pred_ids['uml_id']:
                t_umls = [x.strip() for x in true_ids['uml_id'].split(',')]
                p_umls = [x.strip() for x in pred_ids['uml_id'].split(',')]
                if any(t in p_umls for t in t_umls): return True, "CUI Match"
                
            if true_ids['hpo_id'] and pred_ids['hpo_id']:
                t_hpos = [x.strip() for x in true_ids['hpo_id'].split(',')]
                p_hpos = [x.strip() for x in pred_ids['hpo_id'].split(',')]
                if any(t in p_hpos for t in t_hpos): return True, "HPO Match"
                
            if true_ids['mesh_id'] and pred_ids['mesh_id']:
                t_meshes = [x.strip() for x in true_ids['mesh_id'].split(',')]
                p_meshes = [x.strip() for x in pred_ids['mesh_id'].split(',')]
                if any(t in p_meshes for t in t_meshes): return True, "MeSH Match"
                
            if true_ids['omim_id'] and pred_ids['omim_id']:
                t_omims = [x.strip() for x in true_ids['omim_id'].split(',')]
                p_omims = [x.strip() for x in pred_ids['omim_id'].split(',')]
                if any(t in p_omims for t in t_omims): return True, "OMIM Match"
                
            if true_ids['drugbank_id'] and pred_ids['drugbank_id']:
                t_dbanks = [x.strip() for x in true_ids['drugbank_id'].split(',')]
                p_dbanks = [x.strip() for x in pred_ids['drugbank_id'].split(',')]
                if any(t in p_dbanks for t in t_dbanks): return True, "DrugBank Match"
                
        return False, "No ID Match"

    tp, tn, fp, fn = 0, 0, 0, 0
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fuzzy Evaluating"):
        term = row['norm_term']
        
        gt_uml = str(row.get('uml_id', '')).strip()
        gt_hpo = str(row.get('hpo_id', '')).strip()
        gt_mesh = str(row.get('mesh_id', '')).strip()
        gt_omim = str(row.get('omim_id', '')).strip()
        gt_drugbank = str(row.get('drugbank_id', '')).strip()
        
        gt_uml = gt_uml if gt_uml not in ['nan', '<NA>'] else ''
        gt_hpo = gt_hpo if gt_hpo not in ['nan', '<NA>'] else ''
        gt_mesh = gt_mesh if gt_mesh not in ['nan', '<NA>'] else ''
        gt_omim = gt_omim if gt_omim not in ['nan', '<NA>'] else ''
        gt_drugbank = gt_drugbank if gt_drugbank not in ['nan', '<NA>'] else ''
        
        true_ids = {
            'uml_id': gt_uml,
            'hpo_id': gt_hpo,
            'mesh_id': gt_mesh,
            'omim_id': gt_omim,
            'drugbank_id': gt_drugbank
        }
        
        has_gt = any(true_ids.values())
        
        if not term:
            fn += 1
            results.append({'term': term, 'match': False, 'reason': 'Empty term'})
            continue
            
        best_match_info = process.extractOne(term, choices, scorer=fuzz.WRatio)
        if not best_match_info:
            fn += 1
            results.append({'term': term, 'match': False, 'reason': 'No fuzzy match found'})
            continue
            
        best_match_str = best_match_info[0]
        pred_ids_list = kb[best_match_str]
        
        if not has_gt:
            fp += 1
            results.append({'term': term, 'match': False, 'reason': 'FP (Predicted on NIL)'})
            continue
            
        is_match, match_reason = check_id_match(true_ids, pred_ids_list)
        
        if is_match:
            tp += 1
            results.append({'term': term, 'match': True, 'reason': match_reason, 'matched_to': best_match_str})
        else:
            fn += 1
            fp += 1
            results.append({'term': term, 'match': False, 'reason': 'FP/FN (Wrong ID mapping)', 'matched_to': best_match_str})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    row_accuracy = sum([1 for r in results if r['match']]) / len(df) if len(df) > 0 else 0

    print("\n" + "="*30)
    print("=== Fuzzy Evaluation Results ===")
    print("="*30)
    print(f"Total Unique Entities Tested: {len(df)}")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print("-" * 30)
    print(f"Row Accuracy: {row_accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print("="*30)
    
    return pd.DataFrame(results)

model_name = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
embedder = EmbeddingModels(model_choice=model_name)

def evaluate_sapbert_matching(df, embedder):
    print(f"Original dataframe size: {len(df)}")
    
    df = df.copy()
    df['norm_term'] = df['term'].apply(normalize_text)
    df = df.drop_duplicates(subset=['norm_term']).reset_index(drop=True)
    print(f"Size after deduplicating unique input terms: {len(df)}")

    print("Building KB from truth_qwen and truth_gemma...")
    kb = {}
    
    for idx, row in df.iterrows():
        gt_uml = str(row.get('uml_id', '')).strip()
        gt_hpo = str(row.get('hpo_id', '')).strip()
        gt_mesh = str(row.get('mesh_id', '')).strip()
        gt_omim = str(row.get('omim_id', '')).strip()
        gt_drugbank = str(row.get('drugbank_id', '')).strip()
        
        gt_uml = gt_uml if gt_uml not in ['nan', '<NA>'] else ''
        gt_hpo = gt_hpo if gt_hpo not in ['nan', '<NA>'] else ''
        gt_mesh = gt_mesh if gt_mesh not in ['nan', '<NA>'] else ''
        gt_omim = gt_omim if gt_omim not in ['nan', '<NA>'] else ''
        gt_drugbank = gt_drugbank if gt_drugbank not in ['nan', '<NA>'] else ''
        
        ids = {
            'uml_id': gt_uml, 'hpo_id': gt_hpo, 'mesh_id': gt_mesh,
            'omim_id': gt_omim, 'drugbank_id': gt_drugbank
        }
        
        qwen = normalize_text(row.get('truth_qwen', ''))
        gemma = normalize_text(row.get('truth_gemma', ''))
        
        if qwen:
            if qwen not in kb: kb[qwen] = []
            kb[qwen].append(ids)
        if gemma:
            if gemma not in kb: kb[gemma] = []
            kb[gemma].append(ids)

    choices = list(kb.keys())
    print(f"Total unique truth keys in KB: {len(choices)}")
    
    # 2. Embed the KB choices using SapBERT
    print("Encoding KB choices with SapBERT...")
    choices_embeddings = embedder.encode_text(choices, batch_size=32)
    
    # FIX: Force the output into a proper 2D numpy matrix
    if isinstance(choices_embeddings, list):
        choices_embeddings = np.vstack(choices_embeddings)
    elif hasattr(choices_embeddings, 'cpu'):
        choices_embeddings = choices_embeddings.cpu().numpy()
        
    if choices_embeddings.ndim == 1:
        choices_embeddings = choices_embeddings.reshape(1, -1)
        
    # L2 Normalize for fast Cosine Similarity
    choices_embeddings_norm = choices_embeddings / np.linalg.norm(choices_embeddings, axis=1, keepdims=True)

    
    def check_id_match(true_ids, pred_ids_list):
        for pred_ids in pred_ids_list:
            if true_ids['uml_id'] and pred_ids['uml_id']:
                t_umls = [x.strip() for x in true_ids['uml_id'].split(',')]
                p_umls = [x.strip() for x in pred_ids['uml_id'].split(',')]
                if any(t in p_umls for t in t_umls): return True, "CUI Match"
                
            if true_ids['hpo_id'] and pred_ids['hpo_id']:
                t_hpos = [x.strip() for x in true_ids['hpo_id'].split(',')]
                p_hpos = [x.strip() for x in pred_ids['hpo_id'].split(',')]
                if any(t in p_hpos for t in t_hpos): return True, "HPO Match"
                
            if true_ids['mesh_id'] and pred_ids['mesh_id']:
                t_meshes = [x.strip() for x in true_ids['mesh_id'].split(',')]
                p_meshes = [x.strip() for x in pred_ids['mesh_id'].split(',')]
                if any(t in p_meshes for t in t_meshes): return True, "MeSH Match"
                
            if true_ids['omim_id'] and pred_ids['omim_id']:
                t_omims = [x.strip() for x in true_ids['omim_id'].split(',')]
                p_omims = [x.strip() for x in pred_ids['omim_id'].split(',')]
                if any(t in p_omims for t in t_omims): return True, "OMIM Match"
                
            if true_ids['drugbank_id'] and pred_ids['drugbank_id']:
                t_dbanks = [x.strip() for x in true_ids['drugbank_id'].split(',')]
                p_dbanks = [x.strip() for x in pred_ids['drugbank_id'].split(',')]
                if any(t in p_dbanks for t in t_dbanks): return True, "DrugBank Match"
                
        return False, "No ID Match"

    tp, tn, fp, fn = 0, 0, 0, 0
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="SapBERT Evaluating"):
        term = row['norm_term']
        
        gt_uml = str(row.get('uml_id', '')).strip()
        gt_hpo = str(row.get('hpo_id', '')).strip()
        gt_mesh = str(row.get('mesh_id', '')).strip()
        gt_omim = str(row.get('omim_id', '')).strip()
        gt_drugbank = str(row.get('drugbank_id', '')).strip()
        
        gt_uml = gt_uml if gt_uml not in ['nan', '<NA>'] else ''
        gt_hpo = gt_hpo if gt_hpo not in ['nan', '<NA>'] else ''
        gt_mesh = gt_mesh if gt_mesh not in ['nan', '<NA>'] else ''
        gt_omim = gt_omim if gt_omim not in ['nan', '<NA>'] else ''
        gt_drugbank = gt_drugbank if gt_drugbank not in ['nan', '<NA>'] else ''
        
        true_ids = {
            'uml_id': gt_uml, 'hpo_id': gt_hpo, 'mesh_id': gt_mesh,
            'omim_id': gt_omim, 'drugbank_id': gt_drugbank
        }
        has_gt = any(true_ids.values())
        
        if not term:
            fn += 1
            results.append({'term': term, 'match': False, 'reason': 'Empty term'})
            continue
            
        # Get the pre-calculated embedding from the dataframe
        term_emb = row.get('embedding', None)
        if term_emb is None or (isinstance(term_emb, float) and np.isnan(term_emb)):
            # Fallback if somehow missing
            term_emb = embedder.encode_text([term], batch_size=1)[0]
            if hasattr(term_emb, 'cpu'): term_emb = term_emb.cpu().numpy()
            
        term_emb = np.array(term_emb)
        # Normalize the term embedding for Cosine Similarity
        norm_val = np.linalg.norm(term_emb)
        term_emb_norm = term_emb / norm_val if norm_val > 0 else term_emb
        
        # 3. Calculate Cosine Similarity via Dot Product
        similarities = np.dot(choices_embeddings_norm, term_emb_norm)
        best_match_idx = np.argmax(similarities)
        best_match_str = choices[best_match_idx]
        best_score = similarities[best_match_idx]
        
        # 4. Check IDs against the best match from the KB
        pred_ids_list = kb[best_match_str]
        
        if not has_gt:
            fp += 1
            results.append({'term': term, 'match': False, 'reason': 'FP (Predicted on NIL)'})
            continue
            
        is_match, match_reason = check_id_match(true_ids, pred_ids_list)
        
        if is_match:
            tp += 1
            results.append({'term': term, 'match': True, 'reason': match_reason, 'matched_to': best_match_str, 'score': best_score})
        else:
            fn += 1
            fp += 1
            results.append({'term': term, 'match': False, 'reason': 'FP/FN (Wrong ID mapping)', 'matched_to': best_match_str, 'score': best_score})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    row_accuracy = sum([1 for r in results if r['match']]) / len(df) if len(df) > 0 else 0

    print("\n" + "="*30)
    print("=== SapBERT Evaluation Results ===")
    print("="*30)
    print(f"Total Unique Entities Tested: {len(df)}")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print("-" * 30)
    print(f"Row Accuracy: {row_accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print("="*30)
    
    return pd.DataFrame(results)
