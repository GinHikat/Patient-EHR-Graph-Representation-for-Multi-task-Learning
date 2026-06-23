import os
import sys
import pandas as pd

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from modules.dataset_preprocessing.external.uml import spacy_quickumls
except ImportError:
    sys.path.append(os.path.join(project_root, 'modules'))
    from dataset_preprocessing.external.uml import spacy_quickumls

# Mapping table of UMLS Semantic Types to broad clinical categories
UMLS_STY_TO_CATEGORY = {
    # Disease
    "Disease or Syndrome": "Disease",
    "Neoplastic Process": "Disease",
    "Pathologic Function": "Disease",
    "Mental or Behavioral Dysfunction": "Disease",
    "Cell or Molecular Dysfunction": "Disease",
    "Experimental Model of Disease": "Disease",
    
    # Diagnosis
    "Congenital Abnormality": "Diagnosis",
    "Acquired Abnormality": "Diagnosis",
    "Anatomical Abnormality": "Diagnosis",
    "Injury or Poisoning": "Diagnosis",

    # Phenotype
    "Sign or Symptom": "Phenotype",
    "Finding": "Phenotype",
    "Clinical Attribute": "Phenotype",
    "Organism Attribute": "Phenotype",
    "Physiologic Function": "Phenotype",
    "Organism Function": "Phenotype",
    "Organ or Tissue Function": "Phenotype",
    "Cell Function": "Phenotype",
    "Biologic Function": "Phenotype",

    # Body Parts
    "Anatomical Structure": "Body Parts",
    "Body Location or Region": "Body Parts",
    "Body Part, Organ, or Organ Component": "Body Parts",
    "Body Space or Junction": "Body Parts",
    "Body Substance": "Body Parts",
    "Body System": "Body Parts",
    "Cell": "Body Parts",
    "Cell Component": "Body Parts",
    "Tissue": "Body Parts",
    "Embryonic Structure": "Body Parts",
    "Fully Formed Anatomical Structure": "Body Parts",

    # Drugs
    "Clinical Drug": "Drugs",
    "Pharmacologic Substance": "Drugs",
    "Antibiotic": "Drugs",
    "Biologically Active Substance": "Drugs",
    "Hormone": "Drugs",
    "Vitamin": "Drugs",
    "Immunologic Factor": "Drugs",
    "Organic Chemical": "Drugs",  # Mapped to Drugs to capture common drugs (e.g. metoprolol) in clinical notes

    # Chemicals
    "Chemical": "Chemicals",
    "Chemical Viewed Functionally": "Chemicals",
    "Chemical Viewed Structurally": "Chemicals",
    "Inorganic Chemical": "Chemicals",
    "Element, Ion, or Isotope": "Chemicals",
    "Enzyme": "Chemicals",
    "Amino Acid, Peptide, or Protein": "Chemicals",
    "Nucleic Acid, Nucleoside, or Nucleotide": "Chemicals",
    "Gene or Genome": "Chemicals",
    "Molecular Sequence": "Chemicals",
    "Nucleotide Sequence": "Chemicals",
    "Amino Acid Sequence": "Chemicals",
    "Carbohydrate Sequence": "Chemicals",
    "Receptor": "Chemicals",
    "Substance": "Chemicals",
    "Hazardous or Poisonous Substance": "Chemicals",
    "Indicator, Reagent, or Diagnostic Aid": "Chemicals",

    # Procedures
    "Therapeutic or Preventive Procedure": "Procedures",
    "Diagnostic Procedure": "Procedures",
    "Health Care Activity": "Procedures",
    "Research Activity": "Procedures",
    "Molecular Biology Research Technique": "Procedures",
    "Educational Activity": "Procedures",

    # Labs
    "Laboratory Procedure": "Labs",
    "Laboratory or Test Result": "Labs",

    # Devices
    "Medical Device": "Devices",
    "Drug Delivery Device": "Devices",
    "Research Device": "Devices"
}

# General and clinical stopwords to exclude from single-word concepts
CLINICAL_STOPWORDS = {
    "was", "were", "scheduled", "history", "is", "a", "an", "the", "of", "to", 
    "for", "in", "on", "with", "by", "at", "from", "as", "but", "or", "and", 
    "be", "has", "had", "have", "are", "been", "this", "that", "these", "those", 
    "his", "her", "their", "our", "its", "my", "your", "who", "which", "what", 
    "where", "when", "why", "how", "rule", "out", "showed", "showing", "shows", 
    "noted", "note", "notes", "noting", "admitted", "admit", "admitting", 
    "discharge", "discharged", "summary", "report", "patient", "pt", "history of"
}

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.blank("en")
            _nlp.add_pipe("sentencizer")
        except Exception as e:
            print(f"Warning: Failed to load Spacy sentencizer: {e}")
    return _nlp

def extract_entities(text: str) -> pd.DataFrame:
    """
    Sentence tokenizes the input clinical note/text and feeds each sentence 
    to spacy_quickumls to retrieve mapped UMLS entities. Mapped entities 
    are categorized into Disease, Diagnosis, Phenotype, Body Parts, Drugs, 
    Chemicals, Procedures, Labs, and Devices.
    """
    if not text or not isinstance(text, str):
        return pd.DataFrame(columns=['text', 'term', 'cui', 'similarity', 'type', 'category'])
        
    # nlp = get_nlp()
    if nlp:
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    else:
        # Fallback to simple split if spacy is unavailable
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
    dfs = []
    for sent in sentences:
        try:
            df_sent = spacy_quickumls(sent)
            if not df_sent.empty:
                dfs.append(df_sent)
        except Exception as e:
            print(f"Error extracting from sentence '{sent}': {e}")
            
    if not dfs:
        return pd.DataFrame(columns=['text', 'term', 'cui', 'similarity', 'type', 'category'])
        
    result_df = pd.concat(dfs, ignore_index=True)
    
    # Map the UMLS types to our custom categories
    result_df['category'] = result_df['type'].map(UMLS_STY_TO_CATEGORY).fillna("Other")
    
    return result_df

def get_cui_vocab_codes(cui: str) -> dict:
    """
    Look up a UMLS CUI in the preloaded MRCONSO table and return associated codes 
    for ICD-10, RxNorm, SNOMED CT, and LOINC.
    """
    try:
        from modules.dataset_preprocessing.external.uml import uml
        matching = uml[uml['CUI'] == cui]
        if matching.empty:
            return {}
        
        codes = {}
        for _, row in matching.iterrows():
            sab = row['SAB']
            code = row['CODE']
            if sab in ('ICD10', 'ICD10CM', 'ICD10PCS'):
                codes['icd10'] = code
            elif sab == 'ICD9CM':
                codes['icd9'] = code
            elif sab == 'RXNORM':
                codes['rxnorm'] = code
            elif sab == 'SNOMEDCT_US':
                codes['snomed'] = code
            elif sab in ('LNC', 'LOINC'):
                codes['loinc'] = code
            elif sab == 'MSH':
                codes['mesh'] = code
            elif sab == 'OMIM':
                codes['omim'] = code
            elif sab == 'HPO':
                codes['hpo'] = code
            elif sab == 'DRUGBANK':
                codes['drugbank'] = code
            elif sab == 'PUBCHEM':
                codes['pubchem'] = code
            elif sab in ('MEDLINE', 'PMID', 'PUBMED'):
                codes['pubmed'] = code
        return codes
    except Exception as e:
        print(f"Error looking up vocabulary codes for CUI {cui}: {e}")
        return {}

def extract_entities_umls(text: str) -> dict:
    """
    Sentence tokenizes the input clinical text using SpaCy and runs QuickUMLS 
    concept matching. Returns a dictionary of sentences and a list of 
    de-duplicated and resolved entities, complete with character start/end 
    offsets relative to the original text, category mapping, and external codes.
    """
    if not text or not isinstance(text, str):
        return {"original_text": text, "sentences": [], "entities": []}

    nlp = get_nlp()
    if nlp:
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents if sent.text.strip()]
    else:
        doc = None
        sentences = [s.strip() for s in text.split('.') if s.strip()]

    # Lazy import to prevent index loading at module import time
    from modules.dataset_preprocessing.external.uml import matcher, tui_mapping

    raw_entities = []

    def get_semantic_type(sem_set):
        if not isinstance(sem_set, (set, list)):
            return None
        for tui in sem_set:
            if tui in tui_mapping.index:
                res = tui_mapping.loc[tui, 'sty']
                return res.iloc[0] if isinstance(res, pd.Series) else res
        return None

    if doc:
        for sent in doc.sents:
            sent_text = sent.text
            sent_start = sent.start_char
            try:
                matches = matcher.match(sent_text)
                flat_matches = [item for sublist in matches for item in sublist]
                for item in flat_matches:
                    if item['ngram'].strip().lower() in CLINICAL_STOPWORDS:
                        continue
                    tui_set = item.get('semtypes', set())
                    sty_name = get_semantic_type(tui_set)
                    category = UMLS_STY_TO_CATEGORY.get(sty_name, "Other")
                    raw_entities.append({
                        "start": sent_start + item['start'],
                        "end": sent_start + item['end'],
                        "text": item['ngram'],
                        "canonical_name": item['term'],
                        "cui": item['cui'],
                        "similarity": float(item['similarity']),
                        "type": sty_name or "Unknown",
                        "category": category
                    })
            except Exception as e:
                print(f"Error matching sentence '{sent_text}': {e}")
    else:
        # Fallback to simple split offsets
        current_offset = 0
        for sent_text in sentences:
            idx = text.find(sent_text, current_offset)
            if idx != -1:
                sent_start = idx
                current_offset = sent_start + len(sent_text)
            else:
                sent_start = current_offset
            try:
                matches = matcher.match(sent_text)
                flat_matches = [item for sublist in matches for item in sublist]
                for item in flat_matches:
                    if item['ngram'].strip().lower() in CLINICAL_STOPWORDS:
                        continue
                    tui_set = item.get('semtypes', set())
                    sty_name = get_semantic_type(tui_set)
                    category = UMLS_STY_TO_CATEGORY.get(sty_name, "Other")
                    raw_entities.append({
                        "start": sent_start + item['start'],
                        "end": sent_start + item['end'],
                        "text": item['ngram'],
                        "canonical_name": item['term'],
                        "cui": item['cui'],
                        "similarity": float(item['similarity']),
                        "type": sty_name or "Unknown",
                        "category": category
                    })
            except Exception as e:
                print(f"Error matching sentence '{sent_text}': {e}")

    # Remove overlapping entity matches, prioritize longer span and higher similarity
    raw_entities.sort(key=lambda x: (x['start'], -(x['end'] - x['start'])))

    resolved_entities = []
    last_end = -1
    for ent in raw_entities:
        if ent['start'] >= last_end:
            resolved_entities.append(ent)
            last_end = ent['end']
        else:
            if not resolved_entities:
                continue
            prev = resolved_entities[-1]
            prev_len = prev['end'] - prev['start']
            ent_len = ent['end'] - ent['start']
            # Replace if current is a better match
            if ent['similarity'] > prev['similarity'] or (ent['similarity'] == prev['similarity'] and ent_len > prev_len):
                resolved_entities[-1] = ent
                last_end = ent['end']

    # Fetch ontology codes for each resolved entity
    for ent in resolved_entities:
        ent['codes'] = get_cui_vocab_codes(ent['cui'])

    return {
        "original_text": text,
        "sentences": sentences,
        "entities": resolved_entities
    }

def extract_entities_llm(text: str) -> dict:
    """
    Dummy function for Zero-Shot Medical LLM extraction.
    Currently returns a hardcoded mock response structure.
    """
    return {
        "original_text": text,
        "sentences": [text],
        "entities": [
            {
                "start": 0,
                "end": len(text) if len(text) < 10 else 10,
                "text": text[:10] if len(text) >= 10 else text,
                "canonical_name": "Dummy LLM Concept",
                "cui": "C_DUMMY",
                "similarity": 1.0,
                "type": "Finding",
                "category": "Phenotype",
                "codes": {}
            }
        ]
    }

_dl_extractor = None

_plmicd_model = None
_mlb_classes = None
_vihealthbert_tokenizer = None
_external_kg_cache = None

def extract_entities_dl(text: str, threshold: float = 0.5, model_length: str = "long") -> dict:
    """
    Extract diagnosis categories using ProcDiagExtractor (long) or PLMICDModel (short).
    """
    global _dl_extractor, _plmicd_model, _mlb_classes, _vihealthbert_tokenizer, _external_kg_cache
    
    if model_length == "long":
        from modules.models.extractor import ProcDiagExtractor, diagnosis_dict, diagnosis_icd_dict
        
        if _dl_extractor is None:
            _dl_extractor = ProcDiagExtractor("statedict_900_202")
            
        predictions = _dl_extractor.predict(text, threshold=threshold)
        
        entities = []
        for desc, prob in predictions.items():
            ccsr_category = diagnosis_dict.get(desc, desc)
            icd_codes = diagnosis_icd_dict.get(desc, "")
            entities.append({
                "start": 0,
                "end": len(text),
                "text": text,
                "canonical_name": desc,
                "cui": ccsr_category,
                "similarity": float(prob),
                "type": "Diagnosis",
                "category": "Diagnosis",
                "codes": {
                    "icd10": icd_codes
                }
            })
            
        return {
            "original_text": text,
            "sentences": [text],
            "entities": entities
        }
    else:
        # SHORT MODEL (vihealthbert Kaggle Output)
        print(f"[PLM-ICD Deep Learning] Processing text with threshold={threshold}...")
        import torch
        import json
        import pandas as pd
        from modules.extend.model.plmicd_model import PLMICDModel
        from transformers import AutoTokenizer
        from safetensors.torch import load_file
        
        model_dir = os.path.join(project_root, "modules", "extend", "statedict", "dl_vi", "final_model")
        classes_file = os.path.join(project_root, "modules", "extend", "statedict", "dl_vi", "classes.json")
        
        if _mlb_classes is None:
            print("[PLM-ICD Deep Learning] Loading classes.json...")
            with open(classes_file, "r", encoding="utf-8") as f:
                _mlb_classes = json.load(f)
                
        if _external_kg_cache is None:
            try:
                ekg_path = os.path.join(project_root, "data", "viettel", "mapping", "external_kg.parquet")
                df_ekg = pd.read_parquet(ekg_path)
                # Include Diagnosis, Disease, and Phenotype to capture all condition-like entities
                _external_kg_cache = df_ekg[df_ekg['labels'].isin(['Diagnosis', 'Disease', 'Phenotype'])]
            except Exception as e:
                print(f"Error loading external_kg.parquet: {e}")
                _external_kg_cache = pd.DataFrame()
                
        if _plmicd_model is None:
            print("[PLM-ICD Deep Learning] Loading model state dict...")
            _plmicd_model = PLMICDModel(num_labels=len(_mlb_classes), model_name="vihealthbert")
            state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
            # Strip module. prefix if saved from DataParallel in Kaggle
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            _plmicd_model.load_state_dict(state_dict, strict=False)
            _plmicd_model.eval()
            
        if _vihealthbert_tokenizer is None:
            print("[PLM-ICD Deep Learning] Loading tokenizer...")
            _vihealthbert_tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
        encodings = _vihealthbert_tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            outputs = _plmicd_model(input_ids=encodings['input_ids'], attention_mask=encodings['attention_mask'])
            probs = torch.sigmoid(outputs.logits).squeeze().numpy()
            
        entities = []
        # Support case where probs is 0-d (single class) or 1-d
        if probs.ndim == 0:
            probs = [probs.item()]
            
        for idx, prob in enumerate(probs):
            if prob >= threshold:
                cui_class = _mlb_classes[idx]
                codes = get_cui_vocab_codes(cui_class)
                
                # Default canonical name is the CUI itself
                canonical_name = cui_class
                
                # Attempt to map the CUI to the Vietnamese term
                if not _external_kg_cache.empty:
                    match = _external_kg_cache[_external_kg_cache['uml_id'] == cui_class]
                    if not match.empty:
                        row = match.iloc[0]
                        vn_name = row.get('medgemma_trans')
                        if pd.isna(vn_name) or not vn_name:
                            vn_name = row.get('qwen_trans')
                        if pd.isna(vn_name) or not vn_name:
                            vn_name = row.get('map_trans')
                        if pd.isna(vn_name) or not vn_name:
                            vn_name = row.get('name')
                        
                        if pd.notna(vn_name) and vn_name:
                            canonical_name = str(vn_name)
                            
                # Fallback to English MRCONSO term if it's completely missing from external_kg
                if canonical_name == cui_class:
                    try:
                        from modules.dataset_preprocessing.external.uml import get_uml
                        df_uml = get_uml()
                        if df_uml is not None and not df_uml.empty:
                            match_uml = df_uml[df_uml['CUI'] == cui_class]
                            if not match_uml.empty:
                                canonical_name = match_uml.iloc[0]['STR']
                    except Exception:
                        pass
                
                # Lexical post-matching to highlight the entity if it actually appears in the text
                start_idx = 0
                end_idx = len(text)
                extracted_text = text
                
                # We'll try to find any of the Vietnamese translations in the text
                import re
                potential_terms = [canonical_name]
                if not _external_kg_cache.empty:
                    match = _external_kg_cache[_external_kg_cache['uml_id'] == cui_class]
                    if not match.empty:
                        r = match.iloc[0]
                        for c in ['medgemma_trans', 'qwen_trans', 'map_trans', 'name']:
                            val = r.get(c)
                            if pd.notna(val) and val:
                                potential_terms.append(str(val))
                                
                for term in potential_terms:
                    # Escape special regex chars but allow case-insensitive match
                    match_obj = re.search(re.escape(term), text, re.IGNORECASE)
                    if match_obj:
                        start_idx = match_obj.start()
                        end_idx = match_obj.end()
                        extracted_text = text[start_idx:end_idx]
                        break
                
                entities.append({
                    "start": start_idx,
                    "end": end_idx,
                    "text": extracted_text,
                    "canonical_name": canonical_name,
                    "cui": cui_class,
                    "similarity": float(prob),
                    "type": "Diagnosis",
                    "category": "Diagnosis",
                    "codes": codes
                })
                
        return {
            "original_text": text,
            "sentences": [text],
            "entities": entities
        }

_ner_extractors = {}
_sapbert_embedder = None
_base_df_cache = None
_drug_mapping_cache = None

SEMTYPE_MAPPING = {
    # Disease / Symptom
    "Disease or Syndrome": "Disease/Symptom",
    "Sign or Symptom": "Disease/Symptom",
    "Pathologic Function": "Disease/Symptom",
    "Neoplastic Process": "Disease/Symptom",
    "Mental or Behavioral Dysfunction": "Disease/Symptom",
    "Finding": "Disease/Symptom",
    
    # Procedure / Treatment
    "Therapeutic or Preventive Procedure": "Procedure/Treatment",
    "Diagnostic Procedure": "Procedure/Treatment",
    "Health Care Activity": "Procedure/Treatment",
    "Laboratory Procedure": "Procedure/Treatment",
    "Medical Device": "Procedure/Treatment",
    
    # Drug
    "Pharmacologic Substance": "Drug",
    "Organic Chemical": "Drug",
    "Clinical Drug": "Drug",
    "Antibiotic": "Drug",
    "Biomedical or Dental Material": "Drug",
    "Amino Acid, Peptide, or Protein": "Drug",
    "Biologically Active Substance": "Drug",
    "Immunologic Factor": "Drug",
    "Indicator, Reagent, or Diagnostic Aid": "Drug",
    "Nucleic Acid, Nucleoside, or Nucleotide": "Drug",
    "Vitamin": "Drug"
}

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

def extract_entities_ner(text: str, model_name: str = "vihealthbert") -> dict:
    """
    Extract entities using fine-tuned Vietnamese NER and map to UMLS CUI via SapBERT.
    Compatible with Numpy 1.x by parsing the stringified CSV instead of pickle.
    """
    global _ner_extractors, _sapbert_embedder, _base_df_cache, _drug_mapping_cache
    import sys
    import os
    import numpy as np
    import pandas as pd
    import ast
    from sklearn.metrics.pairwise import cosine_similarity
    from modules.extend.model.inference_ner import NER
    from modules.models.models import EmbeddingModels

    if model_name not in _ner_extractors:
        print(f"Loading NER model {model_name}...")
        _ner_extractors[model_name] = NER(model_name)
    
    if _sapbert_embedder is None:
        print("Loading SapBERT model...")
        _sapbert_embedder = EmbeddingModels(model_choice="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR")
        
    if _base_df_cache is None:
        print("Loading base CSV embeddings...")
        # map_path = os.path.join(project_root, "data", "viettel", "mapping", "mapped_entities_embedded.parquet")
        map_path = os.path.join(project_root, "data", "viettel", "mapping", "mapped_entities_embedded.csv")
        try:
            df = pd.read_csv(map_path)
            # df = df.dropna(subset=['mapped_cui']).reset_index(drop=True)
            
            import re
            def parse_emb(x):
                # Bulletproof regex extraction: regardless of if Parquet saved it as a string,
                # an array containing a string, or a byte string, this will find all the floats.
                floats = re.findall(r'-?\d+\.\d+(?:e-?\d+)?', str(x))
                return np.array([float(f) for f in floats], dtype=np.float32)
            
            df['embedding'] = df['embedding'].apply(parse_emb)
            
            # As we discovered, original_type is much more trustworthy than mapped_type
            df['macro_type'] = df['original_type']
            
            _base_df_cache = df
        except Exception as e:
            print(f"Error loading CSV base entities: {e}")
            _base_df_cache = pd.DataFrame()

    if _drug_mapping_cache is None:
        print("Loading drug mapping parquet...")
        drug_path = os.path.join(project_root, "data", "viettel", "mapping", "drug_mapping.parquet")
        try:
            _drug_mapping_cache = pd.read_parquet(drug_path)
        except Exception as e:
            print(f"Error loading drug mapping: {e}")
            _drug_mapping_cache = pd.DataFrame()

    base_df = _base_df_cache
    drug_mapping_df = _drug_mapping_cache
    
    ner_results = _ner_extractors[model_name].extract_entities(text)
    
    entities = []
    THRESHOLD = 0.8
    
    if not ner_results:
        return {"original_text": text, "sentences": [text], "entities": []}
        
    extracted_texts = [ent["term"] for ent in ner_results]
    extracted_embeddings = _sapbert_embedder.encode_text(extracted_texts, batch_size=32, show_progress=False)
    
    cat_map = {
        "Disease/Symptom": "Disease",
        "Procedure/Treatment": "Procedures",
        "Drug": "Drugs"
    }
    
    for idx, ent in enumerate(ner_results):
        term = ent["term"]
        label = ent["label"]
        offset = ent["offset"]
        
        start = offset[0] if offset and offset[0] is not None else 0
        end = offset[1] if offset and offset[1] is not None else text.find(term) + len(term)
        if start == 0 and end == text.find(term) + len(term):
            start = text.find(term)
        
        emb = extracted_embeddings[idx:idx+1]
        
        mapped_cat = cat_map.get(label, "Other")
        # Use macro_type mapped from the granular mapped_type dictionary
        mask = base_df['macro_type'] == label
        
        cui = f"C_{label.upper().replace('/', '_')}"
        similarity = 1.0
        canonical_name = term
        
        if mask.any():
            type_df = base_df[mask]
            type_embeddings = np.vstack(type_df['embedding'].values)
            sims = cosine_similarity(emb, type_embeddings)[0]
            max_idx = np.argmax(sims)
            max_sim = sims[max_idx]
            
            print(f"DEBUG: Term='{term}', MaxSim={max_sim}, Matched='{type_df.iloc[max_idx]['entity']}', MappedCUI='{type_df.iloc[max_idx]['mapped_cui']}'")
            
            if max_sim > THRESHOLD:
                cui_val = type_df.iloc[max_idx]['mapped_cui']
                if pd.notna(cui_val) and cui_val != "":
                    cui = str(cui_val)
                    similarity = float(max_sim)
                    # Update the canonical name to the retrieved standardized entity
                    canonical_name = str(type_df.iloc[max_idx]['entity'])
                    
        # DRUG FALLBACK LOGIC
        codes = get_cui_vocab_codes(cui) if cui.startswith("C") and not cui.startswith("C_") else {}
        if label == "Drug" and cui == "C_DRUG" and not drug_mapping_df.empty:
            search_term = term.title()
            matched_row = drug_mapping_df[drug_mapping_df['name'] == search_term]
            
            if not matched_row.empty:
                row = matched_row.iloc[0]
                if 'db_id' in row and pd.notna(row['db_id']) and str(row['db_id']).strip() and str(row['db_id']).strip() != "None":
                    codes['drugbank'] = str(row['db_id'])
                if 'pubchem_id' in row and pd.notna(row['pubchem_id']) and str(row['pubchem_id']).strip() and str(row['pubchem_id']).strip() != "None":
                    codes['pubchem'] = str(row['pubchem_id'])
                if 'mesh_id' in row and pd.notna(row['mesh_id']) and str(row['mesh_id']).strip() and str(row['mesh_id']).strip() != "None":
                    codes['mesh'] = str(row['mesh_id'])
                
                if codes:
                    canonical_name = search_term
                    similarity = 1.0
                    print(f"DEBUG: Drug Fallback matched '{term}' to {codes} via drug_mapping.parquet")
        
        entities.append({
            "start": start,
            "end": end,
            "text": term,
            "canonical_name": canonical_name,
            "cui": cui,
            "similarity": similarity,
            "type": label,
            "category": mapped_cat,
            "codes": codes
        })
        
    return {
        "original_text": text,
        "sentences": [text],
        "entities": entities
    }
