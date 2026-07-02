import os
import sys
import re
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

try:
    from langdetect import detect
except ImportError:
    def detect(text):
        return 'en' if re.search(r'[a-zA-Z]', text) else 'vi'

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

UMLS_STY_TO_CATEGORY = {
    "Disease or Syndrome": "Disease", "Neoplastic Process": "Disease", "Pathologic Function": "Disease",
    "Mental or Behavioral Dysfunction": "Disease", "Cell or Molecular Dysfunction": "Disease", "Experimental Model of Disease": "Disease",
    "Congenital Abnormality": "Diagnosis", "Acquired Abnormality": "Diagnosis", "Anatomical Abnormality": "Diagnosis", "Injury or Poisoning": "Diagnosis",
    "Sign or Symptom": "Phenotype", "Finding": "Phenotype", "Clinical Attribute": "Phenotype", "Organism Attribute": "Phenotype",
    "Physiologic Function": "Phenotype", "Organism Function": "Phenotype", "Organ or Tissue Function": "Phenotype", "Cell Function": "Phenotype", "Biologic Function": "Phenotype",
    "Anatomical Structure": "Body Parts", "Body Location or Region": "Body Parts", "Body Part, Organ, or Organ Component": "Body Parts",
    "Body Space or Junction": "Body Parts", "Body Substance": "Body Parts", "Body System": "Body Parts", "Cell": "Body Parts",
    "Cell Component": "Body Parts", "Tissue": "Body Parts", "Embryonic Structure": "Body Parts", "Fully Formed Anatomical Structure": "Body Parts",
    "Clinical Drug": "Drugs", "Pharmacologic Substance": "Drugs", "Antibiotic": "Drugs", "Biologically Active Substance": "Drugs",
    "Hormone": "Drugs", "Vitamin": "Drugs", "Immunologic Factor": "Drugs", "Organic Chemical": "Drugs", 
    "Chemical": "Chemicals", "Chemical Viewed Functionally": "Chemicals", "Chemical Viewed Structurally": "Chemicals",
    "Inorganic Chemical": "Chemicals", "Element, Ion, or Isotope": "Chemicals", "Enzyme": "Chemicals", "Amino Acid, Peptide, or Protein": "Chemicals",
    "Nucleic Acid, Nucleoside, or Nucleotide": "Chemicals", "Gene or Genome": "Chemicals", "Molecular Sequence": "Chemicals",
    "Nucleotide Sequence": "Chemicals", "Amino Acid Sequence": "Chemicals", "Carbohydrate Sequence": "Chemicals", "Receptor": "Chemicals",
    "Substance": "Chemicals", "Hazardous or Poisonous Substance": "Chemicals", "Indicator, Reagent, or Diagnostic Aid": "Chemicals",
    "Therapeutic or Preventive Procedure": "Procedures", "Diagnostic Procedure": "Procedures", "Health Care Activity": "Procedures",
    "Research Activity": "Procedures", "Molecular Biology Research Technique": "Procedures", "Educational Activity": "Procedures",
    "Laboratory Procedure": "Labs", "Laboratory or Test Result": "Labs",
    "Medical Device": "Devices", "Drug Delivery Device": "Devices", "Research Device": "Devices"
}

class EntityExtractor:
    def __init__(self, mode: str, ner_model: str = "phobert", retrieval_model: str = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"):
        """
        EntityExtractor utility for unified medical entity extraction. Supports 4 different extraction modes

        Args:
            mode (str): The extraction strategy to use. Supported modes are:
                - 'quickumls': Spacy-based string matching with UMLS synonyms.
                - 'ner only': Uses a fine-tuned Named Entity Recognition model to extract text spans.
                - 'ner + retrieval': Uses NER to find spans, then SapBERT to map them to canonical CUIs.
                - 'doc_class': Document-level classification to identify implicit conditions.
            ner_model (str): Name or path of the NER model to load (e.g., 'phobert', 'vihealthbert').
            retrieval_model (str): Default SapBERT retrieval model to use for Vietnamese entity mapping.
        """
        self.mode = mode.lower()
        self.ner_model = ner_model
        
        self.retrieval_model_vi = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
        self.retrieval_model_en = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        
        # Caches for lazy loading
        self._ner_instance = None
        self._sapbert_vi = None
        self._sapbert_en = None
        self._doc_classifier_vi = None
        self._doc_classifier_en = None
        self._vihealthbert_tokenizer = None
        self._longformer_tokenizer = None
        
        self._mlb_classes = None
        self._base_df_cache = None
        self._external_kg_cache = None
        self._mrconso_cache = None
        
    def _detect_lang(self, text: str) -> str:
        # Clinical text often confuses langdetect because of Latin/medical terms.
        # A robust way is to check for specific Vietnamese diacritics.
        vi_chars = re.compile(r'[àáãạảăắằẵặẳâấầẫậẩđèéẽẹẻêếềễệểìíĩịỉòóõọỏôốồỗộổơớờỡợởùúũụủưứừữựửỳýỹỵỷ]', re.IGNORECASE)
        if vi_chars.search(text):
            return 'vi'
        return 'en'
            
    def _get_ner_instance(self, lang="vi"):
        ner_lang_mode = "english" if lang == "en" else "vietnamese"
        if self._ner_instance is None or getattr(self._ner_instance, 'mode', '') != ner_lang_mode:
            from modules.extend.model.inference_ner import NER
            self._ner_instance = NER(mode=ner_lang_mode, model_name=self.ner_model)
        return self._ner_instance
        
    def _get_sapbert_instance(self, lang="vi"):
        from modules.models.models import EmbeddingModels
        if lang == "en":
            if self._sapbert_en is None:
                self._sapbert_en = EmbeddingModels(model_choice=self.retrieval_model_en)
            return self._sapbert_en
        else:
            if self._sapbert_vi is None:
                self._sapbert_vi = EmbeddingModels(model_choice=self.retrieval_model_vi)
            return self._sapbert_vi

    def _get_mapped_db(self):
        if self._base_df_cache is None:
            map_path = os.path.join(project_root, "data", "viettel", "mapping", "mapped_entities_embedded.csv")
            try:
                df = pd.read_csv(map_path)
                def parse_emb(x):
                    floats = re.findall(r'-?\d+\.\d+(?:e-?\d+)?', str(x))
                    return np.array([float(f) for f in floats], dtype=np.float32)
                df['embedding'] = df['embedding'].apply(parse_emb)
                df['macro_type'] = df.get('original_type', df.get('mapped_type', 'Other'))
                self._base_df_cache = df
            except Exception as e:
                print(f"Error loading CSV base entities: {e}")
                self._base_df_cache = pd.DataFrame()
        return self._base_df_cache

    def _get_external_kg(self):
        if self._external_kg_cache is None:
            try:
                ekg_path = os.path.join(project_root, "data", "viettel", "mapping", "external_kg.parquet")
                df_ekg = pd.read_parquet(ekg_path)
                self._external_kg_cache = df_ekg[df_ekg['labels'].isin(['Diagnosis', 'Disease', 'Phenotype'])]
            except Exception as e:
                print(f"Error loading external_kg.parquet: {e}")
                self._external_kg_cache = pd.DataFrame()
        return self._external_kg_cache

    def _get_mlb_classes(self, lang="vi"):
        if lang == "en":
            if getattr(self, '_mlb_classes_en', None) is None:
                classes_file = os.path.join(project_root, "modules", "extend", 'model', "statedict", "dl_en", "classes_en.json")
                with open(classes_file, "r", encoding="utf-8") as f:
                    self._mlb_classes_en = json.load(f)
            return self._mlb_classes_en
        else:
            if getattr(self, '_mlb_classes_vi', None) is None:
                classes_file = os.path.join(project_root, "modules", "extend", 'model', "statedict", "dl_vi", "classes.json")
                with open(classes_file, "r", encoding="utf-8") as f:
                    self._mlb_classes_vi = json.load(f)
            return self._mlb_classes_vi

    def _get_doc_classifier(self, lang="vi"):
        classes = self._get_mlb_classes(lang)
        model_dir = os.path.join(project_root, "modules", "extend", 'model', "statedict", "dl_vi", "final_model")
        from transformers import AutoTokenizer
        from safetensors.torch import load_file
        
        if lang == "en":
            if self._doc_classifier_en is None:
                # English model: PLMICDModel (trained with LAAT head)
                from modules.extend.model.training.plmicd_model import PLMICDModel
                model_name = "yikuan8/Clinical-Longformer"
                self._doc_classifier_en = PLMICDModel(num_labels=len(classes), model_name=model_name)
                
                # Load custom weights if available for EN
                en_model_path = os.path.join(project_root, "modules", "extend", 'model', "statedict", "dl_en", "model_state.pt")
                if os.path.exists(en_model_path):
                    import torch
                    state_dict = torch.load(en_model_path, map_location='cpu')
                    # Strip _orig_mod. from keys (e.g. from torch.compile)
                    state_dict = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
                    
                    # Handle shape mismatches (e.g. 202 vs 575 classes)
                    current_state = self._doc_classifier_en.state_dict()
                    for k in list(state_dict.keys()):
                        if k in current_state and state_dict[k].shape != current_state[k].shape:
                            if len(state_dict[k].shape) == 1:
                                new_tensor = current_state[k].clone()
                                min_size = min(new_tensor.shape[0], state_dict[k].shape[0])
                                new_tensor[:min_size] = state_dict[k][:min_size]
                                state_dict[k] = new_tensor
                            elif len(state_dict[k].shape) == 2:
                                new_tensor = current_state[k].clone()
                                min_dim0 = min(new_tensor.shape[0], state_dict[k].shape[0])
                                min_dim1 = min(new_tensor.shape[1], state_dict[k].shape[1])
                                new_tensor[:min_dim0, :min_dim1] = state_dict[k][:min_dim0, :min_dim1]
                                state_dict[k] = new_tensor
                                
                    self._doc_classifier_en.load_state_dict(state_dict, strict=False)
                
                self._doc_classifier_en.eval()
                self._longformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
            return self._doc_classifier_en, self._longformer_tokenizer
        else:
            if self._doc_classifier_vi is None:
                # Vietnamese model: PLMICDModel
                from modules.extend.model.training.plmicd_model import PLMICDModel
                self._doc_classifier_vi = PLMICDModel(num_labels=len(classes), model_name="vihealthbert")
                state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                self._doc_classifier_vi.load_state_dict(state_dict, strict=False)
                self._doc_classifier_vi.eval()
                self._vihealthbert_tokenizer = AutoTokenizer.from_pretrained(model_dir)
            return self._doc_classifier_vi, self._vihealthbert_tokenizer

    def _get_cui_vocab_codes(self, cui: str) -> dict: 
        if self._mrconso_cache is None:
            optimized_path = os.path.join(project_root, 'data', 'UML', "MRCONSO_optimized.parquet")
            if os.path.exists(optimized_path):
                df = pd.read_parquet(optimized_path)
                
                # Pre-compute dictionary for O(1) lookups instead of O(N) pandas filtering
                vocab_dict = {}
                # Filter for relevant vocabularies first
                target_sabs = {'ICD10', 'ICD10CM', 'ICD10PCS', 'ICD9CM', 'RXNORM', 'SNOMEDCT_US', 'LNC', 'LOINC', 'MSH', 'OMIM', 'HPO', 'DRUGBANK', 'PUBCHEM', 'MEDLINE', 'PMID', 'PUBMED'}
                df_filtered = df[df['SAB'].isin(target_sabs)]
                
                for row in df_filtered.itertuples(index=False):
                    row_cui = row.CUI
                    if row_cui not in vocab_dict:
                        vocab_dict[row_cui] = {}
                        
                    sab = row.SAB
                    code = row.CODE
                    if sab in ('ICD10', 'ICD10CM', 'ICD10PCS'):
                        vocab_dict[row_cui]['icd10'] = code
                    elif sab == 'ICD9CM':
                        vocab_dict[row_cui]['icd9'] = code
                    elif sab == 'RXNORM':
                        vocab_dict[row_cui]['rxnorm'] = code
                    elif sab == 'SNOMEDCT_US':
                        vocab_dict[row_cui]['snomed'] = code
                    elif sab in ('LNC', 'LOINC'):
                        vocab_dict[row_cui]['loinc'] = code
                    elif sab == 'MSH':
                        vocab_dict[row_cui]['mesh'] = code
                    elif sab == 'OMIM':
                        vocab_dict[row_cui]['omim'] = code
                    elif sab == 'HPO':
                        vocab_dict[row_cui]['hpo'] = code
                    elif sab == 'DRUGBANK':
                        vocab_dict[row_cui]['drugbank'] = code
                    elif sab == 'PUBCHEM':
                        vocab_dict[row_cui]['pubchem'] = code
                    elif sab in ('MEDLINE', 'PMID', 'PUBMED'):
                        vocab_dict[row_cui]['pubmed'] = code
                        
                self._mrconso_cache = vocab_dict
            else:
                self._mrconso_cache = {}
                
        if not cui:
            return {}
            
        return self._mrconso_cache.get(cui, {})

    def extract(self, text: str, lang:str = '', **kwargs) -> pd.DataFrame:
        if len(lang) < 1:
            lang = self._detect_lang(text)
        
        if self.mode == "quickumls":
            res = self._run_quickumls(text)
        elif self.mode == "ner only":
            res = self._run_ner_only(text, lang)
        elif self.mode == "ner + retrieval":
            res = self._run_ner_retrieval(text, lang)
        elif self.mode == "doc_class":
            res = self._run_doc_class(text, lang, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        return pd.DataFrame(res)

    def _run_quickumls(self, text: str) -> List[Dict[str, Any]]:
        from modules.dataset_preprocessing.external.uml import spacy_quickumls
        import spacy
        try:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
        results = []
        for sent in sentences:
            df_sent = spacy_quickumls(sent)
            if not df_sent.empty:
                for _, row in df_sent.iterrows():
                    cui = row.get('cui')
                    codes = self._get_cui_vocab_codes(cui) if cui else {}
                    extracted_text = row.get('text', row.get('ngram', ''))
                    canonical_term = row.get('term', '')
                    sty_name = row.get('type', 'Unknown')
                    category = UMLS_STY_TO_CATEGORY.get(sty_name, sty_name)
                    
                    results.append({
                        "term": extracted_text,
                        "canonical_name": canonical_term,
                        "offset": (None, None),
                        "label": sty_name,
                        "category": category,
                        "cui": cui,
                        "similarity": row.get('similarity', 1.0),
                        "codes": codes
                    })
                    
        # Deduplicate by term keeping the highest similarity
        dedup_map = {}
        for res in results:
            t = res['term']
            if not t: continue
            if t not in dedup_map:
                dedup_map[t] = res
            else:
                if res['similarity'] > dedup_map[t]['similarity']:
                    dedup_map[t] = res
        results = list(dedup_map.values())

        # Fix offsets
        import re
        
        def get_first_occurrence(res):
            term = res['term']
            if not term: return float('inf')
            match = re.search(re.escape(term), text, re.IGNORECASE)
            return match.start() if match else float('inf')
            
        results.sort(key=get_first_occurrence)
        
        current_idx = 0
        for res in results:
            term = res['term']
            if not term:
                continue
                
            match_obj = re.search(re.escape(term), text[current_idx:], re.IGNORECASE)
            if match_obj:
                start_idx = current_idx + match_obj.start()
                end_idx = current_idx + match_obj.end()
                res['offset'] = (start_idx, end_idx)
                res['term'] = text[start_idx:end_idx] # Exact original case from text
                current_idx = end_idx
                
        # Sort by similarity descending as requested
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def _run_ner_only(self, text: str, lang: str) -> List[Dict[str, Any]]:
        ner = self._get_ner_instance(lang)
        ner_results = ner.extract_entities(text)
        return ner_results

    def _jaccard_similarity(self, str1: str, str2: str) -> float:
        s1 = set(str(str1).lower().split())
        s2 = set(str(str2).lower().split())
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def _run_ner_retrieval(self, text: str, lang: str) -> List[Dict[str, Any]]:
        from sklearn.metrics.pairwise import cosine_similarity
        ner = self._get_ner_instance(lang)
        ner_results = ner.extract_entities(text)
        
        if not ner_results:
            return []
            
        embedder = self._get_sapbert_instance(lang)
        base_df = self._get_mapped_db()
        
        extracted_texts = [ent["term"] for ent in ner_results]
        extracted_embeddings = embedder.encode_text(extracted_texts, batch_size=32, show_progress=False)
        
        results = []
        for idx, ent in enumerate(ner_results):
            term = ent["term"]
            label = ent["label"]
            offset = ent["offset"]
            
            emb = extracted_embeddings[idx:idx+1]
            mask = base_df['macro_type'] == label
            
            cui = None
            similarity = 1.0
            canonical_name = term
            
            if mask.any():
                type_df = base_df[mask]
                type_embeddings = np.vstack(type_df['embedding'].values)
                sims = cosine_similarity(emb, type_embeddings)[0]
                
                valid_indices = np.where(sims > 0.7)[0]
                
                if len(valid_indices) > 0:
                    best_idx = valid_indices[0]
                    best_score = -1.0
                    
                    for v_idx in valid_indices:
                        candidate_entity = str(type_df.iloc[v_idx]['entity'])
                        jaccard_sim = self._jaccard_similarity(term, candidate_entity)
                        
                        if jaccard_sim > best_score:
                            best_score = jaccard_sim
                            best_idx = v_idx
                            
                    cui_val = type_df.iloc[best_idx]['mapped_cui']
                    if pd.notna(cui_val) and cui_val != "":
                        cui = str(cui_val)
                    else:
                        cui = None
                        
                    similarity = float(sims[best_idx])
                    canonical_name = str(type_df.iloc[best_idx]['entity'])
                        
            codes = self._get_cui_vocab_codes(cui) if cui and cui.startswith("C") and not cui.startswith("C_") else {}
            
            results.append({
                "term": term,
                "canonical_name": canonical_name,
                "offset": offset,
                "label": label,
                "cui": cui,
                "similarity": similarity,
                "codes": codes
            })
            
        return results

    def _run_doc_class(self, text: str, lang: str, threshold: float = 0.5, dl_model: str = "auto") -> List[Dict[str, Any]]:
        # Override lang based on the chosen dl_model architecture
        if dl_model == "short":
            lang = "vi"
        elif dl_model == "long":
            lang = "en"
            
        model, tokenizer = self._get_doc_classifier(lang)
        classes = self._get_mlb_classes(lang)
        ekg_df = self._get_external_kg()
        
        encodings = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        # Compute predictions
        with torch.no_grad():
            if lang == "en":
                # PLMICDModel returns SequenceClassifierOutput with logits
                outputs = model(input_ids=encodings['input_ids'], attention_mask=encodings['attention_mask'])
            else:
                # AutoModelForSequenceClassification returns SequenceClassifierOutput
                outputs = model(**encodings)
                    
            probs = torch.sigmoid(outputs.logits).squeeze().numpy()
            
        if probs.ndim == 0:
            probs = [probs.item()]
            
        results = []
        for idx, prob in enumerate(probs):
            if prob >= threshold:
                cui_class = classes[idx]
                codes = self._get_cui_vocab_codes(cui_class)
                canonical_name = cui_class
                
                # Fetch translation
                if not ekg_df.empty:
                    match = ekg_df[ekg_df['uml_id'] == cui_class]
                    if not match.empty:
                        row = match.iloc[0]
                        cols_to_check = ['name', 'medgemma_trans', 'qwen_trans', 'map_trans'] if lang == "en" else ['medgemma_trans', 'qwen_trans', 'map_trans', 'name']
                        for col in cols_to_check:
                            val = row.get(col)
                            if pd.notna(val) and val:
                                canonical_name = str(val)
                                break
                                
                if canonical_name == cui_class:
                    base_df = self._get_mapped_db()
                    match_base = base_df[base_df['mapped_cui'] == cui_class]
                    if not match_base.empty:
                        canonical_name = match_base.iloc[0]['entity']
                        
                # Lexical matching for offset
                start_idx = None
                end_idx = None
                extracted_text = canonical_name
                
                potential_terms = [canonical_name]
                if not ekg_df.empty:
                    match = ekg_df[ekg_df['uml_id'] == cui_class]
                    if not match.empty:
                        r = match.iloc[0]
                        for c in ['medgemma_trans', 'qwen_trans', 'map_trans', 'name']:
                            val = r.get(c)
                            if pd.notna(val) and val:
                                potential_terms.append(str(val))
                                
                for term in potential_terms:
                    match_obj = re.search(re.escape(term), text, re.IGNORECASE)
                    if match_obj:
                        start_idx = match_obj.start()
                        end_idx = match_obj.end()
                        extracted_text = text[start_idx:end_idx]
                        break
                        
                results.append({
                    "term": extracted_text,
                    "canonical_name": canonical_name,
                    "offset": (start_idx, end_idx),
                    "label": "Diagnosis",
                    "cui": cui_class,
                    "similarity": float(prob),
                    "codes": codes
                })
                
        return results
