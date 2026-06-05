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
