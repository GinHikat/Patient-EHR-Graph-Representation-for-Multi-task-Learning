import os
import re
import csv
from collections import defaultdict

bc5cdr_dir = r"d:\Study\Education\Projects\Thesis\data\viettel\vietnamese_ner\training\english\BC5CDR"
chem_dir = r"d:\Study\Education\Projects\Thesis\data\viettel\vietnamese_ner\training\english\ChemDisGene"
out_dir = r"d:\Study\Education\Projects\Thesis\data\viettel\vietnamese_ner\training\english"

# Files to process
bc5cdr_files = [
    os.path.join(bc5cdr_dir, "CDR_TrainingSet.PubTator.txt"),
    os.path.join(bc5cdr_dir, "CDR_DevelopmentSet.PubTator.txt"),
    os.path.join(bc5cdr_dir, "CDR_TestSet.PubTator.txt")
]

chem_files = [
    (os.path.join(chem_dir, "train_abstracts.txt"), os.path.join(chem_dir, "train_relationships.tsv")),
    (os.path.join(chem_dir, "dev_abstracts.txt"), os.path.join(chem_dir, "dev_relationships.tsv")),
    (os.path.join(chem_dir, "test_abstracts.txt"), os.path.join(chem_dir, "test_relationships.tsv"))
]

# Stats
stats = {
    'ner_tokens': 0,
    'ner_entities': 0,
    'ner_documents': 0,
    're_relations': 0,
    'entity_types': set(),
    'relation_types': set()
}

def tokenize_and_tag(text, entities):
    # Sort entities by start offset
    entities = sorted(entities, key=lambda x: x['start'])
    
    char_to_label = ['O'] * len(text)
    
    for ent in entities:
        if ent['type'] not in ['Chemical', 'Disease']:
            continue
            
        stats['entity_types'].add(ent['type'])
        stats['ner_entities'] += 1
        
        start = ent['start']
        end = min(ent['end'], len(text)) # safety clamp
        
        char_to_label[start] = f"B-{ent['type']}"
        for i in range(start + 1, end):
            char_to_label[i] = f"I-{ent['type']}"
            
    tokens = []
    labels = []
    
    # strictly split by whitespace
    for match in re.finditer(r'\S+', text):
        token_str = match.group()
        token_start = match.start()
        token_end = match.end()
        
        token_label = 'O'
        
        # Look for B- tag first
        for i in range(token_start, token_end):
            if char_to_label[i].startswith('B-'):
                token_label = char_to_label[i]
                break
        
        # Look for I- tag if no B-
        if token_label == 'O':
            for i in range(token_start, token_end):
                if char_to_label[i].startswith('I-'):
                    token_label = char_to_label[i]
                    break
                    
        tokens.append(token_str)
        labels.append(token_label)
        
    stats['ner_tokens'] += len(tokens)
    stats['ner_documents'] += 1
    
    return tokens, labels


print("Starting unified dataset generation...")

with open(os.path.join(out_dir, "unified_ner.conll"), "w", encoding="utf-8") as out_conll, \
     open(os.path.join(out_dir, "unified_re.csv"), "w", encoding="utf-8", newline='') as out_csv_file:
     
    csv_writer = csv.writer(out_csv_file)
    csv_writer.writerow(["Input", "Head", "Rel", "Target"])
     
    # Process BC5CDR
    for fpath in bc5cdr_files:
        if not os.path.exists(fpath):
            print(f"Skipping {fpath} (not found)")
            continue
        print(f"Processing {os.path.basename(fpath)} (BC5CDR)")
        with open(fpath, "r", encoding="utf-8") as f:
            docs = f.read().split('\n\n')
            for doc in docs:
                if not doc.strip(): continue
                lines = doc.strip().split('\n')
                
                title = ""
                abstract = ""
                entities = []
                relations = []
                mesh_to_text = defaultdict(set)
                
                for line in lines:
                    parts = line.split('\t')
                    if '|t|' in line:
                        _, title = line.split('|t|', 1)
                    elif '|a|' in line:
                        _, abstract = line.split('|a|', 1)
                    elif len(parts) == 6:
                        pid, start, end, text, etype, mesh = parts
                        # We only care about Chemical and Disease
                        if etype in ['Chemical', 'Disease']:
                            entities.append({'start': int(start), 'end': int(end), 'type': etype, 'text': text, 'mesh': mesh})
                            # In BC5CDR, multiple mesh IDs might be joined by |
                            for m in mesh.split('|'):
                                mesh_to_text[m].add(text)
                    elif len(parts) == 4 and parts[1] == 'CID':
                        pid, rel, mesh1, mesh2 = parts
                        relations.append((mesh1, 'Cause', mesh2))

                if title and abstract:
                    full_text = title + " " + abstract
                else:
                    full_text = title or abstract
                    
                for r in relations:
                    head_mesh, rel_type, target_mesh = r
                    head_texts = mesh_to_text.get(head_mesh, set())
                    target_texts = mesh_to_text.get(target_mesh, set())
                    
                    if head_texts and target_texts:
                        head_str = sorted(head_texts)[0]
                        target_str = sorted(target_texts)[0]
                        csv_writer.writerow([full_text, head_str, rel_type, target_str])
                        stats['re_relations'] += 1
                        stats['relation_types'].add(rel_type)
                    
                tokens, labels = tokenize_and_tag(full_text, entities)
                for t, l in zip(tokens, labels):
                    out_conll.write(f"{t}\t{l}\n")
                out_conll.write("\n")

    # Process ChemDisGene
    for abstract_f, rel_f in chem_files:
        if not os.path.exists(abstract_f) or not os.path.exists(rel_f):
            print(f"Skipping {abstract_f} (not found)")
            continue
        print(f"Processing {os.path.basename(abstract_f)} (ChemDisGene)")
        
        doc_texts = {}
        doc_mesh_to_text = {}
        
        # NER
        with open(abstract_f, "r", encoding="utf-8") as f:
            docs = f.read().split('\n\n')
            for doc in docs:
                if not doc.strip(): continue
                lines = doc.strip().split('\n')
                
                doc_id = ""
                title = ""
                abstract = ""
                entities = []
                mesh_to_text = defaultdict(set)
                
                for line in lines:
                    parts = line.split('\t')
                    if '|t|' in line:
                        doc_id, title = line.split('|t|', 1)
                    elif '|a|' in line:
                        doc_id, abstract = line.split('|a|', 1)
                    elif len(parts) == 6:
                        pid, start, end, text, etype, mesh = parts
                        if etype in ['Chemical', 'Disease']:
                            entities.append({'start': int(start), 'end': int(end), 'type': etype, 'text': text, 'mesh': mesh})
                            for m in mesh.split('|'):
                                mesh_to_text[m].add(text)
                        
                if title and abstract:
                    full_text = title + " " + abstract
                else:
                    full_text = title or abstract
                    
                if doc_id:
                    doc_texts[doc_id] = full_text
                    doc_mesh_to_text[doc_id] = mesh_to_text
                    
                tokens, labels = tokenize_and_tag(full_text, entities)
                for t, l in zip(tokens, labels):
                    out_conll.write(f"{t}\t{l}\n")
                out_conll.write("\n")
                
        # RE
        with open(rel_f, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4: continue
                pid, rel, mesh1, mesh2 = parts
                
                if rel == 'chem_disease:therapeutic':
                    mapped_rel = 'Treat'
                elif rel == 'chem_disease:marker/mechanism':
                    mapped_rel = 'Cause'
                else:
                    continue # Ignore all other relations (including gene-related)
                    
                input_text = doc_texts.get(pid, "")
                mesh_to_text = doc_mesh_to_text.get(pid, {})
                
                head_texts = mesh_to_text.get(mesh1, set())
                target_texts = mesh_to_text.get(mesh2, set())
                
                if input_text and head_texts and target_texts:
                    head_str = sorted(head_texts)[0]
                    target_str = sorted(target_texts)[0]
                    csv_writer.writerow([input_text, head_str, mapped_rel, target_str])
                    stats['re_relations'] += 1
                    stats['relation_types'].add(mapped_rel)

# Write stats
stats_file = os.path.join(out_dir, "dataset_stats.txt")
with open(stats_file, "w", encoding="utf-8") as f:
    f.write("Unified Dataset Statistics\n")
    f.write("==========================\n\n")
    f.write(f"Total Documents processed for NER: {stats['ner_documents']}\n")
    f.write(f"Total Tokens in NER dataset: {stats['ner_tokens']}\n")
    f.write(f"Total Entities in NER dataset: {stats['ner_entities']}\n")
    f.write(f"Entity Types: {', '.join(stats['entity_types'])}\n\n")
    f.write(f"Total Relations in RE dataset: {stats['re_relations']}\n")
    f.write(f"Relation Types: {', '.join(stats['relation_types'])}\n")

print(f"Done! Stats saved to {stats_file}")
