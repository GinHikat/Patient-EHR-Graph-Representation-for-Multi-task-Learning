import json
from collections import defaultdict
import os

def get_conll_tags_and_examples(file_path):
    tags = defaultdict(list)
    if not os.path.exists(file_path): return tags
    with open(file_path, 'r', encoding='utf-8') as f:
        current_sentence = []
        current_entity_words = []
        current_tag = None
        for line in f:
            line = line.strip()
            if not line:
                if current_tag and current_entity_words:
                    tags[current_tag].append((' '.join(current_entity_words), ' '.join([w for w, _ in current_sentence])))
                current_sentence = []
                current_entity_words = []
                current_tag = None
                continue
            parts = line.split('\t') if '\t' in line else line.split(' ')
            if len(parts) >= 2:
                word, tag = parts[0], parts[-1]
                current_sentence.append((word, tag))
                
                if tag.startswith('B-'):
                    if current_tag and current_entity_words:
                        tags[current_tag].append((' '.join(current_entity_words), ' '.join([w for w, _ in current_sentence[:-1]])))
                    current_tag = tag[2:]
                    current_entity_words = [word]
                elif tag.startswith('I-'):
                    tag_type = tag[2:]
                    if current_tag == tag_type:
                        current_entity_words.append(word)
                    else:
                        # Treat first I- after O or different I- as B-
                        if current_tag and current_entity_words:
                            tags[current_tag].append((' '.join(current_entity_words), ' '.join([w for w, _ in current_sentence[:-1]])))
                        current_tag = tag_type
                        current_entity_words = [word]
                else:
                    if current_tag and current_entity_words:
                        tags[current_tag].append((' '.join(current_entity_words), ' '.join([w for w, _ in current_sentence[:-1]])))
                    current_tag = None
                    current_entity_words = []
    return {k: v[0] for k, v in tags.items() if v}

with open('tags_output_utf8.txt', 'w', encoding='utf-8') as out_f:
    out_f.write('=== ViMedNer ===\n')
    vimedner = get_conll_tags_and_examples('ViMedNer/data/train.txt')
    for tag, (ent, sent) in vimedner.items():
        out_f.write(f"- {tag}: '{ent}'\n")
        out_f.write(f"  Example: \"{sent}\"\n\n")

    out_f.write('=== VietBioNER ===\n')
    vietbio = get_conll_tags_and_examples('VietBioNER/data_supervised_learning/train.txt')
    for tag, (ent, sent) in vietbio.items():
        out_f.write(f"- {tag}: '{ent}'\n")
        out_f.write(f"  Example: \"{sent}\"\n\n")

    out_f.write('=== PhoNER_COVID19 ===\n')
    phoner = get_conll_tags_and_examples('PhoNER_COVID19/data/syllable/train_syllable.conll')
    for tag, (ent, sent) in phoner.items():
        out_f.write(f"- {tag}: '{ent}'\n")
        out_f.write(f"  Example: \"{sent}\"\n\n")

    out_f.write('=== vimq ===\n')
    vimq_tags = defaultdict(list)
    if os.path.exists('vimq/data/train.json'):
        data = json.load(open('vimq/data/train.json', encoding='utf-8'))
        for item in data:
            sent = item['sentence'].split()
            for start, end, label in item['seq_label']:
                ent = ' '.join(sent[start:end+1])
                if label not in vimq_tags:
                    vimq_tags[label] = (ent, item['sentence'])
    for tag, (ent, sent) in vimq_tags.items():
        out_f.write(f"- {tag}: '{ent}'\n")
        out_f.write(f"  Example: \"{sent}\"\n\n")
