import json
import os
import random

label_mapping = {
    "ten_benh": "Disease/Symptom",
    "trieu_chung_benh": "Disease/Symptom",
    "nguyen_nhan_benh": "Disease/Symptom",
    "Symptom_and_Disease": "Disease/Symptom",
    "SYMPTOM_AND_DISEASE": "Disease/Symptom",
    "bien_phap_chan_doan": "Procedure/Treatment",
    "bien_phap_dieu_tri": "Procedure/Treatment",
    "DiagnosticProcedure": "Procedure/Treatment",
    "medical_procedure": "Procedure/Treatment",
    "drug": "Drug"
}

def parse_conll(file_path, is_io=False):
    sentences = []
    if not os.path.exists(file_path): return sentences
    with open(file_path, 'r', encoding='utf-8') as f:
        current_sentence = []
        current_tag = None
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                current_tag = None
                continue
            parts = line.split('\t') if '\t' in line else line.split(' ')
            if len(parts) >= 2:
                word, tag = parts[0], parts[-1]
                
                if tag == 'O':
                    current_sentence.append((word, 'O'))
                    current_tag = None
                    continue
                
                if tag.startswith('B-'):
                    tag_name = tag[2:]
                    if tag_name in label_mapping:
                        mapped_tag = label_mapping[tag_name]
                        current_sentence.append((word, f'B-{mapped_tag}'))
                        current_tag = mapped_tag
                    else:
                        current_sentence.append((word, 'O'))
                        current_tag = None
                elif tag.startswith('I-'):
                    tag_name = tag[2:]
                    if tag_name in label_mapping:
                        mapped_tag = label_mapping[tag_name]
                        if is_io:
                            if current_tag == mapped_tag:
                                current_sentence.append((word, f'I-{mapped_tag}'))
                            else:
                                current_sentence.append((word, f'B-{mapped_tag}'))
                                current_tag = mapped_tag
                        else:
                            if current_tag == mapped_tag:
                                current_sentence.append((word, f'I-{mapped_tag}'))
                            else:
                                current_sentence.append((word, f'B-{mapped_tag}'))
                                current_tag = mapped_tag
                    else:
                        current_sentence.append((word, 'O'))
                        current_tag = None
                else:
                    current_sentence.append((word, 'O'))
                    current_tag = None
        if current_sentence:
            sentences.append(current_sentence)
    return sentences

def parse_vimq(file_path):
    sentences = []
    if not os.path.exists(file_path): return sentences
    data = json.load(open(file_path, 'r', encoding='utf-8'))
    for item in data:
        words = item['sentence'].split()
        tags = ['O'] * len(words)
        for start, end, label in item['seq_label']:
            if label in label_mapping:
                mapped_tag = label_mapping[label]
                for i in range(start, end + 1):
                    if i < len(words):
                        if i == start:
                            tags[i] = f'B-{mapped_tag}'
                        else:
                            tags[i] = f'I-{mapped_tag}'
        
        current_sentence = [(w, t) for w, t in zip(words, tags)]
        sentences.append(current_sentence)
    return sentences

# 1. Parse all datasets
all_sentences = []
print("Parsing ViMedNer...")
all_sentences.extend(parse_conll('ViMedNer/data/train.txt'))
all_sentences.extend(parse_conll('ViMedNer/data/dev.txt'))
all_sentences.extend(parse_conll('ViMedNer/data/test.txt'))

print("Parsing VietBioNER...")
all_sentences.extend(parse_conll('VietBioNER/data_supervised_learning/train.txt', is_io=True))
all_sentences.extend(parse_conll('VietBioNER/data_supervised_learning/dev.txt', is_io=True))
all_sentences.extend(parse_conll('VietBioNER/data_supervised_learning/test.txt', is_io=True))

print("Parsing PhoNER_COVID19...")
all_sentences.extend(parse_conll('PhoNER_COVID19/data/syllable/train_syllable.conll'))
all_sentences.extend(parse_conll('PhoNER_COVID19/data/syllable/dev_syllable.conll'))
all_sentences.extend(parse_conll('PhoNER_COVID19/data/syllable/test_syllable.conll'))

print("Parsing vimq...")
all_sentences.extend(parse_vimq('vimq/data/train.json'))
all_sentences.extend(parse_vimq('vimq/data/dev.json'))
all_sentences.extend(parse_vimq('vimq/data/test.json'))

print(f"Total unified sentences: {len(all_sentences)}")

# 2. Export to CoNLL
print("Exporting unified CoNLL to unified_ner_dataset.conll...")
with open('unified_ner_dataset.conll', 'w', encoding='utf-8') as f:
    for sentence in all_sentences:
        for word, tag in sentence:
            f.write(f"{word}\t{tag}\n")
        f.write("\n")

# 3. Export to JSONL (ShareGPT)
print("Exporting unified JSONL to unified_qwen_dataset.jsonl...")
system_prompt = "Bạn là một chuyên gia y tế AI. Nhiệm vụ của bạn là trích xuất các thực thể y tế từ văn bản và trả về dưới dạng JSON list. Các loại thực thể hợp lệ bao gồm: Disease/Symptom, Procedure/Treatment, Drug. Kết quả trả về phải chứa entity (tên thực thể), type (loại thực thể), start_token và end_token (chỉ số của từ bắt đầu và kết thúc trong câu, bắt đầu từ 0)."

random.shuffle(all_sentences)

with open('unified_qwen_dataset.jsonl', 'w', encoding='utf-8') as f:
    for sentence in all_sentences:
        text = ' '.join(word for word, tag in sentence)
        entities = []
        
        current_entity_words = []
        current_type = None
        start_idx = -1
        
        for i, (word, tag) in enumerate(sentence):
            if tag.startswith('B-'):
                if current_entity_words:
                    entities.append({
                        "entity": ' '.join(current_entity_words),
                        "type": current_type,
                        "start_token": start_idx,
                        "end_token": i - 1
                    })
                current_type = tag[2:]
                current_entity_words = [word]
                start_idx = i
            elif tag.startswith('I-'):
                if current_type == tag[2:]:
                    current_entity_words.append(word)
                else: # Invalid I- without B-
                    if current_entity_words:
                        entities.append({
                            "entity": ' '.join(current_entity_words),
                            "type": current_type,
                            "start_token": start_idx,
                            "end_token": i - 1
                        })
                    current_type = tag[2:]
                    current_entity_words = [word]
                    start_idx = i
            else: # O
                if current_entity_words:
                    entities.append({
                        "entity": ' '.join(current_entity_words),
                        "type": current_type,
                        "start_token": start_idx,
                        "end_token": i - 1
                    })
                    current_entity_words = []
                    current_type = None
                    start_idx = -1
        
        # Add the last one if sentence ends on an entity
        if current_entity_words:
            entities.append({
                "entity": ' '.join(current_entity_words),
                "type": current_type,
                "start_token": start_idx,
                "end_token": len(sentence) - 1
            })
            
        json_obj = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
                {"role": "assistant", "content": json.dumps(entities, ensure_ascii=False)}
            ]
        }
        f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

print("Done!")
