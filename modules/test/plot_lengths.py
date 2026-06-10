import os
import json
import matplotlib.pyplot as plt

def get_lengths(file_path):
    lengths = []
    if not os.path.exists(file_path):
        return lengths
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            text = data.get("text", "")
            # Calculate length by number of words
            words = text.split()
            lengths.append(len(words))
    return lengths

base_dir = r"d:\Study\Education\Projects\Thesis\data\viettel\vietnamese_ner\training\vietnamese\document_classification"
files = ["doc_class_train.jsonl", "doc_class_dev.jsonl", "doc_class_test.jsonl"]

all_lengths = []
for file in files:
    path = os.path.join(base_dir, file)
    print(f"Processing {file}...")
    all_lengths.extend(get_lengths(path))

print(f"Total sentences processed: {len(all_lengths)}")
print(f"Max length: {max(all_lengths)} words")
print(f"Average length: {sum(all_lengths)/len(all_lengths):.2f} words")

plt.figure(figsize=(12, 6))
plt.hist(all_lengths, bins=range(0, max(all_lengths) + 5, 2), color='#4CAF50', edgecolor='black', alpha=0.7)
plt.title('Sentence Length Distribution (Document Classification Dataset)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Words', fontsize=12)
plt.ylabel('Frequency (Number of Sentences)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Highlight the 95th percentile
p95 = sorted(all_lengths)[int(len(all_lengths) * 0.95)]
plt.axvline(x=p95, color='red', linestyle='dashed', linewidth=2, label=f'95th Percentile: {p95} words')
plt.legend()

# Save the figure directly to the artifacts directory
output_path = r"C:\Users\admin\.gemini\antigravity-ide\brain\a1a09863-cab0-4005-851c-17c3bfce241b\doc_class_sentence_lengths.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Saved histogram to {output_path}")
