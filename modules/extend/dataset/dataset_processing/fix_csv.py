import re

file_path = r'data\viettel\vietnamese_ner\mapped_entities.csv'

lines = []
with open(file_path, 'r', encoding='utf-8-sig') as f:
    for line in f:
        # Split fused lines like `1.00"ceftriaxone"` or `0.0"cấy tế bào"`
        # We look for a float ending (like 1.00 or 0.0 or 0.75) immediately followed by a quote "
        fixed_line = re.sub(r'(\d+\.\d{1,2})\"', r'\1\n"', line)
        
        # In case there are multiple fusions on one physical line
        split_lines = fixed_line.split('\n')
        for sl in split_lines:
            if sl.strip():
                lines.append(sl.strip() + '\n')

with open(file_path, 'w', encoding='utf-8-sig') as f:
    f.writelines(lines)

print("CSV fixed successfully!")
