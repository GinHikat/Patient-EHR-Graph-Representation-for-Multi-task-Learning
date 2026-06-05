import os
import csv
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def clean_id(corrupt_id):
    # Match standard ICD-10 code format (a letter followed by 2 digits, optionally a dot and 1-2 digits)
    m = re.match(r'^([A-Z]\d{2}(?:\.\d{1,2})?)', corrupt_id)
    if m:
        return m.group(1)
    return corrupt_id

def main():
    file_diseases = os.path.join(os.path.dirname(__file__), 'vietnamese_icd10_diseases.csv')
    
    if not os.path.exists(file_diseases):
        print("[ERROR] Disease CSV file not found.")
        return
        
    print(f"[INFO] Cleaning corrupt IDs in {file_diseases}...")
    rows = []
    cleaned_count = 0
    
    try:
        with open(file_diseases, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                old_id = row.get('ID', '')
                new_id = clean_id(old_id)
                if old_id != new_id:
                    row['ID'] = new_id
                    cleaned_count += 1
                rows.append(row)
                
        if cleaned_count > 0:
            with open(file_diseases, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            print(f"[SUCCESS] Cleaned {cleaned_count} IDs in CSV file.")
        else:
            print("[INFO] No corrupt IDs found in CSV file.")
    except Exception as e:
        print(f"[ERROR] Failed to clean CSV: {e}")

if __name__ == '__main__':
    main()
