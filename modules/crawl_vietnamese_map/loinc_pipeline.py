import os
import csv
import json
import time
import math
import requests
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Concurrency configurations (1 worker for absolute stability)
RETRIES = 5
TIMEOUT = 30
PAGE_SIZE = 100
DELAY_BETWEEN_PAGES = 0.2

def fetch_page_with_retry(session, url):
    time.sleep(DELAY_BETWEEN_PAGES)  # Polite delay
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    for attempt in range(RETRIES):
        try:
            r = session.get(url, headers=headers, timeout=TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if 'results' in data:
                    return data
                else:
                    print(f"[WARN] Invalid response format for {url} (Attempt {attempt+1}/{RETRIES})")
            else:
                print(f"[WARN] HTTP {r.status_code} for {url} (Attempt {attempt+1}/{RETRIES})")
        except Exception as e:
            print(f"[WARN] Request error {e} for {url} (Attempt {attempt+1}/{RETRIES})")
        time.sleep(1.5 ** attempt) # Backoff delay
    return None

def extract_value(field_data):
    if isinstance(field_data, dict):
        return field_data.get('value')
    return field_data

def stream_write_records(records, file_path, fieldnames):
    try:
        with open(file_path, 'a', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for record in records:
                row = {field: extract_value(record.get(field)) for field in fieldnames}
                writer.writerow(row)
    except Exception as e:
        print(f"[WARN] Error stream-writing records to CSV: {e}")

def main():
    start_time = time.time()
    session = requests.Session()
    
    file_csv = os.path.join(os.path.dirname(__file__), 'vietnamese_loinc.csv')
    file_json = os.path.join(os.path.dirname(__file__), 'vietnamese_loinc.json')
    
    fieldnames = [
        'id', 'number', 'long_common_name', 'component', 'shortname',
        'property', 'time_aspect', 'system', 'scale_type', 'method_type', 'common_test_rank'
    ]
    
    # Initialize the output CSV file
    try:
        with open(file_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        print("[INFO] Output CSV file initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize CSV file: {e}")
        return
        
    # Fetch page 1 to determine total count
    print("[INFO] Fetching page 1 to retrieve metadata...")
    first_page_url = f"https://loinc.whiteneuron.com/api/loinc/loinc-table/?page_size={PAGE_SIZE}&page=1&q=&ordering=number"
    first_page_data = fetch_page_with_retry(session, first_page_url)
    
    if not first_page_data:
        print("[ERROR] Failed to retrieve page 1. Exiting.")
        return
        
    total_count = first_page_data.get('count', 0)
    total_pages = math.ceil(total_count / PAGE_SIZE)
    print(f"[INFO] Total records found: {total_count}")
    print(f"[INFO] Total pages to crawl: {total_pages}")
    
    # Track metrics
    total_records_saved = 0
    all_raw_records = []
    
    # Process page 1 data first
    p1_results = first_page_data.get('results', [])
    stream_write_records(p1_results, file_csv, fieldnames)
    all_raw_records.extend(p1_results)
    total_records_saved += len(p1_results)
    print(f"  [Progress] Page 1/{total_pages} processed | {total_records_saved}/{total_count} records saved")
    
    # Loop page-by-page from page 2
    for page in range(2, total_pages + 1):
        url = f"https://loinc.whiteneuron.com/api/loinc/loinc-table/?page_size={PAGE_SIZE}&page={page}&q=&ordering=number"
        page_data = fetch_page_with_retry(session, url)
        
        if not page_data:
            print(f"[ERROR] Failed to retrieve page {page} after retries. Skipping page.")
            continue
            
        results = page_data.get('results', [])
        if not results:
            print(f"[WARN] No records found on page {page}.")
            continue
            
        stream_write_records(results, file_csv, fieldnames)
        all_raw_records.extend(results)
        total_records_saved += len(results)
        
        if page % 10 == 0 or page == total_pages:
            elapsed = time.time() - start_time
            rate = total_records_saved / elapsed if elapsed > 0 else 0
            eta_seconds = (total_count - total_records_saved) / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            print(f"  [Progress] Page {page:4d}/{total_pages:4d} processed | {total_records_saved:6d}/{total_count:6d} records | Speed: {rate:.1f} rec/s | ETA: {eta_minutes:.1f}m")
            
    # Post-processing: Read, sort by LOINC number, and rewrite final sorted files
    print(f"\n[INFO] Initial crawl completed. Exported {total_records_saved} records.")
    print("[INFO] Sorting records by LOINC number...")
    
    flat_records = []
    for r in all_raw_records:
        flat_records.append({field: extract_value(r.get(field)) for field in fieldnames})
        
    flat_records_sorted = sorted(flat_records, key=lambda x: str(x.get('number', '')))
    
    # Write sorted CSV
    try:
        with open(file_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in flat_records_sorted:
                writer.writerow(row)
        print(f"[SUCCESS] Sorted CSV exported to: {file_csv}")
    except Exception as e:
        print(f"[ERROR] Failed writing sorted CSV: {e}")
        
    # Write JSON
    try:
        with open(file_json, 'w', encoding='utf-8') as f:
            json.dump(flat_records_sorted, f, ensure_ascii=False, indent=2)
        print(f"[SUCCESS] JSON exported to: {file_json}")
    except Exception as e:
        print(f"[ERROR] Failed saving JSON: {e}")
        
    print(f"\n[INFO] Full pipeline finished in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
