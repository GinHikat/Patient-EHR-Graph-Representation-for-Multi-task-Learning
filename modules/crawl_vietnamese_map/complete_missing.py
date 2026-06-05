import os
import csv
import json
import time
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Concurrency configurations for recovery phase
MAX_WORKERS = 1
RETRIES = 5
TIMEOUT = 30


# Thread-safe locks
print_lock = threading.Lock()
list_lock = threading.Lock()

def safe_print(msg):
    with print_lock:
        print(msg, flush=True)

# Same parsing logic from icd10_pipeline.py
def fetch_with_retry(session, url):
    time.sleep(0.2) # Small safety delay to prevent rate-limiting
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }


    for attempt in range(RETRIES):
        try:
            r = session.get(url, headers=headers, timeout=TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "success":
                    return data
                else:
                    safe_print(f"[WARN] API status error for {url}: {data.get('message')} (Attempt {attempt+1}/{RETRIES})")
            else:
                safe_print(f"[WARN] HTTP {r.status_code} for {url} (Attempt {attempt+1}/{RETRIES})")
        except Exception as e:
            safe_print(f"[WARN] Request error {e} for {url} (Attempt {attempt+1}/{RETRIES})")
        time.sleep(1.5 ** attempt) # Backoff delay
    return None

def parse_group_description(html_content, group_id, level):
    soup = BeautifulSoup(html_content, 'html.parser')
    for meta in soup.find_all(class_='clipboard-meta'):
        meta.decompose()
    block_div = soup.find('div', class_=lambda c: c and (c.startswith('block-') or c == 'type'))
    if not block_div:
        block_div = soup
        
    row = block_div.find('div', class_='row')
    if not row:
        return None
        
    column_content = row.find('div', class_='column-content')
    if not column_content:
        return None
        
    column_layouts = column_content.find_all('div', class_='column-layout')
    if len(column_layouts) < 2:
        return None
        
    layout_vi = column_layouts[0]
    layout_en = column_layouts[1]
    
    # Parse Names
    h_tags_vi = layout_vi.find_all(['h1', 'h2', 'h3', 'h4'])
    name_vi = ""
    for h in h_tags_vi:
        content_div = h.find(class_='content')
        text = content_div.get_text(strip=True) if content_div else h.get_text(strip=True)
        if text:
            name_vi = text
            break
            
    h_tags_en = layout_en.find_all(['h1', 'h2', 'h3', 'h4'])
    name_en = ""
    for h in h_tags_en:
        content_div = h.find(class_='content')
        text = content_div.get_text(strip=True) if content_div else h.get_text(strip=True)
        if text:
            name_en = text
            break
            
    # Parse Includes / Excludes
    includes_vi, excludes_vi = [], []
    dls_vi = layout_vi.find_all('dl', class_='content')
    for dl in dls_vi:
        dt = dl.find('dt')
        if dt:
            dt_text = dt.get_text(strip=True).lower()
            if 'bao gồm' in dt_text:
                lis = dl.find_all('li')
                for li in lis:
                    content_div = li.find(class_='content')
                    includes_vi.append(content_div.get_text(strip=True) if content_div else li.get_text(strip=True))
            elif 'loại trừ' in dt_text:
                lis = dl.find_all('li')
                for li in lis:
                    content_div = li.find(class_='content')
                    excludes_vi.append(content_div.get_text(strip=True) if content_div else li.get_text(strip=True))
                    
    includes_en, excludes_en = [], []
    dls_en = layout_en.find_all('dl', class_='content')
    for dl in dls_en:
        dt = dl.find('dt')
        if dt:
            dt_text = dt.get_text(strip=True).lower()
            if 'include' in dt_text:
                lis = dl.find_all('li')
                for li in lis:
                    content_div = li.find(class_='content')
                    includes_en.append(content_div.get_text(strip=True) if content_div else li.get_text(strip=True))
            elif 'exclude' in dt_text:
                lis = dl.find_all('li')
                for li in lis:
                    content_div = li.find(class_='content')
                    excludes_en.append(content_div.get_text(strip=True) if content_div else li.get_text(strip=True))
                    
    return {
        'id': group_id,
        'level': level,
        'name_vi': name_vi,
        'name_en': name_en,
        'includes_vi': " | ".join(includes_vi) if includes_vi else "",
        'includes_en': " | ".join(includes_en) if includes_en else "",
        'excludes_vi': " | ".join(excludes_vi) if excludes_vi else "",
        'excludes_en': " | ".join(excludes_en) if excludes_en else ""
    }

def parse_diseases_from_dual_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for meta in soup.find_all(class_='clipboard-meta'):
        meta.decompose()
    diseases = []
    
    disease_divs = soup.find_all('div', class_='disease')
    for div in disease_divs:
        code_tag = div.find('a', class_='code')
        if not code_tag:
            continue
        code = code_tag.get_text(strip=True)
        
        column_content = div.find('div', class_='column-content')
        if not column_content:
            continue
            
        column_layouts = column_content.find_all('div', class_='column-layout')
        if len(column_layouts) < 2:
            continue
            
        layout_vi = column_layouts[0]
        layout_en = column_layouts[1]
        
        # Vietnamese
        h4_vi = layout_vi.find('h4')
        term_vi = ""
        if h4_vi:
            content_div = h4_vi.find(class_='content')
            term_vi = content_div.get_text(strip=True) if content_div else h4_vi.get_text(strip=True)
            
        p_vi = layout_vi.find('p')
        note_vi = p_vi.get_text(strip=True) if p_vi else ""
        
        extra_vi = ""
        extra_div_vi = layout_vi.find('div', class_='extra')
        if extra_div_vi:
            content_div = extra_div_vi.find(class_='content')
            extra_vi = content_div.get_text(strip=True) if content_div else extra_div_vi.get_text(strip=True)
            
        index_terms_vi = []
        terms_list_vi = layout_vi.find('ul', class_='list-terms')
        if terms_list_vi:
            lis = terms_list_vi.find_all('li')
            for li in lis:
                content_div = li.find(class_='content')
                index_terms_vi.append(content_div.get_text(strip=True) if content_div else li.get_text(strip=True))
                
        # English
        h4_en = layout_en.find('h4')
        term_en = ""
        if h4_en:
            content_div = h4_en.find(class_='content')
            term_en = content_div.get_text(strip=True) if content_div else h4_en.get_text(strip=True)
            
        p_en = layout_en.find('p')
        note_en = p_en.get_text(strip=True) if p_en else ""
        
        extra_en = ""
        extra_div_en = layout_en.find('div', class_='extra')
        if extra_div_en:
            content_div = extra_div_en.find(class_='content')
            extra_en = content_div.get_text(strip=True) if content_div else extra_div_en.get_text(strip=True)
            
        index_terms_en = []
        terms_list_en = layout_en.find('ul', class_='list-terms')
        if terms_list_en:
            lis = terms_list_en.find_all('li')
            for li in lis:
                content_div = li.find(class_='content')
                index_terms_en.append(content_div.get_text(strip=True) if content_div else li.get_text(strip=True))
                
        diseases.append({
            'ID': code,
            'term_vi': term_vi,
            'term_en': term_en,
            'note_vi': note_vi,
            'note_en': note_en,
            'additional_info_vi': extra_vi,
            'additional_info_en': extra_en,
            'index_terms_vi': "; ".join(index_terms_vi) if index_terms_vi else "",
            'index_terms_en': "; ".join(index_terms_en) if index_terms_en else ""
        })
    return diseases

# Discovery helper functions for parallelization
def discover_chapter_sections(session, ch_id):
    try:
        ch_url = f"https://ccs.whiteneuron.com/api/ICD10/data/chapter?id={ch_id}&lang=dual"
        ch_data = fetch_with_retry(session, ch_url)
        if not ch_data or 'data' not in ch_data or 'data' not in ch_data['data']:
            safe_print(f"    [WARN] Failed chapter {ch_id}")
            return []
        
        ch_html = ch_data['data']['data']['html']
        ch_soup = BeautifulSoup(ch_html, 'html.parser')
        
        section_ids = []
        for a in ch_soup.find_all('a', class_='code-section'):
            href = a.get('href', '')
            if 'section/' in href:
                section_ids.append(href.split('/')[-1])
        return section_ids
    except Exception as e:
        safe_print(f"    [ERROR] Exception discovering sections for chapter {ch_id}: {e}")
        return []

def discover_section_types(session, sec_id):
    try:
        sec_url = f"https://ccs.whiteneuron.com/api/ICD10/data/section?id={sec_id}&lang=dual"
        sec_data = fetch_with_retry(session, sec_url)
        if not sec_data or 'data' not in sec_data or 'data' not in sec_data['data']:
            safe_print(f"      [WARN] Failed section {sec_id}")
            return []
            
        sec_html = sec_data['data']['data']['html']
        sec_soup = BeautifulSoup(sec_html, 'html.parser')
        
        type_ids = []
        for a in sec_soup.find_all('a', class_='code-type'):
            href = a.get('href', '')
            if 'type/' in href:
                type_ids.append(href.split('/')[-1])
        return type_ids
    except Exception as e:
        safe_print(f"      [ERROR] Exception discovering types for section {sec_id}: {e}")
        return []

def crawl_missing_type_worker(session, t_id):
    try:
        type_url = f"https://ccs.whiteneuron.com/api/ICD10/data/type?id={t_id}&lang=dual"
        t_data = fetch_with_retry(session, type_url)
        if not t_data or 'data' not in t_data or 'data' not in t_data['data']:
            safe_print(f"    [ERROR] Failed to fetch missing type {t_id} after all retries.")
            return None, []
            
        html = t_data['data']['data']['html']
        
        # Parse group
        grp = parse_group_description(html, t_id, 'type')
        
        # Parse diseases
        parsed = parse_diseases_from_dual_html(html)
        return grp, parsed
    except Exception as e:
        safe_print(f"    [ERROR] Exception crawling type {t_id}: {e}")
        return None, []

def main():
    session = requests.Session()
    
    # Connection pooling setup for the session
    adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS * 2)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    file_diseases = os.path.join(os.path.dirname(__file__), 'vietnamese_icd10_diseases.csv')
    file_groups = os.path.join(os.path.dirname(__file__), 'vietnamese_icd10_groups.csv')
    
    # 1. Read already successfully crawled types from groups file
    existing_types = set()
    existing_diseases_ids = set()
    
    if os.path.exists(file_groups):
        safe_print(f"[INFO] Reading existing groups from {file_groups}...")
        try:
            with open(file_groups, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('level') == 'type':
                        existing_types.add(row.get('id'))
        except Exception as e:
            safe_print(f"[ERROR] Failed reading groups: {e}")
            
    if os.path.exists(file_diseases):
        safe_print(f"[INFO] Reading existing disease IDs from {file_diseases}...")
        try:
            with open(file_diseases, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_diseases_ids.add(row.get('ID'))
        except Exception as e:
            safe_print(f"[ERROR] Failed reading diseases: {e}")
            
    safe_print(f"[INFO] Found {len(existing_types)} successfully fetched types.")
    safe_print(f"[INFO] Found {len(existing_diseases_ids)} successfully fetched disease codes.")
    
    # 2. Re-discover the complete set of types to find gaps
    safe_print("\n[INFO] Re-discovering complete ICD-10 hierarchy in parallel (using 4 workers)...")
    root_url = "https://ccs.whiteneuron.com/api/ICD10/root?lang=dual"
    root_data = fetch_with_retry(session, root_url)
    if not root_data or 'data' not in root_data:
        safe_print("[ERROR] Cannot fetch root chapters. Exiting.")
        return
        
    chapters = [item['id'] for item in root_data['data'] if item.get('model') == 'chapter']
    safe_print(f"[INFO] Discovered {len(chapters)} chapters.")
    
    # 2.1 Fetch sections in parallel
    section_ids = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(discover_chapter_sections, session, ch_id): ch_id for ch_id in chapters}
        for future in as_completed(futures):
            ch_id = futures[future]
            try:
                sec_list = future.result()
                section_ids.extend(sec_list)
            except Exception as e:
                safe_print(f"[ERROR] Failed to discover sections for {ch_id}: {e}")
                
    safe_print(f"[INFO] Discovered {len(section_ids)} total sections.")
    
    # 2.2 Fetch types in parallel
    all_type_ids = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(discover_section_types, session, sec_id): sec_id for sec_id in section_ids}
        for future in as_completed(futures):
            sec_id = futures[future]
            try:
                type_list = future.result()
                all_type_ids.extend(type_list)
            except Exception as e:
                safe_print(f"[ERROR] Failed to discover types for {sec_id}: {e}")
                
    # Find missing types
    all_type_ids = list(set(all_type_ids))
    missing_types = [t for t in all_type_ids if t not in existing_types]
    
    safe_print(f"\n[INFO] Complete database contains {len(all_type_ids)} types.")
    safe_print(f"[INFO] Number of missing types due to timeouts/400s: {len(missing_types)}")
    
    if not missing_types:
        safe_print("[SUCCESS] No types are missing! Database is complete.")
    else:
        safe_print(f"[INFO] Starting second-pass crawl with {MAX_WORKERS} workers to fetch {len(missing_types)} missing types...")
        
        new_groups = []
        new_diseases = []
        
        count = 0
        total_missing = len(missing_types)
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(crawl_missing_type_worker, session, t_id): t_id for t_id in missing_types}
            for future in as_completed(futures):
                t_id = futures[future]
                count += 1
                try:
                    grp, parsed = future.result()
                    if grp:
                        with list_lock:
                            new_groups.append(grp)
                    if parsed:
                        with list_lock:
                            new_diseases.extend(parsed)
                        safe_print(f"  [{count}/{total_missing}] Successfully parsed missing type {t_id} ({len(parsed)} leaf codes found).")
                    else:
                        safe_print(f"  [{count}/{total_missing}] Handled missing type {t_id} (0 leaf codes parsed).")
                except Exception as e:
                    safe_print(f"[ERROR] Exception processing missing type {t_id}: {e}")
            
        # Write new groups
        if new_groups:
            try:
                with open(file_groups, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=['id', 'level', 'name_vi', 'name_en', 'includes_vi', 'includes_en', 'excludes_vi', 'excludes_en'])
                    for g in new_groups:
                        writer.writerow(g)
                safe_print(f"[SUCCESS] Appended {len(new_groups)} missing group rows to CSV.")
            except Exception as e:
                safe_print(f"[ERROR] Failed appending groups to CSV: {e}")
                
        # Write new diseases
        if new_diseases:
            try:
                with open(file_diseases, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=['ID', 'term_vi', 'term_en', 'note_vi', 'note_en', 'additional_info_vi', 'additional_info_en', 'index_terms_vi', 'index_terms_en'])
                    for d in new_diseases:
                        writer.writerow(d)
                safe_print(f"[SUCCESS] Appended {len(new_diseases)} missing disease rows to CSV.")
            except Exception as e:
                safe_print(f"[ERROR] Failed appending diseases to CSV: {e}")

    # 3. Clean up, remove duplicates and perform final sort
    safe_print("\n[INFO] Starting final clean-up, deduplication and sorting...")
    
    # Clean diseases
    all_diseases = []
    if os.path.exists(file_diseases):
        with open(file_diseases, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('ID'):
                    all_diseases.append(row)
        # Deduplicate by ID
        unique_diseases = {}
        for row in all_diseases:
            unique_diseases[row['ID']] = row
        diseases_sorted = sorted(unique_diseases.values(), key=lambda x: x['ID'])
        
        # Write sorted CSV
        with open(file_diseases, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['ID', 'term_vi', 'term_en', 'note_vi', 'note_en', 'additional_info_vi', 'additional_info_en', 'index_terms_vi', 'index_terms_en'])
            writer.writeheader()
            for row in diseases_sorted:
                writer.writerow(row)
        safe_print(f"[SUCCESS] Cleaned and sorted {len(diseases_sorted)} diseases in CSV.")
        
        # Write JSON
        json_diseases = file_diseases.replace('.csv', '.json')
        with open(json_diseases, 'w', encoding='utf-8') as f:
            json.dump(diseases_sorted, f, ensure_ascii=False, indent=2)
        safe_print(f"[SUCCESS] Cleaned and sorted diseases exported to JSON.")
        
    # Clean groups
    all_groups = []
    if os.path.exists(file_groups):
        with open(file_groups, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('id'):
                    all_groups.append(row)
        # Deduplicate by (level, id)
        unique_groups = {}
        for row in all_groups:
            unique_groups[(row['level'], row['id'])] = row
            
        # Custom sorting logic: chapters first, then sections, then types, then alphabetically
        level_order = {'chapter': 0, 'section': 1, 'type': 2}
        groups_sorted = sorted(unique_groups.values(), key=lambda x: (level_order.get(x['level'], 3), x['id']))
        
        # Write sorted CSV
        with open(file_groups, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'level', 'name_vi', 'name_en', 'includes_vi', 'includes_en', 'excludes_vi', 'excludes_en'])
            writer.writeheader()
            for row in groups_sorted:
                writer.writerow(row)
        safe_print(f"[SUCCESS] Cleaned and sorted {len(groups_sorted)} groups in CSV.")
        
        # Write JSON
        json_groups = file_groups.replace('.csv', '.json')
        with open(json_groups, 'w', encoding='utf-8') as f:
            json.dump(groups_sorted, f, ensure_ascii=False, indent=2)
        safe_print(f"[SUCCESS] Cleaned and sorted groups exported to JSON.")

if __name__ == '__main__':
    main()
