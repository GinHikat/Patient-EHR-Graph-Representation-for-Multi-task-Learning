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


MAX_WORKERS = 4
RETRIES = 5
TIMEOUT = 30


# Thread-safe locks
print_lock = threading.Lock()
list_lock = threading.Lock()

diseases_list = []
groups_list = []
total_diseases_parsed = 0

def safe_print(msg):
    with print_lock:
        print(msg, flush=True)

def fetch_with_retry(session, url):
    for attempt in range(RETRIES):
        try:
            r = session.get(url, timeout=TIMEOUT)
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
    includes_vi = []
    excludes_vi = []
    
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
                    
    includes_en = []
    excludes_en = []
    
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
        # Extract Code/ID
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
        
        # 1. Parse Vietnamese side
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
                term_text = content_div.get_text(strip=True) if content_div else li.get_text(strip=True)
                index_terms_vi.append(term_text)
                
        # 2. Parse English side
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
                term_text = content_div.get_text(strip=True) if content_div else li.get_text(strip=True)
                index_terms_en.append(term_text)
                
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

def stream_write_group(grp):
    output_file = os.path.join(os.path.dirname(__file__), 'vietnamese_icd10_groups.csv')
    try:
        with open(output_file, 'a', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['id', 'level', 'name_vi', 'name_en', 'includes_vi', 'includes_en', 'excludes_vi', 'excludes_en']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(grp)
    except Exception as e:
        safe_print(f"[WARN] Error stream-writing group to CSV: {e}")

def stream_write_disease(parsed_diseases):
    output_file = os.path.join(os.path.dirname(__file__), 'vietnamese_icd10_diseases.csv')
    try:
        with open(output_file, 'a', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['ID', 'term_vi', 'term_en', 'note_vi', 'note_en', 'additional_info_vi', 'additional_info_en', 'index_terms_vi', 'index_terms_en']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for row in parsed_diseases:
                writer.writerow(row)
    except Exception as e:
        safe_print(f"[WARN] Error stream-writing disease to CSV: {e}")

def process_type(session, type_id):
    global total_diseases_parsed
    try:
        url = f"https://ccs.whiteneuron.com/api/ICD10/data/type?id={type_id}&lang=dual"
        response_data = fetch_with_retry(session, url)
        if not response_data or 'data' not in response_data or 'data' not in response_data['data']:
            safe_print(f"[WARN] No valid type data for {type_id}")
            return 0
        
        html = response_data['data']['data']['html']
        
        # Parse type description
        try:
            grp = parse_group_description(html, type_id, 'type')
            if grp:
                with list_lock:
                    groups_list.append(grp)
                    stream_write_group(grp)
        except Exception as e:
            safe_print(f"[ERROR] Failed to parse group description for type {type_id}: {e}")
                
        try:
            parsed = parse_diseases_from_dual_html(html)
        except Exception as e:
            safe_print(f"[ERROR] Failed to parse diseases from HTML for type {type_id}: {e}")
            parsed = None
        
        if parsed:
            with list_lock:
                diseases_list.extend(parsed)
                total_diseases_parsed += len(parsed)
                stream_write_disease(parsed)
            return len(parsed)
        return 0
    except Exception as e:
        safe_print(f"[ERROR] Exception in process_type for {type_id}: {e}")
        return 0

def process_section(session, section_id):
    try:
        url = f"https://ccs.whiteneuron.com/api/ICD10/data/section?id={section_id}&lang=dual"
        response_data = fetch_with_retry(session, url)
        if not response_data or 'data' not in response_data or 'data' not in response_data['data']:
            safe_print(f"[WARN] No valid section data for {section_id}")
            return []
        
        html = response_data['data']['data']['html']
        
        # Parse section description
        try:
            grp = parse_group_description(html, section_id, 'section')
            if grp:
                with list_lock:
                    groups_list.append(grp)
                    stream_write_group(grp)
        except Exception as e:
            safe_print(f"[ERROR] Failed to parse group description for section {section_id}: {e}")
                
        soup = BeautifulSoup(html, 'html.parser')
        type_ids = []
        for a in soup.find_all('a', class_='code-type'):
            href = a.get('href', '')
            if 'type/' in href:
                type_ids.append(href.split('/')[-1])
                
        return type_ids
    except Exception as e:
        safe_print(f"[ERROR] Exception in process_section for {section_id}: {e}")
        return []

def process_chapter(session, chapter_id):
    try:
        url = f"https://ccs.whiteneuron.com/api/ICD10/data/chapter?id={chapter_id}&lang=dual"
        response_data = fetch_with_retry(session, url)
        if not response_data or 'data' not in response_data or 'data' not in response_data['data']:
            safe_print(f"[WARN] No valid chapter data for {chapter_id}")
            return []
        
        html = response_data['data']['data']['html']
        
        # Parse chapter description
        try:
            grp = parse_group_description(html, chapter_id, 'chapter')
            if grp:
                with list_lock:
                    groups_list.append(grp)
                    stream_write_group(grp)
        except Exception as e:
            safe_print(f"[ERROR] Failed to parse group description for chapter {chapter_id}: {e}")
                
        soup = BeautifulSoup(html, 'html.parser')
        section_ids = []
        for a in soup.find_all('a', class_='code-section'):
            href = a.get('href', '')
            if 'section/' in href:
                section_ids.append(href.split('/')[-1])
                
        return section_ids
    except Exception as e:
        safe_print(f"[ERROR] Exception in process_chapter for {chapter_id}: {e}")
        return []

def main():
    start_time = time.time()
    session = requests.Session()
    
    adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS * 2)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Initialize output files
    file_diseases = os.path.join(os.path.dirname(__file__), 'vietnamese_icd10_diseases.csv')
    file_groups = os.path.join(os.path.dirname(__file__), 'vietnamese_icd10_groups.csv')
    
    try:
        with open(file_diseases, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['ID', 'term_vi', 'term_en', 'note_vi', 'note_en', 'additional_info_vi', 'additional_info_en', 'index_terms_vi', 'index_terms_en']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        with open(file_groups, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['id', 'level', 'name_vi', 'name_en', 'includes_vi', 'includes_en', 'excludes_vi', 'excludes_en']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        safe_print("[INFO] Output CSV files initialized successfully.")
    except Exception as e:
        safe_print(f"[ERROR] Failed to initialize CSV files: {e}")
        return
        
    # 1. Fetch root chapters
    safe_print("[INFO] Fetching root chapters...")
    root_url = "https://ccs.whiteneuron.com/api/ICD10/root?lang=dual"
    root_data = fetch_with_retry(session, root_url)
    
    if not root_data or 'data' not in root_data:
        safe_print("[ERROR] Could not load root chapters. Exiting.")
        return
        
    chapters = [item['id'] for item in root_data['data'] if item.get('model') == 'chapter']
    safe_print(f"[INFO] Found {len(chapters)} chapters.")
    
    # 2. Fetch sections for all chapters in parallel
    safe_print("\n[INFO] Fetching sections definitions for all chapters...")
    section_ids = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_chapter, session, ch_id): ch_id for ch_id in chapters}
        for future in as_completed(futures):
            ch_id = futures[future]
            try:
                sec_list = future.result()
                section_ids.extend(sec_list)
                safe_print(f"  Chapter {ch_id}: Found {len(sec_list)} sections.")
            except Exception as e:
                safe_print(f"[ERROR] Failed to fetch sections for chapter {ch_id}: {e}")
                
    safe_print(f"[INFO] Total sections found: {len(section_ids)}")
    
    # 3. Fetch types for all sections in parallel
    safe_print("\n[INFO] Fetching type definitions for all sections...")
    type_ids = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_section, session, sec_id): sec_id for sec_id in section_ids}
        for future in as_completed(futures):
            sec_id = futures[future]
            try:
                types_list = future.result()
                type_ids.extend(types_list)
            except Exception as e:
                safe_print(f"[ERROR] Failed to fetch types for section {sec_id}: {e}")
                
    safe_print(f"[INFO] Total types found: {len(type_ids)}")
    
    # 4. Fetch and parse diseases for all types in parallel
    safe_print("\n[INFO] Scraping disease leaf codes (A00.0, etc.) from all types...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_type, session, t_id): t_id for t_id in type_ids}
        
        count = 0
        total_types = len(type_ids)
        last_percent = -1
        
        for future in as_completed(futures):
            t_id = futures[future]
            count += 1
            try:
                future.result()
            except Exception as e:
                safe_print(f"[ERROR] Failed processing type {t_id}: {e}")
                
            percent = int((count / total_types) * 100)
            if percent != last_percent and percent % 5 == 0:
                last_percent = percent
                bar_length = 20
                filled_length = int(round(bar_length * percent / 100))
                bar = '#' * filled_length + '-' * (bar_length - filled_length)
                safe_print(f"  [Progress] [{bar}] {percent:3d}% | {count:4d}/{total_types:4d} types | {total_diseases_parsed:5d} codes found")
                
    # 5. Export final sorted data
    safe_print(f"\n[INFO] Completed crawl in {time.time() - start_time:.2f} seconds.")
    safe_print(f"[INFO] Sorting and cleaning final outputs...")
    
    # Sort by ID or id to keep final exports clean
    diseases_sorted = sorted(diseases_list, key=lambda x: x['ID'])
    groups_sorted = sorted(groups_list, key=lambda x: (x['level'], x['id']))
    
    json_diseases = os.path.join(os.path.dirname(__file__), 'vietnamese_icd10_diseases.json')
    json_groups = os.path.join(os.path.dirname(__file__), 'vietnamese_icd10_groups.json')
    
    # Write sorted CSVs
    try:
        with open(file_diseases, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['ID', 'term_vi', 'term_en', 'note_vi', 'note_en', 'additional_info_vi', 'additional_info_en', 'index_terms_vi', 'index_terms_en']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in diseases_sorted:
                writer.writerow(row)
        safe_print(f"[SUCCESS] Sorted diseases CSV exported to: {file_diseases}")
    except Exception as e:
         safe_print(f"[ERROR] Failed writing sorted diseases CSV: {e}")
         
    try:
        with open(file_groups, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['id', 'level', 'name_vi', 'name_en', 'includes_vi', 'includes_en', 'excludes_vi', 'excludes_en']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in groups_sorted:
                writer.writerow(row)
        safe_print(f"[SUCCESS] Sorted groups CSV exported to: {file_groups}")
    except Exception as e:
         safe_print(f"[ERROR] Failed writing sorted groups CSV: {e}")

    # Write JSONs
    try:
        with open(json_diseases, 'w', encoding='utf-8') as f:
            json.dump(diseases_sorted, f, ensure_ascii=False, indent=2)
        safe_print(f"[SUCCESS] Diseases JSON exported to: {json_diseases}")
    except Exception as e:
         safe_print(f"[ERROR] Failed saving diseases JSON: {e}")
         
    try:
        with open(json_groups, 'w', encoding='utf-8') as f:
            json.dump(groups_sorted, f, ensure_ascii=False, indent=2)
        safe_print(f"[SUCCESS] Groups JSON exported to: {json_groups}")
    except Exception as e:
         safe_print(f"[ERROR] Failed saving groups JSON: {e}")

if __name__ == "__main__":
    main()
