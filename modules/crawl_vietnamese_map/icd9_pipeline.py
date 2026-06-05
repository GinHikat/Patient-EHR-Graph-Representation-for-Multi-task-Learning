import os
import csv
import json
import time
import requests
from bs4 import BeautifulSoup
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Concurrency configurations (1 worker for absolute stability)
MAX_WORKERS = 1
RETRIES = 5
TIMEOUT = 30

diseases_list = []
groups_list = []
total_procedures_parsed = 0

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
                    print(f"[WARN] API status error for {url}: {data.get('message')} (Attempt {attempt+1}/{RETRIES})")
            else:
                print(f"[WARN] HTTP {r.status_code} for {url} (Attempt {attempt+1}/{RETRIES})")
        except Exception as e:
            print(f"[WARN] Request error {e} for {url} (Attempt {attempt+1}/{RETRIES})")
        time.sleep(1.5 ** attempt) # Backoff delay
    return None

def parse_group_description(html_content, group_id, level):
    soup = BeautifulSoup(html_content, 'html.parser')
    for meta in soup.find_all(class_='clipboard-meta'):
        meta.decompose()
        
    block_div = soup.find('div', class_=lambda c: c and (c.startswith('block-') or c in ['subsection', 'type']))
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

def parse_procedures_from_dual_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for meta in soup.find_all(class_='clipboard-meta'):
        meta.decompose()
        
    procedures = []
    
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
                index_terms_vi.append(content_div.get_text(strip=True) if content_div else li.get_text(strip=True))
                
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
                index_terms_en.append(content_div.get_text(strip=True) if content_div else li.get_text(strip=True))
                
        procedures.append({
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
    return procedures

def stream_write_group(grp):
    output_file = os.path.join(os.path.dirname(__file__), 'vietnamese_icd9_groups.csv')
    try:
        with open(output_file, 'a', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['id', 'level', 'name_vi', 'name_en', 'includes_vi', 'includes_en', 'excludes_vi', 'excludes_en']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(grp)
    except Exception as e:
        print(f"[WARN] Error stream-writing group to CSV: {e}")

def stream_write_procedure(parsed_procedures):
    output_file = os.path.join(os.path.dirname(__file__), 'vietnamese_icd9_procedures.csv')
    try:
        with open(output_file, 'a', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['ID', 'term_vi', 'term_en', 'note_vi', 'note_en', 'additional_info_vi', 'additional_info_en', 'index_terms_vi', 'index_terms_en']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for row in parsed_procedures:
                writer.writerow(row)
    except Exception as e:
        print(f"[WARN] Error stream-writing procedure to CSV: {e}")

def process_subsection(session, sub_id):
    global total_procedures_parsed
    url = f"https://ccs.whiteneuron.com/api/ICD9/data/subsection?id={sub_id}&lang=dual"
    response_data = fetch_with_retry(session, url)
    if not response_data or 'data' not in response_data or 'data' not in response_data['data']:
        print(f"[WARN] No valid subsection data for {sub_id}")
        return 0
    
    html = response_data['data']['data']['html']
    
    # Parse subsection description
    try:
        grp = parse_group_description(html, sub_id, 'subsection')
        if grp:
            groups_list.append(grp)
            stream_write_group(grp)
    except Exception as e:
        print(f"[ERROR] Failed to parse group description for subsection {sub_id}: {e}")
            
    try:
        parsed = parse_procedures_from_dual_html(html)
    except Exception as e:
        print(f"[ERROR] Failed to parse procedures from HTML for subsection {sub_id}: {e}")
        parsed = None
    
    if parsed:
        diseases_list.extend(parsed)
        total_procedures_parsed += len(parsed)
        stream_write_procedure(parsed)
        return len(parsed)
    return 0

def process_section(session, section_id):
    url = f"https://ccs.whiteneuron.com/api/ICD9/data/section?id={section_id}&lang=dual"
    response_data = fetch_with_retry(session, url)
    if not response_data or 'data' not in response_data or 'data' not in response_data['data']:
        print(f"[WARN] No valid section data for {section_id}")
        return []
    
    html = response_data['data']['data']['html']
    
    # Parse section description
    try:
        grp = parse_group_description(html, section_id, 'section')
        if grp:
            groups_list.append(grp)
            stream_write_group(grp)
    except Exception as e:
        print(f"[ERROR] Failed to parse group description for section {section_id}: {e}")
            
    soup = BeautifulSoup(html, 'html.parser')
    sub_ids = []
    for a in soup.find_all('a', class_='code-subsection'):
        href = a.get('href', '')
        if 'subsection/' in href:
            sub_ids.append(href.split('/')[-1])
            
    return sub_ids

def process_chapter(session, chapter_id):
    url = f"https://ccs.whiteneuron.com/api/ICD9/data/chapter?id={chapter_id}&lang=dual"
    response_data = fetch_with_retry(session, url)
    if not response_data or 'data' not in response_data or 'data' not in response_data['data']:
        print(f"[WARN] No valid chapter data for {chapter_id}")
        return []
    
    html = response_data['data']['data']['html']
    
    # Parse chapter description
    try:
        grp = parse_group_description(html, chapter_id, 'chapter')
        if grp:
            groups_list.append(grp)
            stream_write_group(grp)
    except Exception as e:
        print(f"[ERROR] Failed to parse group description for chapter {chapter_id}: {e}")
            
    soup = BeautifulSoup(html, 'html.parser')
    section_ids = []
    for a in soup.find_all('a', class_='code-section'):
        href = a.get('href', '')
        if 'section/' in href:
            section_ids.append(href.split('/')[-1])
            
    return section_ids

def main():
    start_time = time.time()
    session = requests.Session()
    
    # Initialize output files
    file_procedures = os.path.join(os.path.dirname(__file__), 'vietnamese_icd9_procedures.csv')
    file_groups = os.path.join(os.path.dirname(__file__), 'vietnamese_icd9_groups.csv')
    
    try:
        with open(file_procedures, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['ID', 'term_vi', 'term_en', 'note_vi', 'note_en', 'additional_info_vi', 'additional_info_en', 'index_terms_vi', 'index_terms_en']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        with open(file_groups, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['id', 'level', 'name_vi', 'name_en', 'includes_vi', 'includes_en', 'excludes_vi', 'excludes_en']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        print("[INFO] Output CSV files for ICD-9 initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize CSV files: {e}")
        return
        
    # 1. Fetch root chapters
    print("[INFO] Fetching root chapters...")
    root_url = "https://ccs.whiteneuron.com/api/ICD9/root?lang=dual"
    root_data = fetch_with_retry(session, root_url)
    
    if not root_data or 'data' not in root_data:
        print("[ERROR] Could not load root chapters. Exiting.")
        return
        
    chapters = [item['id'] for item in root_data['data'] if item.get('model') == 'chapter']
    print(f"[INFO] Found {len(chapters)} chapters.")
    
    # 2. Fetch sections for all chapters sequentially
    print("\n[INFO] Fetching sections definitions for all chapters...")
    section_ids = []
    for ch_id in chapters:
        try:
            sec_list = process_chapter(session, ch_id)
            section_ids.extend(sec_list)
            print(f"  Chapter {ch_id}: Found {len(sec_list)} sections.")
        except Exception as e:
            print(f"[ERROR] Failed to fetch sections for chapter {ch_id}: {e}")
                
    print(f"[INFO] Total sections found: {len(section_ids)}")
    
    # 3. Fetch subsections for all sections sequentially
    print("\n[INFO] Fetching subsection definitions for all sections...")
    sub_ids = []
    for idx, sec_id in enumerate(section_ids):
        try:
            sub_list = process_section(session, sec_id)
            sub_ids.extend(sub_list)
            if (idx + 1) % 10 == 0 or (idx + 1) == len(section_ids):
                print(f"  [{idx+1}/{len(section_ids)}] Sections mapped...")
        except Exception as e:
            print(f"[ERROR] Failed to fetch subsections for section {sec_id}: {e}")
                
    print(f"[INFO] Total subsections found: {len(sub_ids)}")
    
    # 4. Fetch and parse procedures for all subsections sequentially
    print("\n[INFO] Scraping procedure leaf codes (00.01, etc.) from all subsections...")
    count = 0
    total_subs = len(sub_ids)
    
    for sub_id in sub_ids:
        count += 1
        try:
            process_subsection(session, sub_id)
        except Exception as e:
            print(f"[ERROR] Failed processing subsection {sub_id}: {e}")
            
        if count % 10 == 0 or count == total_subs:
            print(f"  [Progress] {count:4d}/{total_subs:4d} subsections | {total_procedures_parsed:5d} procedures found")
                
    # 5. Export final sorted data
    print(f"\n[INFO] Completed crawl in {time.time() - start_time:.2f} seconds.")
    print(f"[INFO] Sorting and cleaning final outputs...")
    
    # Sort by ID or id to keep final exports clean
    procedures_sorted = sorted(diseases_list, key=lambda x: x['ID'])
    
    # Custom sorting logic: chapters first, then sections, then subsections, alphabetically
    level_order = {'chapter': 0, 'section': 1, 'subsection': 2}
    groups_sorted = sorted(groups_list, key=lambda x: (level_order.get(x['level'], 3), x['id']))
    
    json_procedures = os.path.join(os.path.dirname(__file__), 'vietnamese_icd9_procedures.json')
    json_groups = os.path.join(os.path.dirname(__file__), 'vietnamese_icd9_groups.json')
    
    # Write sorted CSVs
    try:
        with open(file_procedures, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['ID', 'term_vi', 'term_en', 'note_vi', 'note_en', 'additional_info_vi', 'additional_info_en', 'index_terms_vi', 'index_terms_en']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in procedures_sorted:
                writer.writerow(row)
        print(f"[SUCCESS] Sorted procedures CSV exported to: {file_procedures}")
    except Exception as e:
         print(f"[ERROR] Failed writing sorted procedures CSV: {e}")
         
    try:
        with open(file_groups, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['id', 'level', 'name_vi', 'name_en', 'includes_vi', 'includes_en', 'excludes_vi', 'excludes_en']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in groups_sorted:
                writer.writerow(row)
        print(f"[SUCCESS] Sorted groups CSV exported to: {file_groups}")
    except Exception as e:
         print(f"[ERROR] Failed writing sorted groups CSV: {e}")
 
    # Write JSONs
    try:
        with open(json_procedures, 'w', encoding='utf-8') as f:
            json.dump(procedures_sorted, f, ensure_ascii=False, indent=2)
        print(f"[SUCCESS] Procedures JSON exported to: {json_procedures}")
    except Exception as e:
         print(f"[ERROR] Failed saving procedures JSON: {e}")
         
    try:
        with open(json_groups, 'w', encoding='utf-8') as f:
            json.dump(groups_sorted, f, ensure_ascii=False, indent=2)
        print(f"[SUCCESS] Groups JSON exported to: {json_groups}")
    except Exception as e:
         print(f"[ERROR] Failed saving groups JSON: {e}")

if __name__ == "__main__":
    main()
