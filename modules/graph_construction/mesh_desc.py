import requests
import xml.etree.ElementTree as ET
import json
import sys
import time

def fetch_mesh_info(mesh_id: str, api_key: str = None) -> dict:
    """
    Fetch description and definition for a MeSH ID using NCBI E-utilities.

    Args:
        mesh_id: MeSH unique ID, e.g. "D007592"
        api_key: Optional NCBI API key (increases rate limit from 3 to 10 req/sec)

    Returns:
        dict with keys: mesh_id, name, scope_note (description/definition), tree_numbers, entry_terms
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # --- Step 1: Search for the MeSH term to get its internal UID ---
    search_params = {
        "db": "mesh",
        "term": mesh_id,
        "retmode": "json",
    }
    if api_key:
        search_params["api_key"] = api_key

    search_resp = requests.get(f"{base_url}/esearch.fcgi", params=search_params, timeout=15)
    search_resp.raise_for_status()
    search_data = search_resp.json()

    id_list = search_data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        raise ValueError(f"No MeSH record found for ID: {mesh_id}")

    uid = id_list[0]
    print(f"[+] Found UID: {uid} for MeSH ID: {mesh_id}")

    # Polite delay between requests
    time.sleep(0.4)

    # --- Step 2: Fetch the full record using efetch ---
    fetch_params = {
        "db": "mesh",
        "id": uid,
        "retmode": "xml",
    }
    if api_key:
        fetch_params["api_key"] = api_key

    fetch_resp = requests.get(f"{base_url}/efetch.fcgi", params=fetch_params, timeout=15)
    fetch_resp.raise_for_status()

    # --- Step 3: Parse the XML response ---
    root = ET.fromstring(fetch_resp.text)

    result = {
        "mesh_id": mesh_id,
        "uid": uid,
        "name": None,
        "scope_note": None,       # This is the definition/description
        "tree_numbers": [],
        "entry_terms": [],
        "pharmacological_actions": [],
    }

    # MeSH descriptor name
    name_el = root.find(".//DS_MeshHeading")
    if name_el is not None:
        result["name"] = name_el.text.strip()

    # Scope Note = the definition/description
    scope_el = root.find(".//DS_ScopeNote")
    if scope_el is not None:
        result["scope_note"] = scope_el.text.strip()

    # Tree numbers (MeSH hierarchy positions)
    for tn in root.findall(".//DS_TreeNum"):
        if tn.text:
            result["tree_numbers"].append(tn.text.strip())

    # Entry terms (synonyms)
    for et in root.findall(".//DS_EntryTerm"):
        if et.text:
            result["entry_terms"].append(et.text.strip())

    # Pharmacological actions (for drug entries)
    for pa in root.findall(".//DS_PharmAction"):
        if pa.text:
            result["pharmacological_actions"].append(pa.text.strip())

    return result


def print_mesh_info(info: dict):
    """Pretty-print the MeSH information."""
    print("\n" + "=" * 60)
    print(f"  MeSH ID     : {info['mesh_id']}")
    print(f"  Name        : {info['name'] or 'N/A'}")
    print("=" * 60)

    print("\n📋 DEFINITION / SCOPE NOTE:")
    print("-" * 60)
    if info["scope_note"]:
        # Word-wrap at ~80 chars for readability
        words = info["scope_note"].split()
        line, lines = [], []
        for w in words:
            if sum(len(x) + 1 for x in line) + len(w) > 80:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        print("\n".join(lines))
    else:
        print("  (No scope note available)")

    if info["tree_numbers"]:
        print(f"\n🌳 TREE NUMBERS : {', '.join(info['tree_numbers'])}")

    if info["entry_terms"]:
        print(f"\n🔤 ENTRY TERMS  : {', '.join(info['entry_terms'][:10])}")
        if len(info["entry_terms"]) > 10:
            print(f"                  ... and {len(info['entry_terms']) - 10} more")

    if info["pharmacological_actions"]:
        print(f"\n💊 PHARM ACTIONS: {', '.join(info['pharmacological_actions'])}")

    print("\n🔗 Source URL   :")
    print(f"   https://www.ncbi.nlm.nih.gov/mesh/?term={info['mesh_id']}")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Accept MeSH ID from command line or use default
    mesh_id = sys.argv[1] if len(sys.argv) > 1 else "D007592"

    # Optional: set your NCBI API key here for higher rate limits
    # Get one free at: https://www.ncbi.nlm.nih.gov/account/
    api_key = None  # e.g. api_key = "your_key_here"

    print(f"[*] Fetching MeSH info for: {mesh_id}")

    try:
        info = fetch_mesh_info(mesh_id, api_key=api_key)
        print_mesh_info(info)

        # Optionally save to JSON
        out_file = f"mesh_{mesh_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"[✓] Result saved to: {out_file}")

    except requests.HTTPError as e:
        print(f"[!] HTTP error: {e}")
    except ValueError as e:
        print(f"[!] {e}")
    except Exception as e:
        print(f"[!] Unexpected error: {e}")
        raise