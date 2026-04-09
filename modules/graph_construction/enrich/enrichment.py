import requests
import os, sys

UMLS_API_KEY = os.getenv('UML_API_KEY')
UMLS_BASE = "https://utslogin.nlm.nih.gov/cas/v1"
UMLS_CONTENT = "https://uts-ws.nlm.nih.gov/rest"

def load_ccsr_mappings(dx_path, pr_path):
    """
    Loads and cleans CCSR mapping files for Diagnoses and Procedures.
    Use for ICD 10 only
    """
    # Diagnosis (DX_CCSR)
    dx_ccsr = pd.read_csv(dx_path, skipinitialspace=True)
    
    # Strip single quotes from column names
    dx_ccsr.columns = [c.strip("'") for c in dx_ccsr.columns]
    
    # Select and clean relevant columns
    dx_map = dx_ccsr[['ICD-10-CM CODE', 'Default CCSR CATEGORY IP', 'Default CCSR CATEGORY DESCRIPTION IP']].copy()
    dx_map.rename(columns={
        'ICD-10-CM CODE': 'icd_code',
        'Default CCSR CATEGORY IP': 'ccsr_category',
        'Default CCSR CATEGORY DESCRIPTION IP': 'ccsr_description'
    }, inplace=True)
    
    # Strip single quotes from the actual data values
    for col in dx_map.columns:
        dx_map[col] = dx_map[col].astype(str).str.strip("'")

    # Procedures (PRCCSR)
    pr_ccsr = pd.read_csv(pr_path, skipinitialspace=True)
    pr_ccsr.columns = [c.strip("'") for c in pr_ccsr.columns]
    
    pr_map = pr_ccsr[['ICD-10-PCS', 'PRCCSR', 'PRCCSR DESCRIPTION', 'CLINICAL DOMAIN']].copy()
    pr_map.rename(columns={
        'ICD-10-PCS': 'icd_code',
        'PRCCSR': 'ccsr_category',
        'PRCCSR DESCRIPTION': 'ccsr_description',
        'CLINICAL DOMAIN': 'clinical_domain'
    }, inplace=True)
    
    for col in pr_map.columns:
        pr_map[col] = pr_map[col].astype(str).str.strip("'")
    return dx_map, pr_map

def load_ccs_mappings(dx_9_path, pr_9_path):
    """
    Loads and cleans CCS mapping files for ICD-9.
    Handles the CSV format for diagnoses and the TXT format for procedures.
    Use for ICD 9 only
    """
    # ICD-9 Diagnoses (CSV)
    dx_ccs = pd.read_csv(dx_9_path, skipinitialspace=True)
    
    # Strip single quotes from column names and values
    dx_ccs.columns = [c.strip("'") for c in dx_ccs.columns]
    for col in dx_ccs.columns:
        dx_ccs[col] = dx_ccs[col].astype(str).str.strip("'").str.strip()

    # Rename to match your project's common schema
    dx_map = dx_ccs[['ICD-9-CM CODE', 'CCS CATEGORY', 'CCS CATEGORY DESCRIPTION']].copy()
    dx_map.rename(columns={
        'ICD-9-CM CODE': 'icd_code',
        'CCS CATEGORY': 'ccs_category',  # Keep naming consistent with ICD-10 if desired
        'CCS CATEGORY DESCRIPTION': 'ccs_description'
    }, inplace=True)

    # ICD-9 Procedures (TXT)
    # The TXT file uses a 'Category ID - Description' header followed by code lists
    pr_rows = []
    current_cat_id = None
    current_cat_desc = None
    
    with open(pr_9_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            
            # Check if line starts with a category number (e.g., '1   Incision...')
            match = re.match(r'^(\d+)\s+(.*)', line)
            if match:
                current_cat_id = match.group(1)
                current_cat_desc = match.group(2).strip()
            elif current_cat_id:
                # This line contains ICD codes separated by spaces
                codes = line.strip().split()
                for code in codes:
                    pr_rows.append({
                        'icd_code': code,
                        'ccs_category': current_cat_id,
                        'ccs_description': current_cat_desc
                    })
    
    pr_map = pd.DataFrame(pr_rows)

    return dx_map, pr_map

class DatabaseExtract:
    def __init__(self):
        self._tgt_url = None

    def get_umls_ticket(self, api_key: str) -> str:
        """
        Exchange API key for a Ticket Granting Ticket (TGT).
        Caches the TGT URL to reuse it.
        """
        if self._tgt_url:
            return self._tgt_url

        if not api_key:
            raise ValueError("UMLS API key is missing. Ensure UML_API_KEY is set in your .env or environment.")

        resp = requests.post(
            f"{UMLS_BASE}/api-key",
            data={"apikey": api_key}
        )
        resp.raise_for_status()
        # TGT URL is in the Location header
        self._tgt_url = resp.headers["location"]
        return self._tgt_url

    def get_service_ticket(self, tgt_url: str) -> str:
        """
        Get a single-use service ticket from the TGT.
        """

        resp = requests.post(
            tgt_url,
            data={"service": "http://umlsks.nlm.nih.gov"}
        )
        if resp.status_code != 201:
            # If TGT is expired/invalid, clear it so next call gets a new one
            self._tgt_url = None
            resp.raise_for_status()
        return resp.text

    def fetch_umls_description(self, cui: str, api_key: str = UMLS_API_KEY) -> dict:
        """
        Fetch the definition/description for a UMLS Concept Unique Identifier (CUI).
        
        Args:
            cui: UML ID
            api_key: UMLS API key (defaults to global UMLS_API_KEY)
        
        Returns:
            dict with 'name', 'definitions', 'semantic_types'
        """
        try:
            tgt_url = self.get_umls_ticket(api_key)
            ticket = self.get_service_ticket(tgt_url)

            # Fetch concept metadata
            concept_url = f"{UMLS_CONTENT}/content/current/CUI/{cui}"
            resp = requests.get(concept_url, params={"ticket": ticket})
            resp.raise_for_status()
            concept = resp.json()["result"]

            # Fetch definitions (separate endpoint)
            ticket = self.get_service_ticket(tgt_url)  # tickets are single-use
            defs_url = f"{UMLS_CONTENT}/content/current/CUI/{cui}/definitions"
            defs_resp = requests.get(defs_url, params={"ticket": ticket})
            definitions = []
            if defs_resp.status_code == 200:
                for d in defs_resp.json().get("result", []):
                    definitions.append({
                        "source": d.get("rootSource"),
                        "text": d.get("value")
                    })

            return {
                "cui": cui,
                "name": concept.get("name"),
                "semantic_types": [st["name"] for st in concept.get("semanticTypes", [])],
                "definitions": definitions
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                # Clear TGT maybe?
                self._tgt_url = None
            raise
