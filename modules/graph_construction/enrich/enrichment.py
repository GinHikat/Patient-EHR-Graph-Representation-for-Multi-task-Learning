import requests
import os, sys

UMLS_API_KEY = os.getenv('UML_API_KEY')
UMLS_BASE = "https://utslogin.nlm.nih.gov/cas/v1"
UMLS_CONTENT = "https://uts-ws.nlm.nih.gov/rest"

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
