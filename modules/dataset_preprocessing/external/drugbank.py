import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import sys, os
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

data_dir = 'F:/Din/Study/Education/Projects/Thesis/data'
drugbank_path = os.path.join(data_dir, 'DrugBank')

FILE = os.path.join(drugbank_path, "full database.xml")
ns = "{http://www.drugbank.ca}"

records = []

def parse_dosage(dosage_list):
    
    def convert_to_mg(value, unit):
        if value is None:
            return None
        
        if unit == 'mg':
            return value
        elif unit == 'g':
            return value * 1000
        elif unit in ['mcg', 'ug', 'µg']:
            return value / 1000
        else:
            return None

    def parse_one(text):
        if not isinstance(text, str):
            return None
        
        text_lower = text.lower()
        
        # Extract route
        route_match = re.search(r'\((.*?)\)', text)
        route = route_match.group(1).lower() if route_match else None
        
        # Extract value + unit
        dose_match = re.search(r'(\d+\.?\d*)\s*(mg|g|mcg|ug|µg)\b', text_lower)
        if dose_match:
            value = float(dose_match.group(1))
            unit = dose_match.group(2)
            dose_mg = convert_to_mg(value, unit)
        else:
            value, unit, dose_mg = None, None, None
        
        # Detect concentration (presence of "/")
        is_concentration = '/' in text_lower
        
        # Extract concentration if present
        conc_match = re.search(r'(\d+\.?\d*)\s*(mg|g|mcg|ug|µg)\s*/\s*(\d+\.?\d*)\s*(ml)?', text_lower)
        if conc_match:
            conc_value = float(conc_match.group(1))
            conc_unit = conc_match.group(2)
            conc_vol = float(conc_match.group(3))
            
            conc_mg = convert_to_mg(conc_value, conc_unit)
            
            # Handle "mg/1" → treat as NOT real concentration
            if conc_vol == 1:
                conc_mg, conc_vol = None, None
                is_concentration = False
        else:
            conc_mg, conc_vol = None, None
        
        return {
            'raw': text,
            'route': route,
            'dose_mg': dose_mg,          # normalized to mg
            'conc_mg': conc_mg,          # normalized to mg
            'conc_vol': conc_vol,
            'is_concentration': is_concentration
        }
    
    # Handle non-list safely
    if not isinstance(dosage_list, list):
        return pd.DataFrame(columns=[
            'raw', 'route', 'dose_mg', 'conc_mg', 'conc_vol', 'is_concentration'
        ])
    
    parsed = [p for x in dosage_list if isinstance(x, str) 
          if (p := parse_one(x)) is not None]
    
    df = pd.DataFrame(parsed)
    df = df[df['dose_mg'].notna()]

    max = df['dose_mg'].max()
    min = df['dose_mg'].min()

    return pd.Series({'max_dosage': max, 'min_dosage': min})

# Check if file exists before processing
if not os.path.exists(FILE):
    print(f"Error: DrugBank file not found at {FILE}")
else:
    context = ET.iterparse(FILE, events=("end",))

    for event, elem in tqdm(context):

        if elem.tag == ns + "drug":

            drugbank_id = elem.findtext(ns + "drugbank-id[@primary='true']")

            # fallback if primary attribute missing
            if drugbank_id is None:
                ids = elem.findall(ns + "drugbank-id")
                if ids:
                    drugbank_id = ids[0].text

            # skip corrupted entries
            if drugbank_id is None:
                elem.clear()
                continue

            name = elem.findtext(ns + "name")
            description = elem.findtext(ns + "description")
            indication = elem.findtext(ns + "indication")

            # Dosages
            dosages = [
                f"{d.findtext(ns + 'form')} ({d.findtext(ns + 'route')}) - {d.findtext(ns + 'strength')}"
                for d in elem.findall(f"{ns}dosages/{ns}dosage")
            ]

            # Targets
            targets = [
                gene
                for gene in [
                    t.findtext(f"{ns}polypeptide/{ns}gene-name")
                    for t in elem.findall(f"{ns}targets/{ns}target")
                ]
                if gene
            ]

            # Drug Interactions
            interactions = [
                d.findtext(ns + "drugbank-id")
                for d in elem.findall(f"{ns}drug-interactions/{ns}drug-interaction")
                if d.findtext(ns + "drugbank-id")
            ]

            # External IDs
            external_ids = {}

            for ex in elem.findall(f"{ns}external-identifiers/{ns}external-identifier"):

                resource = ex.findtext(ns + "resource")
                identifier = ex.findtext(ns + "identifier")

                if resource and identifier:
                    external_ids[resource] = identifier

            # Append record
            records.append({
                "drugbank_id": drugbank_id,
                "name": name,
                "description": description,
                "indication": indication,
                "dosages": dosages,
                "interactions": interactions,
                "external_ids": external_ids
            })

            elem.clear()

    df = pd.DataFrame(records)

    df[['max_dosage', 'min_dosage']] = df['dosages'].apply(parse_dosage)

    print("Total drugs:", len(df))
    print("Missing drugbank_id:", df["drugbank_id"].isna().sum())
    # Display sample to verify new fields
    if not df.empty:
        print(df[["drugbank_id", "name", "indication", "dosages"]].head())
    
df.to_csv(os.path.join(data_dir, 'DrugBank', "drugbank.csv"), index=False)
