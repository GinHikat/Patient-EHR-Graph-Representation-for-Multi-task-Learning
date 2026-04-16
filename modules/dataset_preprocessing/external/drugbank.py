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

    print("Total drugs:", len(df))
    print("Missing drugbank_id:", df["drugbank_id"].isna().sum())
    # Display sample to verify new fields
    if not df.empty:
        print(df[["drugbank_id", "name", "indication", "dosages"]].head())
    
df.to_csv(os.path.join(data_dir, "drugbank.csv"), index=False)
