import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

data_dir = 'F:/Din/Study/Education/Projects/Thesis/data'
drugbank_path = os.path.join(data_dir, 'DrugBank')

FILE = os.path.join(drugbank_path, "full database.xml")
ns = "{http://www.drugbank.ca}"

records = []

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

        # ATC Codes
        atc_codes = [
            code.attrib.get("code")
            for code in elem.findall(f"{ns}atc-codes/{ns}atc-code")
            if code.attrib.get("code")
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

        # Enzymes
        enzymes = [
            gene
            for gene in [
                e.findtext(f"{ns}polypeptide/{ns}gene-name")
                for e in elem.findall(f"{ns}enzymes/{ns}enzyme")
            ]
            if gene
        ]

        # Transporters
        transporters = [
            gene
            for gene in [
                t.findtext(f"{ns}polypeptide/{ns}gene-name")
                for t in elem.findall(f"{ns}transporters/{ns}transporter")
            ]
            if gene
        ]

        # Carriers
        carriers = [
            gene
            for gene in [
                c.findtext(f"{ns}polypeptide/{ns}gene-name")
                for c in elem.findall(f"{ns}carriers/{ns}carrier")
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
            "atc_codes": atc_codes,
            "targets": targets,
            "enzymes": enzymes,
            "transporters": transporters,
            "carriers": carriers,
            "interactions": interactions,
            "external_ids": external_ids
        })

        elem.clear()

df = pd.DataFrame(records)

print("Total drugs:", len(df))
print("Missing drugbank_id:", df["drugbank_id"].isna().sum())