import pandas as pd
import numpy as np

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

def parse_faers_xml(file_path):
    rows = []

    for event, elem in ET.iterparse(file_path, events=("end",)):

        if elem.tag == "safetyreport":

            report_id = elem.findtext(".//safetyreportid")
            date = elem.findtext(".//receivedate")

            drugs = []
            reactions = []

            # extract drugs
            for drug in elem.findall(".//drug"):
                drug_name = drug.findtext("medicinalproduct")
                role = drug.findtext("drugcharacterization")
                indication = drug.findtext("drugindication")

                drugs.append((drug_name, role, indication))

            # extract reactions
            for reaction in elem.findall(".//reaction"):
                adr = reaction.findtext("reactionmeddrapt")
                reactions.append(adr)

            # create rows
            for drug_name, role, indication in drugs:
                for adr in reactions:
                    rows.append({
                        "report_id": report_id,
                        "date": date,
                        "drug": drug_name,
                        "role": role,
                        "indication": indication,
                        "reaction": adr
                    })

            elem.clear()

    return pd.DataFrame(rows)

df_1 = parse_faers_xml(os.path.join(faers_path, '1_ADR25Q4.xml'))
df_2 = parse_faers_xml(os.path.join(faers_path, '2_ADR25Q4.xml'))
df_3 = parse_faers_xml(os.path.join(faers_path, '3_ADR25Q4.xml'))

df = pd.concat([df_1, df_2, df_3], axis = 0)

df['role'] = df['role'].apply(int)

df = df[df['role'] < 3]

df = df.drop(['role'], axis = 1)

df['disease'] = df['disease'].apply(lambda x: 'Unknown' if x == 'Product used for unknown indication' else x)

df['drug'] = df['drug'].apply(lambda x: x.capitalize())

df.to_csv(os.path.join(faers_path, 'full.csv'))