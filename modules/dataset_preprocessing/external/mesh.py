import requests
import xml.etree.ElementTree as ET
import pandas as pd

url = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml"
r = requests.get(url, stream=True)

with open("desc2026.xml", "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

tree = ET.parse("desc2026.xml")
root = tree.getroot()

records = []
for desc in root.findall("DescriptorRecord"):
    mesh_id = desc.findtext("DescriptorUI")
    name    = desc.findtext("DescriptorName/String")
    records.append({"mesh_id": mesh_id, "name": name})

mesh = pd.DataFrame(records)
mesh.to_csv("mesh_terms.csv", index=False)
