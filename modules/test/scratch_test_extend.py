import os
import sys
import pandas as pd
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Set working directory to project root
os.chdir(project_root)

try:
    from modules.extend.ner_engine import extract_entities
except Exception as e:
    print(f"Error importing extract_entities: {e}")
    sys.exit(1)

# Comprehensive clinical note containing a wide variety of medical entities
comprehensive_note = (
    "Patient is a 72-year-old female with a long history of Type 2 Diabetes Mellitus, "
    "Hypertension, and severe COPD. She presented to the emergency department complaining of "
    "acute chest tightness, shortness of breath, and bilateral lower extremity edema. "
    "On physical examination, she had tachycardia and diffuse bilateral lung crackles. "
    "An electrocardiogram (ECG) was performed immediately, followed by a chest X-ray. "
    "Laboratory tests revealed a serum creatinine of 1.8 mg/dL, an elevated Troponin I of 4.2 ng/mL, "
    "and a blood glucose of 240 mg/dL. The patient was diagnosed with an acute myocardial infarction. "
    "She was immediately started on Metoprolol 25mg PO, Atorvastatin 80mg daily, and albuterol nebulizer. "
    "A cardiac catheterization was scheduled for the next morning."
)

print(f"Clinical Note for Testing:\n{comprehensive_note}\n")

# Run the extended entity extraction
df = extract_entities(comprehensive_note)

if not df.empty:
    # Sort by category and similarity to see results clearly
    df_sorted = df.sort_values(by=['category', 'similarity'], ascending=[True, False])
    print("--- Extracted and Categorized Entities DataFrame ---")
    print(df_sorted.to_string(index=False))
else:
    print("No entities extracted from the comprehensive note.")
