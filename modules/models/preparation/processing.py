import re
import pandas as pd
from tqdm import tqdm

def structural_split_radiology(text):
    headers = ["EXAMINATION", "INDICATION", "TECHNIQUE", "COMPARISON", "FINDINGS", 'IMPRESSION']
    pattern = r'(?=' + '|'.join(headers) + r')'

    sections = re.split(pattern, text)
    sections = [s.strip() for s in sections if s.strip()]

    return sections 

def structural_split_discharge(text):
    headers = [
        "Allergies", "Attending", "Chief Complaint", "Major Surgical or Invasive Procedure",
        "History of Present Illness", "Past Medical History", "Social History", "Family History",
        "Physical Exam", "Pertinent Results", "Brief Hospital Course", "Discharge Diagnosis",
        "Discharge Medications", "Discharge Disposition", "Discharge Condition", "Discharge Instructions",
        "Followup Instructions"
    ]
    pattern = r'(?=' + '|'.join(headers) + r')'

    sections = re.split(pattern, text)
    sections = [s.strip() for s in sections if s.strip()]

    return sections 

def structural_df_radiology(text):
    sections = structural_split_radiology(text)
    result = {"Text": text}
    for section in sections:
        match = re.match(r'(EXAMINATION|INDICATION|TECHNIQUE|COMPARISON|FINDINGS|IMPRESSION)[:\s]*(.*)', section, re.DOTALL)
        if match:
            key = match.group(1).capitalize()
            value = match.group(2).strip()
            result[key] = value

    df = pd.DataFrame([result])
    
    cols = df.columns
    df[cols] = df[cols].apply(lambda x: x.str.replace('_', '', regex=False).str.strip())

    cols = ["Text", "Examination", "Indication", "Technique", "Comparison", "Findings", "Impression"]

    df['Technique'] = df['Technique'].str.title()

    return df[[c for c in cols if c in df.columns]]

def structural_df_discharge(text):
    headers = [
        "Allergies", "Attending", "Chief Complaint", "Major Surgical or Invasive Procedure",
        "History of Present Illness", "Past Medical History", "Social History", "Family History",
        "Physical Exam", "Pertinent Results", "Brief Hospital Course", "Discharge Diagnosis",
        "Discharge Medications", "Discharge Disposition", "Discharge Condition", "Discharge Instructions",
        "Followup Instructions"
    ]
    sections = structural_split_discharge(text)
    result = {"Text": text}
    pattern = r'(' + '|'.join(headers) + r')[:\s]*(.*)'
    for section in sections:
        match = re.match(pattern, section, re.DOTALL)
        if match:
            key = match.group(1)
            value = match.group(2).strip()
            result[key] = value

    df = pd.DataFrame([result])

    cols = df.columns
    df[cols] = df[cols].apply(lambda x: x.str.replace('_', '', regex=False).str.strip())

    cols = ["Text"] + headers

    return df[[c for c in cols if c in df.columns]]

def batch_extracting_radiology(df):
    structured_rows = []

    for text in tqdm(df['text']):
        row = structural_df_radiology(text)
        structured_rows.append(row.iloc[0] if len(row) > 0 else pd.Series())

    structured_df_all = pd.DataFrame(structured_rows).reset_index(drop=True)
    structured_df_all = structured_df_all.drop(columns=['Text'], errors='ignore')

    df = pd.concat([df.reset_index(drop=True), structured_df_all], axis=1)

    cols = ['Indication', 'Technique', 'Comparison', 'Findings', 'Impression']
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('_', '', regex=False).str.strip()

    return df

def batch_extracting_discharge(df):
    headers = [
        "Allergies", "Attending", "Chief Complaint", "Major Surgical or Invasive Procedure",
        "History of Present Illness", "Past Medical History", "Social History", "Family History",
        "Physical Exam", "Pertinent Results", "Brief Hospital Course", "Discharge Diagnosis",
        "Discharge Medications", "Discharge Disposition", "Discharge Condition", "Discharge Instructions",
        "Followup Instructions"
    ]
    structured_rows = []

    for text in tqdm(df['text']):
        row = structural_df_discharge(text)
        structured_rows.append(row.iloc[0] if len(row) > 0 else pd.Series())

    structured_df_all = pd.DataFrame(structured_rows).reset_index(drop=True)
    structured_df_all = structured_df_all.drop(columns=['Text'], errors='ignore')

    df = pd.concat([df.reset_index(drop=True), structured_df_all], axis=1)

    # Clean the extracted text columns
    for col in headers:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('_', '', regex=False).str.strip()

    return df

