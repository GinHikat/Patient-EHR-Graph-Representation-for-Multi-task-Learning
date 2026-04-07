import re
import pandas as pd

def structural_split(text):
    headers = ["EXAMINATION", "INDICATION", "TECHNIQUE", "COMPARISON", "FINDINGS", 'IMPRESSION']
    pattern = r'(?=' + '|'.join(headers) + r')'

    sections = re.split(pattern, text)
    sections = [s.strip() for s in sections if s.strip()]

    return sections 

def structural_df(text):
    sections = structural_split(text)
    result = {"Text": text}
    for section in sections:
        match = re.match(r'(EXAMINATION|INDICATION|TECHNIQUE|COMPARISON|FINDINGS|IMPRESSION)[:\s]*(.*)', section, re.DOTALL)
        if match:
            key = match.group(1).capitalize()
            value = match.group(2).strip()
            result[key] = value

    df = pd.DataFrame([result])
    cols = ["Text", "Examination", "Indication", "Technique", "Comparison", "Findings", "Impression"]
    return df[[c for c in cols if c in df.columns]]

def batch_extracting_radiology(df):

    structured_rows = []

    for text in tqdm(df['text']):
        row = structural_df(text)
        structured_rows.append(row.iloc[0] if len(row) > 0 else pd.Series())

    structured_df_all = pd.DataFrame(structured_rows).reset_index(drop=True)

    structured_df_all = structured_df_all.drop(columns=['Text'], errors='ignore')

    df = pd.concat([df.reset_index(drop=True), structured_df_all], axis=1)

    cols = ['Indication', 'Technique', 'Comparison', 'Findings', 'Impression']

    df[cols] = df[cols].apply(lambda x: x.str.replace('_', '', regex=False).str.strip())

    return df