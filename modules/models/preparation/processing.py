import re
import pandas as pd
from tqdm import tqdm

class Extractor:
    def __init__(self):
        pass

    def structural_split_radiology(self, text):
        """
        Splits a radiology report into its constituent sections based on predefined headers.

        Args:
            text (str): The raw text of the radiology report.

        Returns:
            list: A list of strings, each containing a section of the report.
        """
        headers = ["EXAMINATION", "INDICATION", "TECHNIQUE", "COMPARISON", "FINDINGS", 'IMPRESSION']
        pattern = r'(?=' + '|'.join(headers) + r')'

        sections = re.split(pattern, text)
        sections = [s.strip() for s in sections if s.strip()]

        return sections 

    def structural_split_discharge(self, text):
        """
        Splits a discharge summary into its constituent sections based on predefined headers.

        Args:
            text (str): The raw text of the discharge summary.

        Returns:
            list: A list of strings, each containing a section of the summary.
        """
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

    def structural_df_radiology(self, text):
        """
        Parses a radiology report into a structured pandas DataFrame.

        Args:
            text (str): The raw text of the radiology report.

        Returns:
            pandas.DataFrame: A DataFrame with columns for each report section with section name as columns.
        """
        sections = self.structural_split_radiology(text)
        result = {"Text": text}
        for section in sections:
            match = re.match(r'(EXAMINATION|INDICATION|TECHNIQUE|COMPARISON|FINDINGS|IMPRESSION)[:\s]*(.*)', section, re.DOTALL)
            if match:
                key = match.group(1).capitalize()
                value = match.group(2).strip()
                result[key] = value

        df = pd.DataFrame([result])
        
        cols = df.columns
        df[cols] = df[cols].apply(lambda x: x.str.replace('_', '', regex=False).str.replace('\n', ' ', regex=False).str.strip())

        cols = ["Text", "Examination", "Indication", "Technique", "Comparison", "Findings", "Impression"]

        df['Technique'] = df['Technique'].str.title()

        return df[[c for c in cols if c in df.columns]]

    def structural_df_discharge(self, text):
        """
        Parses 1 discharge summary into a structured pandas DataFrame.

        Args:
            text (str): The raw text of the discharge summary.

        Returns:
            pandas.DataFrame: A DataFrame with columns for each summary section with section names as columns
        """
        headers = [
            "Allergies", "Attending", "Chief Complaint", "Major Surgical or Invasive Procedure",
            "History of Present Illness", "Past Medical History", "Social History", "Family History",
            "Physical Exam", "Pertinent Results", "Brief Hospital Course", "Discharge Diagnosis",
            "Discharge Medications", "Discharge Disposition", "Discharge Condition", "Discharge Instructions",
            "Followup Instructions"
        ]
        sections = self.structural_split_discharge(text)
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
        df[cols] = df[cols].apply(lambda x: x.str.replace('_', '', regex=False).str.replace('\n', ' ', regex=False).str.strip().str.replace(r'^[^a-zA-Z0-9]+', '', regex=True))

        cols = ["Text"] + headers


        return df[[c for c in cols if c in df.columns]]

    def batch_extracting_radiology(self, df):
        """
        Processes a DataFrame of radiology reports, adding structured columns for each report.

        Args:
            df (pandas.DataFrame): DataFrame containing a 'text' column with raw radiology reports.

        Returns:
            pandas.DataFrame: The original DataFrame enriched with structured report sections.
        """
        structured_rows = []

        for text in tqdm(df['text']):
            row = self.structural_df_radiology(text)
            structured_rows.append(row.iloc[0] if len(row) > 0 else pd.Series())

        structured_df_all = pd.DataFrame(structured_rows).reset_index(drop=True)
        structured_df_all = structured_df_all.drop(columns=['Text'], errors='ignore')

        df = pd.concat([df.reset_index(drop=True), structured_df_all], axis=1)

        cols = ['Indication', 'Technique', 'Comparison', 'Findings', 'Impression']
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('_', '', regex=False).str.strip()

        return df

    def batch_extracting_discharge(self, df):
        """
        Processes a DataFrame of discharge summaries, adding structured columns for each summary.

        Args:
            df (pandas.DataFrame): DataFrame containing a 'text' column with raw discharge summaries.

        Returns:
            pandas.DataFrame: The original DataFrame enriched with structured summary sections.
        """
        headers = [
            "Allergies", "Attending", "Chief Complaint", "Major Surgical or Invasive Procedure",
            "History of Present Illness", "Past Medical History", "Social History", "Family History",
            "Physical Exam", "Pertinent Results", "Brief Hospital Course", "Discharge Diagnosis",
            "Discharge Medications", "Discharge Disposition", "Discharge Condition", "Discharge Instructions",
            "Followup Instructions"
        ]
        structured_rows = []

        for text in tqdm(df['text']):
            row = self.structural_df_discharge(text)
            structured_rows.append(row.iloc[0] if len(row) > 0 else pd.Series())

        structured_df_all = pd.DataFrame(structured_rows).reset_index(drop=True)
        structured_df_all = structured_df_all.drop(columns=['Text'], errors='ignore')

        df = pd.concat([df.reset_index(drop=True), structured_df_all], axis=1)

        # Clean the extracted text columns
        for col in headers:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('_', '', regex=False).str.strip()

        return df

    def procedure_input(self, radiology, discharge):
        """
        Extracts relevant text components for procedure identification from radiology and discharge reports for Procedure extraction model

        Args:
            radiology (str): The raw text of the radiology report.
            discharge (str): The raw text of the discharge summary.

        Returns:
            tuple: A tuple containing (Individual Procedures from Discharge, cleaned Radiology note).
        """

        df_dis = self.structural_df_discharge(discharge)
        df_rad = self.structural_df_radiology(radiology)

        return df_dis['Major Surgical or Invasive Procedure'].iloc[0], (df_rad['Examination'] + ' - ' + df_rad['Indication'] + ' - ' + df_rad['Technique'] + ' - ' + df_rad['Impression']).iloc[0]

    def diagnosis_input(self, radiology, discharge):
        """
        Extracts relevant text components for diagnosis identification from radiology and discharge reports for Diagnosis extraction model

        Args:
            radiology (str): The raw text of the radiology report.
            discharge (str): The raw text of the discharge summary.

        Returns:
            tuple: A tuple containing (cleaned Radiology note, cleaned Discharge note).
        """

        df_dis = self.structural_df_discharge(discharge)
        df_rad = self.structural_df_radiology(radiology)

        return (df_rad['Indication'] + ' - ' + df_rad['Findings'] + ' - ' + df_rad['Impression']).iloc[0], (df_dis['History of Present Illness'] + ' - ' + df_dis['Past Medical History'] + ' - ' + df_dis['Brief Hospital Course'] + ' - ' + df_dis['Discharge Diagnosis']).iloc[0] 