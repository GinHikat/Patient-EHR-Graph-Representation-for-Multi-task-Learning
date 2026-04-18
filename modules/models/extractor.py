import pandas as pd
import numpy as np
import os, sys
import torch
import joblib
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

map_dir = os.path.join(project_root, 'data', 'generalize')
diagnosis_map = pd.read_csv(os.path.join(map_dir, 'diagnosis_map.csv'))
procedure_map = pd.read_csv(os.path.join(map_dir, 'procedure_map.csv'))

diagnosis_dict = diagnosis_map.set_index('ccsr_description')['ccsr_category'].to_dict()
procedure_dict = procedure_map.set_index('ccsr_description')['ccsr_category'].to_dict()

from modules.models.preparation.processing import Extractor
from shared_functions.global_functions import *

extractor = Extractor()

class ProcDiagExtractor:
    def __init__(self, checkpoint):
        self.checkpoint = os.path.join(script_dir, '.models', checkpoint)
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(f"Could not find the model folder at {self.checkpoint}")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        
        self.mlb = joblib.load(os.path.join(self.checkpoint, "mlb.pkl"))

    def load_model(self):
        return AutoModelForSequenceClassification.from_pretrained(self.checkpoint, local_files_only=True)

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)

    def predict(self, text, text_pair=None, threshold=0.5):
        # Tokenize (handle single or pair inputs)
        inputs = self.tokenizer(
            text, 
            text_pair=text_pair,
            return_tensors="pt", 
            truncation=True, 
            max_length=2048, 
            padding="max_length"
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Apply Sigmoid to get independent probabilities
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        
        # Filter by threshold and decode names
        pred_indices = np.where(probs >= threshold)[0]
        pred_labels = self.mlb.classes_[pred_indices]
        
        # Return both labels and their confidence scores
        return dict(zip(pred_labels, probs[pred_indices]))

    def predict_from_source(self, discharge, radiology, threshold = 0.5):
        """
        Processes discharge and radiology text to extract a list of procedures and diagnoses.

        Processes the input texts through an external extractor to a structured format, then converts them into a list of procedures and diagnoses name.

        Args:
            discharge (str): The patient's discharge summary text.
            radiology (str): The patient's radiology report text.

        Returns:
            tuple(list, list): A tuple containing cleaned procedures and diagnoses.
        """
        
        procedure, procedure_input = extractor.procedure_input(radiology, discharge) 
        diagnosis_rad, diagnosis_dis = extractor.diagnosis_input(radiology, discharge)
        
        # Ensure inputs are strings (handles cases where Pandas Series are returned)
        diagnosis_rad = str(diagnosis_rad.iloc[0]) if hasattr(diagnosis_rad, 'iloc') else str(diagnosis_rad)
        diagnosis_dis = str(diagnosis_dis.iloc[0]) if hasattr(diagnosis_dis, 'iloc') else str(diagnosis_dis)

        diagnosis = list(self.predict(diagnosis_rad, diagnosis_dis, threshold).keys())

        mapping = [diagnosis_dict[i] for i in diagnosis]

        return dict({diagnosis:mapping for diagnosis, mapping in zip(diagnosis, mapping)})

    def mapping(self, df):
        """
        Maps extracted procedures and diagnoses from a DataFrame to a Neo4j database.

        Iterates through the DataFrame, processes each record's clinical notes,
        maps the results to CCSR categories, and updates the graph database.

        Args:
            df (pandas.DataFrame): DataFrame containing 'radiology', 'discharge', 
                                   and 'hadm_id' columns.

        Returns:
            Connection from Admission id to generalized Procedure and Diagnosis id in the database
        """

        for i in df.iterrows():
            radiology = i['radiology']
            discharge = i['discharge']

            diagnosis = self.predict_from_source(discharge, radiology)

            # procedure = [procedure_dict[i] for i in procedure]
            diagnosis = [diagnosis_dict[i] for i in diagnosis]

            # query = """
            #     UNWIND $rows AS row

            #     MATCH (a:Admission:Test:MIMIC {id: row.hadm_id})
            #     # MATCH (p:Procedure:Test:MIMIC {id: row.procedure})
            #     MATCH (d:Diagnosis:Test:MIMIC {id: row.diagnosis})

            #     WITH a, row
            #     # UNWIND row.procedure AS procedure
            #     # MERGE (p:Procedure:Test:MIMIC {id: procedure})
            #     # MERGE (a)-[:HAS_PROCEDURE]->(p)

            #     WITH d, row
            #     UNWIND row.diagnosis AS diagnosis
            #     MERGE (diag:Diagnosis:Test:MIMIC {id: diagnosis})
            #     MERGE (d)-[:HAS_DIAGNOSIS]->(diag)

            #     """

            # rows = []
            # rows.append({
            #     "hadm_id": i['hadm_id'],
            #     # "procedure": procedure,
            #     "diagnosis": diagnosis
            # })

            # dml_ddl_neo4j(
            #     query,
            #     progress=False,
            #     rows=rows
            # )

