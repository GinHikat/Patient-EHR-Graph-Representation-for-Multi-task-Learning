import pandas as pd
import numpy as np
import os, sys
from transformers import AutoModel, AutoTokenizer

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

map_dir = os.path.join(project_root, 'data', 'generalize')
diagnosis_map = pd.read_csv(os.path.join(map_dir, 'diagnosis_map.csv'))
procedure_map = pd.read_csv(os.path.join(map_dir, 'procedure_map.csv'))

diagnosis_dict = diagnosis_map.set_index('ccsr_description')['ccsr_category'].to_dict()
procedure_dict = procedure_map.set_index('ccsr_description')['ccsr_category'].to_dict()

from preparation.processing import Extractor
from shared_functions.global_functions import *

extractor = Extractor()

class ProcDiagExtractor:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()

    def load_model(self):
        return AutoModel.from_pretrained(self.checkpoint)

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.checkpoint)

    def extract(self, text):
        """
        Extracts hidden state features from the input text using the loaded model.

        Args:
            text (str): The input text to process.

        Returns:
            torch.Tensor: The last hidden state from the model's output.
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def input_processing(self, discharge, radiology):
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

        procedure = list(procedure).append(self.extract(procedure_input))
        diagnosis = self.extract(diagnosis_rad, diagnosis_dis)

        return procedure, diagnosis

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

            procedure, diagnosis = self.input_processing(discharge, radiology)

            procedure = [procedure_dict[i] for i in procedure]
            diagnosis = [diagnosis_dict[i] for i in diagnosis]

            query = """
                UNWIND $rows AS row

                MATCH (a:Admission:Test:MIMIC {id: row.hadm_id})
                MATCH (p:Procedure:Test:MIMIC {id: row.procedure})
                MATCH (d:Diagnosis:Test:MIMIC {id: row.diagnosis})

                WITH a, row
                UNWIND row.procedure AS procedure
                MERGE (p:Procedure:Test:MIMIC {id: procedure})
                MERGE (a)-[:HAS_PROCEDURE]->(p)

                WITH d, row
                UNWIND row.diagnosis AS diagnosis
                MERGE (diag:Diagnosis:Test:MIMIC {id: diagnosis})
                MERGE (d)-[:HAS_DIAGNOSIS]->(diag)

                """

            rows = []
            rows.append({
                "hadm_id": i['hadm_id'],
                "procedure": procedure,
                "diagnosis": diagnosis
            })

            dml_ddl_neo4j(
                query,
                progress=False,
                rows=rows
            )

