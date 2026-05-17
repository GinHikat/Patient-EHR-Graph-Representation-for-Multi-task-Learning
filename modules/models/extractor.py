import pandas as pd
import numpy as np
import os, sys
import torch
import joblib
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

map_dir = os.path.join(project_root, 'data')
diagnosis_map = pd.read_csv(os.path.join(map_dir, 'diagnosis_map.csv'))
# procedure_map = pd.read_csv(os.path.join(map_dir, 'procedure_map.csv'))

diagnosis_dict = diagnosis_map.set_index('ccsr_description')['ccsr_category'].to_dict()
# procedure_dict = procedure_map.set_index('ccsr_description')['ccsr_category'].to_dict()

from modules.models.preparation.processing import Extractor
from shared_functions.global_functions import *
from modules.models.models import PLMICD_Internal

extractor = Extractor()

class ProcDiagExtractor:
    def __init__(self, checkpoint):
        self.checkpoint = os.path.join(script_dir, '.models', checkpoint)
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(f"Could not find the model folder at {self.checkpoint}")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)
        
        self.mlb = joblib.load(os.path.join(self.checkpoint, "mlb.pkl"))
        num_labels = len(self.mlb.classes_)

        # Load state dict if model_state.pt exists to dynamically detect model type
        state_dict_path = os.path.join(self.checkpoint, "model_state.pt")
        is_plmicd = False
        is_msmn = False
        state_dict = None
        
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location=self.device)
            # Clean torch.compile '_orig_mod.' prefix and handle key mappings
            state_dict = {
                k.replace("_orig_mod.model.", "").replace("_orig_mod.", "").replace(".gamma", ".weight").replace(".beta", ".bias"): v 
                for k, v in state_dict.items()
            }
            
            # Check keys to detect model type
            for key in state_dict.keys():
                if 'laat.' in key:
                    is_plmicd = True
                    break
                elif 'msa.' in key:
                    is_msmn = True
                    break
        else:
            # Fall back to folder name suffix if model_state.pt doesn't exist
            model_tag = checkpoint.split('_')[-1]
            if model_tag == 'plmicd':
                is_plmicd = True
            elif model_tag == 'msmn':
                is_msmn = True

        # Initialize the correct model architecture
        if is_plmicd:
            self.model = PLMICD_Internal(num_labels=num_labels)
        elif is_msmn:
            self.model = MSMN_Internal(num_labels=num_labels)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, local_files_only=True)

        # Load weights if we have a state dict
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()


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

    def predict_batch(self, texts, batch_size=32, threshold=0.5):
        """
        Predicts diagnoses for a list of radiology texts using batched GPU/CPU inference with dynamic padding.
        """
        import re
        parsed_texts = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                parsed_texts.append("")
                continue
            try:
                # Fast inline regex matching structural_df_radiology logic
                match_ind = re.search(r"INDICATION:\s*(.*?)(?=\n[A-Z\s]+:|$)", text, re.IGNORECASE | re.DOTALL)
                match_find = re.search(r"FINDINGS:\s*(.*?)(?=\n[A-Z\s]+:|$)", text, re.IGNORECASE | re.DOTALL)
                match_imp = re.search(r"IMPRESSION:\s*(.*?)(?=\n[A-Z\s]+:|$)", text, re.IGNORECASE | re.DOTALL)
                
                ind = match_ind.group(1).strip() if match_ind else ""
                find = match_find.group(1).strip() if match_find else ""
                imp = match_imp.group(1).strip() if match_imp else ""
                
                combined = " - ".join([x for x in [ind, find, imp] if x])
                parsed_texts.append(combined if combined else text)
            except Exception:
                parsed_texts.append(text)

        all_results = []
        
        # Ensure model is on device
        self.model.to(self.device)
        self.model.eval()

        try:
            from tqdm import tqdm
            pbar = tqdm(range(0, len(parsed_texts), batch_size), desc="Running Batched Inference")
        except ImportError:
            pbar = range(0, len(parsed_texts), batch_size)

        for i in pbar:
            batch = parsed_texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True # Dynamic padding to the longest sequence in the batch
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs_batch = torch.sigmoid(outputs.logits).cpu().numpy()
                
            for probs in probs_batch:
                pred_indices = np.where(probs >= threshold)[0]
                pred_labels = self.mlb.classes_[pred_indices]
                mapping = [diagnosis_dict.get(lbl, lbl) for lbl in pred_labels]
                all_results.append(mapping)
                
        return all_results

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
        
        # Check if discharge is empty/blank/nan
        is_discharge_blank = False
        if discharge is None or (isinstance(discharge, float) and pd.isna(discharge)):
            is_discharge_blank = True
        elif isinstance(discharge, str) and not discharge.strip():
            is_discharge_blank = True

        if is_discharge_blank:
            # If discharge is blank, process only the radiology report (ignore discharge input)
            df_rad = extractor.structural_df_radiology(radiology)
            diagnosis_rad = (df_rad['Indication'] + ' - ' + df_rad['Findings'] + ' - ' + df_rad['Impression']).iloc[0]
            diagnosis_rad = str(diagnosis_rad.iloc[0]) if hasattr(diagnosis_rad, 'iloc') else str(diagnosis_rad)
            
            diagnosis = list(self.predict(diagnosis_rad, text_pair=None, threshold=threshold).keys())
        else:
            procedure, procedure_input = extractor.procedure_input(radiology, discharge) 
            diagnosis_rad, diagnosis_dis = extractor.diagnosis_input(radiology, discharge)
            
            # Ensure inputs are strings (handles cases where Pandas Series are returned)
            diagnosis_rad = str(diagnosis_rad.iloc[0]) if hasattr(diagnosis_rad, 'iloc') else str(diagnosis_rad)
            diagnosis_dis = str(diagnosis_dis.iloc[0]) if hasattr(diagnosis_dis, 'iloc') else str(diagnosis_dis)

            diagnosis = list(self.predict(diagnosis_rad, diagnosis_dis, threshold).keys())

        mapping = [diagnosis_dict.get(i, i) for i in diagnosis]

        return {d: m for d, m in zip(diagnosis, mapping)}

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

            query = """
                UNWIND $rows AS row

                MATCH (a:Admission:Test:MIMIC {id: row.hadm_id})
                # MATCH (p:Procedure:Test:MIMIC {id: row.procedure})
                MATCH (d:Diagnosis:Test:MIMIC {id: row.diagnosis})

                WITH a, row
                # UNWIND row.procedure AS procedure
                # MERGE (p:Procedure:Test:MIMIC {id: procedure})
                # MERGE (a)-[:HAS_PROCEDURE]->(p)

                WITH d, row
                UNWIND row.diagnosis AS diagnosis
                MERGE (diag:Diagnosis:Test:MIMIC {id: diagnosis})
                MERGE (d)-[:HAS_DIAGNOSIS]->(diag)

                """

            rows = []
            rows.append({
                "hadm_id": i['hadm_id'],
                # "procedure": procedure,
                "diagnosis": diagnosis
            })

            dml_ddl_neo4j(
                query,
                progress=False,
                rows=rows
            )

