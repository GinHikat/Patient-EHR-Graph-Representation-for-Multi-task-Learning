import pandas as pd
import numpy as np
import os, sys
from transformers import AutoModel, AutoTokenizer

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

from preparation.processing import Extractor

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
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def input_processing(self, discharge, radiology):
        
        procedure, procedure_input = extractor.procedure_input(radiology, discharge) 
        diagnosis_rad, diagnosis_dis = extractor.diagnosis_input(radiology, discharge)

        procedure = list(procedure).append(self.extract(procedure_input))
        diagnosis = self.extract(diagnosis_rad, diagnosis_dis)

        return procedure, diagnosis