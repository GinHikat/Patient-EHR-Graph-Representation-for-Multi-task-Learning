import os
import sys
import unittest

# Ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.models.extractor import ProcDiagExtractor
from modules.extend.ner_engine import extract_entities_dl

class TestDLExtractor(unittest.TestCase):
    def setUp(self):
        self.sample_sentence = "Patient presents with chest pain and shortness of breath."
        # We also need to test `extract_entities_dl` but it uses the global `_dl_extractor`.
        # However, `extract_entities_dl` needs to be updated to take threshold first.
        # But we can test `ProcDiagExtractor` directly.
        self.extractor = ProcDiagExtractor("statedict_900_202")

    def test_threshold_impact(self):
        # Run predict with a low threshold
        res_low = self.extractor.predict(self.sample_sentence, threshold=0.1)
        self.assertIsInstance(res_low, dict)
        print(f"Results at threshold=0.1: {len(res_low)} entities")

        # Run predict with a high threshold
        res_high = self.extractor.predict(self.sample_sentence, threshold=0.9)
        self.assertIsInstance(res_high, dict)
        print(f"Results at threshold=0.9: {len(res_high)} entities")

        # The number of entities returned at high threshold should be <= the number at low threshold
        self.assertLessEqual(len(res_high), len(res_low))

if __name__ == '__main__':
    unittest.main()
