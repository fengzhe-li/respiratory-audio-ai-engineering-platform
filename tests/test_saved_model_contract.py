import re
import unittest
from pathlib import Path


class SavedModelContractTests(unittest.TestCase):
    MODEL_PATH = Path("models/melspectogramTrained.h5")

    def test_saved_h5_model_contract_matches_refactored_constants(self):
        content = self.MODEL_PATH.read_bytes().decode("latin-1", errors="ignore")
        self.assertIn('"batch_shape": [null, 128, 259, 1]', content)
        self.assertIn('"name": "mSpec_model"', content)
        self.assertIn('"name": "mspec"', content)
        self.assertIn('"units": 6', content)
        self.assertIn('"activation": "softmax"', content)
        self.assertIn('"loss": "sparse_categorical_crossentropy"', content)
        self.assertIn('"class_name": "Nadam"', content)

    def test_saved_h5_has_one_six_unit_output_layer(self):
        content = self.MODEL_PATH.read_bytes().decode("latin-1", errors="ignore")
        six_unit_layers = re.findall(r'"class_name": "Dense".*?"units": 6', content)
        self.assertGreaterEqual(len(six_unit_layers), 1)


if __name__ == "__main__":
    unittest.main()
