import ast
import unittest
from pathlib import Path


class StaticContractTests(unittest.TestCase):
    def test_prediction_output_contract_keys_are_present(self):
        tree = ast.parse(Path("src/deepbreath/inference/predict.py").read_text())
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertTrue(
            {
                "audio_path",
                "top_label",
                "top_confidence",
                "runner_up_label",
                "runner_up_confidence",
                "probabilities",
            }.issubset(constants)
        )

    def test_evaluation_helpers_are_explicit_about_label_type(self):
        source = Path("src/deepbreath/training/evaluate.py").read_text()
        self.assertIn("classification_metrics_encoded", source)
        self.assertIn("classification_metrics_labels", source)
        self.assertIn("confusion_matrix_encoded", source)
        self.assertIn("confusion_matrix_labels", source)


if __name__ == "__main__":
    unittest.main()
