import json
import unittest
from pathlib import Path


class PublicSecurityTests(unittest.TestCase):
    NOTEBOOKS = tuple(Path("notebooks/archive/research").glob("*.ipynb"))
    FORBIDDEN_SNIPPETS = (
        "AI" + "zaSy",
        "C:" + "/Users/",
        "C:" + "\\\\Users\\\\",
        "One" + "Drive",
        "D:" + "\\\\UCL_ML_Project",
    )

    def test_public_notebooks_have_no_outputs_or_execution_counts(self):
        self.assertTrue(self.NOTEBOOKS)
        for notebook in self.NOTEBOOKS:
            data = json.loads(notebook.read_text())
            for cell in data.get("cells", []):
                self.assertEqual(cell.get("outputs", []), [], notebook)
                self.assertIsNone(cell.get("execution_count"), notebook)

    def test_public_notebooks_do_not_contain_known_secret_or_private_path_patterns(self):
        for notebook in self.NOTEBOOKS:
            text = notebook.read_text(errors="ignore")
            for snippet in self.FORBIDDEN_SNIPPETS:
                self.assertNotIn(snippet, text, f"{snippet} found in {notebook}")


if __name__ == "__main__":
    unittest.main()
