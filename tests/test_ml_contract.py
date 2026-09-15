import unittest

from deepbreath import ml_contract


class MLContractTests(unittest.TestCase):
    def test_core_shapes_and_classes(self):
        self.assertEqual(ml_contract.AUDIO_DURATION_SECONDS, 6.0)
        self.assertEqual(ml_contract.EXPECTED_SAMPLE_RATE, 22050)
        self.assertEqual(ml_contract.MEL_SPECTROGRAM_SHAPE, (128, 259))
        self.assertEqual(ml_contract.CNN_INPUT_SHAPE, (128, 259, 1))
        self.assertEqual(ml_contract.NUM_CLASSES, 6)
        self.assertEqual(
            ml_contract.CLASS_NAMES,
            (
                "Bronchiectasis",
                "Bronchiolitis",
                "COPD",
                "Healthy",
                "Pneumonia",
                "URTI",
            ),
        )

    def test_legacy_model_filename_keeps_research_provenance(self):
        self.assertEqual(ml_contract.LEGACY_MODEL_FILENAME, "melspectogramTrained.h5")


if __name__ == "__main__":
    unittest.main()
