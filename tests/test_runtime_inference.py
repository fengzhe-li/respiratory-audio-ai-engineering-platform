import unittest
from pathlib import Path


class RuntimeInferenceTests(unittest.TestCase):
    SAMPLE_AUDIO = Path(
        "/Volumes/TECLAST/MachineLearningProject_UCL/processed_audio_files/"
        "101_1b1_Al_sc_Meditron_0.wav"
    )
    MODEL_PATH = Path("models/melspectogramTrained.h5")

    def test_saved_model_runs_refactored_inference_on_processed_sample(self):
        try:
            import numpy as np
            import librosa
            from tensorflow.keras.models import load_model
            from deepbreath.inference.predict import predict_audio_file
            from deepbreath.ml_contract import (
                CLASS_NAMES,
                EXPECTED_SAMPLE_RATE,
                MEL_SPECTROGRAM_SHAPE,
            )
            from deepbreath.features.mel_spectrogram import extract_mel_spectrogram
        except ImportError as exc:
            self.skipTest(f"Runtime ML dependencies are not installed: {exc}")

        if not self.SAMPLE_AUDIO.exists():
            self.skipTest("External processed sample audio is not available.")

        audio, sample_rate = librosa.load(self.SAMPLE_AUDIO, sr=EXPECTED_SAMPLE_RATE)
        self.assertEqual(sample_rate, EXPECTED_SAMPLE_RATE)
        self.assertAlmostEqual(len(audio) / sample_rate, 6.0, places=5)

        mel = extract_mel_spectrogram(self.SAMPLE_AUDIO)
        self.assertEqual(mel.shape, MEL_SPECTROGRAM_SHAPE)

        model = load_model(self.MODEL_PATH)
        self.assertEqual(model.input_shape, (None, 128, 259, 1))
        self.assertEqual(model.output_shape, (None, 6))

        result = predict_audio_file(model, self.SAMPLE_AUDIO)
        probabilities = result["probabilities"]
        self.assertEqual(tuple(probabilities.keys()), CLASS_NAMES)
        self.assertEqual(len(probabilities), 6)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in probabilities.values()))
        self.assertAlmostEqual(float(np.sum(list(probabilities.values()))), 1.0, places=5)
        self.assertIn(result["top_label"], CLASS_NAMES)
        self.assertIn(result["runner_up_label"], CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()
