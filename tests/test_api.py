import math
import asyncio
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from deepbreath.app.api import app
from deepbreath.app import api as api_module
from deepbreath.ml_contract import AUDIO_DURATION_SECONDS, CLASS_NAMES, EXPECTED_SAMPLE_RATE, NUM_CLASSES


class TestDeepBreathApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

        samples = int(AUDIO_DURATION_SECONDS * EXPECTED_SAMPLE_RATE)
        time_axis = np.linspace(0, AUDIO_DURATION_SECONDS, samples, endpoint=False)
        waveform = (0.05 * np.sin(2 * math.pi * 220 * time_axis)).astype(np.float32)

        fixture = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        fixture.close()
        cls.wav_path = Path(fixture.name)
        sf.write(cls.wav_path, waveform, EXPECTED_SAMPLE_RATE)

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        cls.wav_path.unlink(missing_ok=True)

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_loaded"], body.get("model_error"))
        self.assertEqual(body["model_classes"], NUM_CLASSES)
        self.assertEqual(body["model"], "deepbreath_legacy_cnn")

    def test_predict_valid_wav_returns_six_class_probabilities(self):
        with self.wav_path.open("rb") as audio_file:
            response = self.client.post(
                "/predict",
                files={"file": ("sample.wav", audio_file, "audio/wav")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()

        self.assertIn(body["prediction"], CLASS_NAMES)
        self.assertGreaterEqual(body["confidence"], 0.0)
        self.assertLessEqual(body["confidence"], 1.0)
        self.assertEqual(body["model"], "deepbreath_legacy_cnn")
        self.assertIn("Research prototype only", body["disclaimer"])

        probabilities = body["probabilities"]
        self.assertEqual(set(probabilities), set(CLASS_NAMES))
        self.assertEqual(len(probabilities), NUM_CLASSES)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)
        self.assertAlmostEqual(body["confidence"], probabilities[body["prediction"]], places=6)

    def test_predict_requires_file(self):
        response = self.client.post("/predict")

        self.assertEqual(response.status_code, 422)

    def test_predict_rejects_unsupported_file_type(self):
        response = self.client.post(
            "/predict",
            files={"file": ("sample.txt", b"not audio", "text/plain")},
        )

        self.assertEqual(response.status_code, 415)

    def test_predict_rejects_unreadable_wav(self):
        response = self.client.post(
            "/predict",
            files={"file": ("broken.wav", b"not a wav file", "audio/wav")},
        )

        self.assertEqual(response.status_code, 400)

    def test_predict_rejects_oversized_payload(self):
        with patch.object(api_module, "SETTINGS", replace(api_module.SETTINGS, max_upload_bytes=8)):
            response = self.client.post(
                "/predict",
                files={"file": ("large.wav", b"0123456789", "audio/wav")},
            )
        self.assertEqual(response.status_code, 413)

    def test_oversized_upload_temp_file_is_removed(self):
        original = api_module.tempfile.NamedTemporaryFile
        created = []

        def capture_tempfile(*args, **kwargs):
            handle = original(*args, **kwargs)
            created.append(Path(handle.name))
            return handle

        class FakeUpload:
            async def read(self, _size):
                return b"0123456789"

        with patch.object(api_module, "SETTINGS", replace(api_module.SETTINGS, max_upload_bytes=8)):
            with patch.object(api_module.tempfile, "NamedTemporaryFile", side_effect=capture_tempfile):
                with self.assertRaises(api_module.HTTPException):
                    asyncio.run(api_module._persist_upload(FakeUpload()))
        self.assertEqual(len(created), 1)
        self.assertFalse(created[0].exists())


if __name__ == "__main__":
    unittest.main()
