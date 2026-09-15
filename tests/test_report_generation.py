import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from deepbreath.app.api import app
from deepbreath.reporting.llm_client import LLMConfig
from deepbreath.reporting.report_generator import (
    REPORT_DISCLAIMER,
    StructuredReportInput,
    StructuredReportOutput,
    build_report_prompt,
    generate_structured_report,
)


class MockMessage:
    content = json.dumps(
        {
            "summary": "The model output suggests COPD as the highest probability class.",
            "interpretation": "This is an explanation of model probabilities only.",
            "limitations": "This is not a diagnosis and cannot guide treatment.",
        }
    )


class MockChoice:
    message = MockMessage()


class MockCompletionResponse:
    choices = [MockChoice()]


class MockCompletions:
    def create(self, model, messages):
        self.model = model
        self.messages = messages
        return MockCompletionResponse()


class MockChat:
    def __init__(self):
        self.completions = MockCompletions()


class MockClient:
    def __init__(self):
        self.chat = MockChat()


class ReportGenerationTests(unittest.TestCase):
    def test_report_prompt_contains_safety_constraints(self):
        report_input = StructuredReportInput(
            prediction="COPD",
            confidence=0.8,
            probabilities={"COPD": 0.8, "Healthy": 0.2},
            optional_features={"crackles_total": 2},
        )

        prompt = build_report_prompt(report_input)

        self.assertIn("Do not claim the model diagnosed a patient", prompt)
        self.assertIn("Do not recommend treatment", prompt)
        self.assertIn("Do not invent symptoms", prompt)
        self.assertIn("research prototype", prompt)

    def test_generate_structured_report_uses_mock_client(self):
        config = LLMConfig(api_key="test-key", model="test-model")
        output = generate_structured_report(
            StructuredReportInput(
                prediction="COPD",
                confidence=0.8,
                probabilities={"COPD": 0.8, "Healthy": 0.2},
            ),
            client=MockClient(),
            config=config,
        )

        self.assertEqual(output.disclaimer, REPORT_DISCLAIMER)
        self.assertIn("COPD", output.summary)
        self.assertIn("not a diagnosis", output.limitations)

    def test_missing_api_configuration_is_explicit(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DEEPBREATH_LLM_API_KEY"):
                generate_structured_report(
                    StructuredReportInput(
                        prediction="COPD",
                        confidence=0.8,
                        probabilities={"COPD": 0.8, "Healthy": 0.2},
                    )
                )


class ReportEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_report_endpoint_schema_with_mock_llm(self):
        payload = {
            "prediction": "COPD",
            "confidence": 0.8,
            "probabilities": {"COPD": 0.8, "Healthy": 0.2},
            "optional_features": {"wheezes_total": 1},
        }
        mocked_output = StructuredReportOutput(
            summary="The CNN probability output highlights COPD.",
            interpretation="This is a research explanation of model output only.",
            limitations="This is not a diagnosis and cannot guide treatment.",
        )

        with patch("deepbreath.app.api.generate_structured_report", return_value=mocked_output):
            response = self.client.post("/report", json=payload)

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(set(body), {"summary", "interpretation", "limitations", "disclaimer"})
        self.assertIn("CNN", body["summary"])
        self.assertIn("not a diagnosis", body["limitations"])
        self.assertEqual(body["disclaimer"], REPORT_DISCLAIMER)

    def test_report_endpoint_missing_llm_configuration_returns_503(self):
        payload = {
            "prediction": "COPD",
            "confidence": 0.8,
            "probabilities": {"COPD": 0.8, "Healthy": 0.2},
        }

        with patch.dict(os.environ, {}, clear=True):
            response = self.client.post("/report", json=payload)

        self.assertEqual(response.status_code, 503)
        self.assertIn("LLM reporting is not configured", response.json()["detail"])

    def test_api_without_llm_config_keeps_core_inference_available(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_loaded"], body.get("model_error"))


if __name__ == "__main__":
    unittest.main()
