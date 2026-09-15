import os
import unittest
from unittest.mock import patch

from deepbreath.reporting.llm_client import LLMConfig


class LLMConfigTests(unittest.TestCase):
    def test_missing_required_env_vars_raise_clear_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DEEPBREATH_LLM_API_KEY"):
                LLMConfig.from_env()

    def test_env_config_uses_declared_placeholders(self):
        env = {
            "DEEPBREATH_LLM_API_KEY": "test-key",
            "DEEPBREATH_LLM_BASE_URL": "https://example.invalid/v1",
            "DEEPBREATH_LLM_MODEL": "test-model",
        }
        with patch.dict(os.environ, env, clear=True):
            config = LLMConfig.from_env()
        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.base_url, "https://example.invalid/v1")
        self.assertEqual(config.model, "test-model")


if __name__ == "__main__":
    unittest.main()
