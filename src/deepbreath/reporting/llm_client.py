import os
from dataclasses import dataclass


class LLMConfigurationError(ValueError):
    """Raised when server-side LLM configuration is unavailable or invalid."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = os.getenv("DEEPBREATH_LLM_API_KEY")
        model = os.getenv("DEEPBREATH_LLM_MODEL")
        base_url = os.getenv("DEEPBREATH_LLM_BASE_URL") or None

        missing = [
            name
            for name, value in {
                "DEEPBREATH_LLM_API_KEY": api_key,
                "DEEPBREATH_LLM_MODEL": model,
            }.items()
            if not value
        ]
        if missing:
            raise LLMConfigurationError(
                "Missing LLM configuration: "
                + ", ".join(missing)
                + ". Copy .env.example and set provider-specific values."
            )

        return cls(api_key=api_key, model=model, base_url=base_url)


def build_openai_compatible_client(config: LLMConfig | None = None):
    """Build an OpenAI-compatible client from environment variables."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is required for LLM report generation. "
            "Install project dependencies before using this feature."
        ) from exc

    config = config or LLMConfig.from_env()
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def generate_report(client, prompt: str, model: str | None = None) -> str:
    model = model or LLMConfig.from_env().model
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
