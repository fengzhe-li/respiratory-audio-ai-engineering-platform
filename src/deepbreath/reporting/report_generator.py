from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from deepbreath.reporting.llm_client import (
    LLMConfig,
    build_openai_compatible_client,
    generate_report,
)


REPORT_DISCLAIMER = "Research prototype only. Not for medical diagnosis."
DEFAULT_LIMITATIONS = (
    "This explanation is generated from model probabilities and optional acoustic "
    "features only. It is not a diagnosis, does not include clinical examination, "
    "imaging, laboratory results, or patient history, and must not guide treatment."
)


@dataclass(frozen=True)
class StructuredReportInput:
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    optional_features: dict[str, Any] | None = None


@dataclass(frozen=True)
class StructuredReportOutput:
    summary: str
    interpretation: str
    limitations: str
    disclaimer: str = REPORT_DISCLAIMER


def build_report_prompt(report_input: StructuredReportInput) -> str:
    features = report_input.optional_features or {}
    probabilities_json = json.dumps(report_input.probabilities, indent=2, sort_keys=True)
    features_json = json.dumps(features, indent=2, sort_keys=True)

    return f"""
You are generating an AI explanation for a respiratory sound research prototype.

STRICT SAFETY RULES:
- Do not claim the model diagnosed a patient.
- Do not recommend treatment, medication, or clinical management.
- Do not invent symptoms, vital signs, imaging findings, lab results, demographics, or history.
- Ground the explanation only in the provided prediction, probabilities, and optional features.
- Clearly state that this is a research prototype and not for medical diagnosis.

MODEL OUTPUT:
- Predicted class: {report_input.prediction}
- Confidence: {report_input.confidence:.6f}
- Class probabilities:
{probabilities_json}

OPTIONAL PATIENT-LEVEL ACOUSTIC FEATURES:
{features_json}

Return compact JSON with exactly these keys:
- summary
- interpretation
- limitations

The limitations field must mention that this is not a diagnosis and cannot guide treatment.
""".strip()


def _coerce_report_output(raw_text: str) -> StructuredReportOutput:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return StructuredReportOutput(
            summary=raw_text.strip(),
            interpretation=(
                "The generated explanation should be interpreted only as a research "
                "summary of the provided model output."
            ),
            limitations=DEFAULT_LIMITATIONS,
        )

    return StructuredReportOutput(
        summary=str(parsed.get("summary", "")).strip(),
        interpretation=str(parsed.get("interpretation", "")).strip(),
        limitations=str(parsed.get("limitations", DEFAULT_LIMITATIONS)).strip() or DEFAULT_LIMITATIONS,
    )


def generate_structured_report(
    report_input: StructuredReportInput,
    client: Any | None = None,
    config: LLMConfig | None = None,
) -> StructuredReportOutput:
    """Generate a structured LLM report from existing CNN prediction output."""

    config = config or LLMConfig.from_env()
    client = client or build_openai_compatible_client(config)
    prompt = build_report_prompt(report_input)
    raw_text = generate_report(client, prompt, model=config.model)
    report = _coerce_report_output(raw_text)

    return StructuredReportOutput(
        summary=report.summary,
        interpretation=report.interpretation,
        limitations=report.limitations if report.limitations else DEFAULT_LIMITATIONS,
        disclaimer=REPORT_DISCLAIMER,
    )
