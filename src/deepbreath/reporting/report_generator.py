from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from deepbreath.reporting.llm_client import (
    LLMConfig,
    build_openai_compatible_client,
    generate_report,
)
from deepbreath.ml_contract import CLASS_NAMES


REPORT_DISCLAIMER = "Research prototype only. Not for medical diagnosis."
DEFAULT_LIMITATIONS = (
    "This explanation is generated from model probabilities and optional acoustic "
    "features only. It is not a diagnosis, does not include clinical examination, "
    "imaging, laboratory results, or patient history, and must not guide treatment."
)
MAX_FIELD_LENGTH = 1200
ALLOWED_FEATURES = {"crackles_total", "wheezes_total", "total_cycles"}
UNSAFE_CLAIM = re.compile(r"\b(diagnos(?:e|ed|is)|prescri(?:be|bed|ption)|treat(?:ment|ing)|take\s+medication|rule[ -]?out)\b", re.I)


@dataclass(frozen=True)
class StructuredReportInput:
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    optional_features: dict[str, Any] | None = None

    def validate(self) -> None:
        if self.prediction not in CLASS_NAMES:
            raise ValueError("prediction must be one of the model classes")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and between 0 and 1")
        if set(self.probabilities) != set(CLASS_NAMES):
            raise ValueError("probabilities must contain exactly the six model classes")
        values = list(self.probabilities.values())
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("probabilities must be finite numbers between 0 and 1")
        if abs(sum(values) - 1.0) > 1e-4 or abs(self.probabilities[self.prediction] - self.confidence) > 1e-4:
            raise ValueError("prediction, confidence and probabilities are inconsistent")
        for key, value in (self.optional_features or {}).items():
            if key not in ALLOWED_FEATURES or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value > 1_000_000:
                raise ValueError("optional_features contains an unsupported or invalid value")


@dataclass(frozen=True)
class StructuredReportOutput:
    summary: str
    interpretation: str
    limitations: str
    disclaimer: str = REPORT_DISCLAIMER


def build_report_prompt(report_input: StructuredReportInput) -> str:
    report_input.validate()
    features = report_input.optional_features or {}
    probabilities_json = json.dumps(report_input.probabilities, indent=2, sort_keys=True)
    features_json = json.dumps(features, indent=2, sort_keys=True)

    return f"""
You are generating an AI explanation for a respiratory sound research prototype.

The supplied prediction is unverified client-provided data unless it was produced
by the server endpoint. Never describe it as an authenticated or clinically
validated model result.

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
    if not isinstance(raw_text, str) or not raw_text.strip() or len(raw_text) > 5000:
        raise ValueError("LLM response is empty or oversized")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError("LLM response was not valid JSON")

    if not isinstance(parsed, dict) or set(parsed) != {"summary", "interpretation", "limitations"}:
        raise ValueError("LLM response did not match the required schema")
    if any(not isinstance(parsed[key], str) or not parsed[key].strip() or len(parsed[key]) > MAX_FIELD_LENGTH for key in parsed):
        raise ValueError("LLM response contains invalid fields")
    if any(UNSAFE_CLAIM.search(parsed[key]) for key in ("summary", "interpretation")):
        raise ValueError("LLM response contains unsupported medical claims")
    if "not a diagnosis" not in parsed["limitations"].lower() or "cannot guide treatment" not in parsed["limitations"].lower():
        raise ValueError("LLM response omitted the required research limitations")

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

    report_input.validate()
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
