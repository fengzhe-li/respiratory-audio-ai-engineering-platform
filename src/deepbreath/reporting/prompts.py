def confidence_band(confidence: float) -> str:
    percent = confidence * 100
    if percent >= 75:
        return f"HIGH ({percent:.1f}%). Use as a primary indicator pending confirmation."
    if percent >= 50:
        return f"MODERATE ({percent:.1f}%). Treat as preliminary; further examination recommended."
    return f"LOW ({percent:.1f}%). Result is uncertain; specialist review advised."


def build_llm_prompt(
    prediction: dict[str, object],
    patient_features: dict[str, object] | None = None,
    benchmarks: str | None = None,
    language: str = "English",
) -> str:
    """Create the structured CNN + patient-feature prompt described in the paper."""

    patient_features = patient_features or {}
    return f"""
You are a Senior Respiratory Consultant supporting a research prototype.

STRICT RULES:
- Do not invent vital signs, imaging results, medications, or lab results.
- State that this is AI-assisted and not a standalone diagnosis.
- Ground the report in the CNN prediction and available respiratory sound annotations.

CNN RESULT:
- Top diagnosis: {prediction.get("top_label")}
- Confidence: {confidence_band(float(prediction.get("top_confidence", 0.0)))}
- Differential diagnosis: {prediction.get("runner_up_label")}
- Differential confidence: {float(prediction.get("runner_up_confidence", 0.0)) * 100:.1f}%

PATIENT-LEVEL RESPIRATORY FEATURES:
- Patient ID: {patient_features.get("pid", "unknown")}
- Total cycles: {patient_features.get("total_cycles", "unknown")}
- Crackles total: {patient_features.get("crackles_total", "unknown")}
- Wheezes total: {patient_features.get("wheezes_total", "unknown")}

INSTITUTIONAL BENCHMARKS:
{benchmarks or "No benchmark table was provided."}

TASK:
Generate two outputs in {language}. Separate them with ===SPLIT===.

[PATIENT_SECTION]
Use patient-friendly language. Explain what the sound analysis suggests, what it does not prove,
and red-flag symptoms that require medical review.

[WARD_SECTION]
Use concise clinical language. Include the CNN finding, confidence, differential diagnosis,
available crackle/wheeze evidence, and recommended follow-up checks.
""".strip()
