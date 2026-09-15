from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from deepbreath.app.runtime_config import get_api_settings
from deepbreath.inference.predict import predict_audio_file
from deepbreath.ml_contract import CLASS_NAMES, NUM_CLASSES
from deepbreath.models.load_model import load_trained_model
from deepbreath.reporting.report_generator import (
    REPORT_DISCLAIMER,
    StructuredReportInput,
    generate_structured_report,
)


LOGGER = logging.getLogger("deepbreath.api")
MODEL_ID = "deepbreath_legacy_cnn"
DISCLAIMER = "Research prototype only. Not for medical diagnosis."
SETTINGS = get_api_settings()
ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/vnd.wav",
    "application/octet-stream",
}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_classes: int
    model: str
    model_error: str | None = None


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model: str
    disclaimer: str


class ReportRequest(BaseModel):
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    optional_features: dict[str, Any] | None = None


class ReportResponse(BaseModel):
    summary: str
    interpretation: str
    limitations: str
    disclaimer: str


def default_model_path() -> Path:
    return SETTINGS.model_path


def load_api_model(model_path: Path | None = None) -> Any:
    path = model_path or default_model_path()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    LOGGER.info("Loading DeepBreath model", extra={"model_path": str(path)})
    return load_trained_model(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=SETTINGS.log_level)
    app.state.model = None
    app.state.model_error = None
    try:
        app.state.model = load_api_model()
        LOGGER.info("DeepBreath model loaded", extra={"model": MODEL_ID})
    except Exception as exc:  # pragma: no cover - exercised through health/predict state
        app.state.model_error = str(exc)
        LOGGER.exception("DeepBreath model loading failed")
    yield


app = FastAPI(
    title="DeepBreath API",
    version="0.1.0",
    description="Research prototype API for DeepBreath respiratory sound inference.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model_loaded = app.state.model is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_classes=NUM_CLASSES,
        model=MODEL_ID,
        model_error=app.state.model_error,
    )


def _validate_upload_metadata(file: UploadFile) -> None:
    filename = file.filename or ""
    content_type = file.content_type or ""
    if not filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .wav respiratory audio files are supported.",
        )
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported audio content type. Upload a WAV file.",
        )


def _validate_audio_file(path: Path) -> None:
    try:
        info = sf.info(path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a readable WAV audio file.",
        ) from exc

    if info.frames <= 0 or info.samplerate <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded WAV file does not contain valid audio samples.",
        )
    duration = float(info.frames) / float(info.samplerate)
    if duration <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded WAV file has invalid duration.",
        )
    if duration > SETTINGS.max_audio_duration_seconds:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                "Uploaded WAV duration must be "
                f"{SETTINGS.max_audio_duration_seconds:g} seconds or shorter."
            ),
        )


async def _persist_upload(file: UploadFile) -> Path:
    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(data) > SETTINGS.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file must be {SETTINGS.max_upload_bytes // (1024 * 1024)} MB or smaller.",
        )

    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        temporary.write(data)
        temporary.flush()
        return Path(temporary.name)
    finally:
        temporary.close()


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DeepBreath model is not loaded.",
        )

    _validate_upload_metadata(file)
    temp_path: Path | None = None
    try:
        temp_path = await _persist_upload(file)
        _validate_audio_file(temp_path)
        LOGGER.info("Running DeepBreath inference", extra={"upload_filename": file.filename})
        result = predict_audio_file(
            app.state.model,
            temp_path,
            apply_legacy_preprocessing=True,
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("DeepBreath inference failed", extra={"upload_filename": file.filename})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed for the uploaded WAV file.",
        ) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    probabilities = result["probabilities"]
    return PredictionResponse(
        prediction=str(result["top_label"]),
        confidence=float(result["top_confidence"]),
        probabilities={class_name: float(probabilities[class_name]) for class_name in CLASS_NAMES},
        model=MODEL_ID,
        disclaimer=DISCLAIMER,
    )


@app.post("/report", response_model=ReportResponse)
def report(request: ReportRequest) -> ReportResponse:
    try:
        generated = generate_structured_report(
            StructuredReportInput(
                prediction=request.prediction,
                confidence=request.confidence,
                probabilities=request.probabilities,
                optional_features=request.optional_features,
            )
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM reporting is not configured: {exc}",
        ) from exc
    except Exception as exc:
        LOGGER.exception("DeepBreath report generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report generation failed.",
        ) from exc

    return ReportResponse(
        summary=generated.summary,
        interpretation=generated.interpretation,
        limitations=generated.limitations,
        disclaimer=REPORT_DISCLAIMER,
    )
