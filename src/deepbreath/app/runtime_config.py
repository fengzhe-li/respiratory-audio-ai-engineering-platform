from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from deepbreath.ml_contract import LEGACY_MODEL_FILENAME


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _get_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


@dataclass(frozen=True)
class ApiSettings:
    model_path: Path
    max_upload_bytes: int
    max_audio_duration_seconds: float
    host: str
    port: int
    log_level: str


def get_api_settings() -> ApiSettings:
    default_model_path = _project_root() / "models" / LEGACY_MODEL_FILENAME
    return ApiSettings(
        model_path=Path(os.getenv("DEEPBREATH_MODEL_PATH") or default_model_path),
        max_upload_bytes=_get_int("DEEPBREATH_MAX_UPLOAD_BYTES", 25 * 1024 * 1024),
        max_audio_duration_seconds=_get_float("DEEPBREATH_MAX_AUDIO_DURATION_SECONDS", 60.0),
        host=os.getenv("DEEPBREATH_API_HOST", "0.0.0.0"),
        port=_get_int("DEEPBREATH_API_PORT", 8000),
        log_level=os.getenv("DEEPBREATH_LOG_LEVEL", "INFO").upper(),
    )
