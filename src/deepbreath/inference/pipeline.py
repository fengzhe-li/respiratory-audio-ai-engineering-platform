from pathlib import Path

from deepbreath.inference.predict import predict_audio_file
from deepbreath.models.load_model import load_trained_model


def run_prediction(model_path: str | Path, audio_path: str | Path) -> dict[str, object]:
    model = load_trained_model(model_path)
    return predict_audio_file(model, audio_path)
