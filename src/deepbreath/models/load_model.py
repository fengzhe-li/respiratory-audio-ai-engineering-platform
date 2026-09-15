from pathlib import Path

from tensorflow.keras.models import load_model


def load_trained_model(model_path: str | Path):
    return load_model(model_path)
