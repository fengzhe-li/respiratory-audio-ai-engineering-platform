from pathlib import Path

import librosa
import numpy as np

from deepbreath.data.segmentation import pad_or_clip
from deepbreath.ml_contract import EXPECTED_SAMPLE_RATE, MEL_SPECTROGRAM_SHAPE


def extract_mel_spectrogram_from_audio(sound_array: np.ndarray, sample_rate: int):
    """Extract the Mel spectrogram representation used by the original CNN."""

    mel_features = librosa.feature.melspectrogram(y=sound_array, sr=sample_rate)
    return librosa.power_to_db(mel_features, ref=np.max)


def extract_mel_spectrogram(path: str | Path):
    """Extract Mel features from a legacy six-second processed WAV file."""

    sound_array, sample_rate = librosa.load(path, sr=EXPECTED_SAMPLE_RATE)
    return extract_mel_spectrogram_from_audio(sound_array, sample_rate)


def extract_legacy_preprocessed_mel_spectrogram(path: str | Path):
    """Extract Mel features after applying legacy six-second pad/clip preprocessing."""

    sound_array, sample_rate = librosa.load(path, sr=EXPECTED_SAMPLE_RATE)
    fixed_audio = pad_or_clip(sound_array, sample_rate)
    return extract_mel_spectrogram_from_audio(fixed_audio, sample_rate)


def validate_mel_spectrogram_shape(mel_spectrogram: np.ndarray) -> None:
    if mel_spectrogram.shape != MEL_SPECTROGRAM_SHAPE:
        raise ValueError(
            "Unexpected Mel-spectrogram shape "
            f"{mel_spectrogram.shape}; expected {MEL_SPECTROGRAM_SHAPE}. "
            "Use six-second audio processed by the legacy preprocessing pipeline."
        )


def load_feature_matrix(paths: list[str | Path]) -> np.ndarray:
    return np.array([extract_mel_spectrogram(path) for path in paths])


def add_channel_axis(features: np.ndarray) -> np.ndarray:
    if features.ndim == 3:
        return features[..., np.newaxis]
    return features
