from pathlib import Path

import numpy as np

from deepbreath.features.mel_spectrogram import (
    add_channel_axis,
    extract_mel_spectrogram,
    extract_legacy_preprocessed_mel_spectrogram,
    validate_mel_spectrogram_shape,
)
from deepbreath.ml_contract import CLASS_NAMES


DISEASE_CLASSES = list(CLASS_NAMES)


def predict_audio_file(
    model,
    audio_path: str | Path,
    apply_legacy_preprocessing: bool = False,
) -> dict[str, object]:
    spectrogram = (
        extract_legacy_preprocessed_mel_spectrogram(audio_path)
        if apply_legacy_preprocessing
        else extract_mel_spectrogram(audio_path)
    )
    validate_mel_spectrogram_shape(spectrogram)
    features = add_channel_axis(np.array([spectrogram]))
    probabilities = model.predict({"mspec": features}).flatten()
    ranked_indices = np.argsort(probabilities)[::-1]

    return {
        "audio_path": str(audio_path),
        "top_label": DISEASE_CLASSES[int(ranked_indices[0])],
        "top_confidence": float(probabilities[int(ranked_indices[0])]),
        "runner_up_label": DISEASE_CLASSES[int(ranked_indices[1])],
        "runner_up_confidence": float(probabilities[int(ranked_indices[1])]),
        "probabilities": {
            DISEASE_CLASSES[index]: float(probabilities[index])
            for index in range(len(DISEASE_CLASSES))
        },
    }
