import datetime as dt
import warnings
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from deepbreath.features.mel_spectrogram import add_channel_axis, load_feature_matrix
from deepbreath.models.cnn import build_melspectrogram_cnn, compile_melspectrogram_cnn, default_callbacks


def load_split_labels(train_csv: str | Path, validation_csv: str | Path):
    train = pd.read_csv(train_csv)
    validation = pd.read_csv(validation_csv)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train["disease"])
    y_validation = encoder.transform(validation["disease"])
    return train, validation, y_train, y_validation, encoder


def train_legacy_cnn(
    train_csv: str | Path,
    validation_csv: str | Path,
    processed_audio_dir: str | Path,
    epochs: int = 200,
):
    """Train the notebook CNN using the existing sample-level split."""

    warnings.warn(
        "train_legacy_cnn uses the original sample-level split. It preserves MSc "
        "research provenance but is not suitable for unbiased generalisation evaluation.",
        RuntimeWarning,
        stacklevel=2,
    )
    train, validation, y_train, y_validation, encoder = load_split_labels(train_csv, validation_csv)
    processed_audio_dir = Path(processed_audio_dir)

    x_train = add_channel_axis(
        load_feature_matrix([processed_audio_dir / filename for filename in train["filename"]])
    )
    x_validation = add_channel_axis(
        load_feature_matrix([processed_audio_dir / filename for filename in validation["filename"]])
    )

    model = compile_melspectrogram_cnn(build_melspectrogram_cnn())
    start = dt.datetime.now()
    history = model.fit(
        {"mspec": x_train},
        y_train,
        validation_data=({"mspec": x_validation}, y_validation),
        epochs=epochs,
        verbose=1,
        callbacks=default_callbacks(),
    )
    elapsed = dt.datetime.now() - start
    return model, history, encoder, elapsed
