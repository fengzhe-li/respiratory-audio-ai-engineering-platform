from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tensorflow import keras

from deepbreath.features.mel_spectrogram import add_channel_axis, load_feature_matrix
from deepbreath.ml_contract import CLASS_NAMES
from deepbreath.models.cnn import (
    build_melspectrogram_cnn,
    compile_melspectrogram_cnn,
    default_callbacks,
)


CLASS_TO_INDEX = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}


def encode_labels(labels: pd.Series) -> np.ndarray:
    unknown = sorted(set(labels) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unknown class labels: {unknown}")
    return labels.map(CLASS_TO_INDEX).to_numpy()


def load_features_for_metadata(metadata: pd.DataFrame, processed_audio_dir: str | Path) -> np.ndarray:
    processed_audio_dir = Path(processed_audio_dir)
    paths = [processed_audio_dir / filename for filename in metadata["filename"]]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed audio files: {missing[:5]}")
    return add_channel_axis(load_feature_matrix(paths))


def split_distribution(split_df: pd.DataFrame) -> dict:
    return {
        "patients": int(split_df["pid"].nunique()),
        "samples": int(len(split_df)),
        "base_cycles": int(split_df["base_cycle_id"].nunique()),
        "samples_by_class": {
            class_name: int((split_df["disease"] == class_name).sum()) for class_name in CLASS_NAMES
        },
        "patients_by_class": {
            class_name: int(split_df.loc[split_df["disease"] == class_name, "pid"].nunique())
            for class_name in CLASS_NAMES
        },
    }


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    y_pred = probabilities.argmax(axis=1)
    labels = list(range(len(CLASS_NAMES)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "y_pred": y_pred,
    }


def train_and_evaluate_grouped_split(
    split_metadata: pd.DataFrame,
    features: np.ndarray,
    seed: int,
    epochs: int = 200,
    verbose: int = 1,
) -> dict:
    keras.utils.set_random_seed(seed)

    train_mask = split_metadata["split"] == "train"
    validation_mask = split_metadata["split"] == "validation"
    test_mask = split_metadata["split"] == "test"

    x_train = features[train_mask.to_numpy()]
    x_validation = features[validation_mask.to_numpy()]
    x_test = features[test_mask.to_numpy()]
    y_train = encode_labels(split_metadata.loc[train_mask, "disease"])
    y_validation = encode_labels(split_metadata.loc[validation_mask, "disease"])
    y_test = encode_labels(split_metadata.loc[test_mask, "disease"])

    model = compile_melspectrogram_cnn(build_melspectrogram_cnn())
    history = model.fit(
        {"mspec": x_train},
        y_train,
        validation_data=({"mspec": x_validation}, y_validation),
        epochs=epochs,
        verbose=verbose,
        callbacks=default_callbacks(),
    )

    probabilities = model.predict({"mspec": x_test}, verbose=verbose)
    metrics = evaluate_predictions(y_test, probabilities)
    metrics["history"] = history.history
    metrics["input_shapes"] = {
        "train": list(x_train.shape),
        "validation": list(x_validation.shape),
        "test": list(x_test.shape),
    }
    return metrics
