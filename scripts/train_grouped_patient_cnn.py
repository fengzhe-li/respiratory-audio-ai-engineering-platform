import json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from deepbreath.data.splits import SPLIT_NAMES, assert_no_group_leakage
from deepbreath.features.mel_spectrogram import add_channel_axis, load_feature_matrix
from deepbreath.ml_contract import CLASS_NAMES
from deepbreath.models.cnn import (
    build_melspectrogram_cnn,
    compile_melspectrogram_cnn,
    default_callbacks,
)


CLASS_TO_INDEX = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}


def _load_split_csvs(split_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = {}
    for split_name in SPLIT_NAMES:
        csv_path = split_dir / f"{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing grouped split CSV: {csv_path}")
        frames[split_name] = pd.read_csv(csv_path)

    split_metadata = pd.concat(frames.values(), ignore_index=True)
    assert_no_group_leakage(split_metadata)
    return frames["train"], frames["validation"], frames["test"]


def _encode_labels(labels: pd.Series):
    unknown = sorted(set(labels) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unknown class labels: {unknown}")
    return labels.map(CLASS_TO_INDEX).to_numpy()


def _load_features(split_df: pd.DataFrame, processed_audio_dir: Path):
    paths = [processed_audio_dir / filename for filename in split_df["filename"]]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        preview = missing[:5]
        raise FileNotFoundError(f"Missing processed audio files: {preview}")
    return add_channel_axis(load_feature_matrix(paths))


def _distribution(split_df: pd.DataFrame) -> dict:
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


def _evaluate_predictions(y_true, probabilities) -> dict:
    y_pred = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), average="macro")),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=list(range(len(CLASS_NAMES))), average="weighted")
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
        ).tolist(),
    }


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Train the original DeepBreath CNN architecture on the verified "
            "patient-level grouped split and save metrics separately from legacy results."
        )
    )
    parser.add_argument(
        "--split-dir",
        default="reports/metrics/grouped_patient_split",
        help="Directory containing train/validation/test grouped CSVs.",
    )
    parser.add_argument("--processed-audio-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="reports/metrics/grouped_patient_split",
        help="Directory for grouped evaluation outputs.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--model-output", default="grouped_patient_cnn.keras")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_audio_dir = Path(args.processed_audio_dir)

    train_df, validation_df, test_df = _load_split_csvs(split_dir)
    y_train = _encode_labels(train_df["disease"])
    y_validation = _encode_labels(validation_df["disease"])
    y_test = _encode_labels(test_df["disease"])

    x_train = _load_features(train_df, processed_audio_dir)
    x_validation = _load_features(validation_df, processed_audio_dir)
    x_test = _load_features(test_df, processed_audio_dir)

    model = compile_melspectrogram_cnn(build_melspectrogram_cnn())
    history = model.fit(
        {"mspec": x_train},
        y_train,
        validation_data=({"mspec": x_validation}, y_validation),
        epochs=args.epochs,
        verbose=1,
        callbacks=default_callbacks(),
    )

    probabilities = model.predict({"mspec": x_test})
    metrics = _evaluate_predictions(y_test, probabilities)
    metrics["split_method"] = "patient-level grouped holdout"
    metrics["class_order"] = list(CLASS_NAMES)
    metrics["epochs"] = int(args.epochs)
    metrics["input_shapes"] = {
        "train": list(x_train.shape),
        "validation": list(x_validation.shape),
        "test": list(x_test.shape),
    }
    metrics["split_distribution"] = {
        "train": _distribution(train_df),
        "validation": _distribution(validation_df),
        "test": _distribution(test_df),
    }

    (output_dir / "grouped_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(metrics["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "grouped_confusion_matrix.csv"
    )
    pd.DataFrame(history.history).to_csv(output_dir / "grouped_training_history.csv", index=False)
    probabilities_df = pd.DataFrame(probabilities, columns=CLASS_NAMES)
    probabilities_df.insert(0, "filename", test_df["filename"].to_list())
    probabilities_df.insert(1, "true_label", test_df["disease"].to_list())
    probabilities_df["predicted_label"] = [CLASS_NAMES[index] for index in probabilities.argmax(axis=1)]
    probabilities_df.to_csv(output_dir / "grouped_test_predictions.csv", index=False)

    model.save(output_dir / args.model_output)
    print(f"Grouped patient evaluation complete. Metrics: {output_dir / 'grouped_metrics.json'}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
