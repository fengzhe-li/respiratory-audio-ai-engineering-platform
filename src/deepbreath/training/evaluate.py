import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from deepbreath.ml_contract import CLASS_NAMES


def predict_class_ids(model, features):
    probabilities = model.predict({"mspec": features})
    return np.argmax(probabilities, axis=1), probabilities


def _all_int_labels(values) -> bool:
    return all(isinstance(value, (int, np.integer)) for value in values)


def _all_string_labels(values) -> bool:
    return all(isinstance(value, str) for value in values)


def classification_metrics_encoded(
    y_true,
    y_pred,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> str:
    """Classification report for integer-encoded labels."""

    if not (_all_int_labels(y_true) and _all_int_labels(y_pred)):
        raise TypeError("classification_metrics_encoded expects integer-encoded labels.")
    labels = list(range(len(class_names)))
    return classification_report(y_true, y_pred, labels=labels, target_names=list(class_names))


def classification_metrics_labels(
    y_true,
    y_pred,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> str:
    """Classification report for string labels."""

    if not (_all_string_labels(y_true) and _all_string_labels(y_pred)):
        raise TypeError("classification_metrics_labels expects string class labels.")
    return classification_report(y_true, y_pred, labels=list(class_names))


def confusion_matrix_encoded(
    y_true,
    y_pred,
    class_names: tuple[str, ...] = CLASS_NAMES,
):
    if not (_all_int_labels(y_true) and _all_int_labels(y_pred)):
        raise TypeError("confusion_matrix_encoded expects integer-encoded labels.")
    return confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))


def confusion_matrix_labels(
    y_true,
    y_pred,
    class_names: tuple[str, ...] = CLASS_NAMES,
):
    if not (_all_string_labels(y_true) and _all_string_labels(y_pred)):
        raise TypeError("confusion_matrix_labels expects string class labels.")
    return confusion_matrix(y_true, y_pred, labels=list(class_names))
