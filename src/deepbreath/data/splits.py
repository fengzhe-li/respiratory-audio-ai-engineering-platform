from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from deepbreath.ml_contract import CLASS_NAMES


DEFAULT_GROUPED_SPLIT_SEED = 42
AUGMENTATION_SUFFIXES = (
    "_noise.wav",
    "_pitch_down.wav",
    "_pitch_up.wav",
    "_speed_down.wav",
    "_speed_up.wav",
)
SPLIT_NAMES = ("train", "validation", "test")


def legacy_stratified_split(
    data: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve the original notebook's sample-level stratified split."""

    warnings.warn(
        "legacy_stratified_split reproduces the MSc notebook's sample-level split. "
        "It can leak patient and augmentation information across train/validation.",
        RuntimeWarning,
        stacklevel=2,
    )
    train, validation = train_test_split(
        data,
        stratify=data["disease"],
        random_state=random_state,
        test_size=test_size,
    )
    return train, validation


def save_split(train: pd.DataFrame, validation: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    (output_dir / "training").mkdir(parents=True, exist_ok=True)
    (output_dir / "validation").mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "training" / "train.csv")
    validation.to_csv(output_dir / "validation" / "val.csv")


def extract_patient_id(filename: str) -> int:
    """Parse the ICBHI patient id from a processed cycle filename."""

    try:
        return int(Path(filename).name.split("_", maxsplit=1)[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse patient id from filename: {filename}") from exc


def base_cycle_id(filename: str) -> str:
    """Return the original respiratory cycle id shared by augmented variants."""

    name = Path(filename).name
    for suffix in AUGMENTATION_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    if name.endswith(".wav"):
        return name[:-4]
    return Path(name).stem


def is_augmented_filename(filename: str) -> bool:
    return Path(filename).name.endswith(AUGMENTATION_SUFFIXES)


def prepare_grouped_metadata(label_mapping_csv: str | Path) -> pd.DataFrame:
    """Load label metadata and add explicit patient/base-cycle grouping columns."""

    metadata = pd.read_csv(label_mapping_csv)
    if "disease" not in metadata.columns:
        if "label" not in metadata.columns:
            raise ValueError("Grouped split metadata must include either 'disease' or 'label'.")
        metadata = metadata.rename(columns={"label": "disease"})
    if "filename" not in metadata.columns:
        raise ValueError("Grouped split metadata must include a 'filename' column.")

    metadata = metadata.copy()
    metadata["pid"] = metadata["filename"].map(extract_patient_id).astype("int64")
    metadata["base_cycle_id"] = metadata["filename"].map(base_cycle_id)
    metadata["is_augmented"] = metadata["filename"].map(is_augmented_filename)

    unknown_labels = sorted(set(metadata["disease"]) - set(CLASS_NAMES))
    if unknown_labels:
        raise ValueError(f"Unknown disease labels in metadata: {unknown_labels}")

    patient_label_counts = metadata.groupby("pid")["disease"].nunique()
    multi_label_patients = patient_label_counts[patient_label_counts > 1]
    if not multi_label_patients.empty:
        raise ValueError(
            "Patient-level split requires one disease label per patient. "
            f"Found multi-label patients: {multi_label_patients.index.tolist()}"
        )

    base_cycle_patient_counts = metadata.groupby("base_cycle_id")["pid"].nunique()
    leaking_base_cycles = base_cycle_patient_counts[base_cycle_patient_counts > 1]
    if not leaking_base_cycles.empty:
        raise ValueError(
            "Base respiratory cycle ids must belong to one patient only. "
            f"Found ambiguous ids: {leaking_base_cycles.index.tolist()}"
        )

    return metadata


def _patient_split_counts(n_patients: int) -> tuple[int, int, int]:
    if n_patients < 1:
        return 0, 0, 0
    if n_patients == 1:
        return 1, 0, 0
    if n_patients == 2:
        return 1, 0, 1

    validation = max(1, round(n_patients * 0.15))
    test = max(1, round(n_patients * 0.15))
    train = n_patients - validation - test
    while train < 1:
        if validation >= test and validation > 0:
            validation -= 1
        elif test > 0:
            test -= 1
        train = n_patients - validation - test
    return train, validation, test


def patient_level_grouped_split(
    metadata: pd.DataFrame,
    random_state: int = DEFAULT_GROUPED_SPLIT_SEED,
) -> pd.DataFrame:
    """Create a deterministic patient-level split without crossing augmentation groups.

    The split is class-aware at patient level. It preserves patient isolation first;
    class balance is best-effort because some classes contain very few patients.
    """

    import numpy as np

    required = {"filename", "disease", "pid", "base_cycle_id"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Grouped metadata is missing required columns: {sorted(missing)}")

    patient_label_counts = metadata.groupby("pid")["disease"].nunique()
    multi_label_patients = patient_label_counts[patient_label_counts > 1]
    if not multi_label_patients.empty:
        raise ValueError(
            "Patient-level split requires one disease label per patient. "
            f"Found multi-label patients: {multi_label_patients.index.tolist()}"
        )

    rng = np.random.default_rng(random_state)
    split_by_patient: dict[int, str] = {}
    patient_labels = metadata.groupby("pid")["disease"].first()

    for disease in CLASS_NAMES:
        patients = sorted(patient_labels[patient_labels == disease].index.tolist())
        shuffled = list(rng.permutation(patients))
        train_count, validation_count, test_count = _patient_split_counts(len(shuffled))

        assignments = (
            [("train", pid) for pid in shuffled[:train_count]]
            + [("validation", pid) for pid in shuffled[train_count : train_count + validation_count]]
            + [("test", pid) for pid in shuffled[train_count + validation_count : train_count + validation_count + test_count]]
        )
        for split_name, pid in assignments:
            if int(pid) in split_by_patient:
                raise ValueError(f"Patient {pid} was assigned more than once.")
            split_by_patient[int(pid)] = split_name

    split_metadata = metadata.copy()
    split_metadata["split"] = split_metadata["pid"].map(split_by_patient)
    if split_metadata["split"].isna().any():
        missing_pids = sorted(split_metadata.loc[split_metadata["split"].isna(), "pid"].unique().tolist())
        raise ValueError(f"Some patients were not assigned to a split: {missing_pids}")

    return split_metadata


def validate_grouped_split(
    split_metadata: pd.DataFrame,
    random_seed: int = DEFAULT_GROUPED_SPLIT_SEED,
) -> dict:
    """Return leakage and distribution diagnostics for a grouped split."""

    required = {"filename", "disease", "pid", "base_cycle_id", "split"}
    missing = required - set(split_metadata.columns)
    if missing:
        raise ValueError(f"Split metadata is missing required columns: {sorted(missing)}")

    present_splits = tuple(split_metadata["split"].dropna().unique().tolist())
    unknown_splits = sorted(set(present_splits) - set(SPLIT_NAMES))
    if unknown_splits:
        raise ValueError(f"Unknown split names: {unknown_splits}")

    patient_sets = {
        split_name: set(split_metadata.loc[split_metadata["split"] == split_name, "pid"])
        for split_name in SPLIT_NAMES
    }
    base_cycle_sets = {
        split_name: set(split_metadata.loc[split_metadata["split"] == split_name, "base_cycle_id"])
        for split_name in SPLIT_NAMES
    }

    patient_overlaps = {
        f"{left}_{right}": len(patient_sets[left] & patient_sets[right])
        for left, right in combinations(SPLIT_NAMES, 2)
    }
    base_cycle_overlaps = {
        f"{left}_{right}": len(base_cycle_sets[left] & base_cycle_sets[right])
        for left, right in combinations(SPLIT_NAMES, 2)
    }

    class_distribution = (
        split_metadata.pivot_table(
            index="split",
            columns="disease",
            values="filename",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(index=SPLIT_NAMES, columns=CLASS_NAMES, fill_value=0)
        .astype(int)
    )

    patient_distribution = (
        split_metadata.drop_duplicates(["split", "pid"])
        .pivot_table(
            index="split",
            columns="disease",
            values="pid",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(index=SPLIT_NAMES, columns=CLASS_NAMES, fill_value=0)
        .astype(int)
    )

    return {
        "method": "deterministic class-aware patient-level grouped split",
        "random_seed": int(random_seed),
        "split_names": list(SPLIT_NAMES),
        "total_rows": int(len(split_metadata)),
        "total_patients": int(split_metadata["pid"].nunique()),
        "total_base_cycles": int(split_metadata["base_cycle_id"].nunique()),
        "patient_counts": {
            split_name: int(split_metadata.loc[split_metadata["split"] == split_name, "pid"].nunique())
            for split_name in SPLIT_NAMES
        },
        "sample_counts": {
            split_name: int((split_metadata["split"] == split_name).sum())
            for split_name in SPLIT_NAMES
        },
        "base_cycle_counts": {
            split_name: int(
                split_metadata.loc[split_metadata["split"] == split_name, "base_cycle_id"].nunique()
            )
            for split_name in SPLIT_NAMES
        },
        "patient_overlap_counts": patient_overlaps,
        "base_cycle_overlap_counts": base_cycle_overlaps,
        "class_distribution_samples": class_distribution.to_dict(orient="index"),
        "class_distribution_patients": patient_distribution.to_dict(orient="index"),
        "limitations": [
            "Patient isolation is prioritised over exact class balance.",
            "Classes with very few patients may be absent from validation or test splits.",
        ],
    }


def assert_no_group_leakage(split_metadata: pd.DataFrame) -> None:
    summary = validate_grouped_split(split_metadata)
    patient_overlap = sum(summary["patient_overlap_counts"].values())
    base_cycle_overlap = sum(summary["base_cycle_overlap_counts"].values())
    if patient_overlap != 0 or base_cycle_overlap != 0:
        raise ValueError(
            "Grouped split leakage detected: "
            f"patient_overlap={patient_overlap}, base_cycle_overlap={base_cycle_overlap}"
        )


def save_grouped_split_outputs(
    split_metadata: pd.DataFrame,
    output_dir: str | Path,
    random_seed: int = DEFAULT_GROUPED_SPLIT_SEED,
) -> dict:
    """Save grouped split CSVs and a JSON diagnostic summary."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = validate_grouped_split(split_metadata, random_seed=random_seed)
    assert_no_group_leakage(split_metadata)

    split_metadata.to_csv(output_dir / "all_grouped_split.csv", index=False)
    for split_name in SPLIT_NAMES:
        split_metadata.loc[split_metadata["split"] == split_name].to_csv(
            output_dir / f"{split_name}.csv",
            index=False,
        )

    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
