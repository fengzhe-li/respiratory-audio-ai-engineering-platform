from pathlib import Path

import pandas as pd


ANNOTATION_COLUMNS = ["start", "end", "crackles", "wheezes"]


def filename_parts(filename_stem: str) -> list[str]:
    return filename_stem.split("_")


def load_patient_diagnosis(path: str | Path) -> pd.DataFrame:
    """Load patient-level diagnosis labels from the ICBHI metadata file."""

    diagnosis = pd.read_csv(path, names=["pid", "disease"], header=None)
    diagnosis["pid"] = pd.to_numeric(diagnosis["pid"], errors="coerce")
    diagnosis = diagnosis.dropna(subset=["pid"]).astype({"pid": "int32"})
    return diagnosis


def load_cycle_annotations(audio_dir: str | Path) -> pd.DataFrame:
    """Load respiratory cycle annotations from all ICBHI .txt files."""

    rows: list[pd.DataFrame] = []
    audio_dir = Path(audio_dir)
    for annotation_path in sorted(audio_dir.glob("*.txt")):
        stem = annotation_path.stem
        parts = filename_parts(stem)
        annotation = pd.read_csv(annotation_path, sep="\t", names=ANNOTATION_COLUMNS)
        annotation["pid"] = int(parts[0])
        annotation["mode"] = parts[-2]
        annotation["filename"] = stem
        rows.append(annotation)

    if not rows:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS + ["pid", "mode", "filename"])

    cycles = pd.concat(rows, ignore_index=True)
    cycles["pid"] = cycles["pid"].astype("int32")
    return cycles


def build_cycle_dataframe(audio_dir: str | Path, diagnosis_csv: str | Path) -> pd.DataFrame:
    """Merge cycle-level annotations with patient-level diagnosis labels."""

    cycles = load_cycle_annotations(audio_dir)
    diagnosis = load_patient_diagnosis(diagnosis_csv)
    return pd.merge(cycles, diagnosis, on="pid", how="left")
