from pathlib import Path

import pandas as pd


def extract_patient_id(filename: str) -> int:
    return int(Path(filename).name.split("_")[0])


def build_file_index(processed_audio_dir: str | Path) -> pd.DataFrame:
    processed_audio_dir = Path(processed_audio_dir)
    rows = [
        {"pid": extract_patient_id(path.name), "filename": path.name}
        for path in sorted(processed_audio_dir.glob("*.wav"))
    ]
    return pd.DataFrame(rows)


def attach_disease_labels(files_df: pd.DataFrame, cycle_df: pd.DataFrame) -> pd.DataFrame:
    labels = cycle_df[["pid", "disease"]].drop_duplicates()
    files_df = files_df.copy()
    files_df["pid"] = files_df["pid"].astype("int64")
    return pd.merge(files_df, labels, on="pid", how="left")
