import pandas as pd


def aggregate_patient_features(cycle_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate crackle/wheeze annotations into patient-level indicators."""

    return (
        cycle_df.groupby(["pid", "disease"], dropna=False)
        .agg(
            total_cycles=("filename", "count"),
            crackles_total=("crackles", "sum"),
            wheezes_total=("wheezes", "sum"),
        )
        .reset_index()
    )


def clinical_benchmarks(cycle_df: pd.DataFrame) -> pd.DataFrame:
    """Compute disease-level benchmark counts used to ground LLM reports."""

    patient_features = aggregate_patient_features(cycle_df)
    return (
        patient_features.groupby("disease", dropna=False)
        .agg(
            patients=("pid", "nunique"),
            mean_crackles=("crackles_total", "mean"),
            mean_wheezes=("wheezes_total", "mean"),
            mean_cycles=("total_cycles", "mean"),
        )
        .reset_index()
    )
