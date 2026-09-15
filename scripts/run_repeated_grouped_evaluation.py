import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from deepbreath.data.splits import (
    assert_no_group_leakage,
    patient_level_grouped_split,
    prepare_grouped_metadata,
    validate_grouped_split,
)
from deepbreath.ml_contract import CLASS_NAMES
from deepbreath.training.grouped_evaluation import (
    load_features_for_metadata,
    split_distribution,
    train_and_evaluate_grouped_split,
)


DEFAULT_REPEATED_SEEDS = (7, 21, 42, 84, 168)
METRIC_COLUMNS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")


def _parse_seeds(value: str) -> list[int]:
    return [int(seed.strip()) for seed in value.split(",") if seed.strip()]


def _patient_distribution(metadata: pd.DataFrame) -> pd.DataFrame:
    return (
        metadata.groupby("disease")
        .agg(
            patients=("pid", "nunique"),
            base_cycles=("base_cycle_id", "nunique"),
            samples=("filename", "count"),
            augmented_samples=("is_augmented", "sum"),
        )
        .reindex(CLASS_NAMES)
        .astype(int)
        .reset_index()
    )


def _cv_feasibility(patient_distribution: pd.DataFrame) -> dict:
    min_patients = int(patient_distribution["patients"].min())
    rare_classes = patient_distribution.loc[patient_distribution["patients"] < 5, "disease"].tolist()
    return {
        "grouped_cv_used": False,
        "minimum_patients_per_class": min_patients,
        "rare_classes": rare_classes,
        "reason": (
            "Grouped cross-validation is not used because several diseases have too few "
            "patient groups for statistically meaningful StratifiedGroupKFold. Pneumonia "
            "has only two patients, so repeated grouped holdout is a clearer reliability "
            "probe without pretending fold-level class balance is robust."
        ),
    }


def _metric_summary(per_run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRIC_COLUMNS:
        values = per_run_df[metric]
        rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                "min": values.min(),
                "max": values.max(),
                "ci95_half_width": (
                    1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _per_class_rows(seed: int, report: dict) -> list[dict]:
    rows = []
    for class_name in CLASS_NAMES:
        metrics = report[class_name]
        rows.append(
            {
                "seed": seed,
                "class": class_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1-score"],
                "support": metrics["support"],
            }
        )
    return rows


def _confusion_pairs(confusion: np.ndarray) -> pd.DataFrame:
    rows = []
    for true_index, true_label in enumerate(CLASS_NAMES):
        for pred_index, predicted_label in enumerate(CLASS_NAMES):
            if true_index == pred_index:
                continue
            count = int(confusion[true_index, pred_index])
            if count:
                rows.append(
                    {
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": count,
                    }
                )
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def _markdown_table(df: pd.DataFrame, float_format: str = ".4f") -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in {"seed", "train_patients", "validation_patients", "test_patients", "train_samples", "validation_samples", "test_samples"}:
                values.append(str(int(value)))
            elif isinstance(value, float):
                values.append(format(value, float_format))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    output_dir: Path,
    per_run_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    confusion_pairs_df: pd.DataFrame,
    patient_distribution_df: pd.DataFrame,
    cv_feasibility: dict,
) -> None:
    strongest = (
        per_class_df.groupby("class")["f1"].mean().sort_values(ascending=False).head(2).index.tolist()
    )
    weakest = (
        per_class_df.groupby("class")["f1"].mean().sort_values(ascending=True).head(2).index.tolist()
    )
    lines = [
        "# Grouped Repeated Evaluation Reliability Report",
        "",
        "This report preserves the legacy MSc result and the Phase 3B grouped result. "
        "It investigates patient-independent generalisation stability across fixed grouped holdout seeds.",
        "",
        "## Grouped CV Feasibility",
        "",
        cv_feasibility["reason"],
        "",
        "## Metric Stability",
        "",
        _markdown_table(summary_df),
        "",
        "## Per-Run Metrics",
        "",
        _markdown_table(
            per_run_df[
            [
                "seed",
                "train_patients",
                "validation_patients",
                "test_patients",
                "train_samples",
                "validation_samples",
                "test_samples",
                *METRIC_COLUMNS,
            ]
            ]
        ),
        "",
        "## Patient Distribution Before Balancing",
        "",
        _markdown_table(patient_distribution_df),
        "",
        "## Error Analysis",
        "",
        f"Strongest mean-F1 classes: {', '.join(strongest)}.",
        f"Weakest mean-F1 classes: {', '.join(weakest)}.",
        "",
        "Most common off-diagonal confusion pairs:",
        "",
        _markdown_table(confusion_pairs_df.head(10)) if not confusion_pairs_df.empty else "None.",
        "",
        "## Interpretation",
        "",
        "Equal sample counts do not imply equal patient diversity. The current balancing strategy "
        "creates 624 samples per disease, but some classes are represented by only two or three "
        "patients. Augmentation expands respiratory cycles from the same patients, so it can help "
        "the CNN learn audio transformations while still leaving patient-specific acoustic patterns "
        "as a likely source of overfitting.",
        "",
        "These results should not be presented as clinical reliability claims.",
    ]
    (output_dir / "reliability_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Run repeated patient-level grouped holdout evaluation for DeepBreath "
            "without changing the CNN architecture or legacy artifacts."
        )
    )
    parser.add_argument(
        "--label-mapping-csv",
        default="data/metadata_sample/label_mapping_new.csv",
    )
    parser.add_argument("--processed-audio-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="reports/metrics/grouped_repeated_evaluation",
    )
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_REPEATED_SEEDS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = prepare_grouped_metadata(args.label_mapping_csv).reset_index(drop=True)
    features = load_features_for_metadata(metadata, args.processed_audio_dir)
    patient_distribution_df = _patient_distribution(metadata)
    cv_feasibility = _cv_feasibility(patient_distribution_df)

    per_run_rows = []
    per_class_rows = []
    confusion_total = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    split_summaries = {}

    for seed in seeds:
        print(f"Running grouped holdout seed {seed} ({len(per_run_rows) + 1}/{len(seeds)})")
        split_metadata = patient_level_grouped_split(metadata, random_state=seed)
        assert_no_group_leakage(split_metadata)
        split_summary = validate_grouped_split(split_metadata, random_seed=seed)
        split_summaries[str(seed)] = split_summary

        result = train_and_evaluate_grouped_split(
            split_metadata=split_metadata,
            features=features,
            seed=seed,
            epochs=args.epochs,
            verbose=args.verbose,
        )
        confusion = result["confusion_matrix"]
        confusion_total += confusion

        train_df = split_metadata.loc[split_metadata["split"] == "train"]
        validation_df = split_metadata.loc[split_metadata["split"] == "validation"]
        test_df = split_metadata.loc[split_metadata["split"] == "test"]
        row = {
            "seed": seed,
            "train_patients": split_distribution(train_df)["patients"],
            "validation_patients": split_distribution(validation_df)["patients"],
            "test_patients": split_distribution(test_df)["patients"],
            "train_samples": split_distribution(train_df)["samples"],
            "validation_samples": split_distribution(validation_df)["samples"],
            "test_samples": split_distribution(test_df)["samples"],
            "accuracy": result["accuracy"],
            "balanced_accuracy": result["balanced_accuracy"],
            "macro_f1": result["macro_f1"],
            "weighted_f1": result["weighted_f1"],
        }
        per_run_rows.append(row)
        per_class_rows.extend(_per_class_rows(seed, result["classification_report"]))

    per_run_df = pd.DataFrame(per_run_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    summary_df = _metric_summary(per_run_df)
    confusion_pairs_df = _confusion_pairs(confusion_total)

    per_run_df.to_csv(output_dir / "per_run_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(confusion_total, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix_total.csv"
    )
    confusion_pairs_df.to_csv(output_dir / "confusion_pairs.csv", index=False)
    patient_distribution_df.to_csv(output_dir / "patient_distribution.csv", index=False)

    summary_payload = {
        "seeds": seeds,
        "epochs": args.epochs,
        "metrics": summary_df.to_dict(orient="records"),
        "cv_feasibility": cv_feasibility,
        "split_summaries": split_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_report(
        output_dir,
        per_run_df,
        summary_df,
        per_class_df,
        confusion_pairs_df,
        patient_distribution_df,
        cv_feasibility,
    )
    print(f"Repeated grouped evaluation complete: {output_dir}")


if __name__ == "__main__":
    main()
