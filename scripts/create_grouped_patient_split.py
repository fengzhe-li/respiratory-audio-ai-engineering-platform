from argparse import ArgumentParser
from pathlib import Path

from deepbreath.data.splits import (
    DEFAULT_GROUPED_SPLIT_SEED,
    patient_level_grouped_split,
    prepare_grouped_metadata,
    save_grouped_split_outputs,
)


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Create a leakage-safe patient-level grouped split for DeepBreath. "
            "This does not retrain the model or alter legacy MSc split artifacts."
        )
    )
    parser.add_argument(
        "--label-mapping-csv",
        default="data/metadata_sample/label_mapping_new.csv",
        help="CSV containing filename plus label/disease columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/metrics/grouped_patient_split",
        help="Directory for grouped split CSVs and summary JSON.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_GROUPED_SPLIT_SEED)
    args = parser.parse_args()

    metadata = prepare_grouped_metadata(args.label_mapping_csv)
    split_metadata = patient_level_grouped_split(metadata, random_state=args.seed)
    summary = save_grouped_split_outputs(split_metadata, Path(args.output_dir), random_seed=args.seed)

    print("Grouped patient split created.")
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Patient counts: {summary['patient_counts']}")
    print(f"Patient overlap counts: {summary['patient_overlap_counts']}")
    print(f"Base-cycle overlap counts: {summary['base_cycle_overlap_counts']}")


if __name__ == "__main__":
    main()
