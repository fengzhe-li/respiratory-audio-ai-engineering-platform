from argparse import ArgumentParser

from tensorflow.keras.models import save_model

from deepbreath.training.train import train_legacy_cnn


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Train the legacy DeepBreath CNN from CSV splits. This preserves the MSc "
            "sample-level evaluation path and is not leakage-safe."
        )
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--processed-audio-dir", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    model, _, _, elapsed = train_legacy_cnn(
        args.train_csv,
        args.validation_csv,
        args.processed_audio_dir,
        epochs=args.epochs,
    )
    save_model(model, args.output_model)
    print(f"Training completed in {elapsed}. Saved model to {args.output_model}")


if __name__ == "__main__":
    main()
