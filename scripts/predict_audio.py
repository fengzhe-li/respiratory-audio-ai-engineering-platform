from argparse import ArgumentParser

from deepbreath.inference.pipeline import run_prediction


def main() -> None:
    parser = ArgumentParser(description="Run DeepBreath CNN inference for one WAV file.")
    parser.add_argument("--model", required=True, help="Path to the trained Keras model.")
    parser.add_argument("--audio", required=True, help="Path to a WAV file.")
    args = parser.parse_args()

    result = run_prediction(args.model, args.audio)
    print(result)


if __name__ == "__main__":
    main()
