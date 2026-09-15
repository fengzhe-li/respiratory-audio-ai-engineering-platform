from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_PREDICT_URL = "http://localhost:8000/predict"


def _multipart_body(file_path: Path, boundary: str) -> bytes:
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return header + file_path.read_bytes() + footer


def call_predict_api(file_path: Path, url: str) -> dict:
    boundary = "deepbreath-demo-boundary"
    body = _multipart_body(file_path, boundary)
    request = Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def print_prediction(result: dict) -> None:
    print(f"Predicted class: {result['prediction']}")
    print(f"Confidence: {float(result['confidence']):.4f}")
    print("Class probabilities:")
    for class_name, probability in result["probabilities"].items():
        print(f"  {class_name}: {float(probability):.6f}")
    print(f"Model: {result.get('model', 'unknown')}")
    print(f"Disclaimer: {result.get('disclaimer', 'Research prototype only.')}")


def main() -> None:
    parser = ArgumentParser(description="Call the DeepBreath FastAPI /predict endpoint.")
    parser.add_argument("audio", help="Path to a WAV file to upload to the API.")
    parser.add_argument("--url", default=DEFAULT_PREDICT_URL, help="DeepBreath /predict URL.")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"WAV file not found: {audio_path}")
    if audio_path.suffix.lower() != ".wav":
        raise SystemExit("DeepBreath demo client expects a .wav file.")

    try:
        result = call_predict_api(audio_path, args.url)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"API request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise SystemExit(f"Could not connect to DeepBreath API at {args.url}: {exc}") from exc

    print_prediction(result)


if __name__ == "__main__":
    main()
