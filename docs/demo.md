# DeepBreath Demo Guide

This demo shows DeepBreath as an AI engineering prototype: a respiratory audio
ML pipeline exposed through a FastAPI inference service and packaged for Docker.
It demonstrates software architecture around an existing MSc research model. It
is not a medical diagnostic system.

## Local API Demo

Install the project dependencies and package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Start the FastAPI service:

```bash
uvicorn deepbreath.app.api:app --host 0.0.0.0 --port 8000
```

Open Swagger UI:

```text
http://localhost:8000/docs
```

Check service health:

```bash
curl http://localhost:8000/health
```

## Docker Demo

Build and run the Docker image:

```bash
docker build -t deepbreath .
docker run -p 8000:8000 deepbreath
```

Open Swagger UI:

```text
http://localhost:8000/docs
```

The container packages the legacy saved model at
`models/melspectogramTrained.h5`. The filename is preserved from the original
MSc artifact for provenance.

Optional LLM reporting in Docker:

```bash
docker run -p 8000:8000 \
  -e DEEPBREATH_LLM_API_KEY=your-provider-key \
  -e DEEPBREATH_LLM_BASE_URL=https://api.example.com/v1 \
  -e DEEPBREATH_LLM_MODEL=provider-model-name \
  deepbreath
```

Without these LLM variables, `/health` and `/predict` still work. `/report`
returns a controlled configuration error.

## Uploading A WAV File

Using Swagger UI:

1. Open `http://localhost:8000/docs`.
2. Expand `POST /predict`.
3. Click `Try it out`.
4. Upload a `.wav` respiratory audio file.
5. Click `Execute`.
6. Review the JSON probability response.

Using curl:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/audio.wav"
```

Using the demo client:

```bash
python scripts/demo_predict.py path/to/audio.wav
```

Use `--url` if the API is running somewhere other than `localhost:8000`:

```bash
python scripts/demo_predict.py path/to/audio.wav \
  --url http://localhost:8000/predict
```

## Example API Response

Example output only. This is not a clinical result.

```json
{
  "prediction": "COPD",
  "confidence": 0.8134679794311523,
  "probabilities": {
    "Bronchiectasis": 2.4088222735918485e-11,
    "Bronchiolitis": 0.18653197586536407,
    "COPD": 0.8134679794311523,
    "Healthy": 1.0347745993399826e-9,
    "Pneumonia": 1.3798381326424192e-9,
    "URTI": 1.51135175485706e-8
  },
  "model": "deepbreath_legacy_cnn",
  "disclaimer": "Research prototype only. Not for medical diagnosis."
}
```

The full example is also saved at
`docs/examples/example_prediction.json`.

## Research Prototype Disclaimer

DeepBreath demonstrates an end-to-end AI engineering workflow: audio
preprocessing, Mel-spectrogram CNN inference, patient-independent evaluation,
FastAPI serving, and Docker packaging. The model is not clinically validated and
must not be used for diagnosis, triage, or treatment decisions.
