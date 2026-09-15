# DeepBreath Demo Screenshot Guide

Use real screenshots from a locally running DeepBreath service. Do not create
mock or fake screenshots.

## Screenshot 1: Swagger UI Showing `/predict`

Recommended view:

- Browser open at `http://localhost:8000/docs`.
- `POST /predict` expanded.
- File upload control visible.
- Response schema visible if possible.

Purpose: shows that the model is exposed through a usable FastAPI interface.

## Screenshot 2: WAV Upload And JSON Response

Recommended view:

- Swagger UI after executing `POST /predict`, or terminal with `curl`.
- Uploaded `.wav` filename visible if using Swagger.
- JSON response visible with `prediction`, `confidence`, `probabilities`,
  `model`, and `disclaimer`.

Purpose: shows the end-to-end inference path from uploaded audio to structured
six-class response.

## Screenshot 3: Docker Container Running Healthy

Recommended view:

- Terminal showing:

```bash
docker build -t deepbreath .
docker run -p 8000:8000 deepbreath
docker ps
```

- Container health status should be `healthy` after the healthcheck runs.

Purpose: shows reproducible containerised serving, not only local Python
execution.

## Screenshot 4: Architecture Diagram

Recommended view:

- GitHub-rendered `docs/architecture_diagram.md`, or local Markdown preview.
- Include either the research pipeline diagram or deployment inference service
  diagram.

Purpose: helps a reviewer understand the separation between research code,
inference service, and future LLM reporting.

## Safety Note

Screenshots should present DeepBreath as an AI engineering prototype. Avoid any
phrasing that implies clinical validation, diagnosis, triage, or treatment
recommendation.
