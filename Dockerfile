FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEEPBREATH_MODEL_PATH=/app/models/melspectogramTrained.h5 \
    DEEPBREATH_API_HOST=0.0.0.0 \
    DEEPBREATH_API_PORT=8000 \
    DEEPBREATH_LOG_LEVEL=info

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt pyproject.toml README.md ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-api.txt

COPY src ./src
COPY models/melspectogramTrained.h5 ./models/melspectogramTrained.h5

RUN python -m pip install --no-deps .

RUN useradd --create-home --shell /usr/sbin/nologin deepbreath \
    && chown -R deepbreath:deepbreath /app
USER deepbreath

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail "http://127.0.0.1:${DEEPBREATH_API_PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn deepbreath.app.api:app --host ${DEEPBREATH_API_HOST} --port ${DEEPBREATH_API_PORT} --log-level ${DEEPBREATH_LOG_LEVEL}"]
