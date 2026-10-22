FROM python:3.12-slim

# LightGBM needs OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
COPY ml_backend/requirements.txt ./ml_backend/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ML artifacts (~3 GB): streamed from the public S3 tar and checked against manifest.json.
# Placed before the app code so code-only changes reuse this layer when the build cache is available.
ARG MODEL_BUNDLE_URL=https://s3.hack.week-infra.ru/backend/ml_artifacts_v1.tar
COPY ml_backend/manifest.json ./ml_backend/manifest.json
COPY scripts ./scripts
RUN python -m scripts.fetch_artifacts "$MODEL_BUNDLE_URL" /models/artifacts
ARG TEST_FEATURES_URL=https://s3.hack.week-infra.ru/backend/test_features.sqlite
COPY ml_backend/test_features.sha256 ./ml_backend/test_features.sha256
RUN python -m scripts.fetch_test_features "$TEST_FEATURES_URL" /models/artifacts/test_features.sqlite
ENV ML_ARTIFACTS_DIR=/models/artifacts

COPY app ./app
COPY ml_backend/__init__.py ml_backend/predictor.py ./ml_backend/
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
