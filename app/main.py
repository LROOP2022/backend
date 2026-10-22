"""HTTP adapter for the final ML module (ScoreLookup for IDs, Predictor for feature CSV/JSON)."""
import csv
import io
import logging
import math
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, field_validator

from app.model_assets import load_artifacts
from ml_backend.predictor import MODEL_VERSION, Predictor, ScoreLookup, TestFeaturesLookup

BATCH_SIZE = 1000  # Predictor.predict accepts 1..1000 records
MAX_CSV_BYTES = int(os.getenv("MAX_CSV_BYTES", str(256 * 1024 * 1024)))
logger = logging.getLogger(__name__)
requests_total = Counter("backend_http_requests_total", "HTTP requests", ["method", "route", "status"])
request_duration = Histogram("backend_http_request_seconds", "HTTP request duration", ["method", "route"])


class IdRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: StrictStr | StrictInt

    @field_validator("index")
    @classmethod
    def check_index(cls, value):
        if isinstance(value, bool) or str(value).strip() == "":
            raise ValueError("index must be a nonempty string or integer")
        return value


class RecordsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[dict]


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts = load_artifacts()
    app.state.lookup = ScoreLookup(artifacts)
    app.state.test_features = TestFeaturesLookup(artifacts) if (artifacts / 'test_features.sqlite').is_file() else None
    # Models are loaded once per process, at startup.
    app.state.predictor = Predictor(artifacts, thread_count=int(os.getenv("MODEL_THREADS", "2")))
    app.state.feature_set = set(app.state.predictor.features)
    yield


app = FastAPI(title="Bankruptcy prediction API", version="2.0.0", lifespan=lifespan)
origins = [s.strip() for s in os.getenv("CORS_ORIGINS", "https://web.team-frontend.hack.week-infra.ru").split(",") if s.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.middleware("http")
async def record_http_metrics(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    route = request.scope.get("route")
    path = route.path if route else "unmatched"
    requests_total.labels(request.method, path, str(response.status_code)).inc()
    request_duration.labels(request.method, path).observe(time.monotonic() - start)
    return response


def public_result(row):
    return {"index": row["index"], "probability": row["target"]}


def predict_records(request: Request, records: list[dict]):
    if not 1 <= len(records) <= BATCH_SIZE:
        raise HTTPException(422, f"Provide 1 to {BATCH_SIZE} records")
    try:
        return [public_result(row) for row in request.app.state.predictor.predict(records)]
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(500, "Prediction failed") from exc


@app.get("/health")
def health(request: Request):
    return {"model_ready": True, "lookup_ready": True,
            "test_features_ready": request.app.state.test_features is not None, "model_version": MODEL_VERSION}


@app.get("/actuator/health/liveness")
def liveness():
    return {"status": "UP"}


@app.get("/actuator/health/readiness")
def readiness(request: Request):
    return {"status": "UP" if request.app.state.predictor else "DOWN"}


@app.get("/actuator/prometheus")
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/v1/predict/by-id")
def predict_by_id(payload: IdRequest, request: Request):
    """Stored final forecast for an ID from TEST. Unknown ID -> 404."""
    try:
        return public_result(request.app.state.lookup.predict_by_index(payload.index))
    except KeyError as exc:
        raise HTTPException(404, "Unknown index") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Lookup by index failed")
        raise HTTPException(500, "Lookup failed") from exc


@app.post("/api/v1/predict/by-id/details")
def predict_by_id_details(payload: IdRequest, request: Request):
    """Stored score, source TEST features, and model sensitivity for this ID."""
    source = request.app.state.test_features
    if source is None:
        raise HTTPException(503, "TEST features artifact is not installed")
    try:
        score = public_result(request.app.state.lookup.predict_by_index(payload.index))
        record = source.record_by_index(payload.index, request.app.state.predictor.features)
        explanation = request.app.state.predictor.explain(record)
        return {"index": score['index'], "probability": score['probability'],
                "features": {name: record[name] for name in request.app.state.predictor.features},
                "word_count": record['word_count'], "explanation": explanation}
    except KeyError as exc:
        raise HTTPException(404, "Unknown index") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("ID details failed")
        raise HTTPException(500, "ID details failed") from exc


@app.post("/api/v1/predict/records")
def predict_json(payload: RecordsRequest, request: Request):
    return {"predictions": predict_records(request, payload.records)}


@app.post("/api/v1/predict/explain")
def explain_record(payload: ExplainRequest, request: Request):
    """Sensitivity of one full CSV/JSON record; unavailable for lookup-only IDs."""
    try:
        return request.app.state.predictor.explain(payload.record)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Explanation failed")
        raise HTTPException(500, "Explanation failed") from exc


def parse_csv_row(row: dict, number: int, expected: set) -> dict:
    record = {}
    for key, value in row.items():
        if key in expected:
            if value is None or value.strip() == "":
                raise HTTPException(422, f"CSV row {number}: {key} is empty")
            try:
                parsed = float(value)  # float64, no rounding: exact-match rule needs it
            except ValueError as exc:
                raise HTTPException(422, f"CSV row {number}: {key} must be numeric") from exc
            if not math.isfinite(parsed):
                raise HTTPException(422, f"CSV row {number}: {key} must be finite")
            record[key] = parsed
        else:  # index / word_count
            record[key] = value.strip() or None if key == "index" else (value or None)
    return record


@app.post("/api/v1/predict/csv")
def predict_csv(request: Request, file: UploadFile = File(...)):
    """Sync endpoint: model calls block, so FastAPI runs it in the thread pool, not on the event loop."""
    if file.content_type not in {"text/csv", "application/csv", "application/vnd.ms-excel", "application/octet-stream"}:
        raise HTTPException(422, "Upload a CSV file")
    if file.size is not None and file.size > MAX_CSV_BYTES:
        raise HTTPException(413, f"CSV exceeds {MAX_CSV_BYTES // (1024 * 1024)} MiB")
    stream = io.TextIOWrapper(file.file, encoding="utf-8-sig", newline="")
    reader = csv.DictReader(stream)
    try:
        fieldnames = reader.fieldnames
    except UnicodeDecodeError as exc:
        raise HTTPException(422, "CSV must use UTF-8") from exc
    if fieldnames is None or len(fieldnames) != len(set(fieldnames)):
        raise HTTPException(422, "CSV needs a header with unique columns")
    expected = request.app.state.feature_set
    actual = set(fieldnames)
    extra = actual - expected - {"index", "word_count"}
    if expected - actual or extra:
        raise HTTPException(422, f"CSV columns differ: missing={sorted(expected - actual)[:5]}, unknown={sorted(extra)[:5]}")
    predictions, batch, first = [], [], 2
    try:
        for number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise HTTPException(422, f"CSV row {number} has the wrong number of columns")
            if not batch:
                first = number
            batch.append(parse_csv_row(row, number, expected))
            if len(batch) == BATCH_SIZE:
                predictions.extend(predict_records(request, batch))
                batch = []
        if batch:
            predictions.extend(predict_records(request, batch))
    except UnicodeDecodeError as exc:
        raise HTTPException(422, "CSV must use UTF-8") from exc
    except HTTPException as exc:
        if exc.status_code == 422 and batch and "CSV row" not in str(exc.detail):
            exc.detail = f"Rows {first}-{first + len(batch) - 1}: {exc.detail}"
        raise
    if not predictions:
        raise HTTPException(422, "CSV contains no rows")
    return {"predictions": predictions}
