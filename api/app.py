"""FastAPI service for heart-disease risk prediction.

Endpoints: POST /predict, GET /health, GET /metrics (Prometheus)
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)

from api.schemas import PatientData, PredictionResponse

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("heart-api")

FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak",
            "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "model.joblib"

REQUESTS = Counter("api_requests_total", "Total API requests",
                   ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds",
                    ["endpoint"], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("api_predictions_total", "Predictions by class", ["label"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL))
    app.state.model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        endpoint = request.url.path
        if endpoint != "/metrics":
            REQUESTS.labels(request.method, endpoint, str(status)).inc()
            LATENCY.labels(endpoint).observe(elapsed)
            logger.info("%s %s -> %s in %.1f ms", request.method, endpoint, status,
                        elapsed * 1000)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "model")}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    row = patient.model_dump()
    X = pd.DataFrame([row])[FEATURES]
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)
    confidence = prob if pred == 1 else 1 - prob
    label = "heart disease likely" if pred else "no heart disease"
    PREDICTIONS.labels(str(pred)).inc()
    logger.info("prediction=%d confidence=%.3f input=%s", pred, confidence, row)
    return PredictionResponse(prediction=pred, confidence=round(confidence, 4),
                              probability_disease=round(prob, 4), label=label)
