"""FastAPI wrapper for market-value prediction."""
import os
import sys
import logging
import time
from collections import defaultdict, deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from predict import predict_df
from registry import list_versions, get_model_path, load_model

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "Request latency", ["path"])
PREDICTION_VALUE = Histogram("prediction_value_eur", "Predicted market value", buckets=[1e5, 1e6, 1e7, 1e8, 1e9])

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("marketvalue")

API_KEY = os.getenv("API_KEY")
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "60"))  # requests per minute per IP
_rate_buckets = defaultdict(deque)

MODEL_PATH = os.getenv("MODEL_PATH", None)
if MODEL_PATH:
    os.environ.setdefault("MODEL_PATH", MODEL_PATH)

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(title="Football Market Value Predictor", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    # rate limit by IP
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _rate_buckets[client_ip]
    while bucket and bucket[0] < now - 60:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    bucket.append(now)

    # auth
    if API_KEY and request.url.path not in ("/health", "/docs", "/openapi.json"):
        if request.headers.get("X-API-Key") != API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})

    import time as _t
    start = _t.time()
    logger.info("%s %s", request.method, request.url.path)
    response = await call_next(request)
    dur = _t.time() - start
    REQUEST_LATENCY.labels(path=request.url.path).observe(dur)
    REQUEST_COUNT.labels(method=request.method, path=request.url.path, status=response.status_code).inc()
    logger.info("%s %s -> %s (%.3fs)", request.method, request.url.path, response.status_code, dur)
    return response


class PlayerInput(BaseModel):
    age: float
    height_in_cm: float = Field(180.0)
    international_caps: float = Field(0.0)
    international_goals: float = Field(0.0)
    position: str = Field("Attack")
    sub_position: str = Field("Centre-Forward")
    foot: str = Field("right")
    country_of_citizenship: str = Field("Unknown")
    league: str = Field("Unknown")
    contract_months: float = Field(12.0)
    cum_matches: float = Field(0.0)
    cum_goals: float = Field(0.0)
    cum_assists: float = Field(0.0)
    cum_minutes: float = Field(0.0)
    num_transfers: float = Field(0.0)
    prev_value: float = Field(0.0)
    peak_so_far: float = Field(0.0)
    days_since_prev: float = Field(9999.0)
    player_highest_mv: float = Field(0.0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PlayerInput):
    df = pd.DataFrame([payload.model_dump()])
    val = float(predict_df(df)[0])
    PREDICTION_VALUE.observe(val)
    logger.info("prediction=%s", val)
    return {"predicted_market_value_eur": val}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/models")
def models():
    return list_versions()


@app.get("/models/{version}")
def model_info(version: str):
    reg = list_versions()
    if version not in reg["versions"]:
        raise HTTPException(status_code=404, detail="Unknown version")
    return reg["versions"][version]


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True, log_level=LOG_LEVEL.lower())
