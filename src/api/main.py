"""FastAPI backend for Next Problem.

Run from the repository root:
    uvicorn src.api.main:app --port 8000

Endpoints:
    GET /health                      liveness + model-server status
    GET /recommend?handle=...        ranked "stretch zone" problems for a user
    GET /metrics                     Prometheus metrics (also probes the model server and frontend)
    GET /pipeline                    pipeline stages, latest data/model metrics, service status

The backend never loads the model. It calls the MLflow model server over REST
(MODEL_URL), so the frontend, backend and model server stay loosely coupled.
"""
import json
import logging
import os
import re
import time
from collections import deque
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.api import codeforces_client as cf
from src.api import features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

# ---- Configuration (all overridable via environment variables) ---------------
MODEL_URL = os.getenv("MODEL_URL", "http://127.0.0.1:5001/invocations")
TRAIN_STATS_PATH = os.getenv("TRAIN_STATS_PATH", "data/processed/train_stats.json")
HISTORY_CONTESTS = int(os.getenv("HISTORY_CONTESTS", "60"))  # recent contests used as history
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
HANDLE_RE = re.compile(r"^[A-Za-z0-9_.\-]{3,24}$")
BATCH_SIZE = 5000
TRAIN_METRICS_PATH = os.getenv("TRAIN_METRICS_PATH", "metrics/train_metrics.json")
SPARK_METRICS_PATH = os.getenv("SPARK_METRICS_PATH", "data/processed/spark_metrics.json")
FRONTEND_URL = os.getenv("FRONTEND_URL", "")  # probed on every scrape; empty = not probed
PROBE_TIMEOUT = 3  # seconds

# ---- Prometheus metrics ----------------------------------------------------
REQUESTS = Counter("np_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram(
    "np_request_latency_seconds", "End-to-end request latency", ["endpoint"],
    buckets=(0.05, 0.25, 1, 2.5, 5, 10, 20, 40, 80),
)
ERRORS = Counter("np_errors_total", "Errors by type", ["type"])
MODEL_LATENCY = Histogram("np_model_latency_seconds", "Model server call latency")
LIVE_MEAN = Gauge("np_live_feature_mean", "Mean of recent live values", ["feature"])
TRAIN_MEAN = Gauge("np_train_feature_mean", "Training-set mean", ["feature"])
DRIFT = Gauge(
    "np_feature_mean_shift",
    "Relative shift of live feature mean vs training mean", ["feature"],
)
# The model server and the frontend expose no Prometheus metrics of their own, so the
# backend probes them on every scrape and publishes the result (1 = up, 0 = down).
SERVICE_UP = Gauge("np_service_up", "Health of a dependent service (1 up, 0 down)", ["service"])

# Drift is tracked on user-level features (one value per request).
MONITORED = ["rating", "n_prev_contests"]
_recent = {f: deque(maxlen=50) for f in MONITORED}
TRAIN_STATS: dict = {}
try:
    TRAIN_STATS = json.loads(Path(TRAIN_STATS_PATH).read_text())
    for _f in MONITORED:
        if _f in TRAIN_STATS:
            TRAIN_MEAN.labels(_f).set(TRAIN_STATS[_f])
except (OSError, ValueError):
    log.warning("Could not load training stats from %s; drift gauges disabled", TRAIN_STATS_PATH)


def observe_drift(values: dict) -> None:
    """Update live-mean and relative-shift gauges from the latest request."""
    for name, value in values.items():
        _recent[name].append(float(value))
        live = sum(_recent[name]) / len(_recent[name])
        LIVE_MEAN.labels(name).set(live)
        base = TRAIN_STATS.get(name)
        if base:
            DRIFT.labels(name).set((live - base) / abs(base))


# ---- App ---------------------------------------------------------------------
app = FastAPI(title="Next Problem API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET"], allow_headers=["*"]
)
KNOWN_PATHS = {"/health", "/recommend", "/metrics", "/pipeline"}


@app.middleware("http")
async def track_requests(request: Request, call_next):
    path = request.url.path if request.url.path in KNOWN_PATHS else "other"
    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        ERRORS.labels("unhandled").inc()
        REQUESTS.labels(path, "500").inc()
        raise
    LATENCY.labels(path).observe(time.time() - start)
    REQUESTS.labels(path, str(response.status_code)).inc()
    return response


class ModelServerError(Exception):
    pass


def predict(rows: list) -> list:
    """Probability of solving for each feature row, from the MLflow model server."""
    probs = []
    try:
        for i in range(0, len(rows), BATCH_SIZE):
            chunk = rows[i:i + BATCH_SIZE]
            payload = {"dataframe_split": {
                "columns": features.FEATURE_COLS,
                "data": [[float(r[c]) for c in features.FEATURE_COLS] for r in chunk],
            }}
            with MODEL_LATENCY.time():
                resp = requests.post(MODEL_URL, json=payload, timeout=60)
            resp.raise_for_status()
            probs.extend(float(p[1]) for p in resp.json()["predictions"])
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        raise ModelServerError(str(exc)) from exc
    return probs


def probe(url: str) -> bool:
    """True if `url` answers HTTP 200 within PROBE_TIMEOUT seconds."""
    try:
        return requests.get(url, timeout=PROBE_TIMEOUT).status_code == 200
    except requests.RequestException:
        return False


def model_server_up() -> bool:
    """Ask the MLflow model server's own /health endpoint (derived from MODEL_URL)."""
    up = probe(MODEL_URL.rsplit("/", 1)[0] + "/health")
    SERVICE_UP.labels("model_server").set(int(up))
    return up


@app.get("/health")
def health():
    model_up = model_server_up()
    return {"status": "ok" if model_up else "degraded",
            "model_server": "up" if model_up else "down"}


@app.get("/metrics")
def metrics():
    # Refresh the health gauges so every scrape sees the current state of each service.
    model_server_up()
    if FRONTEND_URL:
        SERVICE_UP.labels("frontend").set(int(probe(FRONTEND_URL)))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/recommend")
def recommend(
    handle: str = Query(..., description="Codeforces handle"),
    k: int = Query(10, ge=1, le=50, description="Number of problems to return"),
    low: float = Query(0.5, ge=0.0, le=1.0, description="Lower solve-probability bound"),
    high: float = Query(0.7, ge=0.0, le=1.0, description="Upper solve-probability bound"),
):
    started = time.time()
    if not HANDLE_RE.match(handle):
        ERRORS.labels("invalid_handle").inc()
        raise HTTPException(400, "Invalid handle: use 3-24 letters, digits, '_', '-' or '.'.")
    if low >= high:
        raise HTTPException(400, "'low' must be smaller than 'high'.")

    # 1. Fetch the user's data from Codeforces.
    try:
        info = cf.get_user_info(handle)
        rating = info.get("rating")
        if rating is None:
            ERRORS.labels("unrated_user").inc()
            raise HTTPException(422, "This user has no rating yet. Play a rated contest first.")
        history = cf.get_user_rating(handle)
        submissions = cf.get_user_status(handle)
        problems = cf.get_problems()
    except cf.HandleNotFound:
        ERRORS.labels("handle_not_found").inc()
        raise HTTPException(404, f"Handle '{handle}' was not found on Codeforces.")
    except cf.CodeforcesError as exc:
        ERRORS.labels("codeforces_api").inc()
        log.error("Codeforces error for %s: %s", handle, exc)
        raise HTTPException(502, "Codeforces is not reachable right now. Try again later.")

    # 2. Build features (same definitions as training).
    by_contest, rated = features.index_problems(problems)
    solved_ever, solved_in_contest = features.solved_sets(submissions)
    contest_ids = [h["contestId"] for h in history][-HISTORY_CONTESTS:]
    n, s = features.tag_history(contest_ids, by_contest, solved_in_contest)
    n_prev = len(contest_ids)

    candidates = [
        p for p in rated
        if (p["contestId"], p["index"]) not in solved_ever
        and rating - 400 <= p["rating"] <= rating + 600
    ]
    rows = [features.build_row(rating, p, n, s, n_prev) for p in candidates]
    observe_drift({"rating": rating, "n_prev_contests": n_prev})

    # 3. Score with the model server.
    try:
        probs = predict(rows) if rows else []
    except ModelServerError as exc:
        ERRORS.labels("model_server").inc()
        log.error("Model server error: %s", exc)
        raise HTTPException(503, "The model service is unavailable. Try again later.")

    # 4. Keep the stretch zone, rank by weakest topics (then closeness to 60%).
    zone = [
        (p, pr, features.weakness(p, n, s))
        for p, pr in zip(candidates, probs) if low <= pr <= high
    ]
    zone.sort(key=lambda t: (-t[2], abs(t[1] - 0.6)))

    recommendations = [
        {
            "contest_id": p["contestId"],
            "index": p["index"],
            "name": p.get("name", "").strip(),
            "rating": p["rating"],
            "tags": p.get("tags", []),
            "p_solve": round(pr, 3),
            "weakness": round(w, 3),
            "url": f"https://codeforces.com/problemset/problem/{p['contestId']}/{p['index']}",
        }
        for p, pr, w in zone[:k]
    ]
    log.info("handle=%s rating=%s candidates=%d in_zone=%d took=%.1fs",
             handle, rating, len(candidates), len(zone), time.time() - started)
    return {
        "handle": handle,
        "rating": rating,
        "history_contests_used": n_prev,
        "weak_tags": features.weak_tags(n, s),
        "candidates_scored": len(candidates),
        "in_zone": len(zone),
        "zone": [low, high],
        "recommendations": recommendations,
    }


# ---- Pipeline status: read-only snapshot for the frontend's pipeline page ----
STAGES = [
    {"name": "Fetch", "tool": "Codeforces API + Python",
     "what": "Downloads contests, standings and rating changes."},
    {"name": "Features", "tool": "PySpark",
     "what": "Builds leak-free per-tag history features and a time-based split, "
             "then runs data quality checks."},
    {"name": "Train", "tool": "scikit-learn + MLflow",
     "what": "Trains an Elo baseline and gradient boosting; every run is tracked."},
    {"name": "Register", "tool": "MLflow Model Registry + DVC",
     "what": "Promotes a model to 'champion' only if it beats the Elo baseline."},
    {"name": "Serve", "tool": "MLflow model server + FastAPI",
     "what": "Serves solve probabilities over REST; the backend ranks problems."},
    {"name": "Monitor", "tool": "Prometheus + Grafana",
     "what": "Tracks latency, errors and feature drift."},
]


def _read_json(path: str):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


@app.get("/pipeline")
def pipeline():
    return {
        "services": health(),
        "stages": STAGES,
        "data": _read_json(SPARK_METRICS_PATH),
        "model": _read_json(TRAIN_METRICS_PATH),
    }
