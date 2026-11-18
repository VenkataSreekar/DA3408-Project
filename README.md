# Next Problem: a personalized Codeforces practice recommender

Enter a Codeforces handle and get unsolved problems you have a **50-70% chance of solving**, focused on your **weakest topics**. A gradient-boosted classifier predicts the solve probability from your rating, the problem's difficulty and tags, and your past success rate per tag.

On held-out test contests (time-based split), the model beats an Elo-style baseline on both acceptance metrics:

| Metric (test) | Elo baseline | Model |
| --- | --- | --- |
| Log loss (lower is better) | 0.24179 | **0.22121** |
| AUC (higher is better) | 0.96459 | **0.96762** |

## Architecture

- **Offline pipeline (DVC):** `fetch` (Codeforces API) → `features` (PySpark, leak-free history, time split, 11 data quality checks) → `train` (Elo + gradient boosting, tracked in MLflow, registered as `champion` only if it beats Elo) → `export_model`.
- **Online system (Docker Compose, five services):** `frontend` (Apache httpd, :8080) → `backend` (FastAPI, :8000) → `model` (MLflow model server, :5001). `prometheus` (:9090) scrapes the backend (which also probes the model server and frontend), Grafana and itself, and evaluates 7 alert rules. `grafana` (:3001) shows the dashboard, including service status and firing alerts.

The services talk only over REST. The design documents (HLD, LLD, test plan and report, user manual) describe the architecture in detail.

## Quick start

Requirements: Docker with Compose, conda, Git and DVC.

```bash
git clone <repository URL> next-problem && cd next-problem
conda env create -f conda.yaml && conda activate nextproblem-env
dvc pull                       # data and trained model (or: dvc repro to rebuild everything)
docker compose up -d --build
docker compose ps              # all five services should be Up
```

Then open **http://localhost:8080**.

| Service | Address |
| --- | --- |
| App | http://localhost:8080 |
| Pipeline status page | http://localhost:8080/pipeline.html |
| API docs (Swagger) | http://localhost:8000/docs |
| Grafana (`admin` / `admin`) | http://localhost:3001 |
| Prometheus (alerts at `/alerts`) | http://localhost:9090 |
| MLflow UI | http://localhost:5000, after `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000` |

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /recommend?handle=&k=10&low=0.5&high=0.7` | Ranked recommendations (errors: 400, 404, 422, 502, 503) |
| `GET /health` | Backend and model-server status |
| `GET /pipeline` | Pipeline stages, data and model metrics, service status |
| `GET /metrics` | Prometheus metrics |

## Pipeline, experiments and CI

```bash
dvc repro                      # rerun only the stages whose inputs changed
dvc dag                        # fetch -> features -> train -> export_model
dvc metrics show               # data and model metrics
mlflow run . -e train --env-manager local   # same training through MLproject
scripts/ci.sh                  # tests -> dvc repro -> metrics -> acceptance gate
```

Enable the pre-commit hook (it runs the tests before every commit) once per clone:

```bash
git config core.hooksPath scripts/git-hooks
```

After retraining, rebuild the images that hold the model and its metrics:

```bash
docker compose up -d --build backend model
```

## Tests

```bash
pytest -v
```

The suite has 35 tests: live features (10), Elo and metrics (5), API with all error paths and service-health probes (13), and Spark leakage, time-split and data quality checks (7). Codeforces and the model server are faked, so the tests need no network.

With the stack running, `scripts/system_tests.sh` runs the end-to-end system tests: health, real recommendations, error codes, monitoring, a backend restart, and a model-server outage that must fire the `ModelServerDown` alert.

## Repository layout

```
src/ingestion/fetch_codeforces.py   download contests, standings, rating changes
src/features/spark_features.py      PySpark feature job and time split
src/training/train.py               Elo + gradient boosting, MLflow tracking and registry
src/training/export_model.py        copy the champion model into model/
src/api/                            FastAPI backend (main.py, features.py, codeforces_client.py)
frontend/                           index.html, pipeline.html, config.js
docker/                             Dockerfiles and backend requirements
monitoring/                         Prometheus config, alert rules, Grafana provisioning and dashboard
tests/                              pytest suite
scripts/                            ci.sh, system_tests.sh and the git pre-commit hook
docs/                               design documents, test report, CI log
dvc.yaml, dvc.lock                  pipeline definition and lock file
MLproject, conda.yaml               MLflow project and pinned environment
docker-compose.yml                  the five services
```

## Known limitations

- Training history covers only the 68 contests in the dataset, while the live service uses up to 15 recent rated contests of any type. The drift gauges show this gap.
- The solve probability is a contest-conditions estimate, and it is likely lower than the chance of solving in untimed practice.
- Unrated users cannot get recommendations.


## Demonstration video

A 3-minute walkthrough of the app, the pipeline status page, MLflow, and monitoring with Grafana and Prometheus:
[Watch the demo on Google Drive](https://drive.google.com/file/d/1ruzrAYbfr3Psq2Lv560Ffdfk_iqAnXBA/view?usp=sharing)