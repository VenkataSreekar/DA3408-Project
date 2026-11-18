# Next Problem: Low-Level Design

Nov 18, 2026 · Venkata Sreekar

## 1. Code structure

The backend exposes four REST endpoints. All of them use HTTP GET and return JSON, except `/metrics`, which returns Prometheus text. The backend reaches the model only through the model server's `POST /invocations`. The architecture and the reasons behind it are in the companion HLD; this document defines the interfaces.

| Module | Responsibility | Calls |
| --- | --- | --- |
| `src/api/main.py` | FastAPI app: the four endpoints, input validation, ranking, Prometheus metrics, health probes of the model server and frontend, request middleware | `codeforces_client`, `features`, model server, frontend |
| `src/api/codeforces_client.py` | Rate-limited, cached access to the Codeforces API; raises typed errors | Codeforces API |
| `src/api/features.py` | Pure functions that build the live feature rows; mirrors the Spark definitions | none |
| `src/ingestion/fetch_codeforces.py` | Offline download of contests, standings and rating changes | Codeforces API |
| `src/features/spark_features.py` | Offline PySpark job: feature table, time split, data quality checks, metrics | none (reads files) |
| `src/training/train.py` | Trains Elo and the boosting models, logs to MLflow, registers the winner | MLflow |
| `src/training/export_model.py` | Copies the `champion` model into `model/` | MLflow registry |
| `frontend/index.html`, `pipeline.html`, `config.js` | Static pages; `config.js` holds every service URL; the main page checks the handle format before sending a request | backend |
| `monitoring/prometheus.yml`, `alert_rules.yml` | Scrape jobs (backend, Prometheus, Grafana) and the 7 alert rules | backend, Grafana |
| `monitoring/grafana/provisioning/` | Data source and the *Next Problem - Monitoring* dashboard | Prometheus |

**Constants shared across modules.** The eight feature names appear as `FEATURE_COLS` in `spark_features.py`, `train.py` and `features.py`. `PRIOR_RATE = 0.3` and `PRIOR_STRENGTH = 5` are defined in both feature modules. The unit test `test_feature_columns_are_identical_in_spark_training_and_api` fails if the three feature lists diverge.

| # | Feature | Type |
| --- | --- | --- |
| 1 | `rating` | float |
| 2 | `difficulty` | float |
| 3 | `rating_diff` | float |
| 4 | `tag_rate_mean` | float |
| 5 | `tag_rate_min` | float |
| 6 | `tag_experience` | float |
| 7 | `n_tags` | float |
| 8 | `n_prev_contests` | float |

The order matters: the model server receives the columns in exactly this order.

## 2. Call sequence for GET /recommend

```text
Browser           Backend              Codeforces API           Model server
    |                 |                       |                       |
    | GET /recommend  |                       |                       |
    |---------------->|                       |                       |
    |                 |                       |                       |
    |                 | [validate; else 400]  |                       |
    |                 | user.info, user.rating|                       |
    |                 | user.status, problems |                       |
    |                 |---------------------->|                       |
    |                 | data, or 404/422/502  |                       |
    |                 |< - - - - - - - - - - -|                       |
    |                 |                       |                       |
    |                 | [build features]      |                       |
    |                 |                       |                       |
    |                 | POST /invocations, batches of 5,000 rows      |
    |                 |---------------------------------------------->|
    |                 | P(solve) per row, or 503                      |
    |                 |< - - - - - - - - - - - - - - - - - - - - - - -|
    |                 |                       |                       |
    |                 | [keep 0.5-0.7,        |                       |
    |                 |  rank, top k]         |                       |
    | 200 JSON        |                       |                       |
    |< - - - - - - - -|                       |                       |
    |                 |                       |                       |
```

Solid arrows are calls and dashed arrows are replies. Each outside call has its own failure code, so the page can tell the user exactly what went wrong. Repeat requests for the same handle within 10 minutes skip the Codeforces round trip, because the client caches the responses.

## 3. GET /recommend

Returns up to k unsolved problems whose predicted solve probability lies in \[low, high\], weakest topics first.

**Request:** `GET http://localhost:8000/recommend?handle=<handle>&k=10&low=0.5&high=0.7`

| Parameter | Type | Required | Default | Rule |
| --- | --- | --- | --- | --- |
| `handle` | string | yes | none | 3-24 characters from `A-Z a-z 0-9 _ - .` (regex `^[A-Za-z0-9_.\-]{3,24}$`) |
| `k` | int | no | 10 | 1 ≤ k ≤ 50 |
| `low` | float | no | 0.5 | 0 ≤ low ≤ 1, and low < high |
| `high` | float | no | 0.7 | 0 ≤ high ≤ 1 |

**Response 200** (`application/json`):

| Field | Type | Meaning |
| --- | --- | --- |
| `handle` | string | The handle as sent |
| `rating` | int | The user's current Codeforces rating |
| `history_contests_used` | int | Number of recent rated contests used as history (at most `HISTORY_CONTESTS`) |
| `weak_tags` | list of {`tag`: string, `rate`: float, `attempts`: int} | Up to 3 tags with the lowest smoothed solve rate, among tags with at least 5 attempts |
| `candidates_scored` | int | Unsolved problems with difficulty from rating − 400 to rating + 600, all scored by the model |
| `in_zone` | int | How many of them fall in \[low, high\] |
| `zone` | \[float, float\] | \[low, high\] as used |
| `recommendations` | list of objects (below), at most k | Sorted by weakness descending, then by distance of `p_solve` from 0.6 |

Each recommendation:

| Field | Type | Meaning |
| --- | --- | --- |
| `contest_id` | int | Codeforces contest id |
| `index` | string | Problem index in the contest, such as `C` |
| `name` | string | Problem name, trimmed |
| `rating` | int | Problem difficulty |
| `tags` | list of string | Codeforces tags (may be empty) |
| `p_solve` | float, 3 decimals | Model's solve probability |
| `weakness` | float, 3 decimals | 1 − mean smoothed solve rate over the problem's tags; 0 for untagged problems |
| `url` | string | `https://codeforces.com/problemset/problem/<contest_id>/<index>` |

Shape of a response (types in place of values):

```json
{
  "handle": "string",
  "rating": 0,
  "history_contests_used": 0,
  "weak_tags": [{"tag": "string", "rate": 0.0, "attempts": 0}],
  "candidates_scored": 0,
  "in_zone": 0,
  "zone": [0.5, 0.7],
  "recommendations": [
    {"contest_id": 0, "index": "string", "name": "string", "rating": 0,
     "tags": ["string"], "p_solve": 0.0, "weakness": 0.0, "url": "string"}
  ]
}
```

**Errors.** Every error body is JSON with a `detail` field. For the backend's own errors `detail` is a message string, which the page shows as is.

| Code | When | `detail` | Error counter (`np_errors_total`) |
| --- | --- | --- | --- |
| 400 | Handle fails the regex | Invalid handle: use 3-24 letters, digits, '\_', '-' or '.'. | `invalid_handle` |
| 400 | low ≥ high | 'low' must be smaller than 'high'. | none |
| 404 | Codeforces reports the handle as not found | Handle '\<handle>' was not found on Codeforces. | `handle_not_found` |
| 422 | The user has no rating | This user has no rating yet. Play a rated contest first. | `unrated_user` |
| 422 | FastAPI validation: `handle` missing, or `k`, `low`, `high` out of range or not numbers | a list of validation errors (FastAPI's standard format) | none |
| 502 | Codeforces unreachable or returned an unexpected error | Codeforces is not reachable right now. Try again later. | `codeforces_api` |
| 503 | Model server unreachable, timed out, or returned a malformed response | The model service is unavailable. Try again later. | `model_server` |

Codeforces and model-server errors are also written to the backend log with the underlying reason.

**Example calls:**

```bash
curl "localhost:8000/recommend?handle=Venkata_Sreekar&k=5"
curl "localhost:8000/recommend?handle=Venkata_Sreekar&low=0.4&high=0.6"
curl -i "localhost:8000/recommend?handle=ab"                  # 400
curl -i "localhost:8000/recommend?handle=zzz_nobody_here_987" # 404
```

## 4. GET /health, GET /pipeline, GET /metrics

These three take no parameters and always return 200 while the backend is running. A failed dependency shows up in the body, not in the status code.

### 4.1 GET /health

Liveness check. The backend calls the model server's `/health` with a 3 s timeout.

| Field | Values |
| --- | --- |
| `status` | `ok` if the model server answered 200, else `degraded` |
| `model_server` | `up` or `down` |

```json
{"status": "ok", "model_server": "up"}
```

### 4.2 GET /pipeline

Read-only snapshot for the Pipeline status page.

| Field | Type | Contents |
| --- | --- | --- |
| `services` | object | The same body as `/health` |
| `stages` | list of {`name`, `tool`, `what`} | The six stages: Fetch, Features, Train, Register, Serve, Monitor |
| `data` | object or null | Contents of `spark_metrics.json`: `rows`, `contests`, `users`, `label_rate`, `split_rows`, `elapsed_seconds`, `rows_per_second`, `quality_checks` ({check name: true/false}) |
| `model` | object or null | Contents of `train_metrics.json`: `elo_baseline`, `best_model` (run id, config, val/test scores), `beats_baseline`, `registered_version`, `git_commit`, `data_md5` |

`data` or `model` is `null` when its file is missing or unreadable. Both files are copied into the backend image at build time, so they show the state at the last `docker compose build`.

### 4.3 GET /metrics

Prometheus exposition format (`text/plain; version=1.0.0; charset=utf-8`), scraped by Prometheus every 15 s. Before answering, it probes the model server's `/health` and, if `FRONTEND_URL` is set, the frontend, each with a 3 s timeout, and updates `np_service_up`. Besides the default process metrics, it exposes:

| Metric | Type | Labels |
| --- | --- | --- |
| `np_requests_total` | counter | `endpoint` (`/health`, `/recommend`, `/metrics`, `/pipeline`, or `other`), `status` |
| `np_request_latency_seconds` | histogram, buckets 0.05, 0.25, 1, 2.5, 5, 10, 20, 40, 80 s | `endpoint` |
| `np_errors_total` | counter | `type` |
| `np_model_latency_seconds` | histogram | none |
| `np_service_up` | gauge, 1 = up, 0 = down | `service` (`model_server`, `frontend`) |
| `np_train_feature_mean` | gauge | `feature` (`rating`, `n_prev_contests`) |
| `np_live_feature_mean` | gauge | `feature` |
| `np_feature_mean_shift` | gauge | `feature` |

Requests to unknown paths are counted under `endpoint="other"`, so the label set stays small. An exception that escapes a handler is counted as status 500 and error type `unhandled`.

## 5. Model server API: POST /invocations

The model server is MLflow's standard scoring server (`mlflow models serve`) running the exported `champion` model. It is a separate service on port 5001, and only the backend calls it. It returns two probabilities per row, and the backend keeps the second one, P(solve).

**Request** (`Content-Type: application/json`), `dataframe_split` format:

| Field | Value |
| --- | --- |
| `dataframe_split.columns` | the eight `FEATURE_COLS`, in the order of section 1 |
| `dataframe_split.data` | one list of 8 floats per (user, problem) pair |

```json
{"dataframe_split": {
  "columns": ["rating", "difficulty", "rating_diff", "tag_rate_mean",
              "tag_rate_min", "tag_experience", "n_tags", "n_prev_contests"],
  "data": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
}}
```

**Response 200:**

```json
{"predictions": [[0.0, 0.0]]}
```

One `[P(not solve), P(solve)]` pair per input row, in the same order. The model was logged with `predict_proba` as its predict function, which is why two values come back.

**How the backend calls it** (`predict()` in `main.py`):

- Rows are sent in batches of 5,000, one POST per batch, each with a 60 s timeout.
- Any network error, non-2xx status, or response without `predictions` raises `ModelServerError`, which becomes HTTP 503 for the client.
- The time of each call is recorded in `np_model_latency_seconds`.

The model server also answers `GET /health` (200 when ready). Docker Compose polls it every 10 s and starts the backend only once it is healthy.

## 6. Internal interfaces

The backend's logic sits in two modules. One wraps the Codeforces API; the other holds pure functions that turn API data into feature rows, so each step can be unit-tested with hand-made data.

### 6.1 Codeforces client (`src/api/codeforces_client.py`)

| Function | Codeforces method | Returns | Cache TTL |
| --- | --- | --- | --- |
| `get_user_info(handle)` | `user.info` | dict of the user's profile, including `rating` (absent if unrated) | 10 min |
| `get_user_rating(handle)` | `user.rating` | list of rating changes, oldest first; each has `contestId` | 10 min |
| `get_user_status(handle)` | `user.status` | list of all the user's submissions | 10 min |
| `get_problems()` | `problemset.problems` | list of all problems, with `contestId`, `index`, `name`, `rating`, `tags` | 6 h |

**Behaviour shared by all four** (`_get()`):

- A lock serialises calls, at least `CF_MIN_INTERVAL` (2.1 s) apart, to respect the rate limit.
- A result is cached under (method, parameters) until its TTL expires.
- Errors:
  - `HandleNotFound`: the API's error comment contains "not found". It is a subclass of `CodeforcesError`.
  - `CodeforcesError`: network failure, invalid JSON, or any other API error.
- A "limit exceeded" reply is retried up to 3 attempts, waiting 2 s and then 4 s.

### 6.2 Feature functions (`src/api/features.py`)

Here `n` and `s` are dictionaries from tag to the number of problems attempted and solved in the history window.

| Function | Input | Output |
| --- | --- | --- |
| `problem_tags(problem)` | problem dict | its tags, or `["none"]` if it has none |
| `index_problems(problems)` | list of problem dicts | (dict contest id → problems, list of rated problems); problems without a rating or contest id are dropped |
| `solved_sets(submissions)` | list of submissions | (set of (contestId, index) ever accepted, set accepted while a CONTESTANT) |
| `tag_history(contest_ids, by_contest, solved_in_contest)` | the history contests | (`n`, `s`) counted over every rated problem of those contests |
| `smoothed_rate(n, s, tag)` | one tag | (s\[tag\] + 5 × 0.3) / (n\[tag\] + 5) |
| `build_row(rating, problem, n, s, n_prev_contests)` | user rating, one candidate problem, history | dict with the eight `FEATURE_COLS` |
| `weakness(problem, n, s)` | one problem, history | 1 − mean `smoothed_rate` over its real tags; 0.0 if untagged |
| `weak_tags(n, s, min_attempts=5, k=3)` | history | up to k {`tag`, `rate`, `attempts`} with the lowest rate, among tags with at least `min_attempts` |

**Feature formulas in `build_row`** (T = the problem's tags, or `none`):

```latex
\text{tag\_rate\_mean} = \frac{1}{|T|}\sum_{t \in T} \text{rate}(t) \qquad \text{tag\_rate\_min} = \min_{t \in T} \text{rate}(t) \qquad \text{tag\_experience} = \frac{1}{|T|}\sum_{t \in T} n_t
```

The remaining features: `rating_diff` = rating − difficulty, `n_tags` = |T|, and `n_prev_contests` = the number of history contests used.

**Ranking in `recommend()`:** keep rows with low ≤ p ≤ high, then sort by the key (−weakness, |p − 0.6|) and take the first k.

## 7. Offline pipeline interfaces

The four DVC stages are command-line scripts that hand over work through files, so each can be run and tested on its own. `dvc.yaml` calls them with the arguments below.

| Stage | Command (as in `dvc.yaml`) | Reads | Writes |
| --- | --- | --- | --- |
| fetch | `python src/ingestion/fetch_codeforces.py --n-contests 70 --users 1500 --out data/raw` | Codeforces API | `data/raw/problems.json`, `contests.json`, `contests/<id>_standings.json`, `contests/<id>_ratings.json` |
| features | `python src/features/spark_features.py --raw data/raw --out data/processed` | `data/raw` | `features.parquet`, `train_stats.json`, `spark_metrics.json` |
| train | `python src/training/train.py --data data/processed/features.parquet --metrics-out metrics/train_metrics.json` | `features.parquet`, `dvc.lock` (for the data md5) | MLflow runs and a registry version; `metrics/train_metrics.json` |
| export\_model | `python src/training/export_model.py` | registry alias `next-problem-classifier@champion` | `model/` (MLmodel, model file, `requirements.txt`, conda env) |

**Script arguments and defaults:**

| Script | Argument | Default |
| --- | --- | --- |
| `fetch_codeforces.py` | `--n-contests` | 70 |
|  | `--users` (cohort members kept per contest) | 1500 |
|  | `--out` | `data/raw` |
| `spark_features.py` | `--raw` | `data/raw` |
|  | `--out` | `data/processed` |
| `train.py` | `--data` | `data/processed/features.parquet` |
|  | `--metrics-out` | `metrics/train_metrics.json` |
|  | `--tracking-uri` | `sqlite:///mlflow.db` |
|  | `--spark-metrics` | `data/processed/spark_metrics.json` |

**Raw file formats** (written by fetch, read by features):

- `<id>_standings.json`: `contest` (with `id` and `startTimeSeconds`), `problems` (each with `index`), and `rows`, where each row is {`handle`, `rank`, `solved`: one boolean per problem, in problem order}.
- `<id>_ratings.json`: the cohort's rating changes, each with `contestId`, `handle`, `oldRating` and `newRating`.

**`features.parquet` columns:** `contest_id`, `start`, `handle`, `rating`, `prob_index`, `difficulty`, `label` (0/1), the eight features, and `split` (`train`, `val` or `test`).

**Data quality checks** (`quality_checks(df) -> dict[str, bool]` in `spark_features.py`): runs 11 checks on the finished table (listed in HLD section 3.3) and returns each result. `main()` writes them to `spark_metrics.json` under `quality_checks`. If any is false, it logs the failed names and exits with code 1, which makes `dvc repro` stop before the train stage.

**`train_metrics.json` keys:** `elo_baseline` {val, test: {log\_loss, auc}}, `best_model` {run\_id, config, val, test}, `beats_baseline`, `registered_version`, `git_commit`, `data_md5`. `scripts/ci.sh` reads `beats_baseline` for the acceptance gate.

**MLproject entry points** run the same scripts: `mlflow run . -e <entry point>` with `main` (`dvc repro`), `fetch`, `features` or `train`.

## 8. Configuration

Every address, path and tuning value is an environment variable or a config file, so moving a service means changing configuration, not code.

**Backend environment variables:**

| Variable | Default in code | Value in Docker Compose | Purpose |
| --- | --- | --- | --- |
| `MODEL_URL` | `http://127.0.0.1:5001/invocations` | `http://model:5001/invocations` | Model server endpoint; `/health` is derived from it |
| `HISTORY_CONTESTS` | 60 | 15 | Recent rated contests used as history |
| `CORS_ORIGINS` | `*` | `*` | Origins allowed to call the API (comma-separated) |
| `TRAIN_STATS_PATH` | `data/processed/train_stats.json` | `/app/train_stats.json` (Dockerfile) | Training means for the drift gauges |
| `TRAIN_METRICS_PATH` | `metrics/train_metrics.json` | `/app/metrics/train_metrics.json` (Dockerfile) | Model results for `/pipeline` |
| `SPARK_METRICS_PATH` | `data/processed/spark_metrics.json` | `/app/metrics/spark_metrics.json` (Dockerfile) | Data metrics for `/pipeline` |
| `CF_MIN_INTERVAL` | 2.1 s | not set | Minimum gap between Codeforces calls |
| `CF_TIMEOUT` | 60 s | not set | Timeout of one Codeforces call |
| `FRONTEND_URL` | empty (not probed) | `http://frontend:80/` | Frontend address probed on every `/metrics` scrape |

**Constants in code:**

| Constant | Value | Where |
| --- | --- | --- |
| `BATCH_SIZE` | 5,000 rows per model-server call | `main.py` |
| User / problem-set cache TTL | 600 s / 21,600 s | `codeforces_client.py` |
| Candidate difficulty window | rating − 400 to rating + 600 | `main.py` |
| Drift window | last 50 requests | `main.py` |
| `PRIOR_RATE`, `PRIOR_STRENGTH` | 0.3, 5 | `features.py`, `spark_features.py` |
| `TRAIN_Q`, `VAL_Q` | 0.70, 0.85 (split quantiles of contest start time) | `spark_features.py` |
| `PROBE_TIMEOUT` | 3 s per health probe | `main.py` |
| `RATING_RANGE`, `DIFFICULTY_RANGE`, `LABEL_RATE_RANGE` | (1, 4500), (800, 3600), (0.05, 0.95) for the quality checks | `spark_features.py` |

**Frontend:** `frontend/config.js` sets `API_URL` (`http://localhost:8000`) and the links to MLflow (5000), Grafana (3001) and Prometheus (9090). Docker Compose mounts it read-only into the container, so editing it needs no rebuild.

**Ports** (host:container, from `docker-compose.yml`): frontend 8080:80, backend 8000:8000, model 5001:5001, Prometheus 9090:9090, Grafana 3001:3000. The MLflow UI (5000) runs on the host.

**Monitoring files:** `monitoring/prometheus.yml` (scrape interval 15 s; jobs `backend`, `prometheus`, `grafana`; rule file), `monitoring/alert_rules.yml` (mounted into the Prometheus container at `/etc/prometheus/alert_rules.yml`), and the Grafana provisioning folder.
