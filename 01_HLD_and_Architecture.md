# Next Problem: Architecture and High-Level Design

Nov 18, 2026 · Venkata Sreekar

## 1. Overview

Next Problem takes a Codeforces handle and returns unsolved problems the user has a 50-70% chance of solving, weakest topics first. A gradient-boosted classifier estimates that chance. On held-out test contests it beats an Elo-style baseline on both acceptance metrics: log loss 0.2212 vs 0.2418, AUC 0.9676 vs 0.9646.

**Problem.** Competitive programmers lose time choosing what to practise. Problems that are too easy teach little, and problems that are too hard frustrate. Codeforces filters only by rating and tag, so it ignores each person's strengths and weaknesses by topic.

**Objective.** Predict P(user solves problem) from the user's pre-contest rating, the problem's difficulty and tags, and the user's past success rate per tag. History features use earlier contests only. Then recommend problems in the "stretch zone" (default 0.5 to 0.7) that target the user's weakest tags.

**Labels need no annotation.** Contest standings record exactly who solved which problem during a contest, so every (user, problem, contest) row has a true label.

**Acceptance criterion.** With a time-based split, the selected model must have lower test log loss AND higher test AUC than the Elo baseline. The training stage checks this automatically, and only a model that passes gets the `champion` alias in the registry.

**Scope.**

- In scope: rated users, problems that have a difficulty rating, and individual solves during rated Codeforces (CF-type) contests.
- Out of scope: unrated users (the API returns 422 for them), gym contests, and educational, ICPC-style and other non-CF contest types in the training data.
- Delivered as a dockerized web app with five services, plus an offline training pipeline that runs locally. No cloud services are used.

## 2. Architecture

```text
+----------------------------------------------------------------------------------+
| OFFLINE PIPELINE (host) - run with: dvc repro                                    |
|                                                                                  |
| +----------------------------------------------------------------------------+   |
| | MLflow tracking + model registry (:5000)                                   |   |
| | tracks params, metrics, permutation importances, git commit, data md5      |   |
| | registers next-problem-classifier; alias "champion" only if it beats Elo   |   |
| +----------------------------------------------------------------------------+   |
|                                                  ^                   |           |
|                            logs runs, registers  |                   | champion  |
|                                                  |                   v           |
| +----------------+  +----------------+  +----------------+  +----------------+   |
| | 1. fetch       |  | 2. features    |  | 3. train       |  | 4. export_model|   |
| | Python +       |->| PySpark:       |->| Elo + 3 HGB    |->| champion ->    |   |
| | Codeforces API |  | leak-free,     |  | runs; Elo gate |  | model/         |   |
| | -> data/raw    |  | time split     |  |                |  |                |   |
| +----------------+  +----------------+  +----------------+  +----------------+   |
|          |                                                           |           |
|          |                                                           |           |
+----------|-----------------------------------------------------------|-----------+
           v                                                           |
  +------------------------------+                                     |
  | Codeforces API (external)    |                  model/ baked into  |
  | public REST, rate-limited    |                  the model image    |
  +------------------------------+                                     |
                                ^ live user data                       |
                                |                                      |
+-------------------------------|--------------------------------------|-----------+
| ONLINE (Docker Compose)       |                                      |           |
|                               ^                                      v           |
| +--------------------+      +--------------------+      +--------------------+   |
| | Frontend :8080     |      | Backend :8000      |      | Model server :5001 |   |
| | Apache httpd: form,| REST | FastAPI: fetches   | REST | MLflow serves the  |   |
| | results, pipeline  |----->| user data, builds  |----->| champion: rows in, |   |
| | status page        |      | features, ranks    |      | P(solve) out       |   |
| +--------------------+      +--------------------+      +--------------------+   |
|                                        ^                                         |
|                                        | scrape                                  |
| +--------------------+      +--------------------+                               |
| | Grafana :3001      |PromQL| Prometheus :9090   |                               |
| | dashboard: traffic,|----->| scrapes /metrics   |                               |
| | errors, drift      |      | every 15 s         |                               |
| +--------------------+      +--------------------+                               |
|                                                                                  |
+----------------------------------------------------------------------------------+
```

Arrows point from the caller to the service it calls. The arrow from export_model to the model server is a build step: the exported model is baked into the model-server image. The browser loads the page from the frontend, and the page's JavaScript then calls the backend directly.

| Block | Responsibility | Talks to |
| --- | --- | --- |
| Frontend (Apache httpd, :8080) | Serves the static page: handle form, results table, error messages, the Pipeline status page and links to MLflow, Grafana and Prometheus | Backend only, over REST; its URL comes from `config.js` |
| Backend (FastAPI, :8000) | Validates input, fetches the user's data, builds the eight features, calls the model, filters the stretch zone and ranks by weakness; exposes `/recommend`, `/health`, `/pipeline`, `/metrics` | Codeforces API, model server |
| Model server (MLflow, :5001) | Serves the `champion` model: feature rows in, solve probabilities out (`POST /invocations`) | Called by the backend |
| Codeforces API (external) | Source of truth for problems, contests, standings, ratings and submissions | Called by `fetch` (offline) and by the backend (online) |
| Prometheus (:9090) | Scrapes and stores the backend's request, latency, error and drift metrics | Backend `/metrics` |
| Grafana (:3001) | Dashboard *Next Problem - Monitoring*, provisioned from files | Prometheus |
| fetch (DVC stage) | Downloads 70 contests for a fixed 10% user cohort into `data/raw` | Codeforces API |
| features (DVC stage, PySpark) | Builds the leak-free feature table and the time-based split | reads `data/raw` |
| train (DVC stage) | Trains Elo and three boosting models, applies the acceptance gate, registers the winner | MLflow |
| export\_model (DVC stage) | Copies the `champion` model into `model/`, the model server's build input | MLflow registry |
| MLflow tracking and registry (:5000, host) | Stores runs, parameters, metrics and artifacts, model versions and the `champion` alias | written by train, read by export\_model |

## 3. Data and feature design

The training table has 339,576 (user, problem, contest) rows from 68 contests and 10,270 users. Every history feature is computed only from contests that started strictly before the row's contest.

### 3.1 Ingestion (`src/ingestion/fetch_codeforces.py`)

- Calls `problemset.problems` (difficulty and tags), `contest.list`, and, for each of the 70 newest finished CF-type contests, `contest.ratingChanges` and `contest.standings`.
- **User cohort:** a handle is kept if crc32(handle) mod 10 = 0, at most 1,500 per contest. The hash picks the same users in every contest and covers all skill levels, unlike taking the top of the standings.
- Only the fields needed are kept: handle, rank, and one solved flag per problem (points > 0).
- It respects the rate limit (2.1 s between calls, up to 4 retries), and it is idempotent: files that already exist are skipped, so an interrupted run resumes. DVC marks `data/raw` as `persist`, so it is not deleted between runs.

### 3.2 Feature job (`src/features/spark_features.py`, PySpark)

1. **Base rows:** standings rows are exploded into one row per problem and joined with problem metadata (difficulty, tags) and the user's pre-contest rating. First-time participants (old rating 0) and problems with no difficulty are dropped. Label = 1 if the problem was solved during the contest.
2. **Per-tag history:** rows are exploded by tag and summed per (user, tag, contest). A window over each (user, tag), ordered by contest start time and ending one row before the current contest, gives prior attempts n and prior solves s.
3. **Smoothing:** tag rate = (s + 5 × 0.3) / (n + 5), so a user with no history on a tag gets a neutral 0.3.
4. **Back to one row per problem:** mean and minimum tag rate and mean prior attempts over the problem's tags. An untagged problem gets the single tag `none`.
5. **Time split:** contests are ordered by start time. The oldest 70% go to train, the next 15% to validation and the newest 15% to test. A contest never straddles two splits.

| Feature | Definition |
| --- | --- |
| `rating` | User's rating before the contest |
| `difficulty` | Problem's difficulty rating |
| `rating_diff` | rating − difficulty |
| `tag_rate_mean` | Mean smoothed solve rate over the problem's tags |
| `tag_rate_min` | Lowest smoothed solve rate among the problem's tags |
| `tag_experience` | Mean number of earlier attempts on the problem's tags |
| `n_tags` | Number of tags (`none` counts as one) |
| `n_prev_contests` | Contests the user played before this one |

### 3.3 Outputs and volumes

| Output | Contents |
| --- | --- |
| `data/processed/features.parquet` | Model-ready rows: features, label, split, and contest/user keys |
| `data/processed/spark_metrics.json` | Row counts and throughput (tracked as DVC metrics) |
| `data/processed/train_stats.json` | Training-set feature means, used as the drift reference |

| Split | Rows |
| --- | --- |
| Train | 263,522 |
| Validation | 44,234 |
| Test | 31,820 |

The positive rate is 35.6%. The job runs on Spark in local mode (`local[*]`, 4 GB driver, 16 shuffle partitions) in 31.1 s, about 10,900 rows/s.

**Leakage guarantees are tested.** `tests/test_spark_features.py` checks that a user's first contest has no history, that history comes only from strictly earlier contests, that flipping a label inside a contest changes no feature of that contest, and that the split is ordered by time.

## 4. Model design

Three gradient-boosting configurations are trained against an Elo baseline. The one with the lowest validation log loss is registered, and it becomes `champion` only if it beats Elo on test.

### 4.1 Baseline

The Elo-style baseline uses rating and difficulty only:

```latex
P(\text{solve}) = \frac{1}{1 + 10^{(\text{difficulty} - \text{rating})/400}}
```

### 4.2 Candidates

scikit-learn `HistGradientBoostingClassifier` on the eight features, `random_state = 42`:

| Run | learning\_rate | max\_iter | max\_depth |
| --- | --- | --- | --- |
| hgb\_0 | 0.10 | 100 | None |
| hgb\_1 (selected) | 0.05 | 200 | 6 |
| hgb\_2 | 0.10 | 200 | 8 |

### 4.3 Selection, acceptance and registration

1. **Select on validation only.** The candidate with the lowest validation log loss wins. Test scores are logged but never used to choose.
2. **Acceptance gate.** The winner must have lower test log loss and higher test AUC than Elo. The result is written to `metrics/train_metrics.json` as `beats_baseline`.
3. **Register.** Every training run registers the winner as a new version of `next-problem-classifier`. Each version is tagged with its git commit, data md5, run id and acceptance result.
4. **Promote.** Only a version that passes gets the `champion` alias. Rolling back means pointing the alias at an earlier version, with no retraining.
5. **Export.** The `export_model` DVC stage downloads `models:/next-problem-classifier@champion` into `model/`, which the model-server image is built from.

The model is logged with an inferred signature and `predict_proba` as its predict function, so the served endpoint returns probabilities. It is serialised with cloudpickle, because MLflow's default skops format refused to load scikit-learn's tree classes. Cloudpickle is acceptable here, since the model is our own artifact.

### 4.4 What MLflow tracks

One parent run, `train_pipeline`, holds four nested runs: `elo_baseline`, `hgb_0`, `hgb_1` and `hgb_2`. Beyond scikit-learn autolog, each run records:

- the git commit and a dirty flag, and the DVC md5 of the feature data (which code and which data made the model);
- the row counts and start-time boundaries of each split;
- validation and test log loss and AUC, and training time;
- permutation importances on a 20,000-row validation sample (`permutation_importance.json`);
- the Spark metrics file, and on the parent run the headline metrics and the summary JSON.

## 5. Online request flow

One `/recommend` call fetches the user's live data from Codeforces, builds the same eight features used in training, scores every candidate on the model server, and keeps the top k in the stretch zone. Each failure point maps to its own HTTP code.

1. **Request.** The page on port 8080 calls `GET /recommend?handle=&k=&low=&high=` on the backend (port 8000). Defaults: k = 10 (1 to 50), low = 0.5, high = 0.7.
2. **Validate.** The handle must be 3-24 letters, digits, `_`, `-` or `.`, and low must be smaller than high. Otherwise it returns **400**.
3. **Fetch from Codeforces.** `user.info`, `user.rating`, `user.status` and `problemset.problems`. Calls are serialised at least 2.1 s apart and cached (user data for 10 minutes, the problem set for 6 hours). An unknown handle returns **404**, a user with no rating **422**, and an unreachable API **502**.
4. **Build features** with `src/api/features.py`, which mirrors the Spark definitions. History is the user's last `HISTORY_CONTESTS` rated contests (15 in Docker Compose). A problem counts as attempted if it was in one of those contests, and as solved if it was accepted while the user was a contestant.
5. **Pick candidates:** rated problems the user has never solved, with difficulty from rating − 400 to rating + 600.
6. **Score.** The backend POSTs the feature rows to the model server's `/invocations` in batches of 5,000 (60 s timeout). If the model server fails, it returns **503**.
7. **Rank.** It keeps problems with low ≤ P(solve) ≤ high, sorts them by weakness (1 − mean smoothed rate over the problem's tags), breaks ties by closeness to 0.6, and returns the top k.
8. **Respond and record.** The JSON holds the user's rating, weakest tags, counts of candidates scored and problems in the zone, and the recommendations: name, rating, tags, P(solve), weakness and a link. Latency, status code, error type and the drift gauges are updated for Prometheus.

The first request for a handle takes about 10 s, almost all of it in the rate-limited Codeforces calls. Repeat requests within 10 minutes are served from the cache.

## 6. Design choices and rationale

The parts talk only over REST and are configured only through environment variables, so any one of them can be rebuilt, moved or replaced without touching the others.

| Tool | Used for | Why this tool |
| --- | --- | --- |
| Git | Source code, configs, `dvc.lock`, tests and docs | Every result can be traced to a commit, and the commit hash is logged in MLflow |
| DVC | Pipeline definition (`dvc.yaml`), data and model versioning, the local remote | Reruns only the stages whose inputs changed, and keeps large data out of Git |
| PySpark | Feature engineering | Window functions express the strictly-earlier rule directly, and the same code scales from local mode to a cluster |
| scikit-learn | Gradient-boosting classifier | Histogram gradient boosting is fast on 260k rows and handles mixed-scale features without preprocessing |
| MLflow | Tracking, model registry, model serving, MLproject | One tool covers experiments, versions, the `champion` alias and a REST model server |
| FastAPI | Backend REST API | Validates query parameters and generates the OpenAPI spec and `/docs` page from the code |
| Apache httpd | Static frontend | Serves plain HTML/JS; the page has no build step |
| Docker Compose | Five services on one network | One command starts the whole system, and each service has its own image |
| Prometheus, Grafana | Metrics collection and dashboards | Pull-based scraping of `/metrics`, with a dashboard provisioned from files |
| pytest | Unit and API tests | Runs locally, in the CI script and in the pre-commit hook |

**Principles followed in the code:**

- **Loose coupling.** The frontend knows only `API_URL` (from `config.js`, mounted into the container), and the backend knows only `MODEL_URL`. The backend never loads the model itself.
- **One feature definition.** The Spark job, the training script and the API each list the same eight `FEATURE_COLS`, and a unit test fails if the three lists ever differ (a guard against train/serve skew).
- **Configuration outside the code.** Ports, URLs, history length, CORS origins and file paths are environment variables with defaults.
- **Clear failure modes.** Every external failure maps to its own HTTP code and error counter, and the page shows a plain message for each.
- **Stateless services.** The backend keeps only a short in-memory cache, so restarting a container loses nothing that matters.

**Taught tools not used.** Kubernetes and Minikube were not needed: the system is five containers on one machine, which Docker Compose runs simply. Ray was not needed either, because the feature job already runs on Spark and training on 263,522 rows fits in memory on one machine.

**Programming paradigm: functional.** Feature construction, the Elo baseline and the metrics are pure functions: their output depends only on their input, with no shared state, which is why each can be unit-tested with a few hand-made rows. The Spark job is a chain of DataFrame transformations, and the backend endpoints are functions. Classes are used only where a library expects them: the FastAPI app and three typed exceptions (`ModelServerError`, `CodeforcesError`, `HandleNotFound`).

**Coding style and documentation.** The Python code follows PEP 8 and is checked with `flake8 --max-line-length=100`. Every module starts with a docstring stating its purpose and usage, functions carry docstrings, and non-obvious rules (the leakage window, smoothing, ranking) are explained in comments where they are applied.

**Git LFS not used.** DVC already versions the large files (raw data, features, the exported model) and keeps them out of Git with content hashes, so Git LFS would duplicate that job. Git holds only code, configuration and `dvc.lock`.

## 7. Versioning, reproducibility and CI

Git versions the code, DVC versions the data and the pipeline, and MLflow versions the experiments and models. Two independent training runs on the same data gave identical results: test log loss 0.22121 and AUC 0.96762.

| What | Versioned by | Where |
| --- | --- | --- |
| Code, configs, tests, docs | Git | the repository |
| Pipeline stages and their dependencies | DVC | `dvc.yaml` |
| Hash of every stage's inputs and outputs | DVC | `dvc.lock`, committed to Git |
| Raw data, features, exported model | DVC cache and remote | `dvc push` / `dvc pull` |
| Experiments: params, metrics, artifacts | MLflow tracking | `mlflow.db` and `mlruns/` |
| Models and the `champion` alias | MLflow Model Registry | `next-problem-classifier` |
| Environment | MLproject + `conda.yaml`, Dockerfiles | the repository |

### 7.1 Pipeline (DVC)

`dvc.yaml` chains four stages: **fetch → features → train → export\_model**. `dvc repro` reruns a stage only when one of its dependencies changed, so on an unchanged pipeline every stage is skipped. `spark_metrics.json` and `train_metrics.json` are DVC metrics, so `dvc metrics show` and `dvc metrics diff` compare runs.

### 7.2 Reproducibility

- Every MLflow run is tagged with the git commit and the DVC md5 of its feature data, which identifies exactly what produced it.
- All randomness is seeded (`random_state = 42` for the models, fixed seeds for the permutation-importance sample).
- The **MLproject** file defines entry points `main` (runs `dvc repro`), `fetch`, `features` and `train`. `conda.yaml` pins Python 3.10, OpenJDK 11 (for Spark) and the exact package versions. `mlflow run .` therefore runs in the same environment on any machine.
- The run through `dvc repro` (registry version 2) and the run through `mlflow run` (version 3) produced the same metrics to five decimals.

### 7.3 Continuous integration

Cloud CI services are not allowed, so CI runs locally in two layers.

1. **On every commit:** a versioned Git hook (`scripts/git-hooks/pre-commit`, enabled with `git config core.hooksPath scripts/git-hooks`) runs the unit tests. A failing test blocks the commit.
2. **Full CI** (`scripts/ci.sh`): unit tests → `dvc repro` → `dvc dag` and `dvc metrics show` → an acceptance gate that exits non-zero unless the model beats Elo. A run is saved as evidence in `docs/ci_run.txt`.

Both are plain scripts, so they would move to a hosted CI runner unchanged.

## 8. Monitoring

The backend exposes Prometheus metrics at `/metrics`. Prometheus scrapes them every 15 s, and a provisioned Grafana dashboard, *Next Problem - Monitoring*, shows service health, traffic, latency, errors by type and feature drift.

| Metric | Type | Labels | What it shows |
| --- | --- | --- | --- |
| `np_requests_total` | Counter | endpoint, status | Requests per endpoint and HTTP status |
| `np_request_latency_seconds` | Histogram | endpoint | End-to-end latency (buckets 0.05 s to 80 s) |
| `np_model_latency_seconds` | Histogram | none | Time spent in model-server calls |
| `np_errors_total` | Counter | type | `invalid_handle`, `handle_not_found`, `unrated_user`, `codeforces_api`, `model_server`, `unhandled` |
| `np_train_feature_mean` | Gauge | feature | Training-set mean, from `train_stats.json` |
| `np_live_feature_mean` | Gauge | feature | Mean over the last 50 requests |
| `np_feature_mean_shift` | Gauge | feature | (live mean − training mean) / training mean |

**Drift.** The three gauges track the user-level features `rating` and `n_prev_contests`, one value per request. A shift near 0 means live users look like the training users. A large, lasting shift means the model is seeing a different population and should be retrained.

**Health.** `/health` reports `ok` when the model server answers its own `/health`, and `degraded` otherwise. Docker Compose also health-checks the model server and starts the backend only after it is healthy.

**Logs.** The backend logs one line per request (handle, rating, candidates, problems in the zone, time taken) and an error line for every Codeforces or model-server failure. `docker compose logs backend` shows them.

## 9. Results

The selected model (hgb\_1) passes the acceptance gate: test log loss is 8.5% lower than Elo's, and test AUC is 0.003 higher.

| Metric | Elo baseline | Model (hgb\_1) | Change |
| --- | --- | --- | --- |
| Validation log loss | 0.23227 | 0.21507 | 7.4% lower |
| Validation AUC | 0.96854 | 0.97010 | +0.0016 |
| Test log loss | 0.24179 | 0.22121 | 8.5% lower |
| Test AUC | 0.96459 | 0.96762 | +0.0030 |

The gap is small in AUC because rating minus difficulty already ranks most pairs correctly. The model's gain is mostly in log loss, meaning its probabilities are better calibrated. Calibration matters here, because the stretch zone is defined on the probability itself.

Source: `metrics/train_metrics.json` (git commit `1c8c823`, registry version 2).

## 10. Known limitations and risks

The largest known risk is a gap between training and serving history: live users have more history than the training rows did, and the drift gauges show it.

| Limitation | Effect | Mitigation |
| --- | --- | --- |
| Training history only covers the 68 contests in the dataset (mean `n_prev_contests` 4.6), while serving uses up to 15 of the user's latest rated contests | `n_prev_contests` and `tag_experience` are larger live than in most training rows; the drift gauge shows a large positive shift | Fetch more contests for training, or lower `HISTORY_CONTESTS`; retrain when the shift persists |
| Training uses CF-type contests only; live history includes every rated contest (for example Educational rounds) | Live tag rates mix in contest types the model never saw | Filter live history to CF-type contests, or add those types to training |
| The label is "solved during a contest" | P(solve) is a contest-pressure estimate and likely understates the chance of solving in untimed practice | Stated in the user manual; the zone bounds can be changed with `low` and `high` |
| The cohort is a 10% hash sample of rated participants; first-time participants are excluded | Users with no rated history are not represented; unrated users get 422 | Acceptable for the use case; documented |
| Problems without a difficulty rating (usually new ones) are skipped | The newest problems are never recommended | Picked up automatically once Codeforces rates them |
| Codeforces allows about one call every 2 s, and the cache lives in one backend process | First request for a handle takes about 10 s; a second backend replica would not share the cache | Cache TTLs; a shared cache would be needed before scaling out |
| The backend and model images hold snapshots of the metrics and the model | After retraining, the running system serves the old model until the images are rebuilt | `docker compose up -d --build backend model` after `dvc repro` |
| Dev-only settings: Grafana `admin`/`admin`, CORS `*`, a cloudpickle model | Not safe on a public network | Change before any deployment beyond localhost |
