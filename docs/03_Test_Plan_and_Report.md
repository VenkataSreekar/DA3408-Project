# Next Problem: Test Plan, Test Cases and Test Report

Nov 18, 2026 · Venkata Sreekar

## 1. Test plan

Testing has three levels: 35 automated tests for code correctness, a model acceptance test against the Elo baseline, and system tests on the running Docker stack. All 35 automated tests pass, and the model passes its acceptance test.

### 1.1 Scope

| In scope | Out of scope |
| --- | --- |
| Live feature construction (`src/api/features.py`) | The Codeforces API itself (external, replaced by fakes) |
| Spark feature job: leakage rules, time split, data quality checks | The internals of the MLflow model server (an off-the-shelf component) |
| Elo baseline, metric helper, feature-list contract | Load and stress testing |
| Every backend endpoint and every error path; service health probes | Browser compatibility beyond a current Chrome/Firefox |
| Model quality against the Elo baseline |  |
| The deployed system: services, page, monitoring, alerts |  |

### 1.2 Test levels

| Level | What | How | Where it runs |
| --- | --- | --- | --- |
| Unit | Feature functions, Elo, metrics, Spark job, data quality checks | pytest with small hand-made inputs whose answers are computed by hand | `tests/test_features.py`, `test_model_utils.py`, `test_spark_features.py` |
| API (integration) | All four endpoints, all error codes, ranking logic | FastAPI `TestClient`; Codeforces and the model server replaced by fakes (pytest `monkeypatch`), so no network is needed | `tests/test_api.py` |
| Model acceptance | Model vs Elo on held-out contests | Automatic check at the end of training, repeated by the CI gate | `train.py`, `scripts/ci.sh` |
| System (end to end) | The running stack: five containers, page, errors, monitoring | Manual checks with the browser and `curl` | `docker compose up -d` |

### 1.3 Approach

- **Leakage is the main risk,** so it gets dedicated tests. A label inside the same contest must never change that contest's features.
- **Expected values are computed by hand.** For example, the smoothed rate with 1 solve in 2 attempts is (1 + 1.5) / (2 + 5) = 2.5 / 7. The tests check the code against that number, not against its own output.
- **Every error path has a test.** Each HTTP error code (400, 404, 422, 502, 503) is forced with a fake and checked.
- **Train/serve skew guard:** a test asserts that the Spark job, training and API use identical feature lists, in the same order.
- **Data quality checks are tested both ways:** they must pass on clean data and must catch a missing value, an impossible difficulty and a contest placed in two splits.

### 1.4 Tools and environment

| Item | Version |
| --- | --- |
| Python | 3.10.22 (conda env `nextproblem`) |
| pytest | 9.2.0 |
| PySpark | local mode, `local[1]` for tests |
| FastAPI test client | Starlette `TestClient` (httpx) |
| OS | Linux (ASUS Zenbook 14) |
| System tests | Docker Compose, 5 services; Chrome/Firefox |

### 1.5 Entry and exit criteria

**Entry:** the code builds, the conda environment from `conda.yaml` is installed, and `dvc repro` has produced the data and model.

**Exit:** all automated tests pass, the model passes acceptance, and every system test case passes.

### 1.6 When tests run

1. **On every commit:** a Git pre-commit hook runs `pytest -q`, and a failing test blocks the commit.
2. **On demand:** `scripts/ci.sh` runs the tests, `dvc repro`, the metrics and the acceptance gate.
3. **Before submission:** the system test cases in section 4, on a fresh `docker compose up -d`; most are automated by `scripts/system_tests.sh`.

## 2. Acceptance criteria

The system is accepted when all ten criteria below hold. All ten are met.

| ID | Criterion | Evidence | Result |
| --- | --- | --- | --- |
| AC-1 | All automated tests pass | 35 of 35 passed (section 5) | Pass |
| AC-2 | The selected model has lower test log loss than Elo | 0.22121 vs 0.24179 | Pass |
| AC-3 | The selected model has higher test AUC than Elo | 0.96762 vs 0.96459 | Pass |
| AC-4 | The split is by time, and no contest is in two splits | T-S5; train/val/test contests ordered by start time | Pass |
| AC-5 | No feature uses information from its own or a later contest | T-S1, T-S2, T-S4 | Pass |
| AC-6 | Every invalid input and every failed dependency returns its documented HTTP code and a readable message | T-A3 to T-A8 | Pass |
| AC-7 | A model gets the `champion` alias only if AC-2 and AC-3 hold | `train.py` gate; CI step 4 printed ACCEPTANCE PASSED | Pass |
| AC-8 | The deployed system serves recommendations for a real rated handle, and monitoring records the traffic | System tests ST-1 to ST-9 | Pass |
| AC-9 | The training data passes every data quality check, and a failed check stops the pipeline | T-S6, T-S7; 11 of 11 checks in `spark_metrics.json` | Pass |
| AC-10 | Every component is monitored, and an outage raises an alert | T-A13; ST-13, ST-14 | Pass |

AC-2 and AC-3 are checked twice: by `train.py`, before it sets the `champion` alias, and by `scripts/ci.sh`, which exits with an error if `beats_baseline` is false.

## 3. Automated test cases

The 35 cases cover four areas, and all of them pass. Each table lists the pytest function, what it checks, the input and the expected result.

### 3.1 Live features: `tests/test_features.py` (10 cases)

| ID | Test | Input | Expected | Result |
| --- | --- | --- | --- | --- |
| T-F1 | `test_untagged_problem_gets_none_tag` | Problem with empty or missing tags | Tags = `["none"]` | Pass |
| T-F2 | `test_index_problems_drops_unrated_and_contestless` | 3 problems: one valid, one without rating, one without contest id | 1 rated problem, indexed under contest 1 only | Pass |
| T-F3 | `test_solved_sets_separates_contest_and_practice` | Accepted in contest, accepted in practice, wrong answer, no contest id | Ever solved = {1A, 1B}; solved in contest = {1A} | Pass |
| T-F4 | `test_tag_history_counts_only_the_given_contests` | History = contest 1 only, 1A solved | n = {math: 1, dp: 1}, s = {math: 1, dp: 0}; contests 2, 3 ignored | Pass |
| T-F5 | `test_smoothed_rate_without_and_with_history` | No history; then 5 solves in 10 attempts | 0.3; then 6.5 / 15 | Pass |
| T-F6 | `test_build_row_single_tag` | Rating 1000, dp problem rated 1000, dp history 0/1 | Exactly the 8 features; rating\_diff 0, tag rates 0.25, experience 1 | Pass |
| T-F7 | `test_build_row_multiple_tags` | Problem with dp and math, rated 1100 | rating\_diff −100; mean of 0.25 and 2.5/6; min 0.25; n\_tags 2 | Pass |
| T-F8 | `test_weakness_is_zero_for_untagged_problems` | Untagged problem | weakness = 0.0 | Pass |
| T-F9 | `test_weakness_is_higher_for_weaker_topic` | dp 0/10 solved, math 9/10 solved | weakness(dp) > weakness(math) | Pass |
| T-F10 | `test_weak_tags_requires_minimum_attempts` | Tag a: 10 attempts; tag b: 2 attempts | Only tag a is reported | Pass |

### 3.2 Model utilities: `tests/test_model_utils.py` (5 cases)

| ID | Test | Input | Expected | Result |
| --- | --- | --- | --- | --- |
| T-M1 | `test_elo_equal_rating_is_fifty_percent` | Rating 1500, difficulty 1500 | P = 0.5 | Pass |
| T-M2 | `test_elo_400_points_above_is_about_91_percent` | Rating 1900, difficulty 1500 | P = 1 / 1.1 ≈ 0.909 | Pass |
| T-M3 | `test_elo_decreases_as_problems_get_harder` | Rating 1500, difficulties 800 to 2400 | P strictly decreasing | Pass |
| T-M4 | `test_score_on_a_perfect_ranking` | Labels \[0,1,0,1\], predictions \[0.1,0.9,0.2,0.8\] | AUC = 1.0, log loss < 0.3, keys {log\_loss, auc} | Pass |
| T-M5 | `test_feature_columns_are_identical_in_spark_training_and_api` | The three `FEATURE_COLS` lists | Identical names and order | Pass |

### 3.3 Backend API: `tests/test_api.py` (13 cases)

Codeforces is replaced by a fake user rated 1000 who played contest 1 and solved 1A in it. The model server is replaced by fixed probabilities.

| ID | Test | Input | Expected | Result |
| --- | --- | --- | --- | --- |
| T-A1 | `test_health_reports_ok_when_model_server_is_up` | Model server answers 200 | `{"status": "ok", "model_server": "up"}` | Pass |
| T-A2 | `test_health_reports_degraded_when_model_server_is_down` | Model server connection error | `{"status": "degraded", "model_server": "down"}` | Pass |
| T-A3 | `test_invalid_handle_returns_400` | handle = `ab` | 400 | Pass |
| T-A4 | `test_bad_probability_zone_returns_400` | low 0.8, high 0.5 | 400 | Pass |
| T-A5 | `test_unknown_handle_returns_404` | Codeforces: handle not found | 404 | Pass |
| T-A6 | `test_unrated_user_returns_422` | User without a rating | 422 | Pass |
| T-A7 | `test_codeforces_outage_returns_502` | Codeforces unreachable | 502 | Pass |
| T-A8 | `test_model_server_outage_returns_503` | Model server refuses the connection | 503 | Pass |
| T-A9 | `test_recommendations_are_in_zone_and_ranked_by_weakness` | 4 candidates, P = 0.60, 0.55, 0.90, 0.65 | 4 scored, 3 in zone; order 1B (dp), 2A (graphs), 3A (untagged); all P in \[0.5, 0.7\] | Pass |
| T-A10 | `test_k_limits_the_number_of_results` | k = 1 | Exactly 1 recommendation | Pass |
| T-A11 | `test_metrics_endpoint_exposes_prometheus_counters` | GET /metrics | Contains `np_requests_total` and `np_errors_total` | Pass |
| T-A12 | `test_pipeline_endpoint_returns_stages_and_service_status` | GET /pipeline, model server up | 6 stages starting with Fetch; status ok; `data` and `model` keys present | Pass |
| T-A13 | `test_metrics_reports_service_health` | GET /metrics; frontend answers 200, model server answers 500 | `np_service_up` is 1 for frontend and 0 for model_server | Pass |

### 3.4 Spark feature job: `tests/test_spark_features.py` (7 cases)

Hand-made data: one user plays contest 1 (problems A solved and B unsolved, both dp), then contest 2 (C: dp; D: dp and math).

| ID | Test | Input | Expected | Result |
| --- | --- | --- | --- | --- |
| T-S1 | `test_first_contest_has_no_history` | Contest 1 rows | experience 0, tag rate 0.3 (prior), 0 earlier contests | Pass |
| T-S2 | `test_history_uses_only_strictly_earlier_contests` | Problem 2C | experience 2 (A, B only); rate 2.5 / 7; 1 earlier contest; rating\_diff 250 | Pass |
| T-S3 | `test_multi_tag_problem_averages_tag_rates` | Problem 2D (dp, math) | mean of 2.5/7 and 0.3; min 0.3; experience 1.0; 2 tags | Pass |
| T-S4 | `test_features_do_not_depend_on_labels_of_the_same_contest` | Same data with 2C solved, then unsolved | All contest-2 features identical (no leakage) | Pass |
| T-S5 | `test_time_split_is_ordered_and_contests_never_straddle` | 10 contests, start times 100 to 1000 | All three splits present; train < val < test by start time | Pass |
| T-S6 | `test_quality_checks_pass_on_clean_data` | Valid 6-row table, one contest per split | All 11 checks pass | Pass |
| T-S7 | `test_quality_checks_catch_bad_data` | Same table with a null tag rate, difficulty 5000, and a contest in two splits | `no_missing_values`, `difficulties_in_range` and `no_contest_in_two_splits` fail; unaffected checks still pass | Pass |

## 4. System test cases

These checks run on the deployed stack after `docker compose up -d`. All 15 have been run and passed. ST-2 to ST-6, ST-9 and ST-14 are automated by `scripts/system_tests.sh`; the page-level parts of ST-3 to ST-6 are quick manual checks in the browser.

| ID | Case | Steps | Expected | Result |
| --- | --- | --- | --- | --- |
| ST-1 | Stack starts | `docker compose up -d`, then `docker compose ps` | 5 services Up; model server healthy | Pass |
| ST-2 | Health check | `curl localhost:8000/health` | `{"status":"ok","model_server":"up"}` | Pass |
| ST-3 | Recommendations for a real user | Open http://localhost:8080, enter `Venkata_Sreekar`, press Get problems | Weakest topics and up to 10 problems, each with solve chance from 50% to 70%; names open the problem on Codeforces | Pass |
| ST-4 | Unknown handle | Enter `zzz_nobody_here_987` | Message: the handle was not found on Codeforces | Pass |
| ST-5 | Invalid handle | Enter `ab` | Message at once, without a request: invalid handle, use 3-24 letters, digits, '\_', '-' or '.' | Pass |
| ST-6 | Backend unavailable | `docker compose stop backend`, search again, then `docker compose start backend` | Message: cannot reach the recommendation service; works again after restart | Pass |
| ST-7 | Pipeline status page | Open http://localhost:8080/pipeline.html | Service cards, six stage cards with real numbers, Elo vs model table | Pass |
| ST-8 | API documentation | Open http://localhost:8000/docs | `/health`, `/metrics`, `/recommend`, `/pipeline` each listed once | Pass |
| ST-9 | Monitoring records traffic | Run ST-3 and ST-4, and `curl "localhost:8000/recommend?handle=ab"`; wait 30 s; open Grafana (http://localhost:3001) and Prometheus targets (http://localhost:9090/targets) | Request count rises; errors by type show `handle_not_found` and `invalid_handle`; drift panels have values; targets backend, prometheus and grafana all UP; service status shows model_server and frontend UP | Pass |
| ST-10 | CI pipeline | `scripts/ci.sh` | Tests pass, DVC stages skipped or rerun, ACCEPTANCE PASSED, CI passed | Pass |
| ST-11 | Reproducible training | `mlflow run . -e train --env-manager local` | Same test metrics as `dvc repro` (log loss 0.22121, AUC 0.96762) | Pass |
| ST-12 | Tests run on commit | `git commit` | Pre-commit hook runs pytest before the commit is recorded | Pass |
| ST-13 | Alert rules load | `docker compose exec prometheus promtool check rules /etc/prometheus/alert_rules.yml` | `SUCCESS: 7 rules found` | Pass |
| ST-14 | Outage raises an alert | `docker compose stop model`; wait 2 minutes; check http://localhost:9090/alerts and the Grafana *Alerts firing* panel; `docker compose start model` | `ModelServerDown` firing; *Alerts firing* at least 1; clears after the restart | Pass |
| ST-15 | Fresh clone works | `git clone` the repository into a new folder, `dvc pull`, `pytest -q`, `docker compose up -d --build` | All data and the model fetched from the DVC remote; 35 tests pass; all five services start | Pass |

Evidence:

- **`docs/system_test_run.txt`** (2026-11-18, commit `b7e9204`): 12 of 12 automated checks passed.
    - ST-3 returned 10 problems for a user rated 1441, all between 50% and 70%, with weakest topics *data structures*, *combinatorics* and *graphs* (559 problems in the zone).
    - ST-9 showed all three scrape targets up, `np_service_up` 1 for the model server and frontend, and the error types `handle_not_found` and `invalid_handle` recorded.
    - In ST-14, `ModelServerDown` was pending after 30 s and firing after 90 s, and cleared 30 s after the model server restarted. While it was down, `/health` reported `degraded`.
- **ST-1 and ST-13:** after rebuilding the stack, all five containers started and the model server was healthy; promtool printed `SUCCESS: 7 rules found`.
- **ST-10:** `docs/ci_run.txt`. 35 tests passed, all 11 data quality checks passed, the model was retrained as registry version 4, and acceptance passed.
- **ST-11:** identical numbers in registry versions 2, 3 and 4.
- **ST-12:** the pre-commit hook ran the test suite before each commit.
- **ST-15:** a fresh clone in `/tmp/np-clean` fetched 188 files with `dvc pull`, passed 35 tests and started all five services.

## 5. Test report

All 35 automated tests passed with 0 failures, and the model passed both acceptance metrics. All 15 system test cases passed, and all ten acceptance criteria are met.

**Run details:** 2026-11-18, Python 3.10.22, pytest 9.2.0, PySpark 3.5.1; the suite took 25.3 s. On the real data, all 11 data quality checks passed (339,576 rows).

### 5.1 Summary

| Level | Cases | Passed | Failed | Not run |
| --- | --- | --- | --- | --- |
| Automated: live features | 10 | 10 | 0 | 0 |
| Automated: model utilities | 5 | 5 | 0 | 0 |
| Automated: backend API | 13 | 13 | 0 | 0 |
| Automated: Spark feature job | 7 | 7 | 0 | 0 |
| **Automated total** | **35** | **35** | **0** | **0** |
| Model acceptance (AC-2, AC-3) | 2 | 2 | 0 | 0 |
| Data quality checks on the real data | 11 | 11 | 0 | 0 |
| System (section 4) | 15 | 15 | 0 | 0 |

Update the system row once the remaining cases are run.

### 5.2 Model acceptance results

| Metric (test split) | Elo baseline | Model (hgb\_1) | Required | Result |
| --- | --- | --- | --- | --- |
| Log loss | 0.24179 | 0.22121 | lower than Elo | Pass |
| AUC | 0.96459 | 0.96762 | higher than Elo | Pass |

The test split holds the newest 15% of contests, 31,820 rows, which no part of training or model selection saw.

### 5.3 Warnings

One warning appears on every run: Starlette's test client says using `httpx` is deprecated. It comes from the test library, not from project code, and does not affect any result.

### 5.4 Defects found and fixed

| # | Defect | Found by | Fix |
| --- | --- | --- | --- |
| 1 | The `/pipeline` endpoint and two config lines were defined twice in `main.py`, so the route was registered twice | Code review | Duplicates removed; commit `3bc6821` |
| 2 | The DVC graph had a disconnected `model.dvc` node, so the exported model was not part of the pipeline | Reviewing `dvc dag` in the CI output | Model export made a DVC stage (`export_model`); commit `4621416` |
| 3 | The fetch script's docstring described top-ranked sampling, but the code samples a hash-based cohort | Code review | Docstring corrected; commit `3bc6821` |

**Open known issue (not a test failure):** live history uses up to 15 recent contests of any type, while training history covers fewer contests and CF-type rounds only. The drift gauges show this. HLD section 10 describes it and its mitigation.

### 5.5 Evidence files

| File | Contents |
| --- | --- |
| `docs/test_report.txt` | Full pytest output, one line per test |
| `docs/test_report.xml` | The same results in JUnit XML format |
| `docs/ci_run.txt` | A full CI run: tests, `dvc repro`, DAG, metrics, acceptance gate |
| `data/processed/spark_metrics.json` | Results of the 11 data quality checks on the real data |
| `docs/system_test_run.txt` | Output of `scripts/system_tests.sh`: the automated system tests |
| `metrics/train_metrics.json` | Model and Elo scores used in 5.2 |

To reproduce: `pytest -v` from the repository root, or `scripts/ci.sh` for the full CI.
