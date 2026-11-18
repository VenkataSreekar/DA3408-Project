# AI Disclosure

**Project:** Next Problem: A Personalized Codeforces Practice Recommender
**Author:** Venkata Sreekar (DA24B030)
**AI tool used:** Claude (Anthropic), through claude.ai

I used claude for the code and documentation also sometimes for debugging. I ran everything on my own machine, tested it, and reviewed the results. All numbers reported in the documents come from those runs, not from the AI.

## What the AI was used for

| Area | What Claude did |
| --- | --- |
| Codeforces data | Explained which API methods to use (`problemset.problems`, `contest.list`, `contest.standings`, `contest.ratingChanges`, `user.info`, `user.rating`, `user.status`), the rate limit, how contest standings give labels, and the hash-based user sample; wrote the fetch script and the backend's Codeforces client |
| Data engineering | Wrote the PySpark feature job: leak-free per-tag history, time-based split, data quality checks |
| Model training | Wrote the training script: Elo baseline, gradient boosting, acceptance gate, MLflow tracking and registry |
| Backend and frontend | Wrote the FastAPI backend, the web pages and the Pipeline status page |
| Testing | Wrote the 35 pytest tests and the system test script |
| Debugging | Diagnosed errors from terminal output I pasted and suggested fixes |
| Documentation | Drafted the HLD, LLD, test plan and report, user manual, README and viva notes, and checked them against the course guidelines |

## Verification

- All code was run on my machine before submission.
- 35 automated tests pass, including leakage tests against expected values worked out by hand.
- `scripts/ci.sh` reruns the pipeline and fails if the model does not beat the Elo baseline.
- `scripts/system_tests.sh` and manual checks in the browser verified the running system, including an outage that fires an alert.
- A fresh clone set up with `dvc pull` and `docker compose up --build` passed the same tests.
- Every reported number comes from files produced by my runs: `metrics/train_metrics.json`, `data/processed/spark_metrics.json`, `docs/test_report.txt`, `docs/ci_run.txt` and `docs/system_test_run.txt`.

I understand the design, the code and the tools used, and I am responsible for the submitted work.
