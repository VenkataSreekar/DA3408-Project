"""Train the Elo baseline and gradient-boosted models, track them in MLflow,
and register the best one.

Usage:
    python src/training/train.py --data data/processed/features.parquet \
        --metrics-out metrics/train_metrics.json

Tracked beyond autolog:
    - git commit hash + dirty flag, and the DVC md5 of the feature file
    - split row counts and the time boundaries between train / val / test
    - permutation importances and the Spark pipeline metrics file
Model selection uses VALIDATION log loss only. Test is reported, never used
to choose. The registered model gets the alias "champion" only if it beats the
Elo baseline on test log loss AND test AUC (the project's acceptance criterion).
"""
import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

# Keep in sync with spark_features.FEATURE_COLS (not imported, to avoid pyspark).
FEATURE_COLS = [
    "rating", "difficulty", "rating_diff",
    "tag_rate_mean", "tag_rate_min", "tag_experience",
    "n_tags", "n_prev_contests",
]
EXPERIMENT = "next-problem"
REGISTERED_NAME = "next-problem-classifier"
CONFIGS = [
    {"learning_rate": 0.10, "max_iter": 100, "max_depth": None},
    {"learning_rate": 0.05, "max_iter": 200, "max_depth": 6},
    {"learning_rate": 0.10, "max_iter": 200, "max_depth": 8},
]


def elo_proba(df: pd.DataFrame) -> np.ndarray:
    """Elo-style baseline: P(solve) = 1 / (1 + 10^((difficulty - rating) / 400))."""
    return (1.0 / (1.0 + 10 ** ((df["difficulty"] - df["rating"]) / 400.0))).to_numpy()


def score(y, p) -> dict:
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)),
    }


def _run(cmd) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def git_info() -> dict:
    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]) or "unknown",
        "git_dirty": str(bool(_run(["git", "status", "--porcelain"]))),
    }


def dvc_data_md5(lock_path: str = "dvc.lock") -> str:
    """md5 of features.parquet as recorded by DVC (identifies the data version)."""
    try:
        lock = yaml.safe_load(Path(lock_path).read_text())
        for out in lock["stages"]["features"]["outs"]:
            if out["path"].endswith("features.parquet"):
                return out["md5"]
    except Exception:
        pass
    return "unknown"


def split_params(parts: dict) -> dict:
    params = {}
    for name, d in parts.items():
        params[f"{name}_rows"] = len(d)
        params[f"{name}_start_min"] = int(d["start"].min())
        params[f"{name}_start_max"] = int(d["start"].max())
    params["n_features"] = len(FEATURE_COLS)
    return params


def log_sklearn_model(model, x_example: pd.DataFrame) -> str:
    """Log the model so that the served endpoint returns probabilities."""
    signature = mlflow.models.infer_signature(x_example, model.predict_proba(x_example))
    kwargs = dict(
        signature=signature,
        input_example=x_example,
        pyfunc_predict_fn="predict_proba",
        serialization_format="cloudpickle",
    )
    try:  # newer MLflow
        info = mlflow.sklearn.log_model(model, name="model", **kwargs)
    except TypeError:  # older MLflow
        info = mlflow.sklearn.log_model(model, artifact_path="model", **kwargs)
    return info.model_uri


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/features.parquet")
    ap.add_argument("--metrics-out", default="metrics/train_metrics.json")
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    ap.add_argument("--spark-metrics", default="data/processed/spark_metrics.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    # float64 everywhere keeps the served model's input schema simple.
    X = {s: d[FEATURE_COLS].astype("float64") for s, d in parts.items()}
    y = {s: d["label"].astype(int).to_numpy() for s, d in parts.items()}
    log.info("Rows: %s", {s: len(d) for s, d in parts.items()})

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    mlflow.sklearn.autolog(
        log_models=False, log_datasets=False, log_post_training_metrics=False
    )

    tags = {**git_info(), "data_md5": dvc_data_md5()}
    data_params = split_params(parts)

    # ---- 1. Elo baseline -------------------------------------------------
    elo = {}
    with mlflow.start_run(run_name="elo_baseline"):
        mlflow.set_tags({**tags, "model_type": "elo_baseline"})
        mlflow.log_params(data_params)
        for s in ("val", "test"):
            elo[s] = score(y[s], elo_proba(parts[s]))
            mlflow.log_metrics({f"{s}_{k}": v for k, v in elo[s].items()})
        if Path(args.spark_metrics).exists():
            mlflow.log_artifact(args.spark_metrics)
    log.info("Elo baseline: %s", elo)

    # ---- 2. Gradient-boosted candidates -----------------------------------
    candidates = []
    for i, cfg in enumerate(CONFIGS):
        with mlflow.start_run(run_name=f"hgb_{i}") as run:
            mlflow.set_tags({**tags, "model_type": "hist_gradient_boosting"})
            mlflow.log_params(data_params)

            model = HistGradientBoostingClassifier(random_state=42, **cfg)
            t0 = time.time()
            model.fit(X["train"], y["train"])
            mlflow.log_metric("train_seconds", time.time() - t0)

            res = {}
            for s in ("val", "test"):
                res[s] = score(y[s], model.predict_proba(X[s])[:, 1])
                mlflow.log_metrics({f"{s}_{k}": v for k, v in res[s].items()})

            # Permutation importance on a validation sample.
            n = min(20000, len(X["val"]))
            idx = np.random.RandomState(0).choice(len(X["val"]), n, replace=False)
            imp = permutation_importance(
                model, X["val"].iloc[idx], y["val"][idx],
                scoring="neg_log_loss", n_repeats=3, random_state=0,
            )
            imp_dict = dict(zip(FEATURE_COLS, map(float, imp.importances_mean)))
            mlflow.log_dict(imp_dict, "permutation_importance.json")
            mlflow.log_metrics({f"perm_imp_{k}": v for k, v in imp_dict.items()})

            model_uri = log_sklearn_model(model, X["val"].head(5))
            candidates.append(
                {"run_id": run.info.run_id, "cfg": cfg, "val": res["val"],
                 "test": res["test"], "model_uri": model_uri}
            )
            log.info("hgb_%d: val=%s test=%s", i, res["val"], res["test"])

    # ---- 3. Select on validation, check acceptance, register -----------------
    best = min(candidates, key=lambda c: c["val"]["log_loss"])
    beats = (
        best["test"]["log_loss"] < elo["test"]["log_loss"]
        and best["test"]["auc"] > elo["test"]["auc"]
    )

    client = MlflowClient()
    client.set_tag(best["run_id"], "selected", "true")
    client.set_tag(best["run_id"], "acceptance_passed", str(beats))
    mv = mlflow.register_model(best["model_uri"], REGISTERED_NAME)
    for k, v in {**tags, "run_id": best["run_id"], "beats_elo_baseline": str(beats)}.items():
        client.set_model_version_tag(REGISTERED_NAME, mv.version, k, v)
    if beats:
        client.set_registered_model_alias(REGISTERED_NAME, "champion", mv.version)
        log.info("Acceptance PASSED: version %s is now 'champion'", mv.version)
    else:
        log.warning("Acceptance FAILED: model does not beat Elo, no champion alias set")

    summary = {
        "elo_baseline": elo,
        "best_model": {
            "run_id": best["run_id"], "config": best["cfg"],
            "val": best["val"], "test": best["test"],
        },
        "beats_baseline": beats,
        "registered_version": int(mv.version),
        "git_commit": tags["git_commit"],
        "data_md5": tags["data_md5"],
    }
    out = Path(args.metrics_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("Summary written to %s", out)


if __name__ == "__main__":
    main()
