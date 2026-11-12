"""Export the registry's 'champion' model into model/ (the folder Docker builds from)."""
import shutil
from pathlib import Path

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
shutil.rmtree("model", ignore_errors=True)
mlflow.artifacts.download_artifacts(
    artifact_uri="models:/next-problem-classifier@champion", dst_path="model")
print("exported:", sorted(p.name for p in Path("model").iterdir()))
