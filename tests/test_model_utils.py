"""Tests for the Elo baseline, metric helper and the shared feature contract."""
import numpy as np
import pandas as pd
import pytest

from src.api.features import FEATURE_COLS as API_COLS
from src.features.spark_features import FEATURE_COLS as SPARK_COLS
from src.training.train import FEATURE_COLS as TRAIN_COLS
from src.training.train import elo_proba, score


def test_elo_equal_rating_is_fifty_percent():
    df = pd.DataFrame({"rating": [1500], "difficulty": [1500]})
    assert elo_proba(df)[0] == pytest.approx(0.5)


def test_elo_400_points_above_is_about_91_percent():
    df = pd.DataFrame({"rating": [1900], "difficulty": [1500]})
    assert elo_proba(df)[0] == pytest.approx(1 / 1.1)


def test_elo_decreases_as_problems_get_harder():
    df = pd.DataFrame({"rating": [1500] * 5, "difficulty": [800, 1200, 1500, 1900, 2400]})
    p = elo_proba(df)
    assert np.all(np.diff(p) < 0)


def test_score_on_a_perfect_ranking():
    result = score([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert result["auc"] == pytest.approx(1.0)
    assert result["log_loss"] < 0.3
    assert set(result) == {"log_loss", "auc"}


def test_feature_columns_are_identical_in_spark_training_and_api():
    """Guards against train/serve skew from mismatched column names or order."""
    assert SPARK_COLS == TRAIN_COLS == API_COLS
