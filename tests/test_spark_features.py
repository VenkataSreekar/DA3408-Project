"""Tests for the Spark feature job, centred on the no-leakage guarantee.

Hand-made data: user u1 plays contest 1 (start=100) and contest 2 (start=200).
  contest 1: A (dp, solved), B (dp, not solved)
  contest 2: C (dp), D (dp + math)
So for contest 2, the dp history is 2 attempts / 1 solve, and math has none.
"""
import pytest
from pyspark.sql import SparkSession

from src.features.spark_features import add_time_split, build_features

SCHEMA = (
    "contest_id long, start long, handle string, rating long, prob_index string, "
    "difficulty long, tags array<string>, label int"
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]").appName("unit-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def features(spark, label_c=1):
    rows = [
        (1, 100, "u1", 1200, "A", 1000, ["dp"], 1),
        (1, 100, "u1", 1200, "B", 1100, ["dp"], 0),
        (2, 200, "u1", 1250, "C", 1000, ["dp"], label_c),
        (2, 200, "u1", 1250, "D", 1200, ["dp", "math"], 0),
    ]
    out = build_features(spark.createDataFrame(rows, SCHEMA))
    return {(r["contest_id"], r["prob_index"]): r.asDict() for r in out.collect()}


def test_first_contest_has_no_history(spark):
    feats = features(spark)
    for key in [(1, "A"), (1, "B")]:
        assert feats[key]["tag_experience"] == 0
        assert feats[key]["tag_rate_mean"] == pytest.approx(0.3)
        assert feats[key]["n_prev_contests"] == 0


def test_history_uses_only_strictly_earlier_contests(spark):
    c = features(spark)[(2, "C")]
    assert c["tag_experience"] == 2                       # A and B only, not C or D
    assert c["tag_rate_mean"] == pytest.approx(2.5 / 7)   # (1 solved + 1.5) / (2 + 5)
    assert c["n_prev_contests"] == 1
    assert c["rating_diff"] == 250


def test_multi_tag_problem_averages_tag_rates(spark):
    d = features(spark)[(2, "D")]
    assert d["tag_rate_mean"] == pytest.approx((2.5 / 7 + 0.3) / 2)
    assert d["tag_rate_min"] == pytest.approx(0.3)
    assert d["tag_experience"] == pytest.approx(1.0)      # (2 + 0) / 2
    assert d["n_tags"] == 2


def test_features_do_not_depend_on_labels_of_the_same_contest(spark):
    """Flipping a label inside contest 2 must not change any contest-2 feature."""
    solved = features(spark, label_c=1)
    unsolved = features(spark, label_c=0)
    for key in [(2, "C"), (2, "D")]:
        for col in ["tag_rate_mean", "tag_rate_min", "tag_experience", "n_prev_contests"]:
            assert solved[key][col] == pytest.approx(unsolved[key][col])


def test_time_split_is_ordered_and_contests_never_straddle(spark):
    df = spark.createDataFrame([(i, i * 100) for i in range(1, 11)],
                               "contest_id long, start long")
    rows = add_time_split(df).collect()
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r["start"])
    assert set(by_split) == {"train", "val", "test"}
    assert max(by_split["train"]) < min(by_split["val"])
    assert max(by_split["val"]) < min(by_split["test"])
