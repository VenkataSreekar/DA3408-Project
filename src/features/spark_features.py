"""PySpark feature job for Next Problem.

Reads raw Codeforces JSON (from fetch_codeforces.py) and writes one row per
(user, problem, contest) with a leak-free feature set and a time-based split.

Usage:
    python src/features/spark_features.py --raw data/raw --out data/processed

Outputs (in --out):
    features.parquet      model-ready rows (columns listed in FEATURE_COLS + meta)
    spark_metrics.json    rows, runtime, throughput (for the evaluation sheet)
    train_stats.json      training-set feature means (used by the drift gauge)
"""
import argparse
import json
import logging
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("spark_features")

# Smoothing so users with no history get a neutral per-tag rate.
PRIOR_RATE = 0.3
PRIOR_STRENGTH = 5
# Fractions of CONTESTS (by time) used for train / val; the rest is test.
TRAIN_Q, VAL_Q = 0.70, 0.85

FEATURE_COLS = [
    "rating", "difficulty", "rating_diff",
    "tag_rate_mean", "tag_rate_min", "tag_experience",
    "n_tags", "n_prev_contests",
]


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("next-problem-features")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )


def load_base(spark: SparkSession, raw: str) -> DataFrame:
    """Return one row per (contest, user, problem).

    Columns: contest_id, start, handle, rating, prob_index, difficulty,
             tags (array), label (1 solved / 0 not).
    """
    standings = spark.read.option("multiLine", True).json(f"{raw}/contests/*_standings.json")
    ratings = spark.read.option("multiLine", True).json(f"{raw}/contests/*_ratings.json")
    problems = spark.read.option("multiLine", True).json(f"{raw}/problems.json")

    # Problem metadata: difficulty + tags (empty tag list -> ["none"]).
    problems = problems.select(
        F.col("contestId").alias("contest_id"),
        F.col("index").alias("prob_index"),
        F.col("rating").alias("difficulty"),
        F.when(F.size("tags") == 0, F.array(F.lit("none"))).otherwise(F.col("tags")).alias("tags"),
    ).filter(F.col("difficulty").isNotNull())

    contest = standings.select(
        F.col("contest.id").alias("contest_id"),
        F.col("contest.startTimeSeconds").alias("start"),
        "problems", "rows",
    )

    # Position of each problem inside the contest (matches position in `solved`).
    prob_pos = (
        contest.select("contest_id", F.posexplode("problems").alias("pos", "p"))
        .select("contest_id", "pos", F.col("p.index").alias("prob_index"))
        .join(problems, ["contest_id", "prob_index"], "inner")
    )

    users = (
        contest.select("contest_id", "start", F.explode("rows").alias("r"))
        .select(
            "contest_id", "start",
            F.col("r.handle").alias("handle"),
            F.posexplode("r.solved").alias("pos", "solved"),
        )
    )

    # Pre-contest rating; oldRating == 0 means a first-time participant -> dropped.
    pre_rating = ratings.select(
        F.col("contestId").alias("contest_id"),
        "handle",
        F.col("oldRating").alias("rating"),
    ).filter(F.col("rating") > 0)

    return (
        users.join(pre_rating, ["contest_id", "handle"], "inner")
        .join(prob_pos, ["contest_id", "pos"], "inner")
        .withColumn("label", F.col("solved").cast("int"))
        .select("contest_id", "start", "handle", "rating",
                "prob_index", "difficulty", "tags", "label")
    )


def build_features(base: DataFrame) -> DataFrame:
    """Add leak-free history features.

    LEAKAGE RULE: every history feature for contest C uses only contests that
    started STRICTLY BEFORE C (window frame ends at -1 row, and rows are
    aggregated per contest first, so the current contest is never included).
    """
    exploded = base.select(
        "contest_id", "start", "handle", "prob_index", "label",
        F.explode("tags").alias("tag"),
    )

    # One row per (user, tag, contest) with that contest's attempts / solves.
    per_ct = exploded.groupBy("handle", "tag", "contest_id", "start").agg(
        F.count("*").alias("n"), F.sum("label").alias("s")
    )
    w_tag = (
        Window.partitionBy("handle", "tag")
        .orderBy("start", "contest_id")
        .rowsBetween(Window.unboundedPreceding, -1)  # strictly earlier contests
    )
    prior = (
        per_ct
        .withColumn("prior_n", F.coalesce(F.sum("n").over(w_tag), F.lit(0)))
        .withColumn("prior_s", F.coalesce(F.sum("s").over(w_tag), F.lit(0)))
        .withColumn(
            "tag_rate",
            (F.col("prior_s") + PRIOR_STRENGTH * PRIOR_RATE)
            / (F.col("prior_n") + PRIOR_STRENGTH),
        )
        .select("handle", "tag", "contest_id", "tag_rate", "prior_n")
    )

    # Collapse tags back to one row per (contest, user, problem).
    per_problem = (
        exploded.join(prior, ["handle", "tag", "contest_id"], "inner")
        .groupBy("contest_id", "handle", "prob_index")
        .agg(
            F.avg("tag_rate").alias("tag_rate_mean"),
            F.min("tag_rate").alias("tag_rate_min"),
            F.avg("prior_n").alias("tag_experience"),
            F.count("*").alias("n_tags"),
        )
    )

    # How many contests the user had played before this one.
    user_contests = base.select("handle", "contest_id", "start").distinct()
    w_user = Window.partitionBy("handle").orderBy("start", "contest_id")
    user_contests = user_contests.withColumn(
        "n_prev_contests", F.row_number().over(w_user) - 1
    ).select("handle", "contest_id", "n_prev_contests")

    return (
        base.drop("tags")
        .join(per_problem, ["contest_id", "handle", "prob_index"], "inner")
        .join(user_contests, ["handle", "contest_id"], "inner")
        .withColumn("rating_diff", F.col("rating") - F.col("difficulty"))
    )


def add_time_split(feat: DataFrame) -> DataFrame:
    """Split by contest start time: oldest 70% train, next 15% val, last 15% test."""
    starts = feat.select("contest_id", "start").distinct()
    q_train, q_val = starts.approxQuantile("start", [TRAIN_Q, VAL_Q], 0.0)
    return feat.withColumn(
        "split",
        F.when(F.col("start") <= q_train, "train")
        .when(F.col("start") <= q_val, "val")
        .otherwise("test"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    t0 = time.time()
    feat = add_time_split(build_features(load_base(spark, args.raw)))
    feat.write.mode("overwrite").parquet(str(out / "features.parquet"))
    elapsed = time.time() - t0

    # Re-read the written data for cheap, reliable statistics.
    df = spark.read.parquet(str(out / "features.parquet"))
    rows = df.count()
    split_counts = {r["split"]: r["count"] for r in df.groupBy("split").count().collect()}
    metrics = {
        "rows": rows,
        "contests": df.select("contest_id").distinct().count(),
        "users": df.select("handle").distinct().count(),
        "label_rate": df.agg(F.avg("label")).first()[0],
        "split_rows": split_counts,
        "elapsed_seconds": round(elapsed, 2),
        "rows_per_second": round(rows / elapsed, 1),
    }
    (out / "spark_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Training-set means for the Prometheus drift gauge.
    train = df.filter(F.col("split") == "train")
    stats = train.agg(*[F.avg(c).alias(c) for c in FEATURE_COLS]).first().asDict()
    (out / "train_stats.json").write_text(json.dumps(stats, indent=2))

    log.info("Metrics: %s", metrics)
    spark.stop()


if __name__ == "__main__":
    main()
