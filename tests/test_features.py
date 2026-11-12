"""Unit tests for src/api/features.py (live feature construction)."""
import pytest

from src.api import features as f

PROBLEMS = [
    {"contestId": 1, "index": "A", "rating": 800, "tags": ["math"]},
    {"contestId": 1, "index": "B", "rating": 1000, "tags": ["dp"]},
    {"contestId": 2, "index": "A", "rating": 1000, "tags": ["graphs"]},
    {"contestId": 3, "index": "A", "rating": 1200, "tags": []},
]


def test_untagged_problem_gets_none_tag():
    assert f.problem_tags({"tags": []}) == ["none"]
    assert f.problem_tags({}) == ["none"]


def test_index_problems_drops_unrated_and_contestless():
    problems = [
        {"contestId": 1, "index": "A", "rating": 800, "tags": ["math"]},
        {"contestId": 1, "index": "B", "tags": ["dp"]},   # no rating
        {"index": "C", "rating": 900},                     # no contestId
    ]
    by_contest, rated = f.index_problems(problems)
    assert len(rated) == 1
    assert list(by_contest.keys()) == [1]


def test_solved_sets_separates_contest_and_practice():
    subs = [
        {"verdict": "OK", "problem": {"contestId": 1, "index": "A"},
         "author": {"participantType": "CONTESTANT"}},
        {"verdict": "OK", "problem": {"contestId": 1, "index": "B"},
         "author": {"participantType": "PRACTICE"}},
        {"verdict": "WRONG_ANSWER", "problem": {"contestId": 1, "index": "C"},
         "author": {"participantType": "CONTESTANT"}},
        {"verdict": "OK", "problem": {"index": "Z"}, "author": {}},  # no contestId
    ]
    ever, in_contest = f.solved_sets(subs)
    assert ever == {(1, "A"), (1, "B")}
    assert in_contest == {(1, "A")}


def test_tag_history_counts_only_the_given_contests():
    by_contest, _ = f.index_problems(PROBLEMS)
    n, s = f.tag_history([1], by_contest, {(1, "A")})
    assert dict(n) == {"math": 1, "dp": 1}      # contests 2 and 3 are ignored
    assert dict(s) == {"math": 1, "dp": 0}


def test_smoothed_rate_without_and_with_history():
    assert f.smoothed_rate({}, {}, "dp") == pytest.approx(0.3)
    assert f.smoothed_rate({"dp": 10}, {"dp": 5}, "dp") == pytest.approx(6.5 / 15)


def test_build_row_single_tag():
    problem = {"contestId": 1, "index": "B", "rating": 1000, "tags": ["dp"]}
    row = f.build_row(1000, problem, {"dp": 1}, {"dp": 0}, n_prev_contests=1)
    assert set(row) == set(f.FEATURE_COLS)      # exactly the features the model expects
    assert row["rating_diff"] == 0
    assert row["tag_rate_mean"] == pytest.approx(0.25)
    assert row["tag_rate_min"] == pytest.approx(0.25)
    assert row["tag_experience"] == 1
    assert row["n_tags"] == 1
    assert row["n_prev_contests"] == 1


def test_build_row_multiple_tags():
    problem = {"contestId": 1, "index": "B", "rating": 1100, "tags": ["dp", "math"]}
    n, s = {"dp": 1, "math": 1}, {"dp": 0, "math": 1}
    row = f.build_row(1000, problem, n, s, n_prev_contests=1)
    assert row["rating_diff"] == -100
    assert row["tag_rate_mean"] == pytest.approx((0.25 + 2.5 / 6) / 2)
    assert row["tag_rate_min"] == pytest.approx(0.25)
    assert row["tag_experience"] == pytest.approx(1.0)
    assert row["n_tags"] == 2


def test_weakness_is_zero_for_untagged_problems():
    assert f.weakness({"tags": []}, {"dp": 10}, {"dp": 0}) == 0.0


def test_weakness_is_higher_for_weaker_topic():
    n, s = {"dp": 10, "math": 10}, {"dp": 0, "math": 9}
    assert f.weakness({"tags": ["dp"]}, n, s) > f.weakness({"tags": ["math"]}, n, s)


def test_weak_tags_requires_minimum_attempts():
    n, s = {"a": 10, "b": 2}, {"a": 0, "b": 0}
    result = f.weak_tags(n, s, min_attempts=5)
    assert [t["tag"] for t in result] == ["a"]
