"""Live feature construction for one user.

MUST mirror src/features/spark_features.py, otherwise the model sees different
numbers at serving time than it saw in training (train/serve skew):

  - a problem counts as "attempted" if it was in a rated contest the user played
    (and has a difficulty rating), "solved" if accepted DURING that contest;
  - per-tag rate = (solved + PRIOR_STRENGTH * PRIOR_RATE) / (attempted + PRIOR_STRENGTH);
  - problems without tags get the single tag "none".

Pure functions only, so they are easy to unit test with hand-made data.
"""
from collections import defaultdict

FEATURE_COLS = [
    "rating", "difficulty", "rating_diff",
    "tag_rate_mean", "tag_rate_min", "tag_experience",
    "n_tags", "n_prev_contests",
]
PRIOR_RATE = 0.3
PRIOR_STRENGTH = 5


def problem_tags(problem: dict) -> list:
    tags = problem.get("tags") or []
    return list(tags) if tags else ["none"]


def index_problems(problems: list):
    """Return (problems grouped by contest id, list of all rated problems)."""
    by_contest = defaultdict(list)
    rated = []
    for p in problems:
        if p.get("rating") is None or p.get("contestId") is None:
            continue
        by_contest[p["contestId"]].append(p)
        rated.append(p)
    return by_contest, rated


def solved_sets(submissions: list):
    """Return (solved ever, solved during a contest) as sets of (contestId, index)."""
    ever, in_contest = set(), set()
    for sub in submissions:
        if sub.get("verdict") != "OK":
            continue
        prob = sub.get("problem", {})
        if prob.get("contestId") is None:
            continue
        key = (prob["contestId"], prob["index"])
        ever.add(key)
        if sub.get("author", {}).get("participantType") == "CONTESTANT":
            in_contest.add(key)
    return ever, in_contest


def tag_history(contest_ids: list, by_contest: dict, solved_in_contest: set):
    """Per-tag attempted (n) and solved (s) counts over the given contests."""
    n, s = defaultdict(int), defaultdict(int)
    for cid in contest_ids:
        for p in by_contest.get(cid, []):
            solved = int((cid, p["index"]) in solved_in_contest)
            for tag in problem_tags(p):
                n[tag] += 1
                s[tag] += solved
    return n, s


def weakness(problem: dict, n: dict, s: dict) -> float:
    """1 - mean smoothed solve rate over the problem's REAL tags (0 if untagged)."""
    tags = problem.get("tags") or []
    if not tags:
        return 0.0
    return 1 - sum(smoothed_rate(n, s, t) for t in tags) / len(tags)


def smoothed_rate(n: dict, s: dict, tag: str) -> float:
    return (s.get(tag, 0) + PRIOR_STRENGTH * PRIOR_RATE) / (n.get(tag, 0) + PRIOR_STRENGTH)


def build_row(rating: float, problem: dict, n: dict, s: dict, n_prev_contests: int) -> dict:
    """Feature row for one (user, candidate problem) pair."""
    tags = problem_tags(problem)
    rates = [smoothed_rate(n, s, t) for t in tags]
    difficulty = problem["rating"]
    return {
        "rating": rating,
        "difficulty": difficulty,
        "rating_diff": rating - difficulty,
        "tag_rate_mean": sum(rates) / len(rates),
        "tag_rate_min": min(rates),
        "tag_experience": sum(n.get(t, 0) for t in tags) / len(tags),
        "n_tags": len(tags),
        "n_prev_contests": n_prev_contests,
    }


def weak_tags(n: dict, s: dict, min_attempts: int = 5, k: int = 3) -> list:
    """The user's weakest tags (lowest smoothed solve rate) with enough history."""
    scored = [
        {"tag": t, "rate": round(smoothed_rate(n, s, t), 3), "attempts": n[t]}
        for t in n if n[t] >= min_attempts
    ]
    return sorted(scored, key=lambda d: d["rate"])[:k]
