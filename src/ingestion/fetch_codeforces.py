"""Fetch raw Codeforces data (problems, contests, standings, rating changes).

Usage:
    python src/ingestion/fetch_codeforces.py --n-contests 70 --users 1500 --out data/raw

Notes:
- Idempotent: files that already exist are skipped, so re-running resumes.
- Polite: sleeps between calls (Codeforces allows ~1 request / 2 s).
- Users are a stable ~10% cohort chosen by hashing the handle (crc32 % 10 == 0),
  so the same users appear across contests and all skill levels are covered.
  At most `--users` cohort members are kept per contest.
"""
import argparse
import json
import logging
import time
from pathlib import Path
import zlib

import requests

BASE = "https://codeforces.com/api/"
SLEEP_SECONDS = 2.1
MAX_RETRIES = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch")


def call(method: str, **params):
    """Call a Codeforces API method with retries; return the 'result' field."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(SLEEP_SECONDS)
            resp = requests.get(BASE + method, params=params, timeout=180)
            data = resp.json()
            if data.get("status") == "OK":
                return data["result"]
            log.warning("%s failed: %s", method, data.get("comment"))
            # Permanent errors (e.g. no rating changes) should not be retried.
            if "not found" in str(data.get("comment", "")).lower():
                return None
        except (requests.RequestException, ValueError) as exc:
            log.warning("%s attempt %d error: %s", method, attempt, exc)
        time.sleep(attempt * 3)
    return None


def save(path: Path, obj) -> None:
    path.write_text(json.dumps(obj))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-contests", type=int, default=70)
    ap.add_argument("--users", type=int, default=1500)
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "contests").mkdir(parents=True, exist_ok=True)

    # 1. Problem metadata (difficulty rating + tags)
    problems_file = out / "problems.json"
    if not problems_file.exists():
        result = call("problemset.problems")
        if result is None:
            raise SystemExit("Could not fetch problemset")
        save(problems_file, result["problems"])
        log.info("Saved %d problems", len(result["problems"]))

    # 2. Contest list, newest finished CF-style contests first
    contests_file = out / "contests.json"
    if not contests_file.exists():
        result = call("contest.list", gym="false")
        if result is None:
            raise SystemExit("Could not fetch contest list")
        save(contests_file, result)
    contests = json.loads(contests_file.read_text())
    candidates = [c for c in contests if c["phase"] == "FINISHED" and c["type"] == "CF"]
    candidates.sort(key=lambda c: c["startTimeSeconds"], reverse=True)

    # 3. Standings + rating changes per contest
    done = 0
    for c in candidates:
        if done >= args.n_contests:
            break
        cid = c["id"]
        st_file = out / "contests" / f"{cid}_standings.json"
        rc_file = out / "contests" / f"{cid}_ratings.json"
        if st_file.exists() and rc_file.exists():
            done += 1
            continue

        ratings = call("contest.ratingChanges", contestId=cid)
        if not ratings:
            log.info("Contest %s has no rating changes, skipping", cid)
            continue
        standings = call("contest.standings", contestId=cid)
        if not standings:
            continue

        # Stable ~10% user cohort (same users across contests, all skill levels)
        cohort = [
            r for r in standings["rows"]
            if zlib.crc32(r["party"]["members"][0]["handle"].encode()) % 10 == 0
        ][: args.users]
        cohort_handles = {r["party"]["members"][0]["handle"] for r in cohort}

        # Keep only cohort users' rating changes
        save(rc_file, [x for x in ratings if x["handle"] in cohort_handles])

        # Keep only what we need: problems + (handle, per-problem points)
        slim = {
            "contest": standings["contest"],
            "problems": standings["problems"],
            "rows": [
                {
                    "handle": r["party"]["members"][0]["handle"],
                    "rank": r["rank"],
                    "solved": [pr["points"] > 0 for pr in r["problemResults"]],
                }
                for r in cohort
            ],
        }
        save(st_file, slim)
        done += 1
        log.info("Contest %s saved (%d/%d)", cid, done, args.n_contests)

    log.info("Finished: %d contests in %s", done, out)


if __name__ == "__main__":
    main()
