"""Thin, polite client for the public Codeforces API, with in-memory caching.

- At most one request every CF_MIN_INTERVAL seconds (Codeforces rate limit).
- Responses are cached with a TTL, so repeated lookups are instant.
"""
import logging
import os
import threading
import time

import requests

log = logging.getLogger("codeforces")

BASE = "https://codeforces.com/api/"
MIN_INTERVAL = float(os.getenv("CF_MIN_INTERVAL", "2.1"))
TIMEOUT = float(os.getenv("CF_TIMEOUT", "60"))
MAX_ATTEMPTS = 3

USER_TTL = 600            # 10 minutes
PROBLEMS_TTL = 6 * 3600   # 6 hours


class CodeforcesError(Exception):
    """The Codeforces API was unreachable or returned an unexpected error."""


class HandleNotFound(CodeforcesError):
    """The requested handle does not exist."""


_lock = threading.Lock()
_last_call = 0.0
_cache: dict = {}


def _get(method: str, ttl: int, **params):
    """GET one API method; returns the 'result' field. Raises CodeforcesError."""
    global _last_call
    key = (method, tuple(sorted(params.items())))
    hit = _cache.get(key)
    if hit and hit[0] > time.time():
        return hit[1]

    comment = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        with _lock:  # serialises calls so the rate limit is respected
            wait = MIN_INTERVAL - (time.time() - _last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                resp = requests.get(BASE + method, params=params, timeout=TIMEOUT)
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                _last_call = time.time()
                raise CodeforcesError(f"Codeforces API unreachable: {exc}") from exc
            _last_call = time.time()

        if data.get("status") == "OK":
            _cache[key] = (time.time() + ttl, data["result"])
            return data["result"]

        comment = str(data.get("comment", ""))
        log.warning("%s failed (attempt %d): %s", method, attempt, comment)
        if "not found" in comment.lower():
            raise HandleNotFound(comment)
        if "limit exceeded" in comment.lower() and attempt < MAX_ATTEMPTS:
            time.sleep(2 * attempt)
            continue
        break
    raise CodeforcesError(comment or "Codeforces API error")


def get_user_info(handle: str) -> dict:
    return _get("user.info", USER_TTL, handles=handle)[0]


def get_user_rating(handle: str) -> list:
    """Rated contests the user took part in, oldest first."""
    return _get("user.rating", USER_TTL, handle=handle)


def get_user_status(handle: str) -> list:
    """All submissions of the user."""
    return _get("user.status", USER_TTL, handle=handle)


def get_problems() -> list:
    """Whole problemset (difficulty rating + tags)."""
    return _get("problemset.problems", PROBLEMS_TTL)["problems"]
