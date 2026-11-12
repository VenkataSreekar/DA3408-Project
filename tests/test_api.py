"""API tests. Codeforces and the model server are replaced by fakes, so these
run offline and fast."""
import pytest
import requests
from fastapi.testclient import TestClient

from src.api import main

client = TestClient(main.app)

# History: the user played contest 1 and solved 1A during the contest.
PROBLEMS = [
    {"contestId": 1, "index": "A", "name": "A1", "rating": 800, "tags": ["math"]},
    {"contestId": 1, "index": "B", "name": "B1", "rating": 1000, "tags": ["dp"]},
    {"contestId": 2, "index": "A", "name": "A2", "rating": 1000, "tags": ["graphs"]},
    {"contestId": 2, "index": "B", "name": "B2", "rating": 1100, "tags": ["math"]},
    {"contestId": 3, "index": "A", "name": "A3", "rating": 1200, "tags": []},
]
SUBMISSIONS = [
    {"verdict": "OK", "problem": {"contestId": 1, "index": "A"},
     "author": {"participantType": "CONTESTANT"}},
]


def fake_codeforces(monkeypatch, rating=1000):
    monkeypatch.setattr(main.cf, "get_user_info", lambda h: {"handle": h, "rating": rating})
    monkeypatch.setattr(main.cf, "get_user_rating", lambda h: [{"contestId": 1}])
    monkeypatch.setattr(main.cf, "get_user_status", lambda h: SUBMISSIONS)
    monkeypatch.setattr(main.cf, "get_problems", lambda: PROBLEMS)


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


def test_health_reports_ok_when_model_server_is_up(monkeypatch):
    monkeypatch.setattr(main.requests, "get", lambda *a, **k: FakeResponse(200))
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_server": "up"}


def test_health_reports_degraded_when_model_server_is_down(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(main.requests, "get", boom)
    body = client.get("/health").json()
    assert body == {"status": "degraded", "model_server": "down"}


def test_invalid_handle_returns_400():
    assert client.get("/recommend", params={"handle": "ab"}).status_code == 400


def test_bad_probability_zone_returns_400():
    r = client.get("/recommend", params={"handle": "alice", "low": 0.8, "high": 0.5})
    assert r.status_code == 400


def test_unknown_handle_returns_404(monkeypatch):
    def not_found(handle):
        raise main.cf.HandleNotFound("handles: User with handle alice not found")
    monkeypatch.setattr(main.cf, "get_user_info", not_found)
    r = client.get("/recommend", params={"handle": "alice"})
    assert r.status_code == 404


def test_unrated_user_returns_422(monkeypatch):
    monkeypatch.setattr(main.cf, "get_user_info", lambda h: {"handle": h})
    assert client.get("/recommend", params={"handle": "alice"}).status_code == 422


def test_codeforces_outage_returns_502(monkeypatch):
    def down(handle):
        raise main.cf.CodeforcesError("unreachable")
    monkeypatch.setattr(main.cf, "get_user_info", down)
    assert client.get("/recommend", params={"handle": "alice"}).status_code == 502


def test_model_server_outage_returns_503(monkeypatch):
    fake_codeforces(monkeypatch)

    def down(rows):
        raise main.ModelServerError("connection refused")
    monkeypatch.setattr(main, "predict", down)
    assert client.get("/recommend", params={"handle": "alice"}).status_code == 503


def test_recommendations_are_in_zone_and_ranked_by_weakness(monkeypatch):
    fake_codeforces(monkeypatch)

    def fake_predict(rows):
        # Candidates in order: 1B, 2A, 2B, 3A (1A was already solved).
        assert len(rows) == 4
        return [0.60, 0.55, 0.90, 0.65]
    monkeypatch.setattr(main, "predict", fake_predict)

    body = client.get("/recommend", params={"handle": "alice", "k": 10}).json()
    assert body["candidates_scored"] == 4
    assert body["in_zone"] == 3                      # 2B (0.90) is outside 50-70%
    order = [(r["contest_id"], r["index"]) for r in body["recommendations"]]
    # weakness: 1B (dp, 0.75) > 2A (graphs, 0.70) > 3A (untagged, 0.0)
    assert order == [(1, "B"), (2, "A"), (3, "A")]
    assert all(0.5 <= r["p_solve"] <= 0.7 for r in body["recommendations"])


def test_k_limits_the_number_of_results(monkeypatch):
    fake_codeforces(monkeypatch)
    monkeypatch.setattr(main, "predict", lambda rows: [0.6, 0.55, 0.9, 0.65])
    body = client.get("/recommend", params={"handle": "alice", "k": 1}).json()
    assert len(body["recommendations"]) == 1


def test_metrics_endpoint_exposes_prometheus_counters():
    text = client.get("/metrics").text
    assert "np_requests_total" in text
    assert "np_errors_total" in text
