import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
import api

client = TestClient(api.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict():
    payload = {
        "age": 24.5, "height_in_cm": 178.0, "international_caps": 54.0,
        "international_goals": 14.0, "position": "Attack",
        "sub_position": "Right Winger", "foot": "left",
        "country_of_citizenship": "England", "league": "GB1",
        "contract_months": 51.7, "cum_matches": 290.0, "cum_goals": 78.0,
        "cum_assists": 72.0, "cum_minutes": 21857.0, "num_transfers": 4.0,
        "prev_value": 130000000.0, "peak_so_far": 150000000.0,
        "days_since_prev": 90.0, "player_highest_mv": 150000000.0,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    val = r.json()["predicted_market_value_eur"]
    assert val > 0
    assert val < 1e9


def test_predict_defaults():
    r = client.post("/predict", json={"age": 20.0})
    assert r.status_code == 200
    assert r.json()["predicted_market_value_eur"] > 0
