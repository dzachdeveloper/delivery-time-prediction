import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)
NORMAL = {"distance_km": 2.0, "prep_time_min": 12.0, "num_items": 2, "hour": 15, "is_raining": 0, "rider_experience_months": 24.0}
RISKY = {"distance_km": 9.0, "prep_time_min": 20.0, "num_items": 6, "hour": 19, "is_raining": 1, "rider_experience_months": 2.0}


# ---------- given tests ----------
def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info_has_buffer():                               # checks TODO 3c
    body = client.get("/model-info").json()
    assert {"model_source", "buffer_minutes", "features"} <= set(body)


def test_predict_returns_valid_response():
    r = client.post("/predict", json=NORMAL)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_minutes"] > 0
    assert body["promised_minutes"] >= body["predicted_minutes"] - 0.5


def test_long_rainy_rush_order_takes_longer():
    normal = client.post("/predict", json=NORMAL).json()
    slow = client.post("/predict", json=RISKY).json()
    assert slow["predicted_minutes"] > normal["predicted_minutes"]


def test_message_is_filled_in():                                # checks TODO 3b
    body = client.post("/predict", json=NORMAL).json()
    assert body["message"] and "TODO" not in body["message"]


# ---------- your tests ----------
@pytest.mark.parametrize("field, value", [
    ("distance_km", 0.05), ("distance_km", 50.1),
    ("prep_time_min", 0.5), ("prep_time_min", 121),
    ("num_items", 0), ("num_items", 51),
    ("hour", -1), ("hour", 24),
    ("is_raining", -1), ("is_raining", 2),
    ("rider_experience_months", -1), ("rider_experience_months", 481),
])
def test_out_of_range_input_is_rejected(field, value):
    r = client.post("/predict", json={**NORMAL, field: value})
    assert r.status_code == 422, f"{field}={value} should be rejected"


@pytest.mark.parametrize("field", list(NORMAL))
def test_missing_field_is_rejected(field):
    request = dict(NORMAL)
    request.pop(field)
    r = client.post("/predict", json=request)
    assert r.status_code == 422, f"request without {field} should be rejected"


def test_longer_distance_never_shortens_prediction():
    distances = [1, 3, 5, 8, 12, 16, 20]
    preds = [client.post("/predict", json={**NORMAL, "distance_km": d}).json()["predicted_minutes"]
             for d in distances]
    for (d1, p1), (d2, p2) in zip(zip(distances, preds), zip(distances[1:], preds[1:])):
        assert p2 >= p1 - 0.5, f"{d2} km predicted {p2} min, faster than {d1} km at {p1} min"
