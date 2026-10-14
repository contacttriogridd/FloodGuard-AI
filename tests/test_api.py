"""API contract: valid requests, 422 on bad input, 413 on oversized payloads, 400 on bad CSV."""
from __future__ import annotations

import importlib
import io
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(bundle_dir):
    os.environ.update({"FG_ARTIFACTS_DIR": str(bundle_dir), "FG_MAX_BATCH": "5", "FG_MAX_CSV_ROWS": "50",
                       "FG_MAX_CSV_BYTES": "6000", "FG_LOCAL_MAX_EVALS": "41"})
    import api.main as main

    main = importlib.reload(main)  # read the env vars above
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def record(factors):
    return {f: 5 for f in factors}


def test_health_and_model_info(client):
    assert client.get("/health").json()["model_loaded"] is True
    info = client.get("/model-info").json()
    assert len(info["factors"]) == 20 and info["bounds"]["Urbanization"] == [0, 16]


def test_valid_prediction(client, record):
    res = client.post("/predict", json=record)
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["flood_probability"] <= 1
    assert body["risk_class"] in {"Low", "Medium", "High"}
    lo, hi = body["confidence_interval"]
    assert lo <= body["flood_probability"] <= hi
    assert len(body["top_drivers"]) == 5 and body["recommended_actions"]


def test_out_of_range_value_is_422(client, record):
    res = client.post("/predict", json={**record, "Urbanization": 17})
    assert res.status_code == 422
    assert res.json()["errors"][0]["field"] == "Urbanization"


def test_extra_field_is_422(client, record):
    assert client.post("/predict", json={**record, "Rainfall": 3}).status_code == 422


def test_non_integer_and_missing_field_are_422(client, record):
    assert client.post("/predict", json={**record, "Siltation": 4.5}).status_code == 422
    partial = dict(record)
    partial.pop("Landslides")
    assert client.post("/predict", json=partial).status_code == 422


def test_batch_limit_is_413(client, record):
    ok = client.post("/predict/batch", json={"records": [record] * 5})
    assert ok.status_code == 200 and len(ok.json()) == 5
    assert client.post("/predict/batch", json={"records": [record] * 6}).status_code == 413


def _csv(factors, rows: int, value: int = 5) -> bytes:
    header = "id," + ",".join(factors)
    body = "\n".join(f"{i}," + ",".join(str(value) for _ in factors) for i in range(rows))
    return f"{header}\n{body}\n".encode()


def test_csv_round_trip(client, factors):
    res = client.post("/predict/csv", files={"file": ("x.csv", io.BytesIO(_csv(factors, 3)), "text/csv")})
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/csv")
    lines = res.text.strip().splitlines()
    assert "flood_probability" in lines[0] and len(lines) == 4


def test_oversized_csv_is_413(client, factors):
    big = _csv(factors, 200)  # > 6000 bytes
    assert client.post("/predict/csv", files={"file": ("big.csv", io.BytesIO(big), "text/csv")}).status_code == 413


def test_bad_csv_is_400(client, factors):
    missing = "id,Urbanization\n1,5\n".encode()
    assert client.post("/predict/csv", files={"file": ("m.csv", io.BytesIO(missing), "text/csv")}).status_code == 400
    bad_value = _csv(factors, 2, value=99)
    assert client.post("/predict/csv", files={"file": ("v.csv", io.BytesIO(bad_value), "text/csv")}).status_code == 400


def test_explain_returns_all_contributions(client, record):
    body = client.post("/explain", json=record).json()
    assert len(body["all_contributions"]) == 20 and len(body["top_drivers"]) == 5
