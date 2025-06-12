import pytest
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_spread_endpoint():
    response = client.get("/spread")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"root", "challenge", "guide"}
    for v in data.values():
        assert "bits" in v and "symbol" in v and "entropy" in v

def test_intent_endpoint():
    response = client.post("/intent", json={"intent": "emergence"})
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"root", "challenge", "guide"}

def test_meaning_endpoint():
    response = client.get("/meaning/000")
    assert response.status_code == 200
    assert "eigenvalues" in response.json()
    assert response.json()["label"] == "origin"

def test_ask_endpoint():
    response = client.post("/ask", json={"question": "What is blocking me?"})
    assert response.status_code == 200
    body = response.json()
    assert "interference_pattern" in response.json()
    assert "spread" in body
    assert "summary" in body
