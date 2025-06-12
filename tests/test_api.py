import pytest
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from app.main import app
import app.api

client = TestClient(app)

@pytest.mark.parametrize(
    "name",
    [
        "answer",
        "simulate",
        "reflect",
        "contemplate",
        "analyze",
        "visualize",
        "teleport_thought",
        "dream",
        "summon",
        "judge_consistency",
        "phase_read",
        "life_path",
        "past_present_future",
        "randaunaut",
        "divine_coords",
        "predict",
        "reveal",
        "warn",
        "insight",
        "symbolize",
        "scrye",
    ],
)
def test_actions_available(name):
    resp = client.get("/actions")
    assert resp.status_code == 200
    assert name in resp.json()


def test_perform_answer():
    resp = client.post("/perform", json={"intent": "answer", "params": {"question": "hi"}})
    assert resp.status_code == 200
    assert "symbol" in resp.json()


@pytest.mark.parametrize(
    "intent,expect_keys",
    [
        ("teleport_thought", {"bell", "bits", "symbol"}),
        ("dream", {"bits", "symbol", "entropy"}),
        ("summon", {"bits", "symbol", "entropy"}),
        ("judge_consistency", {"constant", "balanced"}),
        ("phase_read", {"phase_bits", "phase", "symbol"}),
        ("life_path", {"bits", "symbol", "entropy", "meaning"}),
        ("past_present_future", {"past", "present", "future"}),
        ("randaunaut", {"bits", "symbol", "entropy", "meaning"}),
        ("divine_coords", {"bits", "symbol", "entropy", "meaning"}),
        ("predict", {"bits", "symbol", "entropy", "meaning"}),
        ("reveal", {"bits", "symbol", "entropy", "meaning"}),
        ("warn", {"bits", "symbol", "entropy", "meaning"}),
        ("insight", {"bits", "symbol", "entropy", "meaning"}),
        ("symbolize", {"bits", "symbol", "entropy", "meaning"}),
        ("scrye", {"bits", "symbol", "entropy", "meaning"}),
    ],
)
def test_perform_new_actions(intent, expect_keys):
    resp = client.post("/perform", json={"intent": intent})
    assert resp.status_code == 200
    data = resp.json()
    assert expect_keys <= set(data.keys())


def test_fallback_logic(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad")

    monkeypatch.setitem(app.api.ACTIONS, "dream", {"fn": boom, "type": "quantum"})
    resp = client.post("/perform", json={"intent": "dream"})
    assert resp.status_code == 200
    data = resp.json()
    assert "symbol" in data and "error" in data


def test_test_actions_endpoint():
    resp = client.get("/test-actions")
    assert resp.status_code == 200
    assert "answer" in resp.json()


def test_simulate_and_density_endpoints():
    gates = [{"name": "H", "qubits": [0]}]
    sim = client.post("/simulate", json={"gates": gates})
    assert sim.status_code == 200
    assert "state" in sim.json()

    dens = client.post(
        "/density",
        json={"gates": gates, "noise": {"type": "amplitude", "gamma": 0.2, "qubit": 0}},
    )
    assert dens.status_code == 200
    assert "rho" in dens.json()


def test_entropy_trace_and_log():
    _ = client.get("/spread")
    entropy = client.post("/entropy", json={"subsystem": [0]})
    assert entropy.status_code == 200
    trace = client.get("/trace")
    assert trace.status_code == 200
    log = client.get("/log")
    assert log.status_code == 200


def test_symbols_endpoint():
    resp = client.get("/symbols")
    assert resp.status_code == 200
    assert "000" in resp.json()




