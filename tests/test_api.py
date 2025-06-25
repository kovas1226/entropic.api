import asyncio
import pytest
pytest.importorskip("httpx")
import httpx
from app.api import app
import app.api

async def _request(method: str, url: str, **kwargs):
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        return await ac.request(method, url, **kwargs)

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
    resp = asyncio.run(_request("GET", "/actions"))
    assert resp.status_code == 200
    assert name in resp.json()

def test_perform_answer():
    resp = asyncio.run(_request("POST", "/perform", json={"intent": "answer", "params": {"question": "hi"}}))
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
    resp = asyncio.run(_request("POST", "/perform", json={"intent": intent}))
    assert resp.status_code == 200
    data = resp.json()
    assert expect_keys <= set(data.keys())

def test_fallback_logic(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad")
    monkeypatch.setitem(app.api.ACTIONS, "dream", {"fn": boom, "type": "quantum"})
    resp = asyncio.run(_request("POST", "/perform", json={"intent": "dream"}))
    assert resp.status_code == 200
    data = resp.json()
    assert "symbol" in data and "error" in data

def test_test_actions_endpoint():
    resp = asyncio.run(_request("GET", "/test-actions"))
    assert resp.status_code == 200
    assert "answer" in resp.json()

def test_simulate_and_density_endpoints():
    gates = [{"name": "H", "qubits": [0]}]
    sim = asyncio.run(_request("POST", "/simulate", json={"gates": gates}))
    assert sim.status_code == 200
    assert "state" in sim.json()
    dens = asyncio.run(_request(
        "POST",
        "/density",
        json={"gates": gates, "noise": {"type": "amplitude", "gamma": 0.2, "qubit": 0}},
    ))
    assert dens.status_code == 200
    assert "rho" in dens.json()

def test_entropy_trace_and_log():
    asyncio.run(_request("GET", "/spread"))
    entropy = asyncio.run(_request("POST", "/entropy", json={"subsystem": [0]}))
    assert entropy.status_code == 200
    trace = asyncio.run(_request("GET", "/trace"))
    assert trace.status_code == 200
    log = asyncio.run(_request("GET", "/log"))
    assert log.status_code == 200

def test_symbols_endpoint():
    resp = asyncio.run(_request("GET", "/symbols"))
    assert resp.status_code == 200
    assert "000" in resp.json()

def test_spread_endpoint_structure():
    resp = asyncio.run(_request("GET", "/spread"))
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"root", "challenge", "guide"}
    for part in data.values():
        assert {"bits", "symbol", "entropy"} <= set(part.keys())

def test_ask_endpoint():
    resp = asyncio.run(_request("POST", "/ask", json={"question": "What is blocking me?"}))
    assert resp.status_code == 200
    body = resp.json()
    assert "spread" in body and "summary" in body
    for role in ["root", "challenge", "guide"]:
        assert role in body["spread"]
        block = body["spread"][role]
        assert {"bits", "symbol", "entropy"} <= set(block.keys())

def test_perform_all_registered_actions():
    actions = asyncio.run(_request("GET", "/actions")).json()
    for name in actions:
        payload = {"intent": name, "params": {}}
        if name == "answer":
            payload["params"] = {"question": "hi"}
        elif name == "intent":
            payload["params"] = {"intent": "emergence"}
        elif name in {"simulate", "density", "interpret", "visualize"}:
            payload["params"] = {"gates": []}
        elif name == "entropy":
            payload["params"] = {"subsystem": [0]}
        elif name == "ask":
            payload["params"] = {"question": "test"}
        res = asyncio.run(_request("POST", "/perform", json=payload))
        assert res.status_code == 200
        assert isinstance(res.json(), dict)

def test_life_path():
    res = asyncio.run(_request("POST", "/perform", json={"intent": "life_path"}))
    assert res.status_code == 200
    data = res.json()
    assert "bits" in data and "symbol" in data and "meaning" in data
    assert "label" in data["symbol"]

# ----- NEW TESTS FOR /predict-life -----

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"question": "What should I know this week?"},
        {"question": "Love life?", "seed": 42},
        {"question": "Should I move?", "gates": [{"name": "H", "qubits": [0]}, {"name": "CNOT", "qubits": [1,0]}]},
        {"question": "Career advice?", "symbols": {"010": {"label": "fracture", "tone": "tense", "category": "challenge"}}},
        {"question": "How is my energy?", "seed": 1},
    ]
)
async def test_predict_life_endpoint(payload):
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.post("/predict-life", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], str)
        # Should not be a default or placeholder
        assert "[[" not in data["prediction"]
        # details is optional, but if present, should be a dict
        if "details" in data:
            assert isinstance(data["details"], dict)