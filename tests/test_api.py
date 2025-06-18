import pytest
pytest.importorskip("httpx")
from httpx import AsyncClient
from app.api import app, FateOracle, quantum_seed

@pytest.mark.asyncio
async def test_oracle_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.post("/oracle", json={"question": "How are things?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data and isinstance(data["answer"], str)
    assert len(data["answer"]) > 10

def test_fate_oracle():
    oracle = FateOracle()
    q = "What should I understand about my next relationship?"
    ans = oracle.ask(q)
    assert isinstance(ans, str) and len(ans) > 30

def test_quantum_seed():
    seed, entropy = quantum_seed("test question")
    assert isinstance(seed, int)
    assert isinstance(entropy, float)
