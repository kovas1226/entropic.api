from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_quantum_matrix_calculation():
    response = client.post("/quantum-matrix-calculation", json={"entropy_seed": "test-seed"})
    assert response.status_code == 200
    assert "eigenvalues" in response.json()

def test_quantum_wave_interference():
    response = client.post("/quantum-wave-interference", json={
        "frequencies": [1.0, 2.0, 3.0],
        "time_factor": 0.5
    })
    assert response.status_code == 200
    assert "interference_pattern" in response.json()
