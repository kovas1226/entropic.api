import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
from app.quantum_sim import (
    QuantumCircuit,
    DensityMatrixCircuit,
    H, X, CNOT, TOFFOLI, CRx, CRz,
    is_unitary,
    von_neumann_entropy,
    amplitude_damping,
    phase_damping,
    apply_kraus,
    example_bell_and_qft,
    ghz_circuit,
    grover_search,
    teleport,
    visualize_probabilities,
    bloch_coordinates,
)


def test_gate_unitarity():
    for gate in [H, X, CNOT, TOFFOLI, CRx(0.3), CRz(0.5)]:
        assert is_unitary(gate)


def test_state_normalization_after_gates():
    qc = QuantumCircuit(2)
    qc.apply_gate(H, 0)
    qc.apply_gate(CNOT, [0, 1])
    norm = sum(abs(a)**2 for a in qc.state)
    assert abs(norm - 1.0) < 1e-12


def test_measurement_collapse_and_statistics():
    qc = QuantumCircuit(1)
    qc.apply_gate(H, 0)
    counts = {0: 0, 1: 0}
    for _ in range(100):
        outcome = qc.measure(0)
        counts[outcome] += 1
        qc.state = [1, 0]
        qc.apply_gate(H, 0)
    assert abs(counts[0] - counts[1]) < 40


def test_example_bell_and_qft():
    results, final_state = example_bell_and_qft()
    assert len(results) == 2
    assert abs(sum(abs(a)**2 for a in final_state) - 1.0) < 1e-12


def test_von_neumann_entropy_bell():
    qc = QuantumCircuit(2)
    qc.apply_gate(H, 0)
    qc.apply_gate(CNOT, [1, 0])
    s = von_neumann_entropy(qc.state, [0])
    assert abs(s - 1.0) < 1e-6


def test_amplitude_damping_to_ground():
    dc = DensityMatrixCircuit(1)
    dc.apply_gate(H, 0)
    dc.apply_amplitude_damping(1.0, 0)
    assert abs(dc.rho[0][0] - 1.0) < 1e-12
    outcome = dc.measure(0)
    assert outcome == 0


def test_non_adjacent_cnot():
    qc = QuantumCircuit(3)
    qc.apply_gate(H, 0)
    qc.apply_gate(CNOT, [2, 0])  # control qubit0, target qubit2
    entropy = von_neumann_entropy(qc.state, [0])
    assert entropy > 0.9


def test_phase_damping_off_diagonal():
    dc = DensityMatrixCircuit(1)
    dc.apply_gate(H, 0)
    dc.rho = phase_damping(dc.rho, 1.0, 0)
    assert abs(dc.rho[0][1]) < 1e-12


def test_ghz_state():
    qc = ghz_circuit(3)
    amp0 = qc.state[0]
    amp7 = qc.state[7]
    assert abs(abs(amp0) - 1 / math.sqrt(2)) < 1e-6
    assert abs(abs(amp7) - 1 / math.sqrt(2)) < 1e-6


def test_grover_search():
    outcome, state = grover_search([3], 2, iterations=1)
    assert outcome == 3


def test_teleportation():
    m, final_state = teleport([1 / math.sqrt(2), 1 / math.sqrt(2)])
    idx = (m[1] << 1) | m[0]
    amp0 = final_state[idx]
    amp1 = final_state[4 + idx]
    assert abs(amp0 - 1 / math.sqrt(2)) < 1e-6
    assert abs(amp1 - 1 / math.sqrt(2)) < 1e-6


def test_depolarizing_channel():
    dc = DensityMatrixCircuit(1)
    dc.apply_gate(X, 0)
    dc.apply_depolarizing(1.0, 0)
    # Fully mixed -> diagonal elements 0.5
    assert abs(dc.rho[0][0] - 0.5) < 1e-12
    assert abs(dc.rho[1][1] - 0.5) < 1e-12


def test_visualization_helpers():
    qc = QuantumCircuit(1)
    qc.apply_gate(H, 0)
    import io, sys as _sys
    buf = io.StringIO()
    _stdout = _sys.stdout
    _sys.stdout = buf
    visualize_probabilities(qc.state)
    _sys.stdout = _stdout
    out = buf.getvalue().strip().splitlines()
    assert len(out) == 2
    x, y, z = bloch_coordinates(qc.state)
    assert abs(x - 2.0) < 1e-12 and abs(y) < 1e-12 and abs(z) < 1e-12
