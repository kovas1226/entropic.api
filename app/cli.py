import argparse
from .quantum_sim import (
    QuantumCircuit,
    DensityMatrixCircuit,
    H,
    CNOT,
    qft,
    visualize_probabilities,
    grover_search,
    ghz_circuit,
    amplitude_damping,
)


def run_bell(n_qubits: int = 2):
    qc = QuantumCircuit(n_qubits)
    qc.apply_gate(H, 0)
    qc.apply_gate(CNOT, [1, 0])
    print("Bell state after measurement:")
    print(qc.measure(0), qc.measure(1))


def run_qft(n_qubits: int = 3):
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        if q % 2 == 0:
            qc.apply_gate(H, q)
    qft(qc)
    visualize_probabilities(qc.state)


def run_grover():
    qc_result, state = grover_search([3], 2)
    print("Grover search outcome:", qc_result)
    visualize_probabilities(state)


def run_decoherence():
    dc = DensityMatrixCircuit(2)
    dc.apply_gate(H, 0)
    dc.apply_gate(CNOT, [1, 0])
    dc.apply_amplitude_damping(0.3, 0)
    visualize_probabilities([dc.rho[i][i].real for i in range(4)])


def main():
    parser = argparse.ArgumentParser(description="Simple quantum simulator CLI")
    parser.add_argument(
        "circuit",
        choices=["bell", "qft", "grover", "decoherence"],
        help="Circuit to run",
    )
    args = parser.parse_args()
    if args.circuit == "bell":
        run_bell()
    elif args.circuit == "qft":
        run_qft()
    elif args.circuit == "grover":
        run_grover()
    elif args.circuit == "decoherence":
        run_decoherence()


if __name__ == "__main__":
    main()