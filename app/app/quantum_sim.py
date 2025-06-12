# Simple quantum circuit simulator without external dependencies
"""Quantum Circuit Simulator.

This module implements a lightweight quantum circuit simulator designed for
educational purposes and quick algorithm prototyping.  The implementation is
completely self contained and does not rely on external numerical libraries
such as NumPy or Qiskit.  While small, it tries to model essential quantum
mechanics faithfully: state normalisation is enforced, all standard gates are
checked for unitarity and measurement follows the Born rule.  Two circuit
representations are provided: :class:`QuantumCircuit` for pure state vectors and
``DensityMatrixCircuit`` for mixed state evolution with basic noise models.

The simulator exposes helper routines for common algorithms such as the quantum
Fourier transform (QFT) and also utilities for building new gates.  Additional
reference implementations of Grover's search, the Deutsch–Jozsa algorithm,
quantum teleportation and GHZ state preparation are included at the end of the
file with extensive documentation.  A very small command line interface is
provided in :mod:`app.cli` for running demonstration circuits from the shell.

The intent of this module is clarity over performance.  All linear algebra is
implemented in pure Python using nested lists to represent matrices and state
vectors.  For pedagogical clarity many routines include explicit loops instead
of list comprehensions.  Advanced features such as custom Kraus operators,
entanglement entropy computation and simple visualisation helpers are also
implemented directly in Python.

The remainder of this module acts as a small reference manual.  Each public
function and class is documented with a detailed docstring including usage
examples.  Feel free to read through the source code to better understand how
the various pieces fit together.  None of the routines rely on advanced Python
features; everything is written using basic language constructs so it can be
followed by readers with minimal background.

The simulator supports the following features:

* Arbitrary single-qubit and multi-qubit gates specified as matrices
* Application of gates to non-adjacent qubits in little-endian ordering
* Measurement and collapse of both pure states and density matrices
* Basic noise models such as amplitude and phase damping
* Computation of von Neumann entropy for subsystems
* Simple text-based visualisations of probability distributions and Bloch
  vectors
* Implementations of well known quantum algorithms (QFT, Grover, Deutsch–Jozsa,
  teleportation and GHZ state preparation)

The design goal is to keep the code below 1000 lines while demonstrating real
quantum mechanical effects.  Because the implementation is pure Python it is
not suitable for large numbers of qubits, but it is perfectly adequate for
showcasing algorithms on up to half a dozen qubits which is plenty for many
educational examples.

You are encouraged to play with the functions interactively from a Python shell
or notebook.  For instance::

    >>> from app.quantum_sim import QuantumCircuit, H, CNOT, visualize_probabilities
    >>> qc = QuantumCircuit(2)
    >>> qc.apply_gate(H, 0)
    >>> qc.apply_gate(CNOT, [1, 0])
    >>> visualize_probabilities(qc.state)

The above will display the familiar 50/50 Bell state probabilities.  Additional
examples are provided within the docstrings of the algorithm helper functions
near the end of this file.
"""

import random
import math
import cmath
from typing import Callable, Dict, List, Sequence, Tuple, Optional

# Utility checks

def is_unitary(matrix: Sequence[Sequence[complex]], tol: float = 1e-10) -> bool:
    """Return ``True`` if ``matrix`` is unitary.

    Parameters
    ----------
    matrix:
        Square complex matrix encoded as a list of lists representing the
        operator in the computational basis.
    tol:
        Numerical tolerance for the unitarity check.

    A matrix ``U`` is unitary when ``U\u2020 U = I``.  The function performs a
    straightforward check by explicitly forming the product using pure Python
    arithmetic.  It is therefore not optimised for speed but keeps the
    implementation transparent for educational purposes.
    """
    size = len(matrix)
    for i in range(size):
        for j in range(size):
            val = sum(matrix[k][i].conjugate() * matrix[k][j] for k in range(size))
            if i == j:
                if abs(val - 1) > tol:
                    return False
            else:
                if abs(val) > tol:
                    return False
    return True


def is_hermitian(matrix: Sequence[Sequence[complex]], tol: float = 1e-10) -> bool:
    """Return ``True`` if ``matrix`` is Hermitian.

    This helper is primarily used to sanity check Hamiltonians and observables
    used in demonstrations.  The function simply compares each element to the
    complex conjugate of its transpose element within a tolerance.
    """
    size = len(matrix)
    for i in range(size):
        for j in range(size):
            if abs(matrix[i][j] - matrix[j][i].conjugate()) > tol:
                return False
    return True


def matrix_multiply(a: Sequence[Sequence[complex]], b: Sequence[Sequence[complex]]) -> List[List[complex]]:
    """Return product ``a`` ``b`` of two square matrices.

    The matrices are represented as nested lists ``[[...], [...]]`` and are
    assumed to have equal dimensions.  This naive implementation uses triple
    nested ``for`` loops and therefore has ``O(n^3)`` complexity.  It is,
    however, sufficient for the small matrix sizes considered in this
    simulator.
    """
    size = len(a)
    result = [[0j for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for k in range(size):
            for j in range(size):
                result[i][j] += a[i][k] * b[k][j]
    return result


def compose_gates(*gates: Sequence[Sequence[complex]]) -> List[List[complex]]:
    """Return the composition ``g_n \u2218 ... \u2218 g_0`` as a single matrix."""
    result = gates[0]
    for g in gates[1:]:
        result = matrix_multiply(g, result)
    return result


def normalize_state(state: Sequence[complex]) -> List[complex]:
    """Return the given state vector normalised to unit length."""
    norm = math.sqrt(sum(abs(a) ** 2 for a in state))
    if norm == 0:
        raise ValueError("Zero norm state")
    return [a / norm for a in state]


def von_neumann_entropy(state: Sequence[complex], subsystem: Sequence[int]) -> float:
    """Return the von Neumann entropy of the given subsystem of a pure state."""
    n = int(math.log2(len(state)))
    subsystem = list(subsystem)
    other = [i for i in range(n) if i not in subsystem]
    dim_sub = 1 << len(subsystem)
    rho = [[0j for _ in range(dim_sub)] for _ in range(dim_sub)]
    for idx, amp_i in enumerate(state):
        env_i = tuple(((idx >> q) & 1) for q in other)
        sub_i = sum(((idx >> q) & 1) << p for p, q in enumerate(subsystem))
        for jdx, amp_j in enumerate(state):
            env_j = tuple(((jdx >> q) & 1) for q in other)
            if env_i != env_j:
                continue
            sub_j = sum(((jdx >> q) & 1) << p for p, q in enumerate(subsystem))
            rho[sub_i][sub_j] += amp_i * amp_j.conjugate()
    # analytic eigenvalues for 2x2 matrix, else crude power iteration
    eigs: List[float] = []
    if dim_sub == 2:
        a, b = rho[0]
        c, d = rho[1]
        tr = a + d
        det = a * d - b * c
        term = cmath.sqrt(tr * tr - 4 * det)
        eigs = [((tr + term) / 2).real, ((tr - term) / 2).real]
    else:
        size = dim_sub
        working = [row[:] for row in rho]
        for _ in range(size):
            vec = [random.random() for _ in range(size)]
            for _ in range(30):
                new = [sum(working[i][j] * vec[j] for j in range(size)) for i in range(size)]
                norm = math.sqrt(sum(abs(x) ** 2 for x in new))
                if norm == 0:
                    break
                vec = [x / norm for x in new]
            val = sum(vec[i].conjugate() * sum(working[i][j] * vec[j] for j in range(size)) for i in range(size))
            eigs.append(val.real)
            for i in range(size):
                for j in range(size):
                    working[i][j] -= val * vec[i] * vec[j].conjugate()
    entropy = 0.0
    for lam in eigs:
        if lam > 1e-12:
            entropy -= lam * math.log(lam, 2)
    return entropy


# Basic gate application routines

def apply_single_qubit_gate(state: Sequence[complex], gate: Sequence[Sequence[complex]], qubit: int) -> List[complex]:
    """Apply a single-qubit ``gate`` to ``qubit`` in ``state``.

    Parameters
    ----------
    state:
        Current state vector encoded as a list of complex amplitudes.
    gate:
        ``2x2`` unitary matrix to apply.
    qubit:
        Index of the qubit to operate on with ``0`` being the least significant
        qubit.
    """
    step = 1 << qubit
    new_state = [0j] * len(state)
    for i in range(0, len(state), 2 * step):
        for j in range(step):
            idx0 = i + j
            idx1 = idx0 + step
            a0 = state[idx0]
            a1 = state[idx1]
            new_state[idx0] = gate[0][0] * a0 + gate[0][1] * a1
            new_state[idx1] = gate[1][0] * a0 + gate[1][1] * a1
    return new_state


def apply_multi_qubit_gate(state: Sequence[complex], gate: Sequence[Sequence[complex]], qubits: Sequence[int]) -> List[complex]:
    """Apply ``gate`` acting on the given ``qubits`` of ``state``.

    ``gate`` must be a ``2^k x 2^k`` unitary acting on ``k`` qubits.  Qubits are
    specified as a sequence of indices with the least significant qubit being
    index ``0``.  The implementation works for arbitrary qubit positions and is
    used internally by most high-level gate routines.
    """
    k = len(qubits)
    mask = 0
    for q in qubits:
        mask |= 1 << q

    new_state = [0j] * len(state)
    for basis_index, amplitude in enumerate(state):
        gate_index = 0
        for pos, q in enumerate(qubits):
            if basis_index & (1 << q):
                gate_index |= 1 << pos

        base = basis_index & ~mask
        for target in range(1 << k):
            new_basis = base
            for pos, q in enumerate(qubits):
                if target & (1 << pos):
                    new_basis |= 1 << q
            new_state[new_basis] += gate[target][gate_index] * amplitude

    return new_state


def expand_unitary(gate: Sequence[Sequence[complex]], qubits: Sequence[int], n: int) -> List[List[complex]]:
    """Expand a small ``gate`` acting on ``qubits`` to an ``n``-qubit operator."""
    size = 1 << n
    full = [[0j for _ in range(size)] for _ in range(size)]
    for basis in range(size):
        state = [0j] * size
        state[basis] = 1
        if len(qubits) == 1:
            out = apply_single_qubit_gate(state, gate, qubits[0])
        else:
            out = apply_multi_qubit_gate(state, gate, qubits)
        for j, amp in enumerate(out):
            full[j][basis] = amp
    return full

def expand_matrix(mat: Sequence[Sequence[complex]], qubits: Sequence[int], n: int) -> List[List[complex]]:
    """Expand ``mat`` acting on ``qubits`` to a full ``n``-qubit operator."""
    size = 1 << n
    full = [[0j for _ in range(size)] for _ in range(size)]
    for basis in range(size):
        state = [0j] * size
        state[basis] = 1
        out = apply_multi_qubit_gate(state, mat, qubits) if len(qubits) > 1 else apply_single_qubit_gate(state, mat, qubits[0])
        for j, amp in enumerate(out):
            full[j][basis] = amp
    return full


def apply_density_gate(rho: List[List[complex]], gate: Sequence[Sequence[complex]], qubits: Sequence[int]) -> List[List[complex]]:
    """Apply a unitary ``gate`` to ``rho`` acting on the given ``qubits``."""
    n = int(math.log2(len(rho)))
    U = expand_unitary(gate, qubits, n)
    Udg = [[U[j][i].conjugate() for j in range(len(U))] for i in range(len(U))]
    temp = matrix_multiply(U, rho)
    return matrix_multiply(temp, Udg)

def apply_kraus(
    rho: List[List[complex]],
    operators: Sequence[Sequence[Sequence[complex]]],
    qubits: Sequence[int],
) -> List[List[complex]]:
    r"""Apply a general quantum channel defined by Kraus operators.

    Each element of ``operators`` is a matrix representing one Kraus operator
    acting on a subset of qubits.  The function expands each operator to the
    full system size and evaluates ``\sum_i E_i rho E_i^\dagger``.  No
    assumptions about trace preservation are made.
    """
    n = int(math.log2(len(rho)))
    size = 1 << n
    result = [[0j for _ in range(size)] for _ in range(size)]
    for op in operators:
        E = expand_matrix(op, qubits, n)
        Edg = [[E[j][i].conjugate() for j in range(size)] for i in range(size)]
        temp = matrix_multiply(E, rho)
        contrib = matrix_multiply(temp, Edg)
        for i in range(size):
            for j in range(size):
                result[i][j] += contrib[i][j]
    return result


def amplitude_damping(rho: List[List[complex]], gamma: float, qubit: int) -> List[List[complex]]:
    """Apply amplitude damping channel to ``qubit`` with damping rate ``gamma``.

    This models energy relaxation where ``|1\rangle`` decays to ``|0\rangle``
    with probability ``gamma``.  ``rho`` is assumed to be a full density matrix
    and a new matrix is returned.  The operation is completely positive and
    trace preserving for ``0 \u2264 gamma \u2264 1``.
    """
    size = len(rho)
    new_rho = [[0j for _ in range(size)] for _ in range(size)]
    sqrt1 = math.sqrt(1 - gamma)
    sqrtg = math.sqrt(gamma)
    for i in range(size):
        bi = (i >> qubit) & 1
        for j in range(size):
            bj = (j >> qubit) & 1
            val = rho[i][j]
            # E0 contribution
            ni = i if bi == 0 else i
            nj = j if bj == 0 else j
            fi = 1 if bi == 0 else sqrt1
            fj = 1 if bj == 0 else sqrt1
            new_rho[ni][nj] += fi * val * fj.conjugate()
            # E1 contribution
            if bi == 1 and bj == 1:
                ni = i & ~(1 << qubit)
                nj = j & ~(1 << qubit)
                new_rho[ni][nj] += sqrtg * val * sqrtg
    return new_rho

def phase_damping(rho: List[List[complex]], gamma: float, qubit: int) -> List[List[complex]]:
    """Apply phase damping channel to ``qubit``.

    Phase damping destroys coherences between ``|0\rangle`` and ``|1\rangle``
    without affecting populations.  ``gamma`` corresponds to the dephasing
    probability.  When ``gamma`` is ``1`` all off-diagonal terms vanish.
    """
    size = len(rho)
    new_rho = [[0j for _ in range(size)] for _ in range(size)]
    for i in range(size):
        bi = (i >> qubit) & 1
        for j in range(size):
            bj = (j >> qubit) & 1
            val = rho[i][j]
            if bi != bj:
                new_rho[i][j] += (1 - gamma) * val
            else:
                new_rho[i][j] += val
    return new_rho


def depolarizing_channel(
    rho: List[List[complex]], gamma: float, qubit: int
) -> List[List[complex]]:
    r"""Apply a single-qubit depolarizing channel to ``qubit``.

    With probability ``gamma`` the state is replaced by the maximally mixed
    state on that qubit.  The implementation expands the Kraus operators for the
    ``X``, ``Y`` and ``Z`` errors and mixes them with the identity.
    """
    p = gamma / 4
    ops = [
        [[math.sqrt(1 - 3 * p), 0], [0, math.sqrt(1 - 3 * p)]],
        [[0, math.sqrt(p)], [math.sqrt(p), 0]],
        [[0, -1j * math.sqrt(p)], [1j * math.sqrt(p), 0]],
        [[math.sqrt(p), 0], [0, -math.sqrt(p)]],
    ]
    return apply_kraus(rho, ops, [qubit])


def purity(rho: List[List[complex]]) -> float:
    """Return ``Tr(rho^2)`` measuring how pure a density matrix is."""
    size = len(rho)
    val = 0j
    for i in range(size):
        for j in range(size):
            val += rho[i][j] * rho[j][i]
    return val.real


def fidelity(a: Sequence[complex] | List[List[complex]], b: Sequence[complex] | List[List[complex]]) -> float:
    """Return the fidelity between two states or density matrices."""
    if isinstance(a[0], list):
        size = len(a)  # type: ignore[arg-type]
        prod = 0j
        for i in range(size):
            for j in range(size):
                prod += a[i][j] * b[j][i]  # type: ignore[index]
        return prod.real
    else:
        inner = sum(x.conjugate() * y for x, y in zip(a, b))  # type: ignore[index]
        return abs(inner) ** 2


def operator_distance(a: Sequence[Sequence[complex]], b: Sequence[Sequence[complex]]) -> float:
    """Return the Frobenius distance between two operators."""
    size = len(a)
    diff = 0.0
    for i in range(size):
        for j in range(size):
            diff += abs(a[i][j] - b[i][j]) ** 2
    return math.sqrt(diff)


def trotter_step(state: Sequence[complex], h_terms: Sequence[Tuple[Sequence[Sequence[complex]], Sequence[int]]], dt: float) -> List[complex]:
    """Approximate ``exp(-i H dt)`` on ``state`` with first order Trotterisation."""
    result = list(state)
    for Hmat, qubits in h_terms:
        # simple 2x2 exp for demonstration
        if len(Hmat) == 2:
            a = Hmat[0][0]
            b = Hmat[0][1]
            c = Hmat[1][0]
            d = Hmat[1][1]
            tr = a + d
            det = a * d - b * c
            lam = cmath.sqrt(tr * tr - 4 * det)
            eig1 = (tr + lam) / 2
            eig2 = (tr - lam) / 2
            U = [
                [cmath.exp(-1j * eig1 * dt), 0],
                [0, cmath.exp(-1j * eig2 * dt)],
            ]
        else:
            continue
        result = apply_multi_qubit_gate(result, U, qubits)
    return normalize_state(result)


def measure_density(rho: List[List[complex]], qubit: int) -> Tuple[int, List[List[complex]]]:
    """Perform a projective measurement on ``qubit`` of ``rho``."""
    size = len(rho)
    prob0 = sum(rho[i][i].real for i in range(size) if ((i >> qubit) & 1) == 0)
    outcome = 0 if random.random() < prob0 else 1
    new_rho = [[0j for _ in range(size)] for _ in range(size)]
    for i in range(size):
        if ((i >> qubit) & 1) != outcome:
            continue
        for j in range(size):
            if ((j >> qubit) & 1) != outcome:
                continue
            new_rho[i][j] = rho[i][j]
    norm = sum(new_rho[i][i].real for i in range(size))
    if norm > 0:
        for i in range(size):
            for j in range(size):
                new_rho[i][j] /= norm
    return outcome, new_rho


# Measurement

def measure(state: Sequence[complex], qubit: int) -> Tuple[int, List[complex]]:
    """Measure ``qubit`` in the computational basis and collapse ``state``."""
    prob0 = sum(abs(state[i]) ** 2 for i in range(len(state)) if ((i >> qubit) & 1) == 0)
    outcome = 0 if random.random() < prob0 else 1
    collapsed = list(state)
    for i in range(len(state)):
        if ((i >> qubit) & 1) != outcome:
            collapsed[i] = 0j
    collapsed = normalize_state(collapsed)
    return outcome, collapsed


class QuantumCircuit:
    """Simple pure state simulator using state vectors."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.state: List[complex] = [0j] * (1 << n_qubits)
        self.state[0] = 1.0
        self.history: List[str] = []

    def apply_gate(self, gate: Sequence[Sequence[complex]], qubits: Sequence[int] | int):
        """Apply ``gate`` on the specified ``qubits`` of the circuit state."""
        if isinstance(qubits, int):
            qubits = [qubits]
        if len(qubits) == 1:
            self.state = apply_single_qubit_gate(self.state, gate, qubits[0])
        else:
            self.state = apply_multi_qubit_gate(self.state, gate, qubits)
        self.state = normalize_state(self.state)
        self.history.append(f"gate on {qubits}")

    def measure(self, qubit: int) -> int:
        """Measure ``qubit`` and collapse the internal state."""
        result, self.state = measure(self.state, qubit)
        self.history.append(f"measure {qubit} -> {result}")
        return result


class DensityMatrixCircuit:
    """Simulator using density matrices allowing for noise models."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        size = 1 << n_qubits
        self.rho: List[List[complex]] = [[0j for _ in range(size)] for _ in range(size)]
        self.rho[0][0] = 1.0
        self.history: List[str] = []

    def apply_gate(self, gate: Sequence[Sequence[complex]], qubits: Sequence[int] | int):
        """Apply unitary ``gate`` on ``qubits`` to the density matrix."""
        if isinstance(qubits, int):
            qubits = [qubits]
        self.rho = apply_density_gate(self.rho, gate, qubits)
        self.history.append(f"gate on {qubits}")

    def measure(self, qubit: int) -> int:
        """Measure ``qubit`` in the computational basis."""
        outcome, self.rho = measure_density(self.rho, qubit)
        self.history.append(f"measure {qubit} -> {outcome}")
        return outcome

    def apply_amplitude_damping(self, gamma: float, qubit: int):
        """Apply amplitude damping channel to ``qubit``."""
        self.rho = amplitude_damping(self.rho, gamma, qubit)
        self.history.append(f"amplitude_damping {gamma} on {qubit}")

    def apply_phase_damping(self, gamma: float, qubit: int):
        """Apply phase damping channel to ``qubit``."""
        self.rho = phase_damping(self.rho, gamma, qubit)
        self.history.append(f"phase_damping {gamma} on {qubit}")

    def apply_depolarizing(self, gamma: float, qubit: int):
        """Apply depolarizing noise to ``qubit``."""
        self.rho = depolarizing_channel(self.rho, gamma, qubit)
        self.history.append(f"depolarize {gamma} on {qubit}")


# Standard gates
I = [[1, 0], [0, 1]]
X = [[0, 1], [1, 0]]
Y = [[0, -1j], [1j, 0]]
Z = [[1, 0], [0, -1]]
H = [[1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]]


def Rx(theta: float) -> List[List[complex]]:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return [[c, -1j * s], [-1j * s, c]]


def Ry(theta: float) -> List[List[complex]]:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return [[c, -s], [s, c]]


def Rz(theta: float) -> List[List[complex]]:
    return [[cmath.exp(-1j * theta / 2), 0], [0, cmath.exp(1j * theta / 2)]]


def _controlled(u: Sequence[Sequence[complex]]) -> List[List[complex]]:
    return [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, u[0][0], u[0][1]],
        [0, 0, u[1][0], u[1][1]],
    ]


def multi_controlled(u: Sequence[Sequence[complex]], controls: int) -> List[List[complex]]:
    """Return matrix for gate with given number of control qubits."""
    dim = 2 ** (controls + 1)
    result = [[0j for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        result[i][i] = 1
    base = 2 ** controls
    for r in range(len(u)):
        for c in range(len(u)):
            result[base + r][base + c] = u[r][c]
    return result


def CRx(theta: float) -> List[List[complex]]:
    return _controlled(Rx(theta))


def CRz(theta: float) -> List[List[complex]]:
    return _controlled(Rz(theta))


CNOT = _controlled(X)
TOFFOLI = multi_controlled(X, 2)
SWAP = [
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
]

# Ensure gate unitarity at definition time
for gate in [I, X, Y, Z, H]:
    assert is_unitary(gate)
assert is_unitary(CNOT)
assert is_unitary(TOFFOLI)
assert is_unitary(SWAP)
assert is_unitary(CRx(0.3))
assert is_unitary(CRz(0.2))

GATE_MAP: Dict[str, Callable[..., List[List[complex]]]] = {
    "H": lambda: H,
    "X": lambda: X,
    "Y": lambda: Y,
    "Z": lambda: Z,
    "CNOT": lambda: CNOT,
    "SWAP": lambda: SWAP,
    "TOFFOLI": lambda: TOFFOLI,
    "Rx": Rx,
    "Ry": Ry,
    "Rz": Rz,
    "CRx": CRx,
    "CRz": CRz,
}

def gate_from_name(name: str, params: Sequence[float] | None = None) -> List[List[complex]]:
    """Return a gate matrix by name."""
    if name not in GATE_MAP:
        raise ValueError(f"Unknown gate {name}")
    params = params or []
    return GATE_MAP[name](*params)


# QFT routine

def qft(circuit: QuantumCircuit, qubits: Sequence[int] | None = None):
    if qubits is None:
        qubits = list(range(circuit.n_qubits))
    n = len(qubits)
    for i, q in enumerate(qubits):
        circuit.apply_gate(H, q)
        for j in range(1, n - i):
            target = qubits[i + j]
            angle = math.pi / (2 ** j)
            circuit.apply_gate(CRz(angle), [target, q])
    for i in range(n // 2):
        circuit.apply_gate(SWAP, [qubits[i], qubits[-i - 1]])


# Example circuit demonstrating entanglement, Bell test style measurement, and QFT

def example_bell_and_qft() -> Tuple[List[int], List[complex]]:
    qc = QuantumCircuit(3)
    qc.apply_gate(H, 0)
    qc.apply_gate(CNOT, [1, 0])
    results = [qc.measure(0), qc.measure(1)]
    # Recreate |00> or |11> on first two qubits, then run QFT on all three
    qft(qc)
    final_state = qc.state
    return results, final_state


def visualize_probabilities(state: Sequence[complex]):
    """Print a simple bar chart of computational basis probabilities."""
    probs = [abs(a) ** 2 for a in state]
    for i, p in enumerate(probs):
        bar = '#' * int(p * 40)
        print(f"{i:0{int(math.log2(len(state)))}b}: {bar}")


def bloch_coordinates(state: Sequence[complex], qubit: int = 0) -> Tuple[float, float, float]:
    """Return Bloch-sphere coordinates ``(x, y, z)`` for ``qubit`` of ``state``."""
    dim = len(state)
    n = int(math.log2(dim))
    x = y = z = 0.0
    for idx, amp in enumerate(state):
        bit = (idx >> qubit) & 1
        partner = idx ^ (1 << qubit)
        x += (amp.conjugate() * state[partner]).real
        y += (amp.conjugate() * state[partner]).imag * (-1 if bit else 1)
        z += (1 if bit == 0 else -1) * abs(amp) ** 2
    return (2 * x, 2 * y, z)


# ---------------------------------------------------------------------------
# Quantum algorithms
# ---------------------------------------------------------------------------

def phase_oracle(marked: Sequence[int], n_qubits: int) -> List[List[complex]]:
    """Return a phase oracle that flips the sign of ``|x>`` for marked indices."""
    size = 1 << n_qubits
    U = [[0j for _ in range(size)] for _ in range(size)]
    for i in range(size):
        U[i][i] = -1 if i in marked else 1
    assert is_unitary(U)
    return U


def diffusion_operator(n_qubits: int) -> List[List[complex]]:
    """Return the Grover diffusion operator for ``n_qubits``."""
    size = 1 << n_qubits
    coef = 2 / size
    U = [[coef - (1 if i == j else 0) for j in range(size)] for i in range(size)]
    return U


def grover_search(marked: Sequence[int], n_qubits: int, iterations: Optional[int] = None) -> Tuple[int, List[complex]]:
    """Perform Grover's search for ``marked`` elements.

    Parameters
    ----------
    marked:
        List of basis states (as integers) that satisfy the search predicate.
    n_qubits:
        Number of qubits in the search space ``N=2**n_qubits``.
    iterations:
        Optional number of Grover iterations.  If ``None`` the theoretically
        optimal number is used.

    Returns
    -------
    tuple
        A pair ``(outcome, state_vector)`` giving the measured solution and the
        final state vector after the algorithm.

    Example
    -------
    >>> grover_search([3], 2)
    (3, [...])
    """
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.apply_gate(H, q)
    if iterations is None:
        iterations = max(1, int(math.pi / 4 * math.sqrt((1 << n_qubits) / max(1, len(marked)))))
    oracle = phase_oracle(marked, n_qubits)
    diffusion = diffusion_operator(n_qubits)
    for _ in range(iterations):
        qc.apply_gate(oracle, list(range(n_qubits)))
        qc.apply_gate(diffusion, list(range(n_qubits)))
    outcome_bits = [qc.measure(q) for q in range(n_qubits)]
    outcome = sum(bit << i for i, bit in enumerate(outcome_bits))
    return outcome, qc.state


def dj_oracle(func: Callable[[int], int], n_qubits: int) -> List[List[complex]]:
    """Return Deutsch–Jozsa oracle for boolean function ``func``.

    Example
    -------
    >>> def balanced(x):
    ...     return bin(x).count("1") % 2
    >>> oracle = dj_oracle(balanced, 3)
    """
    dim = 1 << (n_qubits + 1)
    U = [[0j for _ in range(dim)] for _ in range(dim)]
    for x in range(1 << n_qubits):
        fx = func(x) & 1
        for y in (0, 1):
            inp = (x << 1) | y
            out = (x << 1) | (y ^ fx)
            U[out][inp] = 1
    assert is_unitary(U)
    return U


def deutsch_jozsa(func: Callable[[int], int], n_qubits: int) -> bool:
    """Run the Deutsch–Jozsa algorithm.

    Returns ``True`` if ``func`` is constant and ``False`` if it is balanced.
    The provided ``func`` should map integers ``0..2**n_qubits-1`` to ``0`` or
    ``1``.

    Example
    -------
    >>> deutsch_jozsa(lambda x: 0, 3)
    True
    """
    qc = QuantumCircuit(n_qubits + 1)
    qc.apply_gate(X, n_qubits)
    qc.apply_gate(H, n_qubits)
    for q in range(n_qubits):
        qc.apply_gate(H, q)
    qc.apply_gate(dj_oracle(func, n_qubits), list(range(n_qubits + 1)))
    for q in range(n_qubits):
        qc.apply_gate(H, q)
    results = [qc.measure(q) for q in range(n_qubits)]
    return all(r == 0 for r in results)


def teleport(state: Sequence[complex]) -> Tuple[Tuple[int, int], List[complex]]:
    """Teleport a single qubit ``state`` using two ancillary qubits.

    Example
    -------
    >>> teleport([1, 0])
    ((0, 0), [...])
    """
    if abs(sum(abs(a) ** 2 for a in state) - 1) > 1e-10:
        raise ValueError("Input state must be normalised")
    qc = QuantumCircuit(3)
    qc.state = [0j] * 8
    qc.state[0] = state[0]
    qc.state[1] = state[1]
    qc.apply_gate(H, 1)
    qc.apply_gate(CNOT, [2, 1])
    qc.apply_gate(CNOT, [1, 0])
    qc.apply_gate(H, 0)
    m0 = qc.measure(0)
    m1 = qc.measure(1)
    if m1 == 1:
        qc.apply_gate(X, 2)
    if m0 == 1:
        qc.apply_gate(Z, 2)
    return (m0, m1), qc.state


def ghz_circuit(n_qubits: int) -> QuantumCircuit:
    """Prepare an ``n_qubits`` GHZ state and return the circuit.

    Example
    -------
    >>> qc = ghz_circuit(3)
    >>> visualize_probabilities(qc.state)
    """
    qc = QuantumCircuit(n_qubits)
    qc.apply_gate(H, 0)
    for q in range(1, n_qubits):
        qc.apply_gate(CNOT, [q, 0])
    return qc


def example_ghz(n_qubits: int = 3) -> List[complex]:
    """Return the state vector of an ``n_qubits`` GHZ state."""
    qc = ghz_circuit(n_qubits)
    return qc.state


def serialize_circuit(circ: QuantumCircuit) -> List[str]:
    """Return the operation history of ``circ`` as a list of strings."""
    return list(circ.history)


def intent_circuit(ops: Sequence[Tuple[str, Sequence[int], Sequence[float] | None]], n_qubits: int) -> QuantumCircuit:
    """Build a ``QuantumCircuit`` from a high level specification."""
    qc = QuantumCircuit(n_qubits)
    for name, qubits, params in ops:
        qc.apply_gate(gate_from_name(name, params), qubits)
    return qc

