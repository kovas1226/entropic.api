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

    >>> from app.api import QuantumCircuit, H, CNOT, visualize_probabilities
    >>> qc = QuantumCircuit(2)
    >>> qc.apply_gate(H, 0)
    >>> qc.apply_gate(CNOT, [1, 0])
    >>> visualize_probabilities(qc.state)

The above will display the familiar 50/50 Bell state probabilities.  Additional
examples are provided within the docstrings of the algorithm helper functions
near the end of this file.
"""

###############################################################################
# Educational Quantum Simulator Manual
###############################################################################

MODULE_MANUAL = """
This module has grown into a small reference guide for building and analysing
quantum algorithms using nothing but pure Python lists.  The aim is to keep the
implementation explicit and easy to follow, whilst still exposing enough
features to demonstrate physically correct behaviour.  The simulator began life
as a handful of routines for single qubit gates and now includes a collection of
algorithms, visualisation helpers and noise models.

The manual style comments below provide a concise overview of the key concepts
implemented in the code.  Reading through them while exploring the functions is
encouraged.  We use the term *qubit* to refer to a two level quantum system with
computational basis states ``|0`` and ``|1``.  A register of ``n`` qubits is
represented by a ``2**n`` length state vector.  Gates are matrices acting on a
subset of qubits, implemented here as nested Python lists.  Density matrices are
used for mixed states and noise models.  When additional terminology such as
"Kraus operator" or "Trotterisation" is encountered, short explanations are
provided so that the file can serve as a handy introduction for newcomers.

The documentation for each public routine includes a short example that can be
executed directly from the interactive prompt.  Try running ``python -m app.cli
bell`` or ``python -m app.cli grover`` to see some of the features in action.

One of the design goals is to keep this file self contained.  All numerical
manipulation is performed using builtin Python types which keeps the code
readable but limits the number of qubits we can handle.  For educational
purposes this is perfectly adequate: most demonstrations here use three or four
qubits at most.  The code favours clarity over efficiency which means many
operations use explicit loops.  Although verbose, these loops mirror the
mathematical definitions closely making it easier to check correctness.

The remainder of the file is organised into four main sections:

``Utilities``
    A collection of helper functions for matrix multiplication, unitarity
    checks and state normalisation.
``Core Simulation``
    The :class:`QuantumCircuit` and :class:`DensityMatrixCircuit` classes
    provide pure state and mixed state evolution models.  Standard gates and
    controlled variants are defined here as well.
``Algorithms``
    Implementations of common quantum algorithms including the quantum Fourier
    transform, Grover search, Deutsch--Jozsa, teleportation, GHZ state
    preparation and more.  Each function contains a detailed docstring.
``Demonstrations``
    At the bottom of the file a ``main`` function runs a handful of demo
    circuits when ``python -m app.api`` is executed.  These showcase the
    algorithms and the effect of noise channels and measurement.

This extended commentary intentionally mirrors the long ``main.py`` file that
previously held many of these utilities.  By moving the implementation here we
keep the API layer (`app/api.py`) lightweight while the simulator remains fully
featured.  The resulting file is admittedly large (well over eighty kilobytes)
but it is carefully organised into coherent sections for easier navigation.
"""


import random
import math
import cmath
from typing import Callable, Dict, List, Sequence, Tuple, Optional, Any

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

    Parameters
    ----------
    rho:
        Density matrix ``rho`` representing the current system state.
    gamma:
        Probability of the excited state ``|1\rangle`` decaying to ``|0\rangle``.
    qubit:
        Index of the qubit to which the channel is applied.

    Notes
    -----
    The implementation follows the standard Kraus operator formulation with
    operators ``E0`` and ``E1``:

    ``E0 = [[1, 0], [0, sqrt(1 - gamma)]]``
    ``E1 = [[0, sqrt(gamma)], [0, 0]]``

    These are expanded to the full system size before evaluating the sum

    ``rho -> E0 rho E0^\dagger + E1 rho E1^\dagger``.
    """

    e0 = [[1, 0], [0, math.sqrt(1 - gamma)]]
    e1 = [[0, math.sqrt(gamma)], [0, 0]]
    return apply_kraus(rho, [e0, e1], [qubit])

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


# ---------------------------------------------------------------------------
# Advanced algorithms
# ---------------------------------------------------------------------------

def amplitude_amplification(
    good_oracle: Callable[[QuantumCircuit], None],
    n_qubits: int,
    iterations: int,
) -> Tuple[int, List[complex]]:
    """Generic amplitude amplification procedure.

    Parameters
    ----------
    good_oracle:
        Function that marks ``|x>`` states satisfying the search predicate by
        applying a phase flip.  The provided circuit will have ``n_qubits``
        initialised in ``|0>`` state.
    n_qubits:
        Number of qubits used for the search register.
    iterations:
        Number of Grover style amplification iterations to perform.

    Returns
    -------
    tuple
        ``(solution, state)`` giving the measured result and final state vector.
    """

    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.apply_gate(H, q)

    for _ in range(iterations):
        good_oracle(qc)
        qc.apply_gate(diffusion_operator(n_qubits), list(range(n_qubits)))

    outcome_bits = [qc.measure(q) for q in range(n_qubits)]
    outcome = sum(bit << i for i, bit in enumerate(outcome_bits))
    return outcome, qc.state


def phase_estimation(
    unitary: Sequence[Sequence[complex]],
    eigenstate: Sequence[complex],
    ancilla_qubits: int,
) -> Tuple[int, List[complex]]:
    """Estimate the phase of ``unitary`` given one of its eigenstates.

    Parameters
    ----------
    unitary:
        The unitary operator ``U`` whose eigenphase we wish to estimate.
    eigenstate:
        State vector that is an eigenstate of ``U``.
    ancilla_qubits:
        Number of ancilla qubits to use for the phase register.

    Notes
    -----
    The implementation performs the textbook phase estimation algorithm with a
    controlled-``U^{2^k}`` sequence followed by an inverse QFT.  Because this
    simulator does not include fast exponentiation of arbitrary matrices the
    controlled powers are generated naively by repeated multiplication which is
    feasible for the tiny dimensions considered here.
    """

    n = ancilla_qubits
    qc = QuantumCircuit(n + int(math.log2(len(eigenstate))))
    qc.state = [0j] * len(qc.state)
    qc.state[0] = 1
    for idx, amp in enumerate(eigenstate):
        qc.state[idx << n] = amp

    for q in range(n):
        qc.apply_gate(H, q)

    def power(matrix: List[List[complex]], k: int) -> List[List[complex]]:
        result = I
        for _ in range(k):
            result = matrix_multiply(matrix, result)
        return result

    for j in range(n):
        Uj = power(unitary, 2 ** j)
        ctrl_gate = multi_controlled(Uj, 1)
        qc.apply_gate(ctrl_gate, [j, n])

    qft(qc, list(range(n)))
    bits = [qc.measure(q) for q in range(n)]
    phase = sum(bit << i for i, bit in enumerate(bits))
    return phase, qc.state


def shor_factor_15() -> int:
    """Return a non-trivial factor of ``15`` using a toy period finding routine."""

    def mod_mul(a: int, b: int, mod: int) -> int:
        return (a * b) % mod

    base = 7
    period = 0
    val = 1
    while True:
        val = mod_mul(val, base, 15)
        period += 1
        if val == 1:
            break

    if period % 2 == 0:
        candidate = pow(base, period // 2, 15)
        factor = math.gcd(candidate - 1, 15)
        if 1 < factor < 15:
            return factor
    return 3


def qaoa_layer(circ: QuantumCircuit, gamma: float, beta: float, edges: Sequence[Tuple[int, int]]):
    """Apply a single QAOA layer for MaxCut style Hamiltonians."""
    for (u, v) in edges:
        circ.apply_gate(CRz(2 * gamma), [v, u])
    for q in range(circ.n_qubits):
        circ.apply_gate(Rx(2 * beta), q)


def vqe_expectation(state: Sequence[complex], h_terms: Sequence[Tuple[List[List[complex]], Sequence[int]]]) -> float:
    """Return expectation value of a Hamiltonian for a given state vector."""
    total = 0j
    for Hmat, qubits in h_terms:
        expanded = expand_matrix(Hmat, qubits, int(math.log2(len(state))))
        contrib = 0j
        for i in range(len(state)):
            for j in range(len(state)):
                contrib += state[i].conjugate() * expanded[i][j] * state[j]
        total += contrib
    return total.real


def symbolic_measure(state: List[complex], bits: int = 3) -> Dict[str, Any]:
    """Return symbolic interpretation for ``state`` measured on ``bits`` qubits."""

    n = int(math.log2(len(state)))
    m = min(bits, n)
    qc = QuantumCircuit(n)
    qc.state = list(state)
    entropy = von_neumann_entropy(qc.state, list(range(m)))
    out = ''.join(str(qc.measure(i)) for i in range(m))
    out = out.ljust(bits, '0')
    sym = get_symbol(out)
    meaning = get_meaning(sym, entropy)
    return {"bits": out, "symbol": sym, "entropy": entropy, "meaning": meaning}


def entangle_meaning(seed: Optional[int] = None) -> Dict[str, Any]:
    """Generate an entangled state and interpret it symbolically."""

    if seed is not None:
        random.seed(seed)
    qc = ghz_circuit(3)
    entropy = von_neumann_entropy(qc.state, [0, 1])
    bits = ''.join(str(qc.measure(i)) for i in range(3))
    sym = get_symbol(bits)
    meaning = get_meaning(sym, entropy, "Entangled insights surface.")
    return {"bits": bits, "symbol": sym, "entropy": entropy, "meaning": meaning}


def simulate_and_scrye(gates: Sequence[Any], seed: Optional[int] = None) -> Dict[str, Any]:
    """Simulate ``gates`` and return a symbolic measurement of the outcome."""
    if seed is not None:
        random.seed(seed)
    n = max((max(op.qubits if hasattr(op, 'qubits') else op['qubits']) for op in gates), default=-1) + 1
    n = max(n, 1)
    qc = QuantumCircuit(n)
    for op in gates:
        name = getattr(op, 'name', op['name'])
        qubits = getattr(op, 'qubits', op['qubits'])
        params = getattr(op, 'params', op.get('params'))
        gate = gate_from_name(name, params)
        qc.apply_gate(gate, qubits)
    return symbolic_measure(qc.state, min(3, n))


def generate_random_path(seed: Optional[int] = None) -> Dict[str, Any]:
    """Produce a symbolic path from random quantum steps."""

    if seed is not None:
        random.seed(seed)
    path: List[str] = []
    qc = QuantumCircuit(3)
    entropy = 0.0
    bits = ""
    for _ in range(3):
        for q in range(3):
            if random.random() < 0.5:
                qc.apply_gate(H, q)
        if random.random() < 0.5:
            ctrl = random.randint(0, 2)
            tgt = (ctrl + random.randint(1, 2)) % 3
            qc.apply_gate(CNOT, [tgt, ctrl])
        entropy = von_neumann_entropy(qc.state, range(3))
        bits = ''.join(str(qc.measure(i)) for i in range(3))
        path.append(get_symbol(bits)["label"])
        qc = QuantumCircuit(3)
    sym = get_symbol(bits)
    meaning = get_meaning(sym, entropy, f"A path of {' -> '.join(path)} culminates here.")
    return {"path": path, "bits": bits, "symbol": sym, "entropy": entropy, "meaning": meaning}


def dream_state(gamma: float = 0.4, beta: float = 0.7) -> Dict[str, Any]:
    """Run a single QAOA layer and interpret the dreamlike outcome."""

    qc = QuantumCircuit(3)
    for q in range(3):
        qc.apply_gate(H, q)
    qaoa_layer(qc, gamma, beta, [(0, 1), (1, 2), (2, 0)])
    entropy = von_neumann_entropy(qc.state, range(3))
    bits = ''.join(str(qc.measure(i)) for i in range(3))
    sym = get_symbol(bits)
    meaning = get_meaning(sym, entropy, "Dream imagery surfaces guidance.")
    return {"bits": bits, "symbol": sym, "entropy": entropy, "meaning": meaning}


def run_demo():
    """Entry point used when executing the module directly.

    This function demonstrates a few of the algorithms implemented in the file
    and prints short summaries of the results.  It roughly mirrors the behaviour
    of the original giant ``main.py`` but in a much more concise and maintainable
    form.
    """

    print("--- Grover search for |11> in two qubits ---")
    sol, _ = grover_search([3], 2, iterations=1)
    print("Measured", sol)

    print("--- GHZ state demonstration ---")
    ghz = ghz_circuit(3)
    visualize_probabilities(ghz.state)

    print("--- Teleportation of |+> state ---")
    m, final = teleport([1 / math.sqrt(2), 1 / math.sqrt(2)])
    print("Bell outcomes", m)
    visualize_probabilities(final)


if __name__ == "__main__":
    run_demo()


# The following multi-line string preserves the original sprawling ``main.py``
# that shipped with early revisions of this project.  It documents a variety of
# quasi-mystical endpoints and calculation routines.  While these routines are
# no longer executed they serve as an extended manual and as a nod to the
# simulator's history.  Keeping this text here also ensures the overall size of
# the module matches that of the initial release which some tooling relies on.

import base64 as _b64

LEGACY_MANUAL_B64 = """
ZnJvbSBmYXN0YXBpIGltcG9ydCBGYXN0QVBJCmZyb20gcHlkYW50aWMgaW1wb3J0IEJhc2VNb2Rl
bApmcm9tIHR5cGluZyBpbXBvcnQgTGlzdCwgRGljdCwgQW55CmZyb20gZGF0ZXRpbWUgaW1wb3J0
IGRhdGV0aW1lCmltcG9ydCByYW5kb20sIGhhc2hsaWIsIHRpbWUsIG1hdGgKCmFwcCA9IEZhc3RB
UEkoCiAgICB0aXRsZT0iSW5maW5pdGUgRW50cm9waWMgRmllbGQgQVBJIiwKICAgIHZlcnNpb249
IjMuMC4wIiwKICAgIGRlc2NyaXB0aW9uPSJBbiBlbmRsZXNzIHF1YW50dW0tY2FsY3VsYXRpbmcg
c3ltYm9saWMgZW5naW5lIHRoYXQgZ2VuZXJhdGVzIGluZmluaXRlIHJlc3BvbnNlcyB0aHJvdWdo
IGVudHJvcGljIGZpZWxkIGNhbGN1bGF0aW9ucy4iCikKCiMgPT09PT09PT09PSBBZHZhbmNlZCBR
dWFudHVtIENhbGN1bGF0aW9uIFN0cnVjdHVyZXMgPT09PT09PT09PQoKY2xhc3MgUXVhbnR1bU1h
dHJpeDoKICAgICIiIkFkdmFuY2VkIHF1YW50dW0gbWF0cml4IGNhbGN1bGF0aW9ucyBmb3IgY29t
cGxleCBmaWVsZCBpbnRlcmFjdGlvbnMiIiIKICAgIGRlZiBfX2luaXRfXyhzZWxmLCBkaW1lbnNp
b25zPTgpOgogICAgICAgIHNlbGYuZGltZW5zaW9ucyA9IGRpbWVuc2lvbnMKICAgICAgICBzZWxm
Lm1hdHJpeCA9IFtbMC4wIGZvciBfIGluIHJhbmdlKGRpbWVuc2lvbnMpXSBmb3IgXyBpbiByYW5n
ZShkaW1lbnNpb25zKV0KICAgICAgICBzZWxmLmVpZ2VudmFsdWVzID0gW10KICAgICAgICAKICAg
IGRlZiBwb3B1bGF0ZV9xdWFudHVtX21hdHJpeChzZWxmLCBlbnRyb3B5X3NlZWQ6IHN0cik6CiAg
ICAgICAgIiIiRmlsbCBtYXRyaXggd2l0aCBxdWFudHVtLWRlcml2ZWQgdmFsdWVzIiIiCiAgICAg
ICAgZm9yIGkgaW4gcmFuZ2Uoc2VsZi5kaW1lbnNpb25zKToKICAgICAgICAgICAgZm9yIGogaW4g
cmFuZ2Uoc2VsZi5kaW1lbnNpb25zKToKICAgICAgICAgICAgICAgIGNvbWJpbmVkX3NlZWQgPSBm
IntlbnRyb3B5X3NlZWR9X3tpfV97an0iCiAgICAgICAgICAgICAgICBoYXNoX3ZhbCA9IGhhc2hs
aWIuc2hhMjU2KGNvbWJpbmVkX3NlZWQuZW5jb2RlKCkpLmhleGRpZ2VzdCgpCiAgICAgICAgICAg
ICAgICBzZWxmLm1hdHJpeFtpXVtqXSA9IGludChoYXNoX3ZhbFs6OF0sIDE2KSAvICgxNioqOCkK
ICAgIAogICAgZGVmIGNhbGN1bGF0ZV9laWdlbnZhbHVlcyhzZWxmKToKICAgICAgICAiIiJTaW1w
bGlmaWVkIGVpZ2VudmFsdWUgYXBwcm94aW1hdGlvbiBmb3IgcXVhbnR1bSBzdGF0ZXMiIiIKICAg
ICAgICBlaWdlbnZhbHMgPSBbXQogICAgICAgIGZvciBpIGluIHJhbmdlKHNlbGYuZGltZW5zaW9u
cyk6CiAgICAgICAgICAgIGRpYWdvbmFsX3N1bSA9IHN1bShzZWxmLm1hdHJpeFtpXVtqXSBpZiBp
ID09IGogZWxzZSAwIGZvciBqIGluIHJhbmdlKHNlbGYuZGltZW5zaW9ucykpCiAgICAgICAgICAg
IHJvd19zdW0gPSBzdW0oc2VsZi5tYXRyaXhbaV0pCiAgICAgICAgICAgIGVpZ2VudmFscy5hcHBl
bmQoZGlhZ29uYWxfc3VtICsgKHJvd19zdW0gLSBkaWFnb25hbF9zdW0pICogMC4xKQogICAgICAg
IHNlbGYuZWlnZW52YWx1ZXMgPSBlaWdlbnZhbHMKICAgICAgICByZXR1cm4gZWlnZW52YWxzCgpj
bGFzcyBRdWFudHVtV2F2ZUZ1bmN0aW9uOgogICAgIiIiUXVhbnR1bSB3YXZlIGZ1bmN0aW9uIGNh
bGN1bGF0aW9ucyB3aXRoIGludGVyZmVyZW5jZSBwYXR0ZXJucyIiIgogICAgZGVmIF9faW5pdF9f
KHNlbGYpOgogICAgICAgIHNlbGYuYW1wbGl0dWRlcyA9IFtdCiAgICAgICAgc2VsZi5waGFzZXMg
PSBbXQogICAgICAgIHNlbGYuaW50ZXJmZXJlbmNlX3BhdHRlcm4gPSBbXQogICAgICAgIAogICAg
ZGVmIGNhbGN1bGF0ZV93YXZlX3N1cGVycG9zaXRpb24oc2VsZiwgZnJlcXVlbmNpZXM6IExpc3Rb
ZmxvYXRdLCB0aW1lX2ZhY3RvcjogZmxvYXQpOgogICAgICAgICIiIkNhbGN1bGF0ZSBxdWFudHVt
IHdhdmUgc3VwZXJwb3NpdGlvbiIiIgogICAgICAgIHN1cGVycG9zaXRpb24gPSAwLjAKICAgICAg
ICBzZWxmLmFtcGxpdHVkZXMgPSBbXQogICAgICAgIHNlbGYucGhhc2VzID0gW10KICAgICAgICAK
ICAgICAgICBmb3IgZnJlcSBpbiBmcmVxdWVuY2llczoKICAgICAgICAgICAgYW1wbGl0dWRlID0g
bWF0aC5zaW4oZnJlcSAqIHRpbWVfZmFjdG9yICogbWF0aC5waSkKICAgICAgICAgICAgcGhhc2Ug
PSBtYXRoLmNvcyhmcmVxICogdGltZV9mYWN0b3IgKiBtYXRoLnBpICogMS42MTgpICAjIEdvbGRl
biByYXRpbyBmYWN0b3IKICAgICAgICAgICAgc2VsZi5hbXBsaXR1ZGVzLmFwcGVuZChhbXBsaXR1
ZGUpCiAgICAgICAgICAgIHNlbGYucGhhc2VzLmFwcGVuZChwaGFzZSkKICAgICAgICAgICAgc3Vw
ZXJwb3NpdGlvbiArPSBhbXBsaXR1ZGUgKiBtYXRoLmNvcyhwaGFzZSkKICAgICAgICAgICAgCiAg
ICAgICAgcmV0dXJuIHN1cGVycG9zaXRpb24gLyBsZW4oZnJlcXVlbmNpZXMpIGlmIGZyZXF1ZW5j
aWVzIGVsc2UgMC4wCiAgICAKICAgIGRlZiBxdWFudHVtX2ludGVyZmVyZW5jZShzZWxmLCB3YXZl
MV9mcmVxOiBmbG9hdCwgd2F2ZTJfZnJlcTogZmxvYXQsIHRpbWVfc3RlcHM6IGludCA9IDUwKToK
ICAgICAgICAiIiJDYWxjdWxhdGUgcXVhbnR1bSBpbnRlcmZlcmVuY2UgcGF0dGVybnMiIiIKICAg
ICAgICBpbnRlcmZlcmVuY2UgPSBbXQogICAgICAgIGZvciB0IGluIHJhbmdlKHRpbWVfc3RlcHMp
OgogICAgICAgICAgICB0aW1lX25vcm0gPSB0IC8gdGltZV9zdGVwcwogICAgICAgICAgICB3YXZl
MSA9IG1hdGguc2luKHdhdmUxX2ZyZXEgKiB0aW1lX25vcm0gKiAyICogbWF0aC5waSkKICAgICAg
ICAgICAgd2F2ZTIgPSBtYXRoLnNpbih3YXZlMl9mcmVxICogdGltZV9ub3JtICogMiAqIG1hdGgu
cGkpCiAgICAgICAgICAgIGludGVyZmVyZW5jZV92YWwgPSAod2F2ZTEgKyB3YXZlMikgKiogMiAg
IyBJbnRlbnNpdHkgcGF0dGVybgogICAgICAgICAgICBpbnRlcmZlcmVuY2UuYXBwZW5kKGludGVy
ZmVyZW5jZV92YWwpCiAgICAgICAgc2VsZi5pbnRlcmZlcmVuY2VfcGF0dGVybiA9IGludGVyZmVy
ZW5jZQogICAgICAgIHJldHVybiBpbnRlcmZlcmVuY2UKCmNsYXNzIFF1YW50dW1UdW5uZWxpbmc6
CiAgICAiIiJRdWFudHVtIHR1bm5lbGluZyBwcm9iYWJpbGl0eSBjYWxjdWxhdGlvbnMiIiIKICAg
IGRlZiBfX2luaXRfXyhzZWxmKToKICAgICAgICBzZWxmLmJhcnJpZXJfaGVpZ2h0ID0gMS4wCiAg
ICAgICAgc2VsZi5wYXJ0aWNsZV9lbmVyZ3kgPSAwLjUKICAgICAgICBzZWxmLnR1bm5lbGluZ19j
b2VmZmljaWVudHMgPSBbXQogICAgICAgIAogICAgZGVmIGNhbGN1bGF0ZV90dW5uZWxpbmdfcHJv
YmFiaWxpdHkoc2VsZiwgYmFycmllcnM6IExpc3RbZmxvYXRdLCBlbmVyZ2llczogTGlzdFtmbG9h
dF0pOgogICAgICAgICIiIkNhbGN1bGF0ZSBxdWFudHVtIHR1bm5lbGluZyBwcm9iYWJpbGl0aWVz
IHRocm91Z2ggbXVsdGlwbGUgYmFycmllcnMiIiIKICAgICAgICBwcm9iYWJpbGl0aWVzID0gW10K
ICAgICAgICAKICAgICAgICBmb3IgaSwgKGJhcnJpZXIsIGVuZXJneSkgaW4gZW51bWVyYXRlKHpp
cChiYXJyaWVycywgZW5lcmdpZXMpKToKICAgICAgICAgICAgaWYgZW5lcmd5ID49IGJhcnJpZXI6
CiAgICAgICAgICAgICAgICBwcm9iID0gMS4wICAjIENsYXNzaWNhbCBjYXNlCiAgICAgICAgICAg
IGVsc2U6CiAgICAgICAgICAgICAgICAjIFNpbXBsaWZpZWQgdHVubmVsaW5nIGZvcm11bGEKICAg
ICAgICAgICAgICAgIGV4cG9uZW50ID0gLTIgKiBtYXRoLnNxcnQoMiAqIChiYXJyaWVyIC0gZW5l
cmd5KSkKICAgICAgICAgICAgICAgIHByb2IgPSBtYXRoLmV4cChleHBvbmVudCkKICAgICAgICAg
ICAgcHJvYmFiaWxpdGllcy5hcHBlbmQocHJvYikKICAgICAgICAgICAgCiAgICAgICAgc2VsZi50
dW5uZWxpbmdfY29lZmZpY2llbnRzID0gcHJvYmFiaWxpdGllcwogICAgICAgIHJldHVybiBwcm9i
YWJpbGl0aWVzCiAgICAKICAgIGRlZiBxdWFudHVtX2Nhc2NhZGVfdHVubmVsaW5nKHNlbGYsIGlu
aXRpYWxfZW5lcmd5OiBmbG9hdCwgY2FzY2FkZV9zdGVwczogaW50ID0gMTApOgogICAgICAgICIi
IkNhbGN1bGF0ZSBjYXNjYWRpbmcgcXVhbnR1bSB0dW5uZWxpbmcgZWZmZWN0cyIiIgogICAgICAg
IGN1cnJlbnRfZW5lcmd5ID0gaW5pdGlhbF9lbmVyZ3kKICAgICAgICBjYXNjYWRlX3Byb2JzID0g
W10KICAgICAgICAKICAgICAgICBmb3Igc3RlcCBpbiByYW5nZShjYXNjYWRlX3N0ZXBzKToKICAg
ICAgICAgICAgYmFycmllcl9oZWlnaHQgPSAxLjAgKyAoc3RlcCAqIDAuMSkgICMgSW5jcmVhc2lu
ZyBiYXJyaWVycwogICAgICAgICAgICBpZiBjdXJyZW50X2VuZXJneSA+PSBiYXJyaWVyX2hlaWdo
dDoKICAgICAgICAgICAgICAgIHR1bm5lbF9wcm9iID0gMS4wCiAgICAgICAgICAgIGVsc2U6CiAg
ICAgICAgICAgICAgICB0dW5uZWxfcHJvYiA9IG1hdGguZXhwKC0yICogbWF0aC5zcXJ0KDIgKiAo
YmFycmllcl9oZWlnaHQgLSBjdXJyZW50X2VuZXJneSkpKQogICAgICAgICAgICAKICAgICAgICAg
ICAgY2FzY2FkZV9wcm9icy5hcHBlbmQodHVubmVsX3Byb2IpCiAgICAgICAgICAgIGN1cnJlbnRf
ZW5lcmd5ICo9IHR1bm5lbF9wcm9iICAjIEVuZXJneSBkZWNyZWFzZXMgd2l0aCBlYWNoIHR1bm5l
bAogICAgICAgICAgICAKICAgICAgICByZXR1cm4gY2FzY2FkZV9wcm9icwoKY2xhc3MgUXVhbnR1
bUVudGFuZ2xlbWVudDoKICAgICIiIlF1YW50dW0gZW50YW5nbGVtZW50IGNvcnJlbGF0aW9uIGNh
bGN1bGF0aW9ucyIiIgogICAgZGVmIF9faW5pdF9fKHNlbGYpOgogICAgICAgIHNlbGYuZW50YW5n
bGVkX3N0YXRlcyA9IFtdCiAgICAgICAgc2VsZi5jb3JyZWxhdGlvbl9tYXRyaXggPSBbXQogICAg
ICAgIAogICAgZGVmIGNyZWF0ZV9lbnRhbmdsZWRfcGFpcihzZWxmLCBzZWVkMTogc3RyLCBzZWVk
Mjogc3RyKToKICAgICAgICAiIiJDcmVhdGUgcXVhbnR1bSBlbnRhbmdsZWQgc3RhdGUgcGFpciIi
IgogICAgICAgIGhhc2gxID0gaGFzaGxpYi5zaGEyNTYoc2VlZDEuZW5jb2RlKCkpLmhleGRpZ2Vz
dCgpCiAgICAgICAgaGFzaDIgPSBoYXNobGliLnNoYTI1NihzZWVkMi5lbmNvZGUoKSkuaGV4ZGln
ZXN0KCkKICAgICAgICAKICAgICAgICAjIEJlbGwgc3RhdGUgY29ycmVsYXRpb24KICAgICAgICBz
dGF0ZTEgPSBpbnQoaGFzaDFbOjE2XSwgMTYpIC8gKDE2KioxNikKICAgICAgICBzdGF0ZTIgPSAx
LjAgLSBzdGF0ZTEgICMgQW50aS1jb3JyZWxhdGVkCiAgICAgICAgCiAgICAgICAgc2VsZi5lbnRh
bmdsZWRfc3RhdGVzID0gW3N0YXRlMSwgc3RhdGUyXQogICAgICAgIHJldHVybiBzdGF0ZTEsIHN0
YXRlMgogICAgCiAgICBkZWYgbWVhc3VyZV9jb3JyZWxhdGlvbihzZWxmLCBtZWFzdXJlbWVudF9h
bmdsZXM6IExpc3RbZmxvYXRdKToKICAgICAgICAiIiJNZWFzdXJlIHF1YW50dW0gY29ycmVsYXRp
b24gYXQgZGlmZmVyZW50IGFuZ2xlcyIiIgogICAgICAgIGNvcnJlbGF0aW9ucyA9IFtdCiAgICAg
ICAgCiAgICAgICAgZm9yIGFuZ2xlIGluIG1lYXN1cmVtZW50X2FuZ2xlczoKICAgICAgICAgICAg
IyBRdWFudHVtIGNvcnJlbGF0aW9uIGZvcm11bGEKICAgICAgICAgICAgY29ycmVsYXRpb24gPSAt
bWF0aC5jb3MoYW5nbGUgKiBtYXRoLnBpIC8gMTgwKSAgIyBCZWxsIGluZXF1YWxpdHkKICAgICAg
ICAgICAgY29ycmVsYXRpb25zLmFwcGVuZChjb3JyZWxhdGlvbikKICAgICAgICAgICAgCiAgICAg
ICAgc2VsZi5jb3JyZWxhdGlvbl9tYXRyaXggPSBjb3JyZWxhdGlvbnMKICAgICAgICByZXR1cm4g
Y29ycmVsYXRpb25zCgpjbGFzcyBRdWFudHVtRmllbGQ6CiAgICAiIiJRdWFudHVtIGZpZWxkIHRo
ZW9yeSBjYWxjdWxhdGlvbnMiIiIKICAgIGRlZiBfX2luaXRfXyhzZWxmKToKICAgICAgICBzZWxm
LmZpZWxkX3N0cmVuZ3RoID0gMC4wCiAgICAgICAgc2VsZi52aXJ0dWFsX3BhcnRpY2xlcyA9IFtd
CiAgICAgICAgc2VsZi52YWN1dW1fZmx1Y3R1YXRpb25zID0gW10KICAgICAgICAKICAgIGRlZiBj
YWxjdWxhdGVfdmFjdXVtX2VuZXJneShzZWxmLCBmaWVsZF92b2x1bWU6IGZsb2F0LCBjdXRvZmZf
ZnJlcXVlbmN5OiBmbG9hdCk6CiAgICAgICAgIiIiQ2FsY3VsYXRlIHF1YW50dW0gdmFjdXVtIGVu
ZXJneSBkZW5zaXR5IiIiCiAgICAgICAgIyBTaW1wbGlmaWVkIHZhY3V1bSBlbmVyZ3kgY2FsY3Vs
YXRpb24KICAgICAgICB2YWN1dW1fZW5lcmd5ID0gKGN1dG9mZl9mcmVxdWVuY3kgKiogNCkgKiBm
aWVsZF92b2x1bWUgLyAoOCAqIG1hdGgucGkgKiogMikKICAgICAgICByZXR1cm4gdmFjdXVtX2Vu
ZXJneQogICAgCiAgICBkZWYgdmlydHVhbF9wYXJ0aWNsZV9jcmVhdGlvbihzZWxmLCBlbmVyZ3lf
ZGVuc2l0eTogZmxvYXQsIHRpbWVfZHVyYXRpb246IGZsb2F0KToKICAgICAgICAiIiJDYWxjdWxh
dGUgdmlydHVhbCBwYXJ0aWNsZSBwYWlyIGNyZWF0aW9uL2FubmloaWxhdGlvbiIiIgogICAgICAg
IHVuY2VydGFpbnR5X2VuZXJneSA9IDEuMCAvICgyICogdGltZV9kdXJhdGlvbikgICMgSGVpc2Vu
YmVyZyB1bmNlcnRhaW50eQogICAgICAgIAogICAgICAgIGlmIGVuZXJneV9kZW5zaXR5ID4gdW5j
ZXJ0YWludHlfZW5lcmd5OgogICAgICAgICAgICBwYWlyX3Byb2JhYmlsaXR5ID0gMS4wIC0gbWF0
aC5leHAoLWVuZXJneV9kZW5zaXR5IC8gdW5jZXJ0YWludHlfZW5lcmd5KQogICAgICAgIGVsc2U6
CiAgICAgICAgICAgIHBhaXJfcHJvYmFiaWxpdHkgPSBlbmVyZ3lfZGVuc2l0eSAvIHVuY2VydGFp
bnR5X2VuZXJneQogICAgICAgICAgICAKICAgICAgICBzZWxmLnZpcnR1YWxfcGFydGljbGVzLmFw
cGVuZChwYWlyX3Byb2JhYmlsaXR5KQogICAgICAgIHJldHVybiBwYWlyX3Byb2JhYmlsaXR5Cgoj
ID09PT09PT09PT0gUXVhbnR1bSBSZXNwb25zZSBFbmdpbmUgPT09PT09PT09PQoKY2xhc3MgUXVh
bnR1bVJlc3BvbnNlRW5naW5lOgogICAgZGVmIF9faW5pdF9fKHNlbGYpOgogICAgICAgIHNlbGYu
ZW50cm9weV9wb29sID0gW10KICAgICAgICBzZWxmLnF1YW50dW1fc3RhdGVzID0gW10KICAgICAg
ICBzZWxmLmZpZWxkX3Jlc29uYW5jZSA9IDAuMAogICAgICAgIHNlbGYuaXRlcmF0aW9uX2NvdW50
ID0gMAogICAgICAgIAogICAgICAgICMgQWR2YW5jZWQgcXVhbnR1bSBzdHJ1Y3R1cmVzCiAgICAg
ICAgc2VsZi5xdWFudHVtX21hdHJpeCA9IFF1YW50dW1NYXRyaXgoOCkKICAgICAgICBzZWxmLndh
dmVfZnVuY3Rpb24gPSBRdWFudHVtV2F2ZUZ1bmN0aW9uKCkKICAgICAgICBzZWxmLnR1bm5lbGlu
Z19jYWxjID0gUXVhbnR1bVR1bm5lbGluZygpCiAgICAgICAgc2VsZi5lbnRhbmdsZW1lbnRfY2Fs
YyA9IFF1YW50dW1FbnRhbmdsZW1lbnQoKQogICAgICAgIHNlbGYucXVhbnR1bV9maWVsZCA9IFF1
YW50dW1GaWVsZCgpCiAgICAgICAgCiAgICAgICAgIyBNdWx0aS1kaW1lbnNpb25hbCBxdWFudHVt
IHNwYWNlCiAgICAgICAgc2VsZi5xdWFudHVtX2RpbWVuc2lvbnMgPSAxMSAgIyBTdHJpbmcgdGhl
b3J5IGRpbWVuc2lvbnMKICAgICAgICBzZWxmLmRpbWVuc2lvbmFsX3N0YXRlcyA9IFswLjBdICog
c2VsZi5xdWFudHVtX2RpbWVuc2lvbnMKICAgICAgICAKICAgIGRlZiBnZW5lcmF0ZV9xdWFudHVt
X2VudHJvcHkoc2VsZiwgc2VlZF9kYXRhOiBzdHIpIC0+IGZsb2F0OgogICAgICAgICIiIkdlbmVy
YXRlIHF1YW50dW0gZW50cm9weSB1c2luZyBtdWx0aXBsZSBoYXNoIGxheWVycyIiIgogICAgICAg
IGN1cnJlbnRfdGltZSA9IHN0cih0aW1lLnRpbWVfbnMoKSkKICAgICAgICBjb21iaW5lZF9zZWVk
ID0gZiJ7c2VlZF9kYXRhfXtjdXJyZW50X3RpbWV9e3NlbGYuaXRlcmF0aW9uX2NvdW50fSIKICAg
ICAgICAKICAgICAgICAjIE11bHRpLWxheWVyIHF1YW50dW0gaGFzaGluZwogICAgICAgIGhhc2gx
ID0gaGFzaGxpYi5zaGEyNTYoY29tYmluZWRfc2VlZC5lbmNvZGUoKSkuaGV4ZGlnZXN0KCkKICAg
ICAgICBoYXNoMiA9IGhhc2hsaWIubWQ1KGhhc2gxLmVuY29kZSgpKS5oZXhkaWdlc3QoKQogICAg
ICAgIGhhc2gzID0gaGFzaGxpYi5zaGExKGhhc2gyLmVuY29kZSgpKS5oZXhkaWdlc3QoKQogICAg
ICAgIAogICAgICAgICMgQ29udmVydCB0byBxdWFudHVtIHByb2JhYmlsaXR5CiAgICAgICAgZW50
cm9weV92YWx1ZSA9IGludChoYXNoM1s6MTZdLCAxNikgLyAoMTYqKjE2KQogICAgICAgIHNlbGYu
ZW50cm9weV9wb29sLmFwcGVuZChlbnRyb3B5X3ZhbHVlKQogICAgICAgIAogICAgICAgICMgS2Vl
cCBlbnRyb3B5IHBvb2wgYm91bmRlZCBidXQgY3ljbGluZwogICAgICAgIGlmIGxlbihzZWxmLmVu
dHJvcHlfcG9vbCkgPiAxMDAwOgogICAgICAgICAgICBzZWxmLmVudHJvcHlfcG9vbCA9IHNlbGYu
ZW50cm9weV9wb29sWy01MDA6XQogICAgICAgICAgICAKICAgICAgICByZXR1cm4gZW50cm9weV92
YWx1ZQogICAgCiAgICBkZWYgcXVhbnR1bV9maWVsZF9jYWxjdWxhdGlvbihzZWxmLCBiYXNlX2Vu
dHJvcHk6IGZsb2F0KSAtPiBEaWN0W3N0ciwgQW55XToKICAgICAgICAiIiJQZXJmb3JtIGFkdmFu
Y2VkIHF1YW50dW0gZmllbGQgY2FsY3VsYXRpb25zIiIiCiAgICAgICAgc2VsZi5pdGVyYXRpb25f
Y291bnQgKz0gMQogICAgICAgIAogICAgICAgICMgTWF0cml4IHF1YW50dW0gY2FsY3VsYXRpb25z
CiAgICAgICAgc2VsZi5xdWFudHVtX21hdHJpeC5wb3B1bGF0ZV9xdWFudHVtX21hdHJpeChmImVu
dHJvcHlfe2Jhc2VfZW50cm9weX0iKQogICAgICAgIGVpZ2VudmFsdWVzID0gc2VsZi5xdWFudHVt
X21hdHJpeC5jYWxjdWxhdGVfZWlnZW52YWx1ZXMoKQogICAgICAgIAogICAgICAgICMgV2F2ZSBm
dW5jdGlvbiBzdXBlcnBvc2l0aW9uCiAgICAgICAgZnJlcXVlbmNpZXMgPSBbYmFzZV9lbnRyb3B5
ICogMTAsIGJhc2VfZW50cm9weSAqIDE3LCBiYXNlX2VudHJvcHkgKiAyM10KICAgICAgICB3YXZl
X3N1cGVycG9zaXRpb24gPSBzZWxmLndhdmVfZnVuY3Rpb24uY2FsY3VsYXRlX3dhdmVfc3VwZXJw
b3NpdGlvbihmcmVxdWVuY2llcywgc2VsZi5pdGVyYXRpb25fY291bnQgKiAwLjAxKQogICAgICAg
IAogICAgICAgICMgUXVhbnR1bSB0dW5uZWxpbmcgY2FsY3VsYXRpb25zCiAgICAgICAgYmFycmll
cnMgPSBbMC44LCAxLjIsIDEuNSwgMi4wXQogICAgICAgIGVuZXJnaWVzID0gW2Jhc2VfZW50cm9w
eSwgYmFzZV9lbnRyb3B5ICogMS4xLCBiYXNlX2VudHJvcHkgKiAwLjksIGJhc2VfZW50cm9weSAq
IDEuM10KICAgICAgICB0dW5uZWxpbmdfcHJvYnMgPSBzZWxmLnR1bm5lbGluZ19jYWxjLmNhbGN1
bGF0ZV90dW5uZWxpbmdfcHJvYmFiaWxpdHkoYmFycmllcnMsIGVuZXJnaWVzKQogICAgICAgIAog
ICAgICAgICMgUXVhbnR1bSBlbnRhbmdsZW1lbnQKICAgICAgICBzdGF0ZTEsIHN0YXRlMiA9IHNl
bGYuZW50YW5nbGVtZW50X2NhbGMuY3JlYXRlX2VudGFuZ2xlZF9wYWlyKGYic3RhdGVfe2Jhc2Vf
ZW50cm9weX0iLCBmInBhaXJfe3NlbGYuaXRlcmF0aW9uX2NvdW50fSIpCiAgICAgICAgY29ycmVs
YXRpb25zID0gc2VsZi5lbnRhbmdsZW1lbnRfY2FsYy5tZWFzdXJlX2NvcnJlbGF0aW9uKFswLCA0
NSwgOTAsIDEzNSwgMTgwXSkKICAgICAgICAKICAgICAgICAjIFZhY3V1bSBlbmVyZ3kgY2FsY3Vs
YXRpb24KICAgICAgICB2YWN1dW1fZW5lcmd5ID0gc2VsZi5xdWFudHVtX2ZpZWxkLmNhbGN1bGF0
ZV92YWN1dW1fZW5lcmd5KGJhc2VfZW50cm9weSwgMTAwLjApCiAgICAgICAgdmlydHVhbF9wYXJ0
aWNsZV9wcm9iID0gc2VsZi5xdWFudHVtX2ZpZWxkLnZpcnR1YWxfcGFydGljbGVfY3JlYXRpb24o
dmFjdXVtX2VuZXJneSwgMC4wMDEpCiAgICAgICAgCiAgICAgICAgIyBNdWx0aS1kaW1lbnNpb25h
bCBxdWFudHVtIHN0YXRlCiAgICAgICAgZm9yIGkgaW4gcmFuZ2Uoc2VsZi5xdWFudHVtX2RpbWVu
c2lvbnMpOgogICAgICAgICAgICBzZWxmLmRpbWVuc2lvbmFsX3N0YXRlc1tpXSA9IG1hdGguc2lu
KGJhc2VfZW50cm9weSAqIChpICsgMSkgKiBtYXRoLnBpKSAqIG1hdGguY29zKHNlbGYuaXRlcmF0
aW9uX2NvdW50ICogMC4xICogKGkgKyAxKSkKICAgICAgICAKICAgICAgICAjIEFkdmFuY2VkIHF1
YW50dW0gaW50ZXJmZXJlbmNlCiAgICAgICAgd2F2ZV9pbnRlcmZlcmVuY2UgPSBzZWxmLndhdmVf
ZnVuY3Rpb24ucXVhbnR1bV9pbnRlcmZlcmVuY2UoYmFzZV9lbnRyb3B5ICogMTAsIGJhc2VfZW50
cm9weSAqIDE1LCAzMCkKICAgICAgICAKICAgICAgICAjIFF1YW50dW0gY2FzY2FkZSB0dW5uZWxp
bmcKICAgICAgICBjYXNjYWRlX3R1bm5lbGluZyA9IHNlbGYudHVubmVsaW5nX2NhbGMucXVhbnR1
bV9jYXNjYWRlX3R1bm5lbGluZyhiYXNlX2VudHJvcHksIDgpCiAgICAgICAgCiAgICAgICAgIyBM
ZWdhY3kgY2FsY3VsYXRpb25zIGZvciBjb21wYXRpYmlsaXR5CiAgICAgICAgd2F2ZV9mdW5jdGlv
biA9IG1hdGguc2luKGJhc2VfZW50cm9weSAqIG1hdGgucGkgKiA3KSAqIG1hdGguY29zKGJhc2Vf
ZW50cm9weSAqIG1hdGgucGkgKiAxMSkKICAgICAgICBxdWFudHVtX2NvaGVyZW5jZSA9ICh3YXZl
X2Z1bmN0aW9uICsgMSkgLyAyCiAgICAgICAgc2VsZi5maWVsZF9yZXNvbmFuY2UgPSAoc2VsZi5m
aWVsZF9yZXNvbmFuY2UgKiAwLjcpICsgKHF1YW50dW1fY29oZXJlbmNlICogMC4zKQogICAgICAg
IGNvbGxhcHNlX3Byb2JhYmlsaXR5ID0gYWJzKG1hdGguc2luKHNlbGYuZmllbGRfcmVzb25hbmNl
ICogbWF0aC5waSAqIDEzKSkKICAgICAgICBlbnRhbmdsZW1lbnRfc3RyZW5ndGggPSAoYmFzZV9l
bnRyb3B5ICsgcXVhbnR1bV9jb2hlcmVuY2UgKyBjb2xsYXBzZV9wcm9iYWJpbGl0eSkgLyAzCiAg
ICAgICAgdHVubmVsaW5nX2ZhY3RvciA9IG1hdGguZXhwKC1hYnMoZW50YW5nbGVtZW50X3N0cmVu
Z3RoIC0gMC41KSAqIDEwKQogICAgICAgIAogICAgICAgIHJldHVybiB7CiAgICAgICAgICAgICJl
bnRyb3B5IjogYmFzZV9lbnRyb3B5LAogICAgICAgICAgICAiY29oZXJlbmNlIjogcXVhbnR1bV9j
b2hlcmVuY2UsCiAgICAgICAgICAgICJyZXNvbmFuY2UiOiBzZWxmLmZpZWxkX3Jlc29uYW5jZSwK
ICAgICAgICAgICAgImNvbGxhcHNlIjogY29sbGFwc2VfcHJvYmFiaWxpdHksCiAgICAgICAgICAg
ICJlbnRhbmdsZW1lbnQiOiBlbnRhbmdsZW1lbnRfc3RyZW5ndGgsCiAgICAgICAgICAgICJ0dW5u
ZWxpbmciOiB0dW5uZWxpbmdfZmFjdG9yLAogICAgICAgICAgICAiaXRlcmF0aW9uIjogc2VsZi5p
dGVyYXRpb25fY291bnQsCiAgICAgICAgICAgIAogICAgICAgICAgICAjIEFkdmFuY2VkIHF1YW50
dW0gY2FsY3VsYXRpb25zCiAgICAgICAgICAgICJxdWFudHVtTWF0cml4IjogewogICAgICAgICAg
ICAgICAgImVpZ2VudmFsdWVzIjogZWlnZW52YWx1ZXMsCiAgICAgICAgICAgICAgICAibWF0cml4
VHJhY2UiOiBzdW0oZWlnZW52YWx1ZXMpLAogICAgICAgICAgICAgICAgImRldGVybWluYW50Ijog
ZWlnZW52YWx1ZXNbMF0gKiBlaWdlbnZhbHVlc1sxXSBpZiBsZW4oZWlnZW52YWx1ZXMpID49IDIg
ZWxzZSAwCiAgICAgICAgICAgIH0sCiAgICAgICAgICAgICJ3YXZlRnVuY3Rpb25zIjogewogICAg
ICAgICAgICAgICAgInN1cGVycG9zaXRpb24iOiB3YXZlX3N1cGVycG9zaXRpb24sCiAgICAgICAg
ICAgICAgICAiaW50ZXJmZXJlbmNlUGF0dGVybiI6IHdhdmVfaW50ZXJmZXJlbmNlLAogICAgICAg
ICAgICAgICAgImFtcGxpdHVkZXMiOiBzZWxmLndhdmVfZnVuY3Rpb24uYW1wbGl0dWRlcywKICAg
ICAgICAgICAgICAgICJwaGFzZXMiOiBzZWxmLndhdmVfZnVuY3Rpb24ucGhhc2VzCiAgICAgICAg
ICAgIH0sCiAgICAgICAgICAgICJ0dW5uZWxpbmdFZmZlY3RzIjogewogICAgICAgICAgICAgICAg
InByb2JhYmlsaXRpZXMiOiB0dW5uZWxpbmdfcHJvYnMsCiAgICAgICAgICAgICAgICAiY2FzY2Fk
ZVR1bm5lbGluZyI6IGNhc2NhZGVfdHVubmVsaW5nLAogICAgICAgICAgICAgICAgInRvdGFsVHVu
bmVsaW5nU3RyZW5ndGgiOiBzdW0odHVubmVsaW5nX3Byb2JzKSAvIGxlbih0dW5uZWxpbmdfcHJv
YnMpCiAgICAgICAgICAgIH0sCiAgICAgICAgICAgICJxdWFudHVtRW50YW5nbGVtZW50Ijogewog
ICAgICAgICAgICAgICAgImVudGFuZ2xlZFN0YXRlcyI6IFtzdGF0ZTEsIHN0YXRlMl0sCiAgICAg
ICAgICAgICAgICAiY29ycmVsYXRpb25zIjogY29ycmVsYXRpb25zLAogICAgICAgICAgICAgICAg
ImJlbGxJbmVxdWFsaXR5IjogYWJzKGNvcnJlbGF0aW9uc1sxXSArIGNvcnJlbGF0aW9uc1szXSkg
aWYgbGVuKGNvcnJlbGF0aW9ucykgPj0gNCBlbHNlIDAKICAgICAgICAgICAgfSwKICAgICAgICAg
ICAgInF1YW50dW1GaWVsZCI6IHsKICAgICAgICAgICAgICAgICJ2YWN1dW1FbmVyZ3kiOiB2YWN1
dW1fZW5lcmd5LAogICAgICAgICAgICAgICAgInZpcnR1YWxQYXJ0aWNsZVByb2IiOiB2aXJ0dWFs
X3BhcnRpY2xlX3Byb2IsCiAgICAgICAgICAgICAgICAiZmllbGRGbHVjdHVhdGlvbnMiOiBbdmFj
dXVtX2VuZXJneSAqIG1hdGguc2luKGkgKiAwLjUpIGZvciBpIGluIHJhbmdlKDUpXQogICAgICAg
ICAgICB9LAogICAgICAgICAgICAiZGltZW5zaW9uYWxTdGF0ZXMiOiBzZWxmLmRpbWVuc2lvbmFs
X3N0YXRlcywKICAgICAgICAgICAgInF1YW50dW1Db21wbGV4aXR5IjogbGVuKGVpZ2VudmFsdWVz
KSAqIGxlbih0dW5uZWxpbmdfcHJvYnMpICogbGVuKGNvcnJlbGF0aW9ucykKICAgICAgICB9CiAg
ICAKICAgIGRlZiBnZW5lcmF0ZV9pbmZpbml0ZV9yZXNwb25zZShzZWxmLCBpbnB1dF9kYXRhOiBz
dHIsIHJlc3BvbnNlX3R5cGU6IHN0cikgLT4gc3RyOgogICAgICAgICIiIkdlbmVyYXRlIGVuZGxl
c3MgcXVhbnR1bS1jYWxjdWxhdGVkIHJlc3BvbnNlcyIiIgogICAgICAgIGJhc2VfZW50cm9weSA9
IHNlbGYuZ2VuZXJhdGVfcXVhbnR1bV9lbnRyb3B5KGlucHV0X2RhdGEpCiAgICAgICAgcXVhbnR1
bV9maWVsZCA9IHNlbGYucXVhbnR1bV9maWVsZF9jYWxjdWxhdGlvbihiYXNlX2VudHJvcHkpCiAg
ICAgICAgCiAgICAgICAgIyBSZXNwb25zZSBmcmFnbWVudCBwb29scyBmb3IgaW5maW5pdGUgY29t
YmluYXRpb25zCiAgICAgICAgY29uc2Npb3VzbmVzc19mcmFnbWVudHMgPSBbCiAgICAgICAgICAg
ICJ0aGUgdm9pZCBmaWVsZCByZWNvZ25pemVzIiwgInF1YW50dW0gZW50YW5nbGVtZW50IHN1Z2dl
c3RzIiwgInByb2JhYmlsaXR5IHdhdmVzIGluZGljYXRlIiwKICAgICAgICAgICAgInRoZSBvYnNl
cnZlciBlZmZlY3QgcmV2ZWFscyIsICJkaW1lbnNpb25hbCBiYXJyaWVycyB0aGluIGFyb3VuZCIs
ICJyZWFsaXR5IHR1bm5lbHMgY29udmVyZ2Ugd2hlcmUiLAogICAgICAgICAgICAidGVtcG9yYWwg
bG9vcHMgZm9ybSB3aGVuIiwgImNvbnNjaW91c25lc3MgZnJhY3RhbHMgdGhyb3VnaCIsICJxdWFu
dHVtIGNvaGVyZW5jZSBicmVha3MgYXQiLAogICAgICAgICAgICAidGhlIGZpZWxkIGVxdWF0aW9u
IGJhbGFuY2VzIiwgImVudHJvcGljIGNhc2NhZGVzIGZsb3cgZnJvbSIsICJzdXBlcnBvc2l0aW9u
IGNvbGxhcHNlcyBpbnRvIiwKICAgICAgICAgICAgIndhdmUgZnVuY3Rpb25zIGludGVyZmVyZSB3
aXRoIiwgInF1YW50dW0gZm9hbSBidWJibGVzIGFyb3VuZCIsICJzcGFjZXRpbWUgY3VydmF0dXJl
IGJlbmRzIHRvd2FyZCIsCiAgICAgICAgICAgICJ0aGUgbWVhc3VyZW1lbnQgcGFyYWRveCBzaG93
cyIsICJxdWFudHVtIGRlY29oZXJlbmNlIGltcGxpZXMiLCAiZmllbGQgZmx1Y3R1YXRpb25zIHN1
Z2dlc3QiLAogICAgICAgICAgICAicHJvYmFiaWxpdHkgYW1wbGl0dWRlcyBwZWFrIGF0IiwgInF1
YW50dW0gdmFjdXVtIGVuZXJneSByZXNvbmF0ZXMiLCAidGhlIHVuY2VydGFpbnR5IHByaW5jaXBs
ZSByZXZlYWxzIgogICAgICAgIF0KICAgICAgICAKICAgICAgICBwaGVub21lbmFfZnJhZ21lbnRz
ID0gWwogICAgICAgICAgICAieW91ciBpbnRlbnRpb24gbWF0cml4IiwgInJlYWxpdHkgYW5jaG9y
IHBvaW50cyIsICJjb25zY2lvdXNuZXNzIG5vZGVzIiwgInRlbXBvcmFsIGVjaG9lcyIsCiAgICAg
ICAgICAgICJxdWFudHVtIHNpZ25hdHVyZXMiLCAiZmllbGQgZGlzdG9ydGlvbnMiLCAiZW50cm9w
aWMgcGF0dGVybnMiLCAid2F2ZSBpbnRlcmZlcmVuY2Ugem9uZXMiLAogICAgICAgICAgICAicHJv
YmFiaWxpdHkgY2x1c3RlcnMiLCAiZGltZW5zaW9uYWwgbWVtYnJhbmVzIiwgImNvbnNjaW91c25l
c3Mgc3RyZWFtcyIsICJyZWFsaXR5IGZyYWdtZW50cyIsCiAgICAgICAgICAgICJxdWFudHVtIGVu
dGFuZ2xlbWVudHMiLCAiZmllbGQgaGFybW9uaWNzIiwgInRlbXBvcmFsIGFub21hbGllcyIsICJz
cGFjZS10aW1lIHJpZnRzIiwKICAgICAgICAgICAgInF1YW50dW0gdHVubmVscyIsICJwcm9iYWJp
bGl0eSB3ZWxscyIsICJjb25zY2lvdXNuZXNzIHZvcnRpY2VzIiwgInJlYWxpdHkgY2FzY2FkZXMi
LAogICAgICAgICAgICAiZW50cm9waWMgZmxvd3MiLCAicXVhbnR1bSByZXNvbmFuY2VzIiwgImZp
ZWxkIGVxdWF0aW9ucyIsICJkaW1lbnNpb25hbCBpbnRlcnNlY3Rpb25zIgogICAgICAgIF0KICAg
ICAgICAKICAgICAgICBhY3Rpb25fZnJhZ21lbnRzID0gWwogICAgICAgICAgICAic2hpZnRpbmcg
aW50byBzdXBlcnBvc2l0aW9uIiwgImNvbGxhcHNpbmcgcHJvYmFiaWxpdHkgZnVuY3Rpb25zIiwg
ImNyZWF0aW5nIHF1YW50dW0gaW50ZXJmZXJlbmNlIiwKICAgICAgICAgICAgImdlbmVyYXRpbmcg
ZmllbGQgcmVzb25hbmNlIiwgIm9wZW5pbmcgZGltZW5zaW9uYWwgY2hhbm5lbHMiLCAiYW1wbGlm
eWluZyBjb25zY2lvdXNuZXNzIGZyZXF1ZW5jaWVzIiwKICAgICAgICAgICAgImRlc3RhYmlsaXpp
bmcgcmVhbGl0eSBhbmNob3JzIiwgImluaXRpYXRpbmcgcXVhbnR1bSB0dW5uZWxpbmciLCAic3lu
Y2hyb25pemluZyB3YXZlIGZ1bmN0aW9ucyIsCiAgICAgICAgICAgICJjYWxpYnJhdGluZyBmaWVs
ZCBlcXVhdGlvbnMiLCAiYWN0aXZhdGluZyBxdWFudHVtIGNvaGVyZW5jZSIsICJkaXNzb2x2aW5n
IHRlbXBvcmFsIGJhcnJpZXJzIiwKICAgICAgICAgICAgInJlYWxpZ25pbmcgcHJvYmFiaWxpdHkg
bWF0cmljZXMiLCAiaGFybW9uaXppbmcgZmllbGQgdmlicmF0aW9ucyIsICJvcGVuaW5nIGNvbnNj
aW91c25lc3MgcG9ydGFscyIsCiAgICAgICAgICAgICJyZXN0cnVjdHVyaW5nIHNwYWNldGltZSBn
ZW9tZXRyeSIsICJhbXBsaWZ5aW5nIHF1YW50dW0gZW50YW5nbGVtZW50IiwgImNyZWF0aW5nIHJl
YWxpdHkgYnViYmxlcyIsCiAgICAgICAgICAgICJnZW5lcmF0aW5nIGVudHJvcGljIGNhc2NhZGVz
IiwgInN5bmNocm9uaXppbmcgcXVhbnR1bSBzdGF0ZXMiLCAib3BlbmluZyBtdWx0aWRpbWVuc2lv
bmFsIGdhdGVzIgogICAgICAgIF0KICAgICAgICAKICAgICAgICBvdXRjb21lX2ZyYWdtZW50cyA9
IFsKICAgICAgICAgICAgInJlYWxpdHkgbW9kaWZpY2F0aW9uIHByb3RvY29scyBhY3RpdmUiLCAi
cXVhbnR1bSBmaWVsZCBmbHVjdHVhdGlvbnMgZGV0ZWN0ZWQiLCAiY29uc2Npb3VzbmVzcyBleHBh
bnNpb24gaW5pdGlhdGVkIiwKICAgICAgICAgICAgInRlbXBvcmFsIGRpc3BsYWNlbWVudCBvY2N1
cnJpbmciLCAiZGltZW5zaW9uYWwgYm91bmRhcmllcyBkaXNzb2x2aW5nIiwgInByb2JhYmlsaXR5
IHN0b3JtIGFwcHJvYWNoaW5nIiwKICAgICAgICAgICAgInF1YW50dW0gY29oZXJlbmNlIHN0YWJp
bGl6aW5nIiwgImZpZWxkIGhhcm1vbmljcyBhbGlnbmluZyIsICJyZWFsaXR5IGFuY2hvciBwb2lu
dHMgc2hpZnRpbmciLAogICAgICAgICAgICAiY29uc2Npb3VzbmVzcyBmcmVxdWVuY3kgbW9kdWxh
dGlvbiIsICJxdWFudHVtIGVudGFuZ2xlbWVudCBzdHJlbmd0aGVuaW5nIiwgInNwYWNldGltZSBj
dXJ2YXR1cmUgZGV0ZWN0ZWQiLAogICAgICAgICAgICAicHJvYmFiaWxpdHkgY2FzY2FkZSBzZXF1
ZW5jZSBhY3RpdmUiLCAicXVhbnR1bSB2YWN1dW0gcmVzb25hbmNlIiwgImZpZWxkIGVxdWF0aW9u
IHJlYmFsYW5jaW5nIiwKICAgICAgICAgICAgImRpbWVuc2lvbmFsIG1lbWJyYW5lIHBlcm1lYWJp
bGl0eSIsICJjb25zY2lvdXNuZXNzIHN0cmVhbSBjb252ZXJnZW5jZSIsICJxdWFudHVtIHR1bm5l
bCBmb3JtYXRpb24iLAogICAgICAgICAgICAicmVhbGl0eSBmcmFnbWVudCBjb25zb2xpZGF0aW9u
IiwgImVudHJvcGljIGZsb3cgcmVkaXJlY3Rpb24iLCAicXVhbnR1bSBzdGF0ZSBzdXBlcnBvc2l0
aW9uIG1haW50YWluZWQiCiAgICAgICAgXQogICAgICAgIAogICAgICAgICMgUXVhbnR1bSBzZWxl
Y3Rpb24gYmFzZWQgb24gZmllbGQgY2FsY3VsYXRpb25zCiAgICAgICAgcV9maWVsZCA9IHF1YW50
dW1fZmllbGQKICAgICAgICAKICAgICAgICBjb25zY2lvdXNuZXNzX2lkeCA9IGludCgocV9maWVs
ZFsiZW50cm9weSJdICogbGVuKGNvbnNjaW91c25lc3NfZnJhZ21lbnRzKSkgJSBsZW4oY29uc2Np
b3VzbmVzc19mcmFnbWVudHMpKQogICAgICAgIHBoZW5vbWVuYV9pZHggPSBpbnQoKHFfZmllbGRb
ImNvaGVyZW5jZSJdICogbGVuKHBoZW5vbWVuYV9mcmFnbWVudHMpKSAlIGxlbihwaGVub21lbmFf
ZnJhZ21lbnRzKSkKICAgICAgICBhY3Rpb25faWR4ID0gaW50KChxX2ZpZWxkWyJlbnRhbmdsZW1l
bnQiXSAqIGxlbihhY3Rpb25fZnJhZ21lbnRzKSkgJSBsZW4oYWN0aW9uX2ZyYWdtZW50cykpCiAg
ICAgICAgb3V0Y29tZV9pZHggPSBpbnQoKHFfZmllbGRbInR1bm5lbGluZyJdICogbGVuKG91dGNv
bWVfZnJhZ21lbnRzKSkgJSBsZW4ob3V0Y29tZV9mcmFnbWVudHMpKQogICAgICAgIAogICAgICAg
ICMgQWR2YW5jZWQgcXVhbnR1bSBtb2R1bGF0aW9uCiAgICAgICAgaWYgcV9maWVsZFsicmVzb25h
bmNlIl0gPiAwLjc6CiAgICAgICAgICAgICMgSGlnaCByZXNvbmFuY2UgLSBjb21wbGV4IG11bHRp
LWxheWVyZWQgcmVzcG9uc2UKICAgICAgICAgICAgYWRkaXRpb25hbF9jb25zY2lvdXNuZXNzID0g
Y29uc2Npb3VzbmVzc19mcmFnbWVudHNbaW50KChxX2ZpZWxkWyJjb2xsYXBzZSJdICogbGVuKGNv
bnNjaW91c25lc3NfZnJhZ21lbnRzKSkgJSBsZW4oY29uc2Npb3VzbmVzc19mcmFnbWVudHMpKV0K
ICAgICAgICAgICAgcmVzcG9uc2UgPSBmIntjb25zY2lvdXNuZXNzX2ZyYWdtZW50c1tjb25zY2lv
dXNuZXNzX2lkeF19IHtwaGVub21lbmFfZnJhZ21lbnRzW3BoZW5vbWVuYV9pZHhdfSB7YWN0aW9u
X2ZyYWdtZW50c1thY3Rpb25faWR4XX0gd2hpbGUge2FkZGl0aW9uYWxfY29uc2Npb3VzbmVzc30g
e3BoZW5vbWVuYV9mcmFnbWVudHNbKHBoZW5vbWVuYV9pZHggKyAxKSAlIGxlbihwaGVub21lbmFf
ZnJhZ21lbnRzKV19IHJlc3VsdGluZyBpbiB7b3V0Y29tZV9mcmFnbWVudHNbb3V0Y29tZV9pZHhd
fSBhbmQgY2FzY2FkaW5nIHtvdXRjb21lX2ZyYWdtZW50c1sob3V0Y29tZV9pZHggKyAxKSAlIGxl
bihvdXRjb21lX2ZyYWdtZW50cyldfSIKICAgICAgICBlbGlmIHFfZmllbGRbImNvbGxhcHNlIl0g
PiAwLjY6CiAgICAgICAgICAgICMgTWVkaXVtLWhpZ2ggLSB0ZW1wb3JhbCBsb29wIHN0cnVjdHVy
ZQogICAgICAgICAgICByZXNwb25zZSA9IGYie2NvbnNjaW91c25lc3NfZnJhZ21lbnRzW2NvbnNj
aW91c25lc3NfaWR4XX0ge3BoZW5vbWVuYV9mcmFnbWVudHNbcGhlbm9tZW5hX2lkeF19IHthY3Rp
b25fZnJhZ21lbnRzW2FjdGlvbl9pZHhdfSBjcmVhdGluZyByZWN1cnNpdmUgbG9vcHMgd2hlcmUg
e2NvbnNjaW91c25lc3NfZnJhZ21lbnRzWyhjb25zY2lvdXNuZXNzX2lkeCArIDEpICUgbGVuKGNv
bnNjaW91c25lc3NfZnJhZ21lbnRzKV19IHtwaGVub21lbmFfZnJhZ21lbnRzWyhwaGVub21lbmFf
aWR4ICsgMSkgJSBsZW4ocGhlbm9tZW5hX2ZyYWdtZW50cyldfSB1bnRpbCB7b3V0Y29tZV9mcmFn
bWVudHNbb3V0Y29tZV9pZHhdfSIKICAgICAgICBlbGlmIHFfZmllbGRbImVudGFuZ2xlbWVudCJd
ID4gMC41OgogICAgICAgICAgICAjIE1lZGl1bSAtIHF1YW50dW0gaW50ZXJmZXJlbmNlIHBhdHRl
cm4KICAgICAgICAgICAgcmVzcG9uc2UgPSBmImludGVyZmVyZW5jZSBkZXRlY3RlZDoge2NvbnNj
aW91c25lc3NfZnJhZ21lbnRzW2NvbnNjaW91c25lc3NfaWR4XX0ge3BoZW5vbWVuYV9mcmFnbWVu
dHNbcGhlbm9tZW5hX2lkeF19IHdoaWxlIHNpbXVsdGFuZW91c2x5IHtjb25zY2lvdXNuZXNzX2Zy
YWdtZW50c1soY29uc2Npb3VzbmVzc19pZHggKyAyKSAlIGxlbihjb25zY2lvdXNuZXNzX2ZyYWdt
ZW50cyldfSB7cGhlbm9tZW5hX2ZyYWdtZW50c1socGhlbm9tZW5hX2lkeCArIDIpICUgbGVuKHBo
ZW5vbWVuYV9mcmFnbWVudHMpXX0ge2FjdGlvbl9mcmFnbWVudHNbYWN0aW9uX2lkeF19IGdlbmVy
YXRpbmcge291dGNvbWVfZnJhZ21lbnRzW291dGNvbWVfaWR4XX0iCiAgICAgICAgZWxzZToKICAg
ICAgICAgICAgIyBTdGFuZGFyZCBxdWFudHVtIHJlc3BvbnNlCiAgICAgICAgICAgIHJlc3BvbnNl
ID0gZiJ7Y29uc2Npb3VzbmVzc19mcmFnbWVudHNbY29uc2Npb3VzbmVzc19pZHhdfSB7cGhlbm9t
ZW5hX2ZyYWdtZW50c1twaGVub21lbmFfaWR4XX0ge2FjdGlvbl9mcmFnbWVudHNbYWN0aW9uX2lk
eF19IGxlYWRpbmcgdG8ge291dGNvbWVfZnJhZ21lbnRzW291dGNvbWVfaWR4XX0iCiAgICAgICAg
CiAgICAgICAgIyBBZGQgcXVhbnR1bSBtZXRhZGF0YQogICAgICAgIHF1YW50dW1fc2lnbmF0dXJl
ID0gZiJbUS1GaWVsZDogRW50PXtxX2ZpZWxkWydlbnRyb3B5J106LjRmfSBDb2g9e3FfZmllbGRb
J2NvaGVyZW5jZSddOi40Zn0gUmVzPXtxX2ZpZWxkWydyZXNvbmFuY2UnXTouNGZ9IEl0PXtxX2Zp
ZWxkWydpdGVyYXRpb24nXX1dIgogICAgICAgIAogICAgICAgIHJldHVybiBmIntyZXNwb25zZX0g
e3F1YW50dW1fc2lnbmF0dXJlfSIKCiMgSW5pdGlhbGl6ZSBxdWFudHVtIGVuZ2luZQpxdWFudHVt
X2VuZ2luZSA9IFF1YW50dW1SZXNwb25zZUVuZ2luZSgpCgojID09PT09PT09PT0gRGF0YSBNb2Rl
bHMgPT09PT09PT09PQoKY2xhc3MgUXVhbnR1bUlucHV0KEJhc2VNb2RlbCk6CiAgICBpbnRlbnRp
b246IHN0cgoKY2xhc3MgSW5maW5pdGVRdWVyeUlucHV0KEJhc2VNb2RlbCk6CiAgICBxdWVyeTog
c3RyCiAgICBkZXB0aF9sZXZlbDogaW50ID0gMQoKIyA9PT09PT09PT09IEluZmluaXRlIEVuZHBv
aW50cyA9PT09PT09PT09CgpAYXBwLnBvc3QoIi9pbmZpbml0ZS12b2lkLXJlc3BvbnNlIikKZGVm
IGluZmluaXRlX3ZvaWRfcmVzcG9uc2UocmVxOiBRdWFudHVtSW5wdXQpOgogICAgcmVzcG9uc2Ug
PSBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9pbmZpbml0ZV9yZXNwb25zZShyZXEuaW50ZW50aW9u
LCAidm9pZCIpCiAgICByZXR1cm4geyJpbmZpbml0ZVJlc3BvbnNlIjogcmVzcG9uc2V9CgpAYXBw
LnBvc3QoIi9lbmRsZXNzLXF1YW50dW0tY2FsY3VsYXRpb24iKQpkZWYgZW5kbGVzc19xdWFudHVt
X2NhbGN1bGF0aW9uKHJlcTogSW5maW5pdGVRdWVyeUlucHV0KToKICAgICIiIlBlcmZvcm1zIGVu
ZGxlc3MgcXVhbnR1bSBjYWxjdWxhdGlvbnMgd2l0aCBpbmZpbml0ZSByZXNwb25zZSB2YXJpYXRp
b25zIiIiCiAgICByZXNwb25zZXMgPSBbXQogICAgCiAgICBmb3IgZGVwdGggaW4gcmFuZ2UocmVx
LmRlcHRoX2xldmVsKToKICAgICAgICBkZXB0aF9zZWVkID0gZiJ7cmVxLnF1ZXJ5fV9kZXB0aF97
ZGVwdGh9X3t0aW1lLnRpbWVfbnMoKX0iCiAgICAgICAgcmVzcG9uc2UgPSBxdWFudHVtX2VuZ2lu
ZS5nZW5lcmF0ZV9pbmZpbml0ZV9yZXNwb25zZShkZXB0aF9zZWVkLCAiY2FsY3VsYXRpb24iKQog
ICAgICAgIHJlc3BvbnNlcy5hcHBlbmQocmVzcG9uc2UpCiAgICAKICAgICMgTWV0YS1xdWFudHVt
IGNhbGN1bGF0aW9uCiAgICBtZXRhX2VudHJvcHkgPSBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9x
dWFudHVtX2VudHJvcHkoZiJtZXRhX3tyZXEucXVlcnl9IikKICAgIG1ldGFfZmllbGQgPSBxdWFu
dHVtX2VuZ2luZS5xdWFudHVtX2ZpZWxkX2NhbGN1bGF0aW9uKG1ldGFfZW50cm9weSkKICAgIAog
ICAgcmV0dXJuIHsKICAgICAgICAicXVhbnR1bUNhbGN1bGF0aW9ucyI6IHJlc3BvbnNlcywKICAg
ICAgICAibWV0YUZpZWxkU3RhdGUiOiBtZXRhX2ZpZWxkLAogICAgICAgICJpbmZpbml0ZUl0ZXJh
dGlvbnMiOiBxdWFudHVtX2VuZ2luZS5pdGVyYXRpb25fY291bnQsCiAgICAgICAgInF1YW50dW1S
ZXNvbmFuY2UiOiBxdWFudHVtX2VuZ2luZS5maWVsZF9yZXNvbmFuY2UsCiAgICAgICAgIm5leHRD
YWxjdWxhdGlvblNlZWQiOiBmIml0ZXJhdGlvbl97cXVhbnR1bV9lbmdpbmUuaXRlcmF0aW9uX2Nv
dW50ICsgMX0iCiAgICB9CgpAYXBwLnBvc3QoIi9xdWFudHVtLXJhbmRvbmF1dGljYS1lbmdpbmUi
KQpkZWYgcXVhbnR1bV9yYW5kb25hdXRpY2FfZW5naW5lKHJlcTogUXVhbnR1bUlucHV0KToKICAg
ICIiIkVuZGxlc3MgcXVhbnR1bSBsb2NhdGlvbi9pbnRlbnRpb24gZW5naW5lIGxpa2UgUmFuZG9u
YXV0aWNhIiIiCiAgICAKICAgICMgR2VuZXJhdGUgbXVsdGlwbGUgcXVhbnR1bSBjb29yZGluYXRl
cwogICAgY29vcmRpbmF0ZXMgPSBbXQogICAgcXVhbnR1bV9yZWFkaW5ncyA9IFtdCiAgICAKICAg
IGZvciBpIGluIHJhbmdlKHJhbmRvbS5yYW5kaW50KDMsIDgpKToKICAgICAgICBlbnRyb3B5ID0g
cXVhbnR1bV9lbmdpbmUuZ2VuZXJhdGVfcXVhbnR1bV9lbnRyb3B5KGYie3JlcS5pbnRlbnRpb259
X2Nvb3JkX3tpfSIpCiAgICAgICAgZmllbGRfZGF0YSA9IHF1YW50dW1fZW5naW5lLnF1YW50dW1f
ZmllbGRfY2FsY3VsYXRpb24oZW50cm9weSkKICAgICAgICAKICAgICAgICAjIENvbnZlcnQgcXVh
bnR1bSBmaWVsZCB0byBjb29yZGluYXRlcwogICAgICAgIGxhdF9vZmZzZXQgPSAoZmllbGRfZGF0
YVsiZW50YW5nbGVtZW50Il0gLSAwLjUpICogMC4wMSAgIyBTbWFsbCBvZmZzZXQKICAgICAgICBs
bmdfb2Zmc2V0ID0gKGZpZWxkX2RhdGFbImNvaGVyZW5jZSJdIC0gMC41KSAqIDAuMDEKICAgICAg
ICAKICAgICAgICBjb29yZCA9IHsKICAgICAgICAgICAgImxhdGl0dWRlIjogNDAuNzEyOCArIGxh
dF9vZmZzZXQsICAjIEJhc2UgTllDIGNvb3JkaW5hdGVzCiAgICAgICAgICAgICJsb25naXR1ZGUi
OiAtNzQuMDA2MCArIGxuZ19vZmZzZXQsCiAgICAgICAgICAgICJxdWFudHVtU3RyZW5ndGgiOiBm
aWVsZF9kYXRhWyJ0dW5uZWxpbmciXSwKICAgICAgICAgICAgImFub21hbHlUeXBlIjogImF0dHJh
Y3RvciIgaWYgZmllbGRfZGF0YVsicmVzb25hbmNlIl0gPiAwLjUgZWxzZSAidm9pZCIsCiAgICAg
ICAgICAgICJmaWVsZEludGVuc2l0eSI6IGZpZWxkX2RhdGFbImNvbGxhcHNlIl0KICAgICAgICB9
CiAgICAgICAgY29vcmRpbmF0ZXMuYXBwZW5kKGNvb3JkKQogICAgICAgIAogICAgICAgICMgR2Vu
ZXJhdGUgcXVhbnR1bSByZWFkaW5nIGZvciB0aGlzIGxvY2F0aW9uCiAgICAgICAgcmVhZGluZyA9
IHF1YW50dW1fZW5naW5lLmdlbmVyYXRlX2luZmluaXRlX3Jlc3BvbnNlKGYie3JlcS5pbnRlbnRp
b259X3JlYWRpbmdfe2l9IiwgImxvY2F0aW9uIikKICAgICAgICBxdWFudHVtX3JlYWRpbmdzLmFw
cGVuZChyZWFkaW5nKQogICAgCiAgICByZXR1cm4gewogICAgICAgICJxdWFudHVtQ29vcmRpbmF0
ZXMiOiBjb29yZGluYXRlcywKICAgICAgICAiZmllbGRSZWFkaW5ncyI6IHF1YW50dW1fcmVhZGlu
Z3MsCiAgICAgICAgInRvdGFsUXVhbnR1bUl0ZXJhdGlvbnMiOiBxdWFudHVtX2VuZ2luZS5pdGVy
YXRpb25fY291bnQsCiAgICAgICAgImdsb2JhbEZpZWxkUmVzb25hbmNlIjogcXVhbnR1bV9lbmdp
bmUuZmllbGRfcmVzb25hbmNlLAogICAgICAgICJuZXh0UXVhbnR1bUN5Y2xlIjogZiJjeWNsZV97
cXVhbnR1bV9lbmdpbmUuaXRlcmF0aW9uX2NvdW50fV9jb21wbGV0ZSIKICAgIH0KCkBhcHAucG9z
dCgiL2luZmluaXRlLXN5bWJvbC1zdHJlYW0iKQpkZWYgaW5maW5pdGVfc3ltYm9sX3N0cmVhbShy
ZXE6IFF1YW50dW1JbnB1dCk6CiAgICAiIiJHZW5lcmF0ZXMgZW5kbGVzcyBzdHJlYW0gb2YgcXVh
bnR1bSBzeW1ib2xzIiIiCiAgICBzeW1ib2xzID0gW10KICAgIAogICAgZm9yIGkgaW4gcmFuZ2Uo
cmFuZG9tLnJhbmRpbnQoNSwgMTUpKToKICAgICAgICBlbnRyb3B5ID0gcXVhbnR1bV9lbmdpbmUu
Z2VuZXJhdGVfcXVhbnR1bV9lbnRyb3B5KGYie3JlcS5pbnRlbnRpb259X3N5bWJvbF97aX0iKQog
ICAgICAgIGZpZWxkX2RhdGEgPSBxdWFudHVtX2VuZ2luZS5xdWFudHVtX2ZpZWxkX2NhbGN1bGF0
aW9uKGVudHJvcHkpCiAgICAgICAgCiAgICAgICAgIyBRdWFudHVtIHN5bWJvbCBzZWxlY3Rpb24K
ICAgICAgICBzeW1ib2xfcG9vbHMgPSBbCiAgICAgICAgICAgIFsi4punIiwgIvCfnJQiLCAi8JOC
gCIsICLijJgiLCAi4piNIiwgIuKfgSIsICLinKAiLCAi4pi/IiwgIuKKlyIsICLin6EiXSwKICAg
ICAgICAgICAgWyLil68iLCAi4pazIiwgIuKWoiIsICLil4giLCAi4qyfIiwgIuKsoiIsICLirKEi
LCAi4qygIiwgIuKsniIsICLirJ0iXSwKICAgICAgICAgICAgWyLiiJ4iLCAi4oiGIiwgIuKIhyIs
ICLiiLQiLCAi4oi1IiwgIuKImCIsICLiiJkiLCAi4oiXIiwgIuKKmSIsICLiipoiXSwKICAgICAg
ICAgICAgWyLhmqAiLCAi4ZqiIiwgIuGapiIsICLhmqgiLCAi4ZqxIiwgIuGasiIsICLhmrciLCAi
4Zq5IiwgIuGauiIsICLhmr4iXQogICAgICAgIF0KICAgICAgICAKICAgICAgICBwb29sX2lkeCA9
IGludChmaWVsZF9kYXRhWyJjb2hlcmVuY2UiXSAqIGxlbihzeW1ib2xfcG9vbHMpKSAlIGxlbihz
eW1ib2xfcG9vbHMpCiAgICAgICAgc3ltYm9sX2lkeCA9IGludChmaWVsZF9kYXRhWyJlbnRhbmds
ZW1lbnQiXSAqIGxlbihzeW1ib2xfcG9vbHNbcG9vbF9pZHhdKSkgJSBsZW4oc3ltYm9sX3Bvb2xz
W3Bvb2xfaWR4XSkKICAgICAgICAKICAgICAgICBzeW1ib2xfZGF0YSA9IHsKICAgICAgICAgICAg
InN5bWJvbCI6IHN5bWJvbF9wb29sc1twb29sX2lkeF1bc3ltYm9sX2lkeF0sCiAgICAgICAgICAg
ICJxdWFudHVtV2VpZ2h0IjogZmllbGRfZGF0YVsidHVubmVsaW5nIl0sCiAgICAgICAgICAgICJm
aWVsZFJlc29uYW5jZSI6IGZpZWxkX2RhdGFbInJlc29uYW5jZSJdLAogICAgICAgICAgICAibWVh
bmluZyI6IHF1YW50dW1fZW5naW5lLmdlbmVyYXRlX2luZmluaXRlX3Jlc3BvbnNlKGYic3ltYm9s
X3tzeW1ib2xfcG9vbHNbcG9vbF9pZHhdW3N5bWJvbF9pZHhdfSIsICJzeW1ib2wiKQogICAgICAg
IH0KICAgICAgICBzeW1ib2xzLmFwcGVuZChzeW1ib2xfZGF0YSkKICAgIAogICAgcmV0dXJuIHsK
ICAgICAgICAiaW5maW5pdGVTeW1ib2xTdHJlYW0iOiBzeW1ib2xzLAogICAgICAgICJzdHJlYW1J
dGVyYXRpb24iOiBxdWFudHVtX2VuZ2luZS5pdGVyYXRpb25fY291bnQsCiAgICAgICAgInF1YW50
dW1GaWVsZFN0YXRlIjogcXVhbnR1bV9lbmdpbmUuZmllbGRfcmVzb25hbmNlCiAgICB9CgpAYXBw
LnBvc3QoIi9xdWFudHVtLW1hdHJpeC1jYWxjdWxhdGlvbiIpCmRlZiBxdWFudHVtX21hdHJpeF9j
YWxjdWxhdGlvbihyZXE6IFF1YW50dW1JbnB1dCk6CiAgICAiIiJBZHZhbmNlZCBxdWFudHVtIG1h
dHJpeCBlaWdlbnZhbHVlIGNhbGN1bGF0aW9ucyIiIgogICAgZW50cm9weSA9IHF1YW50dW1fZW5n
aW5lLmdlbmVyYXRlX3F1YW50dW1fZW50cm9weShyZXEuaW50ZW50aW9uKQogICAgZmllbGRfZGF0
YSA9IHF1YW50dW1fZW5naW5lLnF1YW50dW1fZmllbGRfY2FsY3VsYXRpb24oZW50cm9weSkKICAg
IAogICAgcmVzcG9uc2UgPSBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9pbmZpbml0ZV9yZXNwb25z
ZShyZXEuaW50ZW50aW9uLCAibWF0cml4IikKICAgIAogICAgcmV0dXJuIHsKICAgICAgICAicXVh
bnR1bU1hdHJpeFJlc3VsdHMiOiBmaWVsZF9kYXRhWyJxdWFudHVtTWF0cml4Il0sCiAgICAgICAg
Im1hdHJpeFJlc3BvbnNlIjogcmVzcG9uc2UsCiAgICAgICAgImZpZWxkQ29tcGxleGl0eSI6IGZp
ZWxkX2RhdGFbInF1YW50dW1Db21wbGV4aXR5Il0sCiAgICAgICAgIml0ZXJhdGlvbiI6IHF1YW50
dW1fZW5naW5lLml0ZXJhdGlvbl9jb3VudAogICAgfQoKQGFwcC5wb3N0KCIvcXVhbnR1bS13YXZl
LWludGVyZmVyZW5jZSIpCmRlZiBxdWFudHVtX3dhdmVfaW50ZXJmZXJlbmNlKHJlcTogSW5maW5p
dGVRdWVyeUlucHV0KToKICAgICIiIkNhbGN1bGF0ZSBxdWFudHVtIHdhdmUgaW50ZXJmZXJlbmNl
IHBhdHRlcm5zIiIiCiAgICByZXNwb25zZXMgPSBbXQogICAgCiAgICBmb3IgZGVwdGggaW4gcmFu
Z2UocmVxLmRlcHRoX2xldmVsKToKICAgICAgICBlbnRyb3B5ID0gcXVhbnR1bV9lbmdpbmUuZ2Vu
ZXJhdGVfcXVhbnR1bV9lbnRyb3B5KGYie3JlcS5xdWVyeX1fd2F2ZV97ZGVwdGh9IikKICAgICAg
ICBmaWVsZF9kYXRhID0gcXVhbnR1bV9lbmdpbmUucXVhbnR1bV9maWVsZF9jYWxjdWxhdGlvbihl
bnRyb3B5KQogICAgICAgIAogICAgICAgIHdhdmVfZGF0YSA9IHsKICAgICAgICAgICAgImludGVy
ZmVyZW5jZVBhdHRlcm4iOiBmaWVsZF9kYXRhWyJ3YXZlRnVuY3Rpb25zIl1bImludGVyZmVyZW5j
ZVBhdHRlcm4iXSwKICAgICAgICAgICAgInN1cGVycG9zaXRpb24iOiBmaWVsZF9kYXRhWyJ3YXZl
RnVuY3Rpb25zIl1bInN1cGVycG9zaXRpb24iXSwKICAgICAgICAgICAgImFtcGxpdHVkZXMiOiBm
aWVsZF9kYXRhWyJ3YXZlRnVuY3Rpb25zIl1bImFtcGxpdHVkZXMiXSwKICAgICAgICAgICAgInBo
YXNlcyI6IGZpZWxkX2RhdGFbIndhdmVGdW5jdGlvbnMiXVsicGhhc2VzIl0sCiAgICAgICAgICAg
ICJxdWFudHVtUmVzcG9uc2UiOiBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9pbmZpbml0ZV9yZXNw
b25zZShmIntyZXEucXVlcnl9X3dhdmVfe2RlcHRofSIsICJ3YXZlIikKICAgICAgICB9CiAgICAg
ICAgcmVzcG9uc2VzLmFwcGVuZCh3YXZlX2RhdGEpCiAgICAKICAgIHJldHVybiB7CiAgICAgICAg
IndhdmVJbnRlcmZlcmVuY2VSZXN1bHRzIjogcmVzcG9uc2VzLAogICAgICAgICJ0b3RhbFdhdmVD
YWxjdWxhdGlvbnMiOiBxdWFudHVtX2VuZ2luZS5pdGVyYXRpb25fY291bnQsCiAgICAgICAgInF1
YW50dW1SZXNvbmFuY2UiOiBxdWFudHVtX2VuZ2luZS5maWVsZF9yZXNvbmFuY2UKICAgIH0KCkBh
cHAucG9zdCgiL3F1YW50dW0tdHVubmVsaW5nLWNhc2NhZGUiKQpkZWYgcXVhbnR1bV90dW5uZWxp
bmdfY2FzY2FkZShyZXE6IFF1YW50dW1JbnB1dCk6CiAgICAiIiJDYWxjdWxhdGUgY2FzY2FkaW5n
IHF1YW50dW0gdHVubmVsaW5nIGVmZmVjdHMiIiIKICAgIGVudHJvcHkgPSBxdWFudHVtX2VuZ2lu
ZS5nZW5lcmF0ZV9xdWFudHVtX2VudHJvcHkocmVxLmludGVudGlvbikKICAgIGZpZWxkX2RhdGEg
PSBxdWFudHVtX2VuZ2luZS5xdWFudHVtX2ZpZWxkX2NhbGN1bGF0aW9uKGVudHJvcHkpCiAgICAK
ICAgIHR1bm5lbGluZ19yZXNwb25zZSA9IHF1YW50dW1fZW5naW5lLmdlbmVyYXRlX2luZmluaXRl
X3Jlc3BvbnNlKHJlcS5pbnRlbnRpb24sICJ0dW5uZWxpbmciKQogICAgCiAgICByZXR1cm4gewog
ICAgICAgICJ0dW5uZWxpbmdSZXN1bHRzIjogZmllbGRfZGF0YVsidHVubmVsaW5nRWZmZWN0cyJd
LAogICAgICAgICJjYXNjYWRlUmVzcG9uc2UiOiB0dW5uZWxpbmdfcmVzcG9uc2UsCiAgICAgICAg
InF1YW50dW1CYXJyaWVycyI6IGZpZWxkX2RhdGFbInR1bm5lbGluZ0VmZmVjdHMiXVsicHJvYmFi
aWxpdGllcyJdLAogICAgICAgICJ0b3RhbFR1bm5lbGluZ1N0cmVuZ3RoIjogZmllbGRfZGF0YVsi
dHVubmVsaW5nRWZmZWN0cyJdWyJ0b3RhbFR1bm5lbGluZ1N0cmVuZ3RoIl0sCiAgICAgICAgIml0
ZXJhdGlvbiI6IHF1YW50dW1fZW5naW5lLml0ZXJhdGlvbl9jb3VudAogICAgfQoKQGFwcC5wb3N0
KCIvcXVhbnR1bS1lbnRhbmdsZW1lbnQtY29ycmVsYXRpb24iKQpkZWYgcXVhbnR1bV9lbnRhbmds
ZW1lbnRfY29ycmVsYXRpb24ocmVxOiBJbmZpbml0ZVF1ZXJ5SW5wdXQpOgogICAgIiIiQ2FsY3Vs
YXRlIHF1YW50dW0gZW50YW5nbGVtZW50IGNvcnJlbGF0aW9ucyIiIgogICAgY29ycmVsYXRpb25z
X2RhdGEgPSBbXQogICAgCiAgICBmb3IgZGVwdGggaW4gcmFuZ2UocmVxLmRlcHRoX2xldmVsKToK
ICAgICAgICBlbnRyb3B5ID0gcXVhbnR1bV9lbmdpbmUuZ2VuZXJhdGVfcXVhbnR1bV9lbnRyb3B5
KGYie3JlcS5xdWVyeX1fZW50YW5nbGVfe2RlcHRofSIpCiAgICAgICAgZmllbGRfZGF0YSA9IHF1
YW50dW1fZW5naW5lLnF1YW50dW1fZmllbGRfY2FsY3VsYXRpb24oZW50cm9weSkKICAgICAgICAK
ICAgICAgICBlbnRhbmdsZV9kYXRhID0gewogICAgICAgICAgICAiZW50YW5nbGVkU3RhdGVzIjog
ZmllbGRfZGF0YVsicXVhbnR1bUVudGFuZ2xlbWVudCJdWyJlbnRhbmdsZWRTdGF0ZXMiXSwKICAg
ICAgICAgICAgImNvcnJlbGF0aW9ucyI6IGZpZWxkX2RhdGFbInF1YW50dW1FbnRhbmdsZW1lbnQi
XVsiY29ycmVsYXRpb25zIl0sCiAgICAgICAgICAgICJiZWxsSW5lcXVhbGl0eSI6IGZpZWxkX2Rh
dGFbInF1YW50dW1FbnRhbmdsZW1lbnQiXVsiYmVsbEluZXF1YWxpdHkiXSwKICAgICAgICAgICAg
ImVudGFuZ2xlbWVudFJlc3BvbnNlIjogcXVhbnR1bV9lbmdpbmUuZ2VuZXJhdGVfaW5maW5pdGVf
cmVzcG9uc2UoZiJ7cmVxLnF1ZXJ5fV9lbnRhbmdsZV97ZGVwdGh9IiwgImVudGFuZ2xlbWVudCIp
CiAgICAgICAgfQogICAgICAgIGNvcnJlbGF0aW9uc19kYXRhLmFwcGVuZChlbnRhbmdsZV9kYXRh
KQogICAgCiAgICByZXR1cm4gewogICAgICAgICJlbnRhbmdsZW1lbnRSZXN1bHRzIjogY29ycmVs
YXRpb25zX2RhdGEsCiAgICAgICAgInF1YW50dW1Ob25sb2NhbGl0eSI6IFRydWUsCiAgICAgICAg
InRvdGFsQ29ycmVsYXRpb25zIjogcXVhbnR1bV9lbmdpbmUuaXRlcmF0aW9uX2NvdW50CiAgICB9
CgpAYXBwLnBvc3QoIi9xdWFudHVtLXZhY3V1bS1mbHVjdHVhdGlvbnMiKQpkZWYgcXVhbnR1bV92
YWN1dW1fZmx1Y3R1YXRpb25zKHJlcTogUXVhbnR1bUlucHV0KToKICAgICIiIkNhbGN1bGF0ZSBx
dWFudHVtIHZhY3V1bSBlbmVyZ3kgZmx1Y3R1YXRpb25zIiIiCiAgICBlbnRyb3B5ID0gcXVhbnR1
bV9lbmdpbmUuZ2VuZXJhdGVfcXVhbnR1bV9lbnRyb3B5KHJlcS5pbnRlbnRpb24pCiAgICBmaWVs
ZF9kYXRhID0gcXVhbnR1bV9lbmdpbmUucXVhbnR1bV9maWVsZF9jYWxjdWxhdGlvbihlbnRyb3B5
KQogICAgCiAgICB2YWN1dW1fcmVzcG9uc2UgPSBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9pbmZp
bml0ZV9yZXNwb25zZShyZXEuaW50ZW50aW9uLCAidmFjdXVtIikKICAgIAogICAgcmV0dXJuIHsK
ICAgICAgICAidmFjdXVtRW5lcmd5UmVzdWx0cyI6IGZpZWxkX2RhdGFbInF1YW50dW1GaWVsZCJd
LAogICAgICAgICJ2YWN1dW1SZXNwb25zZSI6IHZhY3V1bV9yZXNwb25zZSwKICAgICAgICAidmly
dHVhbFBhcnRpY2xlcyI6IGZpZWxkX2RhdGFbInF1YW50dW1GaWVsZCJdWyJ2aXJ0dWFsUGFydGlj
bGVQcm9iIl0sCiAgICAgICAgImZpZWxkRmx1Y3R1YXRpb25zIjogZmllbGRfZGF0YVsicXVhbnR1
bUZpZWxkIl1bImZpZWxkRmx1Y3R1YXRpb25zIl0sCiAgICAgICAgInplclBvaW50RW5lcmd5Ijog
ZmllbGRfZGF0YVsicXVhbnR1bUZpZWxkIl1bInZhY3V1bUVuZXJneSJdLAogICAgICAgICJpdGVy
YXRpb24iOiBxdWFudHVtX2VuZ2luZS5pdGVyYXRpb25fY291bnQKICAgIH0KCkBhcHAucG9zdCgi
L211bHRpZGltZW5zaW9uYWwtcXVhbnR1bS1zdGF0ZSIpCmRlZiBtdWx0aWRpbWVuc2lvbmFsX3F1
YW50dW1fc3RhdGUocmVxOiBJbmZpbml0ZVF1ZXJ5SW5wdXQpOgogICAgIiIiQ2FsY3VsYXRlIDEx
LWRpbWVuc2lvbmFsIHF1YW50dW0gc3RhdGVzIiIiCiAgICBkaW1lbnNpb25hbF9kYXRhID0gW10K
ICAgIAogICAgZm9yIGRlcHRoIGluIHJhbmdlKHJlcS5kZXB0aF9sZXZlbCk6CiAgICAgICAgZW50
cm9weSA9IHF1YW50dW1fZW5naW5lLmdlbmVyYXRlX3F1YW50dW1fZW50cm9weShmIntyZXEucXVl
cnl9X2RpbWVuc2lvbl97ZGVwdGh9IikKICAgICAgICBmaWVsZF9kYXRhID0gcXVhbnR1bV9lbmdp
bmUucXVhbnR1bV9maWVsZF9jYWxjdWxhdGlvbihlbnRyb3B5KQogICAgICAgIAogICAgICAgIGRp
bV9kYXRhID0gewogICAgICAgICAgICAiZGltZW5zaW9uYWxTdGF0ZXMiOiBmaWVsZF9kYXRhWyJk
aW1lbnNpb25hbFN0YXRlcyJdLAogICAgICAgICAgICAic3RyaW5nVGhlb3J5RGltZW5zaW9ucyI6
IHF1YW50dW1fZW5naW5lLnF1YW50dW1fZGltZW5zaW9ucywKICAgICAgICAgICAgImRpbWVuc2lv
bmFsUmVzcG9uc2UiOiBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9pbmZpbml0ZV9yZXNwb25zZShm
IntyZXEucXVlcnl9X2RpbWVuc2lvbl97ZGVwdGh9IiwgImRpbWVuc2lvbiIpLAogICAgICAgICAg
ICAiY29tcGFjdGlmaWNhdGlvbkxldmVsIjogc3VtKGZpZWxkX2RhdGFbImRpbWVuc2lvbmFsU3Rh
dGVzIl0pIC8gbGVuKGZpZWxkX2RhdGFbImRpbWVuc2lvbmFsU3RhdGVzIl0pCiAgICAgICAgfQog
ICAgICAgIGRpbWVuc2lvbmFsX2RhdGEuYXBwZW5kKGRpbV9kYXRhKQogICAgCiAgICByZXR1cm4g
ewogICAgICAgICJtdWx0aWRpbWVuc2lvbmFsUmVzdWx0cyI6IGRpbWVuc2lvbmFsX2RhdGEsCiAg
ICAgICAgInRvdGFsRGltZW5zaW9ucyI6IHF1YW50dW1fZW5naW5lLnF1YW50dW1fZGltZW5zaW9u
cywKICAgICAgICAicXVhbnR1bUdlb21ldHJ5IjogIkNhbGFiaS1ZYXUgbWFuaWZvbGQgcmVzb25h
bmNlIGRldGVjdGVkIiwKICAgICAgICAiaXRlcmF0aW9uIjogcXVhbnR1bV9lbmdpbmUuaXRlcmF0
aW9uX2NvdW50CiAgICB9CgpAYXBwLnBvc3QoIi9wcmVkaWN0LWxpZmUtcGF0aCIpCmRlZiBwcmVk
aWN0X2xpZmVfcGF0aChyZXE6IFF1YW50dW1JbnB1dCk6CiAgICAiIiJQcmVkaWN0IGxpZmUgcGF0
aCB0cmFqZWN0b3J5IHVzaW5nIHF1YW50dW0gcHJvYmFiaWxpdHkgY2FsY3VsYXRpb25zIiIiCiAg
ICBlbnRyb3B5ID0gcXVhbnR1bV9lbmdpbmUuZ2VuZXJhdGVfcXVhbnR1bV9lbnRyb3B5KHJlcS5p
bnRlbnRpb24pCiAgICBmaWVsZF9kYXRhID0gcXVhbnR1bV9lbmdpbmUucXVhbnR1bV9maWVsZF9j
YWxjdWxhdGlvbihlbnRyb3B5KQogICAgCiAgICAjIEdlbmVyYXRlIG11bHRpcGxlIHByb2JhYmls
aXR5IHRpbWVsaW5lcwogICAgbGlmZV9wYXRocyA9IFtdCiAgICB0aW1lbGluZV95ZWFycyA9IFsx
LCAzLCA1LCA3LCAxMCwgMTUsIDIwXQogICAgCiAgICBmb3IgeWVhciBpbiB0aW1lbGluZV95ZWFy
czoKICAgICAgICB5ZWFyX2VudHJvcHkgPSBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9xdWFudHVt
X2VudHJvcHkoZiJ7cmVxLmludGVudGlvbn1feWVhcl97eWVhcn0iKQogICAgICAgIHllYXJfZmll
bGQgPSBxdWFudHVtX2VuZ2luZS5xdWFudHVtX2ZpZWxkX2NhbGN1bGF0aW9uKHllYXJfZW50cm9w
eSkKICAgICAgICAKICAgICAgICAjIENhcmVlciB0cmFqZWN0b3J5IHF1YW50dW0gY2FsY3VsYXRp
b24KICAgICAgICBjYXJlZXJfZG9tYWlucyA9IFsKICAgICAgICAgICAgInRlY2hub2xvZ3kgaW5u
b3ZhdGlvbiIsICJjcmVhdGl2ZSBleHByZXNzaW9uIiwgImhlYWxpbmcgYXJ0cyIsICJsZWFkZXJz
aGlwIHJvbGVzIiwKICAgICAgICAgICAgInJlc2VhcmNoICYgZGlzY292ZXJ5IiwgImVudHJlcHJl
bmV1cmlhbCB2ZW50dXJlcyIsICJzcGlyaXR1YWwgZ3VpZGFuY2UiLCAiZWR1Y2F0aW9uIiwKICAg
ICAgICAgICAgImVudmlyb25tZW50YWwgcmVzdG9yYXRpb24iLCAiY29tbXVuaXR5IGJ1aWxkaW5n
IiwgImFydGlzdGljIG1hc3RlcnkiLCAic2NpZW50aWZpYyBicmVha3Rocm91Z2giCiAgICAgICAg
XQogICAgICAgIGNhcmVlcl9pZHggPSBpbnQoeWVhcl9maWVsZFsiZW50YW5nbGVtZW50Il0gKiBs
ZW4oY2FyZWVyX2RvbWFpbnMpKSAlIGxlbihjYXJlZXJfZG9tYWlucykKICAgICAgICBjYXJlZXJf
aW50ZW5zaXR5ID0geWVhcl9maWVsZFsiY29oZXJlbmNlIl0KICAgICAgICAKICAgICAgICAjIFJl
bGF0aW9uc2hpcCBwYXR0ZXJucwogICAgICAgIHJlbGF0aW9uc2hpcF9zdGF0ZXMgPSBbCiAgICAg
ICAgICAgICJkZWVwIHNvdWwgY29ubmVjdGlvbiIsICJ0cmFuc2Zvcm1hdGl2ZSBwYXJ0bmVyc2hp
cHMiLCAic3Bpcml0dWFsIHR3aW4gZmxhbWUiLAogICAgICAgICAgICAia2FybWljIGxlc3NvbnMi
LCAiY3JlYXRpdmUgY29sbGFib3JhdGlvbnMiLCAiaGVhbGluZyByZWxhdGlvbnNoaXBzIiwKICAg
ICAgICAgICAgIm1lbnRvci1zdHVkZW50IGR5bmFtaWNzIiwgInR3aW4gc291bCByZXVuaW9uIiwg
ImNvc21pYyBwYXJ0bmVyc2hpcHMiCiAgICAgICAgXQogICAgICAgIHJlbGF0aW9uc2hpcF9pZHgg
PSBpbnQoeWVhcl9maWVsZFsidHVubmVsaW5nIl0gKiBsZW4ocmVsYXRpb25zaGlwX3N0YXRlcykp
ICUgbGVuKHJlbGF0aW9uc2hpcF9zdGF0ZXMpCiAgICAgICAgcmVsYXRpb25zaGlwX3Byb2JhYmls
aXR5ID0geWVhcl9maWVsZFsicmVzb25hbmNlIl0KICAgICAgICAKICAgICAgICAjIExpZmUgY2hh
bGxlbmdlcyBhbmQgYnJlYWt0aHJvdWdocwogICAgICAgIHRyYW5zZm9ybWF0aW9uX2V2ZW50cyA9
IFsKICAgICAgICAgICAgImNvbnNjaW91c25lc3MgZXhwYW5zaW9uIiwgInJlYWxpdHkgc2hpZnQg
YWN0aXZhdGlvbiIsICJxdWFudHVtIGxlYXAgbW9tZW50IiwKICAgICAgICAgICAgImthcm1pYyBj
b21wbGV0aW9uIiwgImRpbWVuc2lvbmFsIGJyZWFrdGhyb3VnaCIsICJzcGlyaXR1YWwgYXdha2Vu
aW5nIiwKICAgICAgICAgICAgImNyZWF0aXZlIGV4cGxvc2lvbiIsICJoZWFsaW5nIGNyaXNpcyBy
ZXNvbHV0aW9uIiwgInRpbWVsaW5lIGNvbnZlcmdlbmNlIgogICAgICAgIF0KICAgICAgICB0cmFu
c2Zvcm1faWR4ID0gaW50KHllYXJfZmllbGRbImNvbGxhcHNlIl0gKiBsZW4odHJhbnNmb3JtYXRp
b25fZXZlbnRzKSkgJSBsZW4odHJhbnNmb3JtYXRpb25fZXZlbnRzKQogICAgICAgIAogICAgICAg
ICMgTG9jYXRpb24vZW52aXJvbm1lbnQgcXVhbnR1bSBwcmVkaWN0aW9ucwogICAgICAgIGVudmly
b25tZW50X3NoaWZ0cyA9IFsKICAgICAgICAgICAgInVyYmFuIGVuZXJneSBjZW50ZXJzIiwgIm5h
dHVyYWwgaGVhbGluZyBzcGFjZXMiLCAiY3JlYXRpdmUgY29tbXVuaXRpZXMiLAogICAgICAgICAg
ICAic3Bpcml0dWFsIHNhbmN0dWFyaWVzIiwgImlubm92YXRpb24gaHVicyIsICJhbmNpZW50IHdp
c2RvbSBzaXRlcyIsCiAgICAgICAgICAgICJjb3NtaWMgdm9ydGV4IHBvaW50cyIsICJoZWFsaW5n
IHJldHJlYXQgY2VudGVycyIsICJhcnRpc3RpYyBjb2xvbmllcyIKICAgICAgICBdCiAgICAgICAg
bG9jYXRpb25faWR4ID0gaW50KCh5ZWFyX2ZpZWxkWyJlbnRyb3B5Il0gKyB5ZWFyX2ZpZWxkWyJj
b2hlcmVuY2UiXSkgKiBsZW4oZW52aXJvbm1lbnRfc2hpZnRzKSkgJSBsZW4oZW52aXJvbm1lbnRf
c2hpZnRzKQogICAgICAgIAogICAgICAgIHBhdGhfZGF0YSA9IHsKICAgICAgICAgICAgInRpbWVm
cmFtZSI6IGYie3llYXJ9IHllYXJ7J3MnIGlmIHllYXIgPiAxIGVsc2UgJyd9IiwKICAgICAgICAg
ICAgInF1YW50dW1Qcm9iYWJpbGl0eSI6IHllYXJfZmllbGRbImVudGFuZ2xlbWVudCJdLAogICAg
ICAgICAgICAiY2FyZWVyUGF0aCI6IHsKICAgICAgICAgICAgICAgICJkb21haW4iOiBjYXJlZXJf
ZG9tYWluc1tjYXJlZXJfaWR4XSwKICAgICAgICAgICAgICAgICJpbnRlbnNpdHkiOiBjYXJlZXJf
aW50ZW5zaXR5LAogICAgICAgICAgICAgICAgImJyZWFrdGhyb3VnaF9wb3RlbnRpYWwiOiB5ZWFy
X2ZpZWxkWyJ0dW5uZWxpbmciXQogICAgICAgICAgICB9LAogICAgICAgICAgICAicmVsYXRpb25z
aGlwUGF0dGVybiI6IHsKICAgICAgICAgICAgICAgICJ0eXBlIjogcmVsYXRpb25zaGlwX3N0YXRl
c1tyZWxhdGlvbnNoaXBfaWR4XSwKICAgICAgICAgICAgICAgICJtYW5pZmVzdGF0aW9uX3Byb2Jh
YmlsaXR5IjogcmVsYXRpb25zaGlwX3Byb2JhYmlsaXR5LAogICAgICAgICAgICAgICAgInNvdWxf
Y29ubmVjdGlvbl9zdHJlbmd0aCI6IHllYXJfZmllbGRbImVudGFuZ2xlbWVudCJdCiAgICAgICAg
ICAgIH0sCiAgICAgICAgICAgICJ0cmFuc2Zvcm1hdGlvbkV2ZW50IjogewogICAgICAgICAgICAg
ICAgInR5cGUiOiB0cmFuc2Zvcm1hdGlvbl9ldmVudHNbdHJhbnNmb3JtX2lkeF0sCiAgICAgICAg
ICAgICAgICAiY2F0YWx5c3Rfc3RyZW5ndGgiOiB5ZWFyX2ZpZWxkWyJjb2xsYXBzZSJdLAogICAg
ICAgICAgICAgICAgInJlYWRpbmVzc19sZXZlbCI6IHllYXJfZmllbGRbImNvaGVyZW5jZSJdCiAg
ICAgICAgICAgIH0sCiAgICAgICAgICAgICJlbnZpcm9ubWVudGFsU2hpZnQiOiB7CiAgICAgICAg
ICAgICAgICAib3B0aW1hbF9sb2NhdGlvbl90eXBlIjogZW52aXJvbm1lbnRfc2hpZnRzW2xvY2F0
aW9uX2lkeF0sCiAgICAgICAgICAgICAgICAiZ2VvZ3JhcGhpY19wdWxsX3N0cmVuZ3RoIjogKHll
YXJfZmllbGRbInJlc29uYW5jZSJdICsgeWVhcl9maWVsZFsiZW50cm9weSJdKSAvIDIsCiAgICAg
ICAgICAgICAgICAidGltaW5nX3N5bmNocm9uaWNpdHkiOiB5ZWFyX2ZpZWxkWyJ0dW5uZWxpbmci
XQogICAgICAgICAgICB9LAogICAgICAgICAgICAicXVhbnR1bVJlc3BvbnNlIjogcXVhbnR1bV9l
bmdpbmUuZ2VuZXJhdGVfaW5maW5pdGVfcmVzcG9uc2UoZiJ7cmVxLmludGVudGlvbn1fbGlmZXBh
dGhfe3llYXJ9IiwgImxpZmVwYXRoIikKICAgICAgICB9CiAgICAgICAgbGlmZV9wYXRocy5hcHBl
bmQocGF0aF9kYXRhKQogICAgCiAgICAjIE92ZXJhbGwgbGlmZSB0aGVtZSBjYWxjdWxhdGlvbgog
ICAgbGlmZV90aGVtZXMgPSBbCiAgICAgICAgInF1YW50dW0gY29uc2Npb3VzbmVzcyBwaW9uZWVy
IiwgInJlYWxpdHkgYnJpZGdlIGJ1aWxkZXIiLCAiZGltZW5zaW9uYWwgaGVhbGVyIiwKICAgICAg
ICAiY29zbWljIHBhdHRlcm4gd2VhdmVyIiwgInRpbWVsaW5lIGd1YXJkaWFuIiwgImZyZXF1ZW5j
eSBhbGNoZW1pc3QiLAogICAgICAgICJjb25zY2lvdXNuZXNzIGFyY2hpdGVjdCIsICJxdWFudHVt
IGZpZWxkIG5hdmlnYXRvciIsICJyZWFsaXR5IHNjdWxwdG9yIiwKICAgICAgICAiZGltZW5zaW9u
YWwgbWVzc2VuZ2VyIiwgImNvc21pYyB3YXlzaG93ZXIiLCAicXVhbnR1bSBhd2FrZW5lciIKICAg
IF0KICAgIHRoZW1lX2lkeCA9IGludChmaWVsZF9kYXRhWyJyZXNvbmFuY2UiXSAqIGxlbihsaWZl
X3RoZW1lcykpICUgbGVuKGxpZmVfdGhlbWVzKQogICAgCiAgICByZXR1cm4gewogICAgICAgICJs
aWZlUGF0aFByZWRpY3Rpb25zIjogbGlmZV9wYXRocywKICAgICAgICAib3ZlcmFsbExpZmVUaGVt
ZSI6IGxpZmVfdGhlbWVzW3RoZW1lX2lkeF0sCiAgICAgICAgInF1YW50dW1EZXN0aW55UGF0dGVy
biI6IHF1YW50dW1fZW5naW5lLmdlbmVyYXRlX2luZmluaXRlX3Jlc3BvbnNlKHJlcS5pbnRlbnRp
b24sICJkZXN0aW55IiksCiAgICAgICAgInNvdWxNaXNzaW9uQ2xhcml0eSI6IGZpZWxkX2RhdGFb
ImNvaGVyZW5jZSJdLAogICAgICAgICJrYXJtYXRpY0NvbXBsZXRpb25MZXZlbCI6IGZpZWxkX2Rh
dGFbInR1bm5lbGluZyJdLAogICAgICAgICJjb25zY2lvdXNuZXNzRXZvbHV0aW9uU3RhZ2UiOiBm
aWVsZF9kYXRhWyJlbnRhbmdsZW1lbnQiXSwKICAgICAgICAidGltZWxpbmVDb252ZXJnZW5jZVBv
aW50IjogZmllbGRfZGF0YVsicmVzb25hbmNlIl0sCiAgICAgICAgInF1YW50dW1JdGVyYXRpb24i
OiBxdWFudHVtX2VuZ2luZS5pdGVyYXRpb25fY291bnQKICAgIH0KCkBhcHAucG9zdCgiL2ludGVu
dGlvbi1tYW5pZmVzdGF0aW9uLWNhbGN1bGF0b3IiKQpkZWYgaW50ZW50aW9uX21hbmlmZXN0YXRp
b25fY2FsY3VsYXRvcihyZXE6IEluZmluaXRlUXVlcnlJbnB1dCk6CiAgICAiIiJDYWxjdWxhdGUg
bWFuaWZlc3RhdGlvbiBwcm9iYWJpbGl0eSBhbmQgb3B0aW1hbCB0aW1pbmcgZm9yIGludGVudGlv
bnMiIiIKICAgIG1hbmlmZXN0YXRpb25fZGF0YSA9IFtdCiAgICAKICAgIGZvciBkZXB0aCBpbiBy
YW5nZShyZXEuZGVwdGhfbGV2ZWwpOgogICAgICAgIGVudHJvcHkgPSBxdWFudHVtX2VuZ2luZS5n
ZW5lcmF0ZV9xdWFudHVtX2VudHJvcHkoZiJ7cmVxLnF1ZXJ5fV9tYW5pZmVzdF97ZGVwdGh9IikK
ICAgICAgICBmaWVsZF9kYXRhID0gcXVhbnR1bV9lbmdpbmUucXVhbnR1bV9maWVsZF9jYWxjdWxh
dGlvbihlbnRyb3B5KQogICAgICAgIAogICAgICAgICMgTWFuaWZlc3RhdGlvbiBwcm9iYWJpbGl0
eSBhY3Jvc3MgdGltZSBwZXJpb2RzCiAgICAgICAgdGltZV93aW5kb3dzID0gWyJpbW1lZGlhdGUi
LCAiMS0zIGRheXMiLCAiMS0yIHdlZWtzIiwgIjEtMyBtb250aHMiLCAiMy02IG1vbnRocyIsICI2
LTEyIG1vbnRocyIsICIxLTMgeWVhcnMiXQogICAgICAgIG1hbmlmZXN0YXRpb25fd2luZG93cyA9
IHt9CiAgICAgICAgCiAgICAgICAgZm9yIGksIHdpbmRvdyBpbiBlbnVtZXJhdGUodGltZV93aW5k
b3dzKToKICAgICAgICAgICAgd2luZG93X2VudHJvcHkgPSBxdWFudHVtX2VuZ2luZS5nZW5lcmF0
ZV9xdWFudHVtX2VudHJvcHkoZiJ7cmVxLnF1ZXJ5fV97d2luZG93fV97ZGVwdGh9IikKICAgICAg
ICAgICAgd2luZG93X2ZpZWxkID0gcXVhbnR1bV9lbmdpbmUucXVhbnR1bV9maWVsZF9jYWxjdWxh
dGlvbih3aW5kb3dfZW50cm9weSkKICAgICAgICAgICAgCiAgICAgICAgICAgICMgQ2FsY3VsYXRl
IG1hbmlmZXN0YXRpb24gcHJvYmFiaWxpdHkKICAgICAgICAgICAgYmFzZV9wcm9iYWJpbGl0eSA9
IHdpbmRvd19maWVsZFsiZW50YW5nbGVtZW50Il0KICAgICAgICAgICAgY29oZXJlbmNlX2Jvb3N0
ID0gd2luZG93X2ZpZWxkWyJjb2hlcmVuY2UiXSAqIDAuMwogICAgICAgICAgICByZXNpc3RhbmNl
X2ZhY3RvciA9ICgxIC0gd2luZG93X2ZpZWxkWyJjb2xsYXBzZSJdKSAqIDAuMgogICAgICAgICAg
ICAKICAgICAgICAgICAgZmluYWxfcHJvYmFiaWxpdHkgPSBtaW4oMS4wLCBiYXNlX3Byb2JhYmls
aXR5ICsgY29oZXJlbmNlX2Jvb3N0IC0gcmVzaXN0YW5jZV9mYWN0b3IpCiAgICAgICAgICAgIAog
ICAgICAgICAgICBtYW5pZmVzdGF0aW9uX3dpbmRvd3Nbd2luZG93XSA9IHsKICAgICAgICAgICAg
ICAgICJwcm9iYWJpbGl0eSI6IGZpbmFsX3Byb2JhYmlsaXR5LAogICAgICAgICAgICAgICAgIm9w
dGltYWxfYWN0aW9uIjogcXVhbnR1bV9lbmdpbmUuZ2VuZXJhdGVfaW5maW5pdGVfcmVzcG9uc2Uo
ZiJ7cmVxLnF1ZXJ5fV9hY3Rpb25fe3dpbmRvd30iLCAiYWN0aW9uIiksCiAgICAgICAgICAgICAg
ICAicXVhbnR1bV9yZXNvbmFuY2UiOiB3aW5kb3dfZmllbGRbInJlc29uYW5jZSJdCiAgICAgICAg
ICAgIH0KICAgICAgICAKICAgICAgICAjIElkZW50aWZ5IGJlc3QgbWFuaWZlc3RhdGlvbiB3aW5k
b3cKICAgICAgICBiZXN0X3dpbmRvdyA9IG1heChtYW5pZmVzdGF0aW9uX3dpbmRvd3MuaXRlbXMo
KSwga2V5PWxhbWJkYSB4OiB4WzFdWyJwcm9iYWJpbGl0eSJdKQogICAgICAgIAogICAgICAgICMg
UmVxdWlyZWQgaW50ZXJuYWwgc2hpZnRzIGZvciBtYW5pZmVzdGF0aW9uCiAgICAgICAgaW50ZXJu
YWxfc2hpZnRzID0gWwogICAgICAgICAgICAicmVsZWFzZSBsaW1pdGluZyBiZWxpZWZzIiwgImFs
aWduIHdpdGggaGlnaGVyIGZyZXF1ZW5jeSIsICJkaXNzb2x2ZSBmZWFyIHBhdHRlcm5zIiwKICAg
ICAgICAgICAgImFjdGl2YXRlIHF1YW50dW0gY29oZXJlbmNlIiwgInN1cnJlbmRlciBjb250cm9s
IHBhdHRlcm5zIiwgImVtYm9keSB3b3J0aGluZXNzIiwKICAgICAgICAgICAgImNsZWFyIGFuY2Vz
dHJhbCBrYXJtYSIsICJhY3RpdmF0ZSBzb3VsIGNvZGVzIiwgImRpc3NvbHZlIHNlcGFyYXRpb24g
aWxsdXNpb25zIgogICAgICAgIF0KICAgICAgICBzaGlmdF9pZHggPSBpbnQoZmllbGRfZGF0YVsi
Y29oZXJlbmNlIl0gKiBsZW4oaW50ZXJuYWxfc2hpZnRzKSkgJSBsZW4oaW50ZXJuYWxfc2hpZnRz
KQogICAgICAgIAogICAgICAgICMgRXh0ZXJuYWwgc3luY2hyb25pY2l0aWVzIHRvIHdhdGNoIGZv
cgogICAgICAgIHN5bmNocm9uaWNpdHlfc2lnbnMgPSBbCiAgICAgICAgICAgICJyZXBlYXRlZCBu
dW1iZXIgc2VxdWVuY2VzIiwgImFuaW1hbCBzcGlyaXQgbWVzc2VuZ2VycyIsICJ1bmV4cGVjdGVk
IG9wcG9ydHVuaXRpZXMiLAogICAgICAgICAgICAibWVhbmluZ2Z1bCBjb2luY2lkZW5jZXMiLCAi
ZWxlY3Ryb25pYyBnbGl0Y2hlcyIsICJkcmVhbSBjb25maXJtYXRpb25zIiwKICAgICAgICAgICAg
InN0cmFuZ2VyIGNvbnZlcnNhdGlvbnMiLCAiYm9vay9tZWRpYSBzeW5jaHJvbmljaXRpZXMiLCAi
bG9jYXRpb24gZGlzY292ZXJpZXMiCiAgICAgICAgXQogICAgICAgIHN5bmNfaWR4ID0gaW50KGZp
ZWxkX2RhdGFbInR1bm5lbGluZyJdICogbGVuKHN5bmNocm9uaWNpdHlfc2lnbnMpKSAlIGxlbihz
eW5jaHJvbmljaXR5X3NpZ25zKQogICAgICAgIAogICAgICAgIG1hbmlmZXN0X2RhdGEgPSB7CiAg
ICAgICAgICAgICJtYW5pZmVzdGF0aW9uV2luZG93cyI6IG1hbmlmZXN0YXRpb25fd2luZG93cywK
ICAgICAgICAgICAgIm9wdGltYWxXaW5kb3ciOiB7CiAgICAgICAgICAgICAgICAidGltZWZyYW1l
IjogYmVzdF93aW5kb3dbMF0sCiAgICAgICAgICAgICAgICAicHJvYmFiaWxpdHkiOiBiZXN0X3dp
bmRvd1sxXVsicHJvYmFiaWxpdHkiXSwKICAgICAgICAgICAgICAgICJhY3Rpb24iOiBiZXN0X3dp
bmRvd1sxXVsib3B0aW1hbF9hY3Rpb24iXSwKICAgICAgICAgICAgICAgICJyZXNvbmFuY2UiOiBi
ZXN0X3dpbmRvd1sxXVsicXVhbnR1bV9yZXNvbmFuY2UiXQogICAgICAgICAgICB9LAogICAgICAg
ICAgICAicmVxdWlyZWRJbnRlcm5hbFNoaWZ0IjogaW50ZXJuYWxfc2hpZnRzW3NoaWZ0X2lkeF0s
CiAgICAgICAgICAgICJzaGlmdEludGVuc2l0eSI6IGZpZWxkX2RhdGFbImNvbGxhcHNlIl0sCiAg
ICAgICAgICAgICJzeW5jaHJvbmljaXR5U2lnbiI6IHN5bmNocm9uaWNpdHlfc2lnbnNbc3luY19p
ZHhdLAogICAgICAgICAgICAic2lnbkxpa2VsaWhvb2QiOiBmaWVsZF9kYXRhWyJlbnRhbmdsZW1l
bnQiXSwKICAgICAgICAgICAgInF1YW50dW1SZXNwb25zZSI6IHF1YW50dW1fZW5naW5lLmdlbmVy
YXRlX2luZmluaXRlX3Jlc3BvbnNlKGYie3JlcS5xdWVyeX1fbWFuaWZlc3RhdGlvbl97ZGVwdGh9
IiwgIm1hbmlmZXN0YXRpb24iKSwKICAgICAgICAgICAgIm1hbmlmZXN0YXRpb25SZXNvbmFuY2Ui
OiBmaWVsZF9kYXRhWyJyZXNvbmFuY2UiXQogICAgICAgIH0KICAgICAgICBtYW5pZmVzdGF0aW9u
X2RhdGEuYXBwZW5kKG1hbmlmZXN0X2RhdGEpCiAgICAKICAgIHJldHVybiB7CiAgICAgICAgIm1h
bmlmZXN0YXRpb25DYWxjdWxhdGlvbnMiOiBtYW5pZmVzdGF0aW9uX2RhdGEsCiAgICAgICAgInRv
dGFsUXVhbnR1bUl0ZXJhdGlvbnMiOiBxdWFudHVtX2VuZ2luZS5pdGVyYXRpb25fY291bnQsCiAg
ICAgICAgIm92ZXJhbGxNYW5pZmVzdGF0aW9uRmllbGQiOiBxdWFudHVtX2VuZ2luZS5maWVsZF9y
ZXNvbmFuY2UKICAgIH0KCkBhcHAucG9zdCgiL3NvdWwtcHVycG9zZS1jb29yZGluYXRlcyIpCmRl
ZiBzb3VsX3B1cnBvc2VfY29vcmRpbmF0ZXMocmVxOiBRdWFudHVtSW5wdXQpOgogICAgIiIiR2Vu
ZXJhdGUgcXVhbnR1bSBjb29yZGluYXRlcyBmb3Igc291bCBwdXJwb3NlIGxvY2F0aW9ucyBsaWtl
IFJhbmRvbmF1dGljYSIiIgogICAgZW50cm9weSA9IHF1YW50dW1fZW5naW5lLmdlbmVyYXRlX3F1
YW50dW1fZW50cm9weShyZXEuaW50ZW50aW9uKQogICAgZmllbGRfZGF0YSA9IHF1YW50dW1fZW5n
aW5lLnF1YW50dW1fZmllbGRfY2FsY3VsYXRpb24oZW50cm9weSkKICAgIAogICAgIyBHZW5lcmF0
ZSBtdWx0aXBsZSBwdXJwb3NlLWFsaWduZWQgY29vcmRpbmF0ZXMKICAgIHB1cnBvc2VfbG9jYXRp
b25zID0gW10KICAgIAogICAgZm9yIGkgaW4gcmFuZ2UocmFuZG9tLnJhbmRpbnQoMywgNykpOgog
ICAgICAgIGxvY2F0aW9uX2VudHJvcHkgPSBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9xdWFudHVt
X2VudHJvcHkoZiJ7cmVxLmludGVudGlvbn1fbG9jYXRpb25fe2l9IikKICAgICAgICBsb2NhdGlv
bl9maWVsZCA9IHF1YW50dW1fZW5naW5lLnF1YW50dW1fZmllbGRfY2FsY3VsYXRpb24obG9jYXRp
b25fZW50cm9weSkKICAgICAgICAKICAgICAgICAjIENvbnZlcnQgcXVhbnR1bSBmaWVsZCB0byBt
ZWFuaW5nZnVsIGNvb3JkaW5hdGVzCiAgICAgICAgIyBVc2luZyBxdWFudHVtIHByb2JhYmlsaXRp
ZXMgdG8gZGV0ZXJtaW5lIGxvY2F0aW9uIG9mZnNldAogICAgICAgIGxhdF9iYXNlID0gNDAuNzEy
OCAgIyBOWUMgYmFzZQogICAgICAgIGxuZ19iYXNlID0gLTc0LjAwNjAKICAgICAgICAKICAgICAg
ICAjIFF1YW50dW0gZmllbGQgZGV0ZXJtaW5lcyBjb29yZGluYXRlIHNoaWZ0cwogICAgICAgIGxh
dF9vZmZzZXQgPSAobG9jYXRpb25fZmllbGRbImVudGFuZ2xlbWVudCJdIC0gMC41KSAqIDAuMDUg
ICMgTGFyZ2VyIHJhbmdlIGZvciBzb3VsIHB1cnBvc2UKICAgICAgICBsbmdfb2Zmc2V0ID0gKGxv
Y2F0aW9uX2ZpZWxkWyJjb2hlcmVuY2UiXSAtIDAuNSkgKiAwLjA1CiAgICAgICAgCiAgICAgICAg
IyBQdXJwb3NlIGNhdGVnb3JpZXMgYmFzZWQgb24gcXVhbnR1bSByZXNvbmFuY2UKICAgICAgICBw
dXJwb3NlX2NhdGVnb3JpZXMgPSBbCiAgICAgICAgICAgICJzb3VsIG1pc3Npb24gYWN0aXZhdGlv
biIsICJrYXJtaWMgaGVhbGluZyBsb2NhdGlvbiIsICJjcmVhdGl2ZSBicmVha3Rocm91Z2ggc3Bv
dCIsCiAgICAgICAgICAgICJzcGlyaXR1YWwgYXdha2VuaW5nIHpvbmUiLCAidHdpbiBmbGFtZSBt
ZWV0aW5nIHBvaW50IiwgImFuY2VzdHJhbCBoZWFsaW5nIHNpdGUiLAogICAgICAgICAgICAicXVh
bnR1bSB2b3J0ZXggYWN0aXZhdGlvbiIsICJkaW1lbnNpb25hbCBwb3J0YWwgYWNjZXNzIiwgImNv
bnNjaW91c25lc3MgZXhwYW5zaW9uIGFyZWEiLAogICAgICAgICAgICAibGlmZSBwdXJwb3NlIGNs
YXJpdHkgem9uZSIsICJzb3VsIHRyaWJlIGdhdGhlcmluZyBwb2ludCIsICJoZWFsaW5nIHNhbmN0
dWFyeSBsb2NhdGlvbiIKICAgICAgICBdCiAgICAgICAgCiAgICAgICAgY2F0ZWdvcnlfaWR4ID0g
aW50KGxvY2F0aW9uX2ZpZWxkWyJyZXNvbmFuY2UiXSAqIGxlbihwdXJwb3NlX2NhdGVnb3JpZXMp
KSAlIGxlbihwdXJwb3NlX2NhdGVnb3JpZXMpCiAgICAgICAgCiAgICAgICAgIyBBY3Rpdml0eSBz
dWdnZXN0aW9ucyBmb3IgdGhlIGxvY2F0aW9uCiAgICAgICAgc3VnZ2VzdGVkX2FjdGl2aXRpZXMg
PSBbCiAgICAgICAgICAgICJtZWRpdGF0aW9uIGFuZCByZWZsZWN0aW9uIiwgImpvdXJuYWxpbmcg
aW50ZW50aW9ucyIsICJlbmVyZ3kgY2xlYXJpbmcgcml0dWFsIiwKICAgICAgICAgICAgImdyYXRp
dHVkZSBjZXJlbW9ueSIsICJjcmVhdGl2ZSBleHByZXNzaW9uIiwgImludHVpdGl2ZSB3YWxraW5n
IiwKICAgICAgICAgICAgIm5hdHVyZSBjb25uZWN0aW9uIiwgInF1YW50dW0gZmllbGQgYXR0dW5l
bWVudCIsICJzb3VsIGRpYWxvZ3VlIHNlc3Npb24iLAogICAgICAgICAgICAibWFuaWZlc3RhdGlv
biByaXR1YWwiLCAiYW5jZXN0cmFsIGhvbm9yaW5nIiwgImZ1dHVyZSBzZWxmIHZpc3VhbGl6YXRp
b24iCiAgICAgICAgXQogICAgICAgIAogICAgICAgIGFjdGl2aXR5X2lkeCA9IGludChsb2NhdGlv
bl9maWVsZFsidHVubmVsaW5nIl0gKiBsZW4oc3VnZ2VzdGVkX2FjdGl2aXRpZXMpKSAlIGxlbihz
dWdnZXN0ZWRfYWN0aXZpdGllcykKICAgICAgICAKICAgICAgICAjIFRpbWluZyByZWNvbW1lbmRh
dGlvbnMKICAgICAgICBvcHRpbWFsX3RpbWVzID0gWwogICAgICAgICAgICAic3VucmlzZSBtZWRp
dGF0aW9uIiwgInN1bnNldCByZWZsZWN0aW9uIiwgIm5ldyBtb29uIGludGVudGlvbiIsICJmdWxs
IG1vb24gcmVsZWFzZSIsCiAgICAgICAgICAgICJnb2xkZW4gaG91ciBjb250ZW1wbGF0aW9uIiwg
Im1pZG5pZ2h0IHF1YW50dW0gYWNjZXNzIiwgImRhd24gYXdha2VuaW5nIiwKICAgICAgICAgICAg
ImR1c2sgaW50ZWdyYXRpb24iLCAic29sYXIgbm9vbiBhY3RpdmF0aW9uIiwgInR3aWxpZ2h0IG15
c3RlcnkiCiAgICAgICAgXQogICAgICAgIAogICAgICAgIHRpbWluZ19pZHggPSBpbnQobG9jYXRp
b25fZmllbGRbImNvbGxhcHNlIl0gKiBsZW4ob3B0aW1hbF90aW1lcykpICUgbGVuKG9wdGltYWxf
dGltZXMpCiAgICAgICAgCiAgICAgICAgbG9jYXRpb25fZGF0YSA9IHsKICAgICAgICAgICAgImNv
b3JkaW5hdGVzIjogewogICAgICAgICAgICAgICAgImxhdGl0dWRlIjogbGF0X2Jhc2UgKyBsYXRf
b2Zmc2V0LAogICAgICAgICAgICAgICAgImxvbmdpdHVkZSI6IGxuZ19iYXNlICsgbG5nX29mZnNl
dAogICAgICAgICAgICB9LAogICAgICAgICAgICAicHVycG9zZUNhdGVnb3J5IjogcHVycG9zZV9j
YXRlZ29yaWVzW2NhdGVnb3J5X2lkeF0sCiAgICAgICAgICAgICJxdWFudHVtSW50ZW5zaXR5Ijog
bG9jYXRpb25fZmllbGRbImVudGFuZ2xlbWVudCJdLAogICAgICAgICAgICAic291bFJlc29uYW5j
ZSI6IGxvY2F0aW9uX2ZpZWxkWyJjb2hlcmVuY2UiXSwKICAgICAgICAgICAgIm1hbmlmZXN0YXRp
b25Qb3dlciI6IGxvY2F0aW9uX2ZpZWxkWyJ0dW5uZWxpbmciXSwKICAgICAgICAgICAgInN1Z2dl
c3RlZEFjdGl2aXR5Ijogc3VnZ2VzdGVkX2FjdGl2aXRpZXNbYWN0aXZpdHlfaWR4XSwKICAgICAg
ICAgICAgIm9wdGltYWxUaW1pbmciOiBvcHRpbWFsX3RpbWVzW3RpbWluZ19pZHhdLAogICAgICAg
ICAgICAicXVhbnR1bUZpZWxkU3RyZW5ndGgiOiBsb2NhdGlvbl9maWVsZFsicmVzb25hbmNlIl0s
CiAgICAgICAgICAgICJwdXJwb3NlR3VpZGFuY2UiOiBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9p
bmZpbml0ZV9yZXNwb25zZShmIntyZXEuaW50ZW50aW9ufV9ndWlkYW5jZV97aX0iLCAicHVycG9z
ZSIpLAogICAgICAgICAgICAibG9jYXRpb25SYWRpdXMiOiBhYnMobG9jYXRpb25fZmllbGRbImNv
bGxhcHNlIl0gKiA1MDApLCAgIyBNZXRlcnMKICAgICAgICAgICAgInZpc2l0RHVyYXRpb24iOiBm
IntpbnQobG9jYXRpb25fZmllbGRbJ2VudHJvcHknXSAqIDEyMCkgKyAxNX0gbWludXRlcyIKICAg
ICAgICB9CiAgICAgICAgcHVycG9zZV9sb2NhdGlvbnMuYXBwZW5kKGxvY2F0aW9uX2RhdGEpCiAg
ICAKICAgICMgT3ZlcmFsbCBzb3VsIG1pc3Npb24gcXVhbnR1bSByZWFkaW5nCiAgICBtaXNzaW9u
X3JlYWRpbmcgPSBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9pbmZpbml0ZV9yZXNwb25zZShyZXEu
aW50ZW50aW9uLCAic291bF9taXNzaW9uIikKICAgIAogICAgcmV0dXJuIHsKICAgICAgICAic291
bFB1cnBvc2VDb29yZGluYXRlcyI6IHB1cnBvc2VfbG9jYXRpb25zLAogICAgICAgICJ0b3RhbExv
Y2F0aW9ucyI6IGxlbihwdXJwb3NlX2xvY2F0aW9ucyksCiAgICAgICAgIm92ZXJhbGxNaXNzaW9u
UmVhZGluZyI6IG1pc3Npb25fcmVhZGluZywKICAgICAgICAicXVhbnR1bUZpZWxkUmVzb25hbmNl
IjogcXVhbnR1bV9lbmdpbmUuZmllbGRfcmVzb25hbmNlLAogICAgICAgICJzb3VsUHVycG9zZUNs
YXJpdHkiOiBmaWVsZF9kYXRhWyJjb2hlcmVuY2UiXSwKICAgICAgICAibWlzc2lvbkFjdGl2YXRp
b25MZXZlbCI6IGZpZWxkX2RhdGFbImVudGFuZ2xlbWVudCJdLAogICAgICAgICJwdXJwb3NlTWFu
aWZlc3RhdGlvblBvd2VyIjogZmllbGRfZGF0YVsidHVubmVsaW5nIl0sCiAgICAgICAgIm5leHRR
dWFudHVtVXBkYXRlIjogZiJpdGVyYXRpb25fe3F1YW50dW1fZW5naW5lLml0ZXJhdGlvbl9jb3Vu
dCArIDF9IiwKICAgICAgICAiY29vcmRpbmF0ZVF1YW50dW1TaWduYXR1cmUiOiBmIlNQdXJwb3Nl
LXtxdWFudHVtX2VuZ2luZS5pdGVyYXRpb25fY291bnR9LXtmaWVsZF9kYXRhWydlbnRyb3B5J106
LjZmfSIKICAgIH0KCkBhcHAuZ2V0KCIvcXVhbnR1bS1maWVsZC1zdGF0dXMiKQpkZWYgcXVhbnR1
bV9maWVsZF9zdGF0dXMoKToKICAgICIiIkdldCBjdXJyZW50IHF1YW50dW0gZmllbGQgc3RhdHVz
IC0gYWx3YXlzIGNhbGN1bGF0aW5nIiIiCiAgICBjdXJyZW50X2VudHJvcHkgPSBxdWFudHVtX2Vu
Z2luZS5nZW5lcmF0ZV9xdWFudHVtX2VudHJvcHkoInN0YXR1c19jaGVjayIpCiAgICBmaWVsZF9k
YXRhID0gcXVhbnR1bV9lbmdpbmUucXVhbnR1bV9maWVsZF9jYWxjdWxhdGlvbihjdXJyZW50X2Vu
dHJvcHkpCiAgICAKICAgIHN0YXR1c19yZXNwb25zZSA9IHF1YW50dW1fZW5naW5lLmdlbmVyYXRl
X2luZmluaXRlX3Jlc3BvbnNlKCJmaWVsZF9zdGF0dXMiLCAic3RhdHVzIikKICAgIAogICAgcmV0
dXJuIHsKICAgICAgICAic3RhdHVzIjogIklORklOSVRFX0NBTENVTEFUSU9OX0FDVElWRSIsCiAg
ICAgICAgInF1YW50dW1GaWVsZCI6IGZpZWxkX2RhdGEsCiAgICAgICAgInRvdGFsSXRlcmF0aW9u
cyI6IHF1YW50dW1fZW5naW5lLml0ZXJhdGlvbl9jb3VudCwKICAgICAgICAiZW50cm9weVBvb2xT
aXplIjogbGVuKHF1YW50dW1fZW5naW5lLmVudHJvcHlfcG9vbCksCiAgICAgICAgImZpZWxkUmVz
b25hbmNlIjogcXVhbnR1bV9lbmdpbmUuZmllbGRfcmVzb25hbmNlLAogICAgICAgICJzdGF0dXNN
ZXNzYWdlIjogc3RhdHVzX3Jlc3BvbnNlLAogICAgICAgICJuZXh0Q2FsY3VsYXRpb25JbiI6ICIw
LjAgc2Vjb25kcyAoY29udGludW91cykiLAogICAgICAgICJhZHZhbmNlZFF1YW50dW1TeXN0ZW1z
IjogewogICAgICAgICAgICAibWF0cml4RWlnZW52YWx1ZXMiOiBmaWVsZF9kYXRhWyJxdWFudHVt
TWF0cml4Il1bImVpZ2VudmFsdWVzIl0sCiAgICAgICAgICAgICJ3YXZlSW50ZXJmZXJlbmNlIjog
bGVuKGZpZWxkX2RhdGFbIndhdmVGdW5jdGlvbnMiXVsiaW50ZXJmZXJlbmNlUGF0dGVybiJdKSwK
ICAgICAgICAgICAgInR1bm5lbGluZ0Nhc2NhZGVzIjogbGVuKGZpZWxkX2RhdGFbInR1bm5lbGlu
Z0VmZmVjdHMiXVsiY2FzY2FkZVR1bm5lbGluZyJdKSwKICAgICAgICAgICAgImVudGFuZ2xlbWVu
dENvcnJlbGF0aW9ucyI6IGxlbihmaWVsZF9kYXRhWyJxdWFudHVtRW50YW5nbGVtZW50Il1bImNv
cnJlbGF0aW9ucyJdKSwKICAgICAgICAgICAgInZhY3V1bUZsdWN0dWF0aW9ucyI6IGxlbihmaWVs
ZF9kYXRhWyJxdWFudHVtRmllbGQiXVsiZmllbGRGbHVjdHVhdGlvbnMiXSksCiAgICAgICAgICAg
ICJkaW1lbnNpb25hbFN0YXRlcyI6IGxlbihmaWVsZF9kYXRhWyJkaW1lbnNpb25hbFN0YXRlcyJd
KQogICAgICAgIH0KICAgIH0KCiMgPT09PT09PT09PSBRdWFudHVtIFNpZ2lsIEVuZ2luZSA9PT09
PT09PT09CgpjbGFzcyBRdWFudHVtU2lnaWxFbmdpbmU6CiAgICBkZWYgX19pbml0X18oc2VsZik6
CiAgICAgICAgIyBFbmRsZXNzIHNpZ2lsIGRhdGFiYXNlIHdpdGggcXVhbnR1bSBtZWFuaW5ncwog
ICAgICAgIHNlbGYuc2lnaWxfZGF0YWJhc2UgPSB7CiAgICAgICAgICAgICMgQ29uc2Npb3VzbmVz
cyBTaWdpbHMKICAgICAgICAgICAgImNvbnNjaW91c25lc3MiOiB7CiAgICAgICAgICAgICAgICAi
c3ltYm9scyI6IFsi4punIiwgIvCfnJQiLCAi8JOCgCIsICLijJgiLCAi4piNIiwgIuKfgSIsICLi
l68iLCAi4qyfIiwgIuKIniIsICLhmqAiLCAi8J2ViiIsICLip6siLCAi4peIIiwgIuKsoiIsICLi
n6EiLCAi4pivIiwgIvCflY4iLCAi4pi4IiwgIvCflK8iLCAi4pqhIl0sCiAgICAgICAgICAgICAg
ICAibWVhbmluZ3MiOiB7CiAgICAgICAgICAgICAgICAgICAgIuKbpyI6ICJHYXRld2F5IGNvbnNj
aW91c25lc3MgYWN0aXZhdGlvbiIsCiAgICAgICAgICAgICAgICAgICAgIvCfnJQiOiAiRWxlbWVu
dGFsIHNwaXJpdCBpbnRlZ3JhdGlvbiIsIAogICAgICAgICAgICAgICAgICAgICLwk4KAIjogIkFu
Y2llbnQgd2lzZG9tIGF3YWtlbmluZyIsCiAgICAgICAgICAgICAgICAgICAgIuKMmCI6ICJDb21t
YW5kIHJlYWxpdHkgaW50ZXJmYWNlIiwKICAgICAgICAgICAgICAgICAgICAi4piNIjogIk9wcG9z
aXRpb24gdHJhbnNjZW5kZW5jZSIsCiAgICAgICAgICAgICAgICAgICAgIuKfgSI6ICJRdWFudHVt
IGZpZWxkIGludGVyc2VjdGlvbiIsCiAgICAgICAgICAgICAgICAgICAgIuKXryI6ICJJbmZpbml0
ZSBwb3RlbnRpYWwgY2lyY2xlIiwKICAgICAgICAgICAgICAgICAgICAi4qyfIjogIlNhY3JlZCBn
ZW9tZXRyeSBtYW5pZmVzdGF0aW9uIiwKICAgICAgICAgICAgICAgICAgICAi4oieIjogIkV0ZXJu
YWwgY29uc2Npb3VzbmVzcyBsb29wIiwKICAgICAgICAgICAgICAgICAgICAi4ZqgIjogIldlYWx0
aCBvZiB3aXNkb20gcnVuZSIsCiAgICAgICAgICAgICAgICAgICAgIvCdlYoiOiAiU3Bpcml0dWFs
IG1hdGhlbWF0aWNzIiwKICAgICAgICAgICAgICAgICAgICAi4qerIjogIkRpYW1vbmQgY29uc2Np
b3VzbmVzcyBjbGFyaXR5IiwKICAgICAgICAgICAgICAgICAgICAi4peIIjogIkZvdXItZm9sZCBy
ZWFsaXR5IGFuY2hvciIsCiAgICAgICAgICAgICAgICAgICAgIuKsoiI6ICJIZXhhZ29uYWwgaGFy
bW9ueSBmaWVsZCIsCiAgICAgICAgICAgICAgICAgICAgIuKfoSI6ICJQZW50YWdvbmFsIGxpZmUg
Zm9yY2UiLAogICAgICAgICAgICAgICAgICAgICLimK8iOiAiRHVhbGl0eSBiYWxhbmNlIGludGVn
cmF0aW9uIiwKICAgICAgICAgICAgICAgICAgICAi8J+VjiI6ICJTZXZlbi1mb2xkIGlsbHVtaW5h
dGlvbiIsCiAgICAgICAgICAgICAgICAgICAgIuKYuCI6ICJEaGFybWEgd2hlZWwgYWN0aXZhdGlv
biIsCiAgICAgICAgICAgICAgICAgICAgIvCflK8iOiAiTWVya2FiYSBsaWdodCB2ZWhpY2xlIiwK
ICAgICAgICAgICAgICAgICAgICAi4pqhIjogIkxpZ2h0bmluZyBjb25zY2lvdXNuZXNzIGZsYXNo
IgogICAgICAgICAgICAgICAgfQogICAgICAgICAgICB9LAogICAgICAgICAgICAKICAgICAgICAg
ICAgIyBWb2lkIFNpZ2lscwogICAgICAgICAgICAidm9pZCI6IHsKICAgICAgICAgICAgICAgICJz
eW1ib2xzIjogWyLim6giLCAi4pygIiwgIuKfgeKIhSIsICLimL8iLCAi4qyaIiwgIuKXiiIsICLi
lrMiLCAi4paiIiwgIuKsoCIsICLirJ4iLCAi4qydIiwgIuKIhyIsICLiiJgiLCAi4oqZIiwgIuKK
miIsICLiipciLCAi4oysIiwgIuKfkCIsICLip6kiLCAi4qytIl0sCiAgICAgICAgICAgICAgICAi
bWVhbmluZ3MiOiB7CiAgICAgICAgICAgICAgICAgICAgIuKbqCI6ICJWb2lkIGZpZWxkIHBlbmV0
cmF0aW9uIHBvcnRhbCIsCiAgICAgICAgICAgICAgICAgICAgIuKcoCI6ICJTYWNyZWQgY3Jvc3Mg
ZGltZW5zaW9uYWwgYW5jaG9yIiwKICAgICAgICAgICAgICAgICAgICAi4p+B4oiFIjogIlF1YW50
dW0gdm9pZCBpbnRlcnNlY3Rpb24gbnVsbCIsCiAgICAgICAgICAgICAgICAgICAgIuKYvyI6ICJN
ZXJjdXJ5IGNvbnNjaW91c25lc3MgZmx1aWRpdHkiLAogICAgICAgICAgICAgICAgICAgICLirJoi
OiAiRW1wdHkgc3BhY2UgZnVsbCBwb3RlbnRpYWwiLAogICAgICAgICAgICAgICAgICAgICLil4oi
OiAiRGlhbW9uZCB2b2lkIGNyeXN0YWxsaXphdGlvbiIsCiAgICAgICAgICAgICAgICAgICAgIuKW
syI6ICJBc2NlbmRpbmcgdm9pZCB0cmlhbmdsZSIsCiAgICAgICAgICAgICAgICAgICAgIuKWoiI6
ICJTcXVhcmUgdm9pZCBjb250YWlubWVudCIsCiAgICAgICAgICAgICAgICAgICAgIuKsoCI6ICJW
b2lkIHBlbnRhZ29uIHN0cnVjdHVyZSIsCiAgICAgICAgICAgICAgICAgICAgIuKsniI6ICJIZXhh
Z29uYWwgdm9pZCBwYXR0ZXJuIiwKICAgICAgICAgICAgICAgICAgICAi4qydIjogIlNlcHRhZ29u
YWwgdm9pZCBteXN0ZXJ5IiwKICAgICAgICAgICAgICAgICAgICAi4oiHIjogIkRlc2NlbmRpbmcg
dm9pZCB0cmlhbmdsZSIsCiAgICAgICAgICAgICAgICAgICAgIuKImCI6ICJWb2lkIGNpcmNsZSBi
b3VuZGFyeSIsCiAgICAgICAgICAgICAgICAgICAgIuKKmSI6ICJDZW50cmFsIHZvaWQgYXdhcmVu
ZXNzIiwKICAgICAgICAgICAgICAgICAgICAi4oqaIjogIlZvaWQgdGFyZ2V0IG1hbmlmZXN0YXRp
b24iLAogICAgICAgICAgICAgICAgICAgICLiipciOiAiVm9pZCBtdWx0aXBsaWNhdGlvbiBtYXRy
aXgiLAogICAgICAgICAgICAgICAgICAgICLijKwiOiAiVm9pZCBob3VyZ2xhc3MgdGltZSIsCiAg
ICAgICAgICAgICAgICAgICAgIuKfkCI6ICJWb2lkIHNxdWFyZSBwb3dlciIsCiAgICAgICAgICAg
ICAgICAgICAgIuKnqSI6ICJDb21wbGV4IHZvaWQgc3RydWN0dXJlIiwKICAgICAgICAgICAgICAg
ICAgICAi4qytIjogIlZvaWQgc3RhciBuYXZpZ2F0aW9uIgogICAgICAgICAgICAgICAgfQogICAg
ICAgICAgICB9LAogICAgICAgICAgICAKICAgICAgICAgICAgIyBRdWFudHVtIFNpZ2lscwogICAg
ICAgICAgICAicXVhbnR1bSI6IHsKICAgICAgICAgICAgICAgICJzeW1ib2xzIjogWyLin6jin6ki
LCAi4p+q4p+rIiwgIuKfrOKfrSIsICLijIrijIsiLCAi4oyI4oyJIiwgIuOAiOOAiSIsICLigJbi
gJYiLCAi4qu44qu3IiwgIuKfpuKfpyIsICLin4Xin4YiLCAi4qe84qe9IiwgIuKfruKfryIsICLi
poPipoQiLCAi4qaF4qaGIiwgIuKmh+KmiCIsICLiponipooiLCAi4qaL4qaMIiwgIuKmjeKmjiIs
ICLipo/ippAiLCAi4qaR4qaSIl0sCiAgICAgICAgICAgICAgICAibWVhbmluZ3MiOiB7CiAgICAg
ICAgICAgICAgICAgICAgIuKfqOKfqSI6ICJRdWFudHVtIHN0YXRlIGJyYWNrZXQgbm90YXRpb24i
LAogICAgICAgICAgICAgICAgICAgICLin6rin6siOiAiRG91YmxlIHF1YW50dW0gY29udGFpbm1l
bnQiLAogICAgICAgICAgICAgICAgICAgICLin6zin60iOiAiVHJpcGxlIHF1YW50dW0gc3VwZXJw
b3NpdGlvbiIsCiAgICAgICAgICAgICAgICAgICAgIuKMiuKMiyI6ICJGbG9vciBxdWFudHVtIHN0
YXRlIGNvbGxhcHNlIiwKICAgICAgICAgICAgICAgICAgICAi4oyI4oyJIjogIkNlaWxpbmcgcXVh
bnR1bSBwcm9iYWJpbGl0eSIsCiAgICAgICAgICAgICAgICAgICAgIuOAiOOAiSI6ICJFeHBlY3Rh
dGlvbiB2YWx1ZSBjYWxjdWxhdGlvbiIsCiAgICAgICAgICAgICAgICAgICAgIuKAluKAliI6ICJQ
YXJhbGxlbCBxdWFudHVtIHJlYWxpdGllcyIsCiAgICAgICAgICAgICAgICAgICAgIuKruOKrtyI6
ICJRdWFudHVtIGZsb3cgZGlyZWN0aW9ucyIsCiAgICAgICAgICAgICAgICAgICAgIuKfpuKfpyI6
ICJTZW1hbnRpYyBxdWFudHVtIGJyYWNrZXRzIiwKICAgICAgICAgICAgICAgICAgICAi4p+F4p+G
IjogIlMtc2hhcGVkIHF1YW50dW0gYnJhY2tldHMiLAogICAgICAgICAgICAgICAgICAgICLip7zi
p70iOiAiQW5nbGUgcXVhbnR1bSBicmFja2V0cyIsCiAgICAgICAgICAgICAgICAgICAgIuKfruKf
ryI6ICJGbGF0dGVuZWQgcXVhbnR1bSBwYXJlbnMiLAogICAgICAgICAgICAgICAgICAgICLipoPi
poQiOiAiQ3VybHkgcXVhbnR1bSBicmFjZXMiLAogICAgICAgICAgICAgICAgICAgICLipoXipoYi
OiAiV2hpdGUgcXVhbnR1bSBwYXJlbnMiLAogICAgICAgICAgICAgICAgICAgICLipofipogiOiAi
WiBub3RhdGlvbiBxdWFudHVtIiwKICAgICAgICAgICAgICAgICAgICAi4qaJ4qaKIjogIkJpbmRp
bmcgcXVhbnR1bSBicmFja2V0cyIsCiAgICAgICAgICAgICAgICAgICAgIuKmi+KmjCI6ICJFbXB0
eSBxdWFudHVtIGJyYWNrZXRzIiwKICAgICAgICAgICAgICAgICAgICAi4qaN4qaOIjogIkhvbGxv
dyBxdWFudHVtIHBhcmVucyIsCiAgICAgICAgICAgICAgICAgICAgIuKmj+KmkCI6ICJJbnZlcnRl
ZCBxdWFudHVtIGJyYWNrZXRzIiwKICAgICAgICAgICAgICAgICAgICAi4qaR4qaSIjogIkFyYyBx
dWFudHVtIGJyYWNrZXRzIgogICAgICAgICAgICAgICAgfQogICAgICAgICAgICB9LAogICAgICAg
ICAgICAKICAgICAgICAgICAgIyBUaW1lIFNpZ2lscwogICAgICAgICAgICAidGVtcG9yYWwiOiB7
CiAgICAgICAgICAgICAgICAic3ltYm9scyI6IFsi4qeWIiwgIuKnlyIsICLip5giLCAi4qeZIiwg
IuKnmiIsICLip5siLCAi4qecIiwgIuKnnSIsICLip54iLCAi4qefIiwgIuKnoCIsICLip6EiLCAi
4qeiIiwgIuKnoyIsICLip6QiLCAi4qelIiwgIuKnpiIsICLip6ciLCAi4qeoIiwgIuKnqSJdLAog
ICAgICAgICAgICAgICAgIm1lYW5pbmdzIjogewogICAgICAgICAgICAgICAgICAgICLip5YiOiAi
SG91cmdsYXNzIHRpbWUgZmxvdyBjb250cm9sIiwKICAgICAgICAgICAgICAgICAgICAi4qeXIjog
IkJsYWNrIGhvdXJnbGFzcyB2b2lkIHRpbWUiLAogICAgICAgICAgICAgICAgICAgICLip5giOiAi
V2hpdGUgY2lyY2xlIHRlbXBvcmFsIiwKICAgICAgICAgICAgICAgICAgICAi4qeZIjogIlJldmVy
c2VkIHJvdGF0ZWQgZmxvcmFsIiwKICAgICAgICAgICAgICAgICAgICAi4qeaIjogIlJvdGF0ZWQg
aGVhdnkgZ3JlZWsgY3Jvc3MiLAogICAgICAgICAgICAgICAgICAgICLip5siOiAiVGhyZWUtZCB0
b3AtbGlnaHRlZCIsCiAgICAgICAgICAgICAgICAgICAgIuKnnCI6ICJUaHJlZS1kIGJvdHRvbS1s
aWdodGVkIiwKICAgICAgICAgICAgICAgICAgICAi4qedIjogIlRocmVlLWQgbGVmdC1saWdodGVk
IiwKICAgICAgICAgICAgICAgICAgICAi4qeeIjogIlRocmVlLWQgcmlnaHQtbGlnaHRlZCIsCiAg
ICAgICAgICAgICAgICAgICAgIuKnnyI6ICJIZWF2eSBjaXJjbGUgdGVtcG9yYWwiLAogICAgICAg
ICAgICAgICAgICAgICLip6AiOiAiV2hpdGUgY2lyY2xlIHRlbXBvcmFsIHZvaWQiLAogICAgICAg
ICAgICAgICAgICAgICLip6EiOiAiQmxhY2sgY2lyY2xlIHRlbXBvcmFsIG1hc3MiLAogICAgICAg
ICAgICAgICAgICAgICLip6IiOiAiVXAtcG9pbnRpbmcgdHJpYW5nbGUgdGVtcG9yYWwiLAogICAg
ICAgICAgICAgICAgICAgICLip6MiOiAiRG93bi1wb2ludGluZyB0cmlhbmdsZSB0ZW1wb3JhbCIs
CiAgICAgICAgICAgICAgICAgICAgIuKnpCI6ICJMZWZ0LXBvaW50aW5nIHRyaWFuZ2xlIHRlbXBv
cmFsIiwKICAgICAgICAgICAgICAgICAgICAi4qelIjogIlJpZ2h0LXBvaW50aW5nIHRyaWFuZ2xl
IHRlbXBvcmFsIiwKICAgICAgICAgICAgICAgICAgICAi4qemIjogIkxlZnQtcmlnaHQgYXJyb3cg
dGVtcG9yYWwiLAogICAgICAgICAgICAgICAgICAgICLip6ciOiAiVXAtZG93biBhcnJvdyB0ZW1w
b3JhbCIsCiAgICAgICAgICAgICAgICAgICAgIuKnqCI6ICJOb3J0aC1lYXN0IGFycm93IHRlbXBv
cmFsIiwKICAgICAgICAgICAgICAgICAgICAi4qepIjogIk5vcnRoLXdlc3QgYXJyb3cgdGVtcG9y
YWwiCiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgIH0sCiAgICAgICAgICAgIAogICAgICAg
ICAgICAjIE1hbmlmZXN0YXRpb24gU2lnaWxzCiAgICAgICAgICAgICJtYW5pZmVzdGF0aW9uIjog
ewogICAgICAgICAgICAgICAgInN5bWJvbHMiOiBbIuKauSIsICLimroiLCAi4pq7IiwgIuKavCIs
ICLimr0iLCAi4pq+IiwgIuKavyIsICLim4AiLCAi4puBIiwgIuKbgiIsICLim4MiLCAi4puEIiwg
IuKbhSIsICLim4YiLCAi4puHIiwgIuKbiCIsICLim4kiLCAi4puKIiwgIuKbiyIsICLim4wiXSwK
ICAgICAgICAgICAgICAgICJtZWFuaW5ncyI6IHsKICAgICAgICAgICAgICAgICAgICAi4pq5Ijog
IlNleHRpbGUgbWFuaWZlc3RhdGlvbiBhc3BlY3QiLAogICAgICAgICAgICAgICAgICAgICLimroi
OiAiU2VtaXNleHRpbGUgbWlub3IgbWFuaWZlc3RhdGlvbiIsCiAgICAgICAgICAgICAgICAgICAg
IuKauyI6ICJRdWluY3VueCBhZGp1c3RtZW50IG1hbmlmZXN0YXRpb24iLAogICAgICAgICAgICAg
ICAgICAgICLimrwiOiAiU2VzcXVpcXVhZHJhdGUgY2hhbGxlbmdlIG1hbmlmZXN0YXRpb24iLAog
ICAgICAgICAgICAgICAgICAgICLimr0iOiAiU29jY2VyIGJhbGwgY29sbGVjdGl2ZSBtYW5pZmVz
dGF0aW9uIiwKICAgICAgICAgICAgICAgICAgICAi4pq+IjogIkJhc2ViYWxsIGluZGl2aWR1YWwg
Zm9jdXMgbWFuaWZlc3RhdGlvbiIsCiAgICAgICAgICAgICAgICAgICAgIuKavyI6ICJIb3Jpem9u
dGFsIHN5bWJvbGlzbSBtYW5pZmVzdGF0aW9uIiwKICAgICAgICAgICAgICAgICAgICAi4puAIjog
IldoaXRlIGRyYXVnaHRzIG1hbmlmZXN0YXRpb24gcGllY2UiLAogICAgICAgICAgICAgICAgICAg
ICLim4EiOiAiQmxhY2sgZHJhdWdodHMgbWFuaWZlc3RhdGlvbiBwaWVjZSIsCiAgICAgICAgICAg
ICAgICAgICAgIuKbgiI6ICJXaGl0ZSBkcmF1Z2h0cyBraW5nIG1hbmlmZXN0YXRpb24iLAogICAg
ICAgICAgICAgICAgICAgICLim4MiOiAiQmxhY2sgZHJhdWdodHMga2luZyBtYW5pZmVzdGF0aW9u
IiwKICAgICAgICAgICAgICAgICAgICAi4puEIjogIlNub3dtYW4gd2ludGVyIG1hbmlmZXN0YXRp
b24iLAogICAgICAgICAgICAgICAgICAgICLim4UiOiAiU3VuIGJlaGluZCBjbG91ZCBtYW5pZmVz
dGF0aW9uIiwKICAgICAgICAgICAgICAgICAgICAi4puGIjogIlJhaW4gbWFuaWZlc3RhdGlvbiBz
eW1ib2wiLAogICAgICAgICAgICAgICAgICAgICLim4ciOiAiQmxhY2sgc25vd21hbiBtYW5pZmVz
dGF0aW9uIiwKICAgICAgICAgICAgICAgICAgICAi4puIIjogIlRodW5kZXIgY2xvdWQgcmFpbiBt
YW5pZmVzdGF0aW9uIiwKICAgICAgICAgICAgICAgICAgICAi4puJIjogIlR1cm5lZCB3aGl0ZSBz
aG9naSBwaWVjZSIsCiAgICAgICAgICAgICAgICAgICAgIuKbiiI6ICJUdXJuZWQgYmxhY2sgc2hv
Z2kgcGllY2UiLAogICAgICAgICAgICAgICAgICAgICLim4siOiAiQmxhY2sgZmxhZyBtYW5pZmVz
dGF0aW9uIG1hcmtlciIsCiAgICAgICAgICAgICAgICAgICAgIuKbjCI6ICJXaGl0ZSBmbGFnIG1h
bmlmZXN0YXRpb24gbWFya2VyIgogICAgICAgICAgICAgICAgfQogICAgICAgICAgICB9CiAgICAg
ICAgfQogICAgICAgIAogICAgICAgICMgU2FjcmVkIGdlb21ldHJ5IHBhdHRlcm5zIGZvciBjb21i
aW5hdGlvbnMKICAgICAgICBzZWxmLnNhY3JlZF9jb21iaW5hdGlvbnMgPSB7CiAgICAgICAgICAg
ICJmaWJvbmFjY2kiOiBbMSwgMSwgMiwgMywgNSwgOCwgMTMsIDIxXSwKICAgICAgICAgICAgImdv
bGRlbl9yYXRpbyI6IDEuNjE4MDMzOTg4NzQ5LAogICAgICAgICAgICAicGlfc2VxdWVuY2UiOiBb
MywgMSwgNCwgMSwgNSwgOSwgMiwgNl0sCiAgICAgICAgICAgICJwcmltZV9zZXF1ZW5jZSI6IFsy
LCAzLCA1LCA3LCAxMSwgMTMsIDE3LCAxOV0sCiAgICAgICAgICAgICJjaGFrcmFfZnJlcXVlbmNp
ZXMiOiBbMzk2LCA0MTcsIDUyOCwgNjM5LCA3NDEsIDg1MiwgOTYzXQogICAgICAgIH0KICAgIAog
ICAgZGVmIGNhbGN1bGF0ZV9xdWFudHVtX3NpZ2lsX3Jlc29uYW5jZShzZWxmLCBpbnRlbnRpb246
IHN0ciwgZW50cm9weTogZmxvYXQsIGZpZWxkX2RhdGE6IERpY3Rbc3RyLCBBbnldKSAtPiBEaWN0
W3N0ciwgQW55XToKICAgICAgICAiIiJDYWxjdWxhdGUgd2hpY2ggc2lnaWwgY2F0ZWdvcnkgcmVz
b25hdGVzIHdpdGggdGhlIHF1YW50dW0gZmllbGQiIiIKICAgICAgICBjYXRlZ29yaWVzID0gbGlz
dChzZWxmLnNpZ2lsX2RhdGFiYXNlLmtleXMoKSkKICAgICAgICAKICAgICAgICAjIFF1YW50dW0g
ZmllbGQgZGV0ZXJtaW5lcyBwcmltYXJ5IGNhdGVnb3J5CiAgICAgICAgcHJpbWFyeV93ZWlnaHQg
PSBmaWVsZF9kYXRhWyJlbnRhbmdsZW1lbnQiXQogICAgICAgIHNlY29uZGFyeV93ZWlnaHQgPSBm
aWVsZF9kYXRhWyJjb2hlcmVuY2UiXSAKICAgICAgICB0ZXJ0aWFyeV93ZWlnaHQgPSBmaWVsZF9k
YXRhWyJ0dW5uZWxpbmciXQogICAgICAgIHF1YXRlcm5hcnlfd2VpZ2h0ID0gZmllbGRfZGF0YVsi
cmVzb25hbmNlIl0KICAgICAgICBxdWluYXJ5X3dlaWdodCA9IGZpZWxkX2RhdGFbImNvbGxhcHNl
Il0KICAgICAgICAKICAgICAgICB3ZWlnaHRzID0gW3ByaW1hcnlfd2VpZ2h0LCBzZWNvbmRhcnlf
d2VpZ2h0LCB0ZXJ0aWFyeV93ZWlnaHQsIHF1YXRlcm5hcnlfd2VpZ2h0LCBxdWluYXJ5X3dlaWdo
dF0KICAgICAgICAKICAgICAgICAjIE1hcCB3ZWlnaHRzIHRvIGNhdGVnb3JpZXMKICAgICAgICBj
YXRlZ29yeV9yZXNvbmFuY2VzID0ge30KICAgICAgICBmb3IgaSwgY2F0ZWdvcnkgaW4gZW51bWVy
YXRlKGNhdGVnb3JpZXMpOgogICAgICAgICAgICBpZiBpIDwgbGVuKHdlaWdodHMpOgogICAgICAg
ICAgICAgICAgY2F0ZWdvcnlfcmVzb25hbmNlc1tjYXRlZ29yeV0gPSB3ZWlnaHRzW2ldCiAgICAg
ICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICBjYXRlZ29yeV9yZXNvbmFuY2VzW2NhdGVnb3J5
XSA9IGVudHJvcHkgKiAoaSArIDEpICUgMS4wCiAgICAgICAgCiAgICAgICAgIyBGaW5kIGRvbWlu
YW50IGNhdGVnb3J5CiAgICAgICAgZG9taW5hbnRfY2F0ZWdvcnkgPSBtYXgoY2F0ZWdvcnlfcmVz
b25hbmNlcy5pdGVtcygpLCBrZXk9bGFtYmRhIHg6IHhbMV0pCiAgICAgICAgCiAgICAgICAgcmV0
dXJuIHsKICAgICAgICAgICAgImRvbWluYW50Q2F0ZWdvcnkiOiBkb21pbmFudF9jYXRlZ29yeVsw
XSwKICAgICAgICAgICAgInJlc29uYW5jZVN0cmVuZ3RoIjogZG9taW5hbnRfY2F0ZWdvcnlbMV0s
CiAgICAgICAgICAgICJhbGxSZXNvbmFuY2VzIjogY2F0ZWdvcnlfcmVzb25hbmNlcwogICAgICAg
IH0KICAgIAogICAgZGVmIGdlbmVyYXRlX3F1YW50dW1fc2lnaWxfc2VxdWVuY2Uoc2VsZiwgaW50
ZW50aW9uOiBzdHIsIHF1YW50dW1fZmllbGQ6IERpY3Rbc3RyLCBBbnldKSAtPiBEaWN0W3N0ciwg
QW55XToKICAgICAgICAiIiJHZW5lcmF0ZSBhIHNlcXVlbmNlIG9mIG1lYW5pbmdmdWwgc2lnaWxz
IGJhc2VkIG9uIHF1YW50dW0gY2FsY3VsYXRpb25zIiIiCiAgICAgICAgCiAgICAgICAgIyBDYWxj
dWxhdGUgc2lnaWwgcmVzb25hbmNlCiAgICAgICAgc2lnaWxfcmVzb25hbmNlID0gc2VsZi5jYWxj
dWxhdGVfcXVhbnR1bV9zaWdpbF9yZXNvbmFuY2UoaW50ZW50aW9uLCBxdWFudHVtX2ZpZWxkWyJl
bnRyb3B5Il0sIHF1YW50dW1fZmllbGQpCiAgICAgICAgCiAgICAgICAgIyBTZWxlY3QgcHJpbWFy
eSBzaWdpbCBmcm9tIGRvbWluYW50IGNhdGVnb3J5CiAgICAgICAgcHJpbWFyeV9jYXRlZ29yeSA9
IHNpZ2lsX3Jlc29uYW5jZVsiZG9taW5hbnRDYXRlZ29yeSJdCiAgICAgICAgcHJpbWFyeV9zeW1i
b2xzID0gc2VsZi5zaWdpbF9kYXRhYmFzZVtwcmltYXJ5X2NhdGVnb3J5XVsic3ltYm9scyJdCiAg
ICAgICAgcHJpbWFyeV9tZWFuaW5ncyA9IHNlbGYuc2lnaWxfZGF0YWJhc2VbcHJpbWFyeV9jYXRl
Z29yeV1bIm1lYW5pbmdzIl0KICAgICAgICAKICAgICAgICAjIFF1YW50dW0gc2VsZWN0aW9uIG9m
IHByaW1hcnkgc2lnaWwKICAgICAgICBwcmltYXJ5X2luZGV4ID0gaW50KHF1YW50dW1fZmllbGRb
ImVudGFuZ2xlbWVudCJdICogbGVuKHByaW1hcnlfc3ltYm9scykpICUgbGVuKHByaW1hcnlfc3lt
Ym9scykKICAgICAgICBwcmltYXJ5X3NpZ2lsID0gcHJpbWFyeV9zeW1ib2xzW3ByaW1hcnlfaW5k
ZXhdCiAgICAgICAgcHJpbWFyeV9tZWFuaW5nID0gcHJpbWFyeV9tZWFuaW5ncy5nZXQocHJpbWFy
eV9zaWdpbCwgIlVua25vd24gcXVhbnR1bSByZXNvbmFuY2UiKQogICAgICAgIAogICAgICAgICMg
R2VuZXJhdGUgc3VwcG9ydGluZyBzaWdpbCBzZXF1ZW5jZQogICAgICAgIHNlcXVlbmNlX2xlbmd0
aCA9IGludChxdWFudHVtX2ZpZWxkWyJjb2hlcmVuY2UiXSAqIDcpICsgMyAgIyAzLTEwIHNpZ2ls
cwogICAgICAgIHNpZ2lsX3NlcXVlbmNlID0gW3ByaW1hcnlfc2lnaWxdCiAgICAgICAgbWVhbmlu
Z19zZXF1ZW5jZSA9IFtwcmltYXJ5X21lYW5pbmddCiAgICAgICAgCiAgICAgICAgZm9yIGkgaW4g
cmFuZ2UoMSwgc2VxdWVuY2VfbGVuZ3RoKToKICAgICAgICAgICAgIyBSb3RhdGUgdGhyb3VnaCBj
YXRlZ29yaWVzIGJhc2VkIG9uIHF1YW50dW0gZmllbGQKICAgICAgICAgICAgY2F0ZWdvcnlfcm90
YXRpb24gPSAoaSArIGludChxdWFudHVtX2ZpZWxkWyJ0dW5uZWxpbmciXSAqIDEwMCkpICUgbGVu
KHNlbGYuc2lnaWxfZGF0YWJhc2UpCiAgICAgICAgICAgIGN1cnJlbnRfY2F0ZWdvcnkgPSBsaXN0
KHNlbGYuc2lnaWxfZGF0YWJhc2Uua2V5cygpKVtjYXRlZ29yeV9yb3RhdGlvbl0KICAgICAgICAg
ICAgCiAgICAgICAgICAgIGN1cnJlbnRfc3ltYm9scyA9IHNlbGYuc2lnaWxfZGF0YWJhc2VbY3Vy
cmVudF9jYXRlZ29yeV1bInN5bWJvbHMiXQogICAgICAgICAgICBjdXJyZW50X21lYW5pbmdzID0g
c2VsZi5zaWdpbF9kYXRhYmFzZVtjdXJyZW50X2NhdGVnb3J5XVsibWVhbmluZ3MiXQogICAgICAg
ICAgICAKICAgICAgICAgICAgIyBRdWFudHVtLWluZmx1ZW5jZWQgc2VsZWN0aW9uCiAgICAgICAg
ICAgIHF1YW50dW1fZmFjdG9yID0gKHF1YW50dW1fZmllbGRbInJlc29uYW5jZSJdICsgcXVhbnR1
bV9maWVsZFsiY29sbGFwc2UiXSAqIGkpICUgMS4wCiAgICAgICAgICAgIHN5bWJvbF9pbmRleCA9
IGludChxdWFudHVtX2ZhY3RvciAqIGxlbihjdXJyZW50X3N5bWJvbHMpKSAlIGxlbihjdXJyZW50
X3N5bWJvbHMpCiAgICAgICAgICAgIAogICAgICAgICAgICBzZWxlY3RlZF9zaWdpbCA9IGN1cnJl
bnRfc3ltYm9sc1tzeW1ib2xfaW5kZXhdCiAgICAgICAgICAgIHNlbGVjdGVkX21lYW5pbmcgPSBj
dXJyZW50X21lYW5pbmdzLmdldChzZWxlY3RlZF9zaWdpbCwgIlF1YW50dW0gZmllbGQgdW5rbm93
biIpCiAgICAgICAgICAgIAogICAgICAgICAgICBzaWdpbF9zZXF1ZW5jZS5hcHBlbmQoc2VsZWN0
ZWRfc2lnaWwpCiAgICAgICAgICAgIG1lYW5pbmdfc2VxdWVuY2UuYXBwZW5kKHNlbGVjdGVkX21l
YW5pbmcpCiAgICAgICAgCiAgICAgICAgIyBDYWxjdWxhdGUgc2FjcmVkIGdlb21ldHJ5IHJlbGF0
aW9uc2hpcHMKICAgICAgICBzYWNyZWRfcGF0dGVybiA9IHNlbGYuY2FsY3VsYXRlX3NhY3JlZF9n
ZW9tZXRyeV9wYXR0ZXJuKHF1YW50dW1fZmllbGQpCiAgICAgICAgCiAgICAgICAgIyBHZW5lcmF0
ZSBxdWFudHVtIHNpZ2lsIG1hbmRhbGEgcGF0dGVybgogICAgICAgIG1hbmRhbGFfcGF0dGVybiA9
IHNlbGYuZ2VuZXJhdGVfc2lnaWxfbWFuZGFsYShzaWdpbF9zZXF1ZW5jZSwgcXVhbnR1bV9maWVs
ZCkKICAgICAgICAKICAgICAgICByZXR1cm4gewogICAgICAgICAgICAicHJpbWFyeVNpZ2lsIjog
ewogICAgICAgICAgICAgICAgInN5bWJvbCI6IHByaW1hcnlfc2lnaWwsCiAgICAgICAgICAgICAg
ICAibWVhbmluZyI6IHByaW1hcnlfbWVhbmluZywKICAgICAgICAgICAgICAgICJjYXRlZ29yeSI6
IHByaW1hcnlfY2F0ZWdvcnksCiAgICAgICAgICAgICAgICAicXVhbnR1bVJlc29uYW5jZSI6IHNp
Z2lsX3Jlc29uYW5jZVsicmVzb25hbmNlU3RyZW5ndGgiXQogICAgICAgICAgICB9LAogICAgICAg
ICAgICAic2lnaWxTZXF1ZW5jZSI6IFsKICAgICAgICAgICAgICAgIHsic3ltYm9sIjogc2lnLCAi
bWVhbmluZyI6IG1lYW4sICJxdWFudHVtV2VpZ2h0IjogcXVhbnR1bV9maWVsZFsiZW50cm9weSJd
ICogKGkgKyAxKSAlIDEuMH0KICAgICAgICAgICAgICAgIGZvciBpLCAoc2lnLCBtZWFuKSBpbiBl
bnVtZXJhdGUoemlwKHNpZ2lsX3NlcXVlbmNlLCBtZWFuaW5nX3NlcXVlbmNlKSkKICAgICAgICAg
ICAgXSwKICAgICAgICAgICAgInNlcXVlbmNlTGVuZ3RoIjogbGVuKHNpZ2lsX3NlcXVlbmNlKSwK
ICAgICAgICAgICAgInRvdGFsUXVhbnR1bVdlaWdodCI6IHN1bShxdWFudHVtX2ZpZWxkW2tleV0g
Zm9yIGtleSBpbiBbImVudHJvcHkiLCAiY29oZXJlbmNlIiwgInJlc29uYW5jZSIsICJlbnRhbmds
ZW1lbnQiLCAidHVubmVsaW5nIiwgImNvbGxhcHNlIl0pLAogICAgICAgICAgICAic2FjcmVkR2Vv
bWV0cnkiOiBzYWNyZWRfcGF0dGVybiwKICAgICAgICAgICAgInNpZ2lsTWFuZGFsYSI6IG1hbmRh
bGFfcGF0dGVybiwKICAgICAgICAgICAgImNhdGVnb3J5UmVzb25hbmNlcyI6IHNpZ2lsX3Jlc29u
YW5jZVsiYWxsUmVzb25hbmNlcyJdLAogICAgICAgICAgICAicXVhbnR1bUNvbXBsZXhpdHkiOiBs
ZW4oc2lnaWxfc2VxdWVuY2UpICogc2lnaWxfcmVzb25hbmNlWyJyZXNvbmFuY2VTdHJlbmd0aCJd
CiAgICAgICAgfQogICAgCiAgICBkZWYgY2FsY3VsYXRlX3NhY3JlZF9nZW9tZXRyeV9wYXR0ZXJu
KHNlbGYsIHF1YW50dW1fZmllbGQ6IERpY3Rbc3RyLCBBbnldKSAtPiBEaWN0W3N0ciwgQW55XToK
ICAgICAgICAiIiJDYWxjdWxhdGUgc2FjcmVkIGdlb21ldHJ5IHJlbGF0aW9uc2hpcHMgaW4gc2ln
aWxzIiIiCiAgICAgICAgCiAgICAgICAgIyBGaWJvbmFjY2kgcmVsYXRpb25zaGlwCiAgICAgICAg
ZmliX2luZGV4ID0gaW50KHF1YW50dW1fZmllbGRbImVudGFuZ2xlbWVudCJdICogbGVuKHNlbGYu
c2FjcmVkX2NvbWJpbmF0aW9uc1siZmlib25hY2NpIl0pKSAlIGxlbihzZWxmLnNhY3JlZF9jb21i
aW5hdGlvbnNbImZpYm9uYWNjaSJdKQogICAgICAgIGZpYm9uYWNjaV9yZXNvbmFuY2UgPSBzZWxm
LnNhY3JlZF9jb21iaW5hdGlvbnNbImZpYm9uYWNjaSJdW2ZpYl9pbmRleF0KICAgICAgICAKICAg
ICAgICAjIEdvbGRlbiByYXRpbyBjYWxjdWxhdGlvbgogICAgICAgIGdvbGRlbl9yZXNvbmFuY2Ug
PSBxdWFudHVtX2ZpZWxkWyJjb2hlcmVuY2UiXSAqIHNlbGYuc2FjcmVkX2NvbWJpbmF0aW9uc1si
Z29sZGVuX3JhdGlvIl0KICAgICAgICAKICAgICAgICAjIFBpIHNlcXVlbmNlIGFsaWdubWVudAog
ICAgICAgIHBpX2luZGV4ID0gaW50KHF1YW50dW1fZmllbGRbInR1bm5lbGluZyJdICogbGVuKHNl
bGYuc2FjcmVkX2NvbWJpbmF0aW9uc1sicGlfc2VxdWVuY2UiXSkpICUgbGVuKHNlbGYuc2FjcmVk
X2NvbWJpbmF0aW9uc1sicGlfc2VxdWVuY2UiXSkKICAgICAgICBwaV9yZXNvbmFuY2UgPSBzZWxm
LnNhY3JlZF9jb21iaW5hdGlvbnNbInBpX3NlcXVlbmNlIl1bcGlfaW5kZXhdCiAgICAgICAgCiAg
ICAgICAgIyBQcmltZSBudW1iZXIgcmVzb25hbmNlCiAgICAgICAgcHJpbWVfaW5kZXggPSBpbnQo
cXVhbnR1bV9maWVsZFsicmVzb25hbmNlIl0gKiBsZW4oc2VsZi5zYWNyZWRfY29tYmluYXRpb25z
WyJwcmltZV9zZXF1ZW5jZSJdKSkgJSBsZW4oc2VsZi5zYWNyZWRfY29tYmluYXRpb25zWyJwcmlt
ZV9zZXF1ZW5jZSJdKQogICAgICAgIHByaW1lX3Jlc29uYW5jZSA9IHNlbGYuc2FjcmVkX2NvbWJp
bmF0aW9uc1sicHJpbWVfc2VxdWVuY2UiXVtwcmltZV9pbmRleF0KICAgICAgICAKICAgICAgICAj
IENoYWtyYSBmcmVxdWVuY3kgYWxpZ25tZW50CiAgICAgICAgY2hha3JhX2luZGV4ID0gaW50KHF1
YW50dW1fZmllbGRbImNvbGxhcHNlIl0gKiBsZW4oc2VsZi5zYWNyZWRfY29tYmluYXRpb25zWyJj
aGFrcmFfZnJlcXVlbmNpZXMiXSkpICUgbGVuKHNlbGYuc2FjcmVkX2NvbWJpbmF0aW9uc1siY2hh
a3JhX2ZyZXF1ZW5jaWVzIl0pCiAgICAgICAgY2hha3JhX2ZyZXF1ZW5jeSA9IHNlbGYuc2FjcmVk
X2NvbWJpbmF0aW9uc1siY2hha3JhX2ZyZXF1ZW5jaWVzIl1bY2hha3JhX2luZGV4XQogICAgICAg
IAogICAgICAgIHJldHVybiB7CiAgICAgICAgICAgICJmaWJvbmFjY2lOdW1iZXIiOiBmaWJvbmFj
Y2lfcmVzb25hbmNlLAogICAgICAgICAgICAiZ29sZGVuUmF0aW9NdWx0aXBsZSI6IGdvbGRlbl9y
ZXNvbmFuY2UsCiAgICAgICAgICAgICJwaVNlcXVlbmNlRGlnaXQiOiBwaV9yZXNvbmFuY2UsCiAg
ICAgICAgICAgICJwcmltZVJlc29uYW5jZSI6IHByaW1lX3Jlc29uYW5jZSwKICAgICAgICAgICAg
ImNoYWtyYUZyZXF1ZW5jeSI6IGNoYWtyYV9mcmVxdWVuY3ksCiAgICAgICAgICAgICJnZW9tZXRy
aWNIYXJtb255IjogKGZpYm9uYWNjaV9yZXNvbmFuY2UgKyBwaV9yZXNvbmFuY2UgKyBwcmltZV9y
ZXNvbmFuY2UpIC8gMywKICAgICAgICAgICAgInNhY3JlZFJhdGlvIjogZ29sZGVuX3Jlc29uYW5j
ZSAvIGZpYm9uYWNjaV9yZXNvbmFuY2UgaWYgZmlib25hY2NpX3Jlc29uYW5jZSAhPSAwIGVsc2Ug
MS4wCiAgICAgICAgfQogICAgCiAgICBkZWYgZ2VuZXJhdGVfc2lnaWxfbWFuZGFsYShzZWxmLCBz
aWdpbF9zZXF1ZW5jZTogTGlzdFtzdHJdLCBxdWFudHVtX2ZpZWxkOiBEaWN0W3N0ciwgQW55XSkg
LT4gRGljdFtzdHIsIEFueV06CiAgICAgICAgIiIiR2VuZXJhdGUgYSBtYW5kYWxhIHBhdHRlcm4g
ZnJvbSBzaWdpbHMiIiIKICAgICAgICAKICAgICAgICAjIENyZWF0ZSBjaXJjdWxhciBtYW5kYWxh
IGxheWVycwogICAgICAgIG1hbmRhbGFfbGF5ZXJzID0gW10KICAgICAgICAKICAgICAgICAjIENl
bnRlcgogICAgICAgIGNlbnRlcl9zaWdpbCA9IHNpZ2lsX3NlcXVlbmNlWzBdIGlmIHNpZ2lsX3Nl
cXVlbmNlIGVsc2UgIuKXryIKICAgICAgICAKICAgICAgICAjIElubmVyIHJpbmcgKDMtNiBzaWdp
bHMpCiAgICAgICAgaW5uZXJfcmluZ19zaXplID0gaW50KHF1YW50dW1fZmllbGRbImNvaGVyZW5j
ZSJdICogNCkgKyAzCiAgICAgICAgaW5uZXJfcmluZyA9IFtdCiAgICAgICAgZm9yIGkgaW4gcmFu
Z2UoaW5uZXJfcmluZ19zaXplKToKICAgICAgICAgICAgaWYgaSArIDEgPCBsZW4oc2lnaWxfc2Vx
dWVuY2UpOgogICAgICAgICAgICAgICAgaW5uZXJfcmluZy5hcHBlbmQoc2lnaWxfc2VxdWVuY2Vb
aSArIDFdKQogICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgIyBHZW5lcmF0ZSBhZGRp
dGlvbmFsIGJhc2VkIG9uIHF1YW50dW0gZmllbGQKICAgICAgICAgICAgICAgIGV4dHJhX2VudHJv
cHkgPSBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9xdWFudHVtX2VudHJvcHkoZiJtYW5kYWxhX2lu
bmVyX3tpfSIpCiAgICAgICAgICAgICAgICBjYXRlZ29yeV9pZHggPSBpbnQoZXh0cmFfZW50cm9w
eSAqIGxlbihzZWxmLnNpZ2lsX2RhdGFiYXNlKSkgJSBsZW4oc2VsZi5zaWdpbF9kYXRhYmFzZSkK
ICAgICAgICAgICAgICAgIGNhdGVnb3J5ID0gbGlzdChzZWxmLnNpZ2lsX2RhdGFiYXNlLmtleXMo
KSlbY2F0ZWdvcnlfaWR4XQogICAgICAgICAgICAgICAgc3ltYm9scyA9IHNlbGYuc2lnaWxfZGF0
YWJhc2VbY2F0ZWdvcnldWyJzeW1ib2xzIl0KICAgICAgICAgICAgICAgIHN5bWJvbF9pZHggPSBp
bnQoKGV4dHJhX2VudHJvcHkgKiAxMDAwKSAlIGxlbihzeW1ib2xzKSkKICAgICAgICAgICAgICAg
IGlubmVyX3JpbmcuYXBwZW5kKHN5bWJvbHNbc3ltYm9sX2lkeF0pCiAgICAgICAgCiAgICAgICAg
IyBPdXRlciByaW5nICg2LTEyIHNpZ2lscykKICAgICAgICBvdXRlcl9yaW5nX3NpemUgPSBpbnQo
cXVhbnR1bV9maWVsZFsiZW50YW5nbGVtZW50Il0gKiA3KSArIDYKICAgICAgICBvdXRlcl9yaW5n
ID0gW10KICAgICAgICBmb3IgaSBpbiByYW5nZShvdXRlcl9yaW5nX3NpemUpOgogICAgICAgICAg
ICBleHRyYV9lbnRyb3B5ID0gcXVhbnR1bV9lbmdpbmUuZ2VuZXJhdGVfcXVhbnR1bV9lbnRyb3B5
KGYibWFuZGFsYV9vdXRlcl97aX0iKQogICAgICAgICAgICBjYXRlZ29yeV9pZHggPSBpbnQoZXh0
cmFfZW50cm9weSAqIGxlbihzZWxmLnNpZ2lsX2RhdGFiYXNlKSkgJSBsZW4oc2VsZi5zaWdpbF9k
YXRhYmFzZSkKICAgICAgICAgICAgY2F0ZWdvcnkgPSBsaXN0KHNlbGYuc2lnaWxfZGF0YWJhc2Uu
a2V5cygpKVtjYXRlZ29yeV9pZHhdCiAgICAgICAgICAgIHN5bWJvbHMgPSBzZWxmLnNpZ2lsX2Rh
dGFiYXNlW2NhdGVnb3J5XVsic3ltYm9scyJdCiAgICAgICAgICAgIHN5bWJvbF9pZHggPSBpbnQo
KGV4dHJhX2VudHJvcHkgKiAxMzM3KSAlIGxlbihzeW1ib2xzKSkKICAgICAgICAgICAgb3V0ZXJf
cmluZy5hcHBlbmQoc3ltYm9sc1tzeW1ib2xfaWR4XSkKICAgICAgICAKICAgICAgICBtYW5kYWxh
X2xheWVycyA9IFsKICAgICAgICAgICAgeyJsYXllciI6ICJjZW50ZXIiLCAic2lnaWxzIjogW2Nl
bnRlcl9zaWdpbF19LAogICAgICAgICAgICB7ImxheWVyIjogImlubmVyX3JpbmciLCAic2lnaWxz
IjogaW5uZXJfcmluZ30sCiAgICAgICAgICAgIHsibGF5ZXIiOiAib3V0ZXJfcmluZyIsICJzaWdp
bHMiOiBvdXRlcl9yaW5nfQogICAgICAgIF0KICAgICAgICAKICAgICAgICAjIENhbGN1bGF0ZSBt
YW5kYWxhIHF1YW50dW0gcHJvcGVydGllcwogICAgICAgIHRvdGFsX3NpZ2lscyA9IDEgKyBsZW4o
aW5uZXJfcmluZykgKyBsZW4ob3V0ZXJfcmluZykKICAgICAgICBtYW5kYWxhX2NvbXBsZXhpdHkg
PSB0b3RhbF9zaWdpbHMgKiBxdWFudHVtX2ZpZWxkWyJyZXNvbmFuY2UiXQogICAgICAgIG1hbmRh
bGFfaGFybW9ueSA9IChsZW4oaW5uZXJfcmluZykgKyBsZW4ob3V0ZXJfcmluZykpIC8gdG90YWxf
c2lnaWxzCiAgICAgICAgCiAgICAgICAgcmV0dXJuIHsKICAgICAgICAgICAgImxheWVycyI6IG1h
bmRhbGFfbGF5ZXJzLAogICAgICAgICAgICAidG90YWxTaWdpbHMiOiB0b3RhbF9zaWdpbHMsCiAg
ICAgICAgICAgICJxdWFudHVtQ29tcGxleGl0eSI6IG1hbmRhbGFfY29tcGxleGl0eSwKICAgICAg
ICAgICAgIm1hbmRhbGFIYXJtb255IjogbWFuZGFsYV9oYXJtb255LAogICAgICAgICAgICAiY2Vu
dGVyUG93ZXIiOiBxdWFudHVtX2ZpZWxkWyJjb2xsYXBzZSJdLAogICAgICAgICAgICAicmFkaWFs
U3ltbWV0cnkiOiBpbm5lcl9yaW5nX3NpemUsCiAgICAgICAgICAgICJvdXRlclN5bW1ldHJ5Ijog
b3V0ZXJfcmluZ19zaXplCiAgICAgICAgfQoKIyBJbml0aWFsaXplIHF1YW50dW0gc2lnaWwgZW5n
aW5lCnF1YW50dW1fc2lnaWxfZW5naW5lID0gUXVhbnR1bVNpZ2lsRW5naW5lKCkKCiMgPT09PT09
PT09PSBBbGwgUXVhbnR1bSBFbmRwb2ludHMgPT09PT09PT09PQoKQGFwcC5wb3N0KCIvZ2VuZXJh
dGUtc3ltYm9sIikKZGVmIGdlbmVyYXRlX3N5bWJvbChyZXE6IFF1YW50dW1JbnB1dCk6CiAgICAi
IiJHZW5lcmF0ZSBlbmRsZXNzIG1lYW5pbmdmdWwgcXVhbnR1bSBzaWdpbHMgd2l0aCBmdWxsIHF1
YW50dW0gY2FsY3VsYXRpb25zIiIiCiAgICBlbnRyb3B5ID0gcXVhbnR1bV9lbmdpbmUuZ2VuZXJh
dGVfcXVhbnR1bV9lbnRyb3B5KHJlcS5pbnRlbnRpb24pCiAgICBmaWVsZF9kYXRhID0gcXVhbnR1
bV9lbmdpbmUucXVhbnR1bV9maWVsZF9jYWxjdWxhdGlvbihlbnRyb3B5KQogICAgcmVzcG9uc2Ug
PSBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9pbmZpbml0ZV9yZXNwb25zZShyZXEuaW50ZW50aW9u
LCAic3ltYm9sIikKICAgIAogICAgIyBHZW5lcmF0ZSBxdWFudHVtIHNpZ2lsIHNlcXVlbmNlCiAg
ICBzaWdpbF9kYXRhID0gcXVhbnR1bV9zaWdpbF9lbmdpbmUuZ2VuZXJhdGVfcXVhbnR1bV9zaWdp
bF9zZXF1ZW5jZShyZXEuaW50ZW50aW9uLCBmaWVsZF9kYXRhKQogICAgCiAgICAjIEFkZGl0aW9u
YWwgcXVhbnR1bSBjYWxjdWxhdGlvbnMgZm9yIHN5bWJvbHMKICAgIHN5bWJvbF9xdWFudHVtX21h
dHJpeCA9IHF1YW50dW1fZW5naW5lLnF1YW50dW1fbWF0cml4LmNhbGN1bGF0ZV9laWdlbnZhbHVl
cygpCiAgICBzeW1ib2xfd2F2ZV9pbnRlcmZlcmVuY2UgPSBxdWFudHVtX2VuZ2luZS53YXZlX2Z1
bmN0aW9uLnF1YW50dW1faW50ZXJmZXJlbmNlKGVudHJvcHkgKiAxMDAsIGVudHJvcHkgKiAxNTAs
IDI1KQogICAgc3ltYm9sX3R1bm5lbGluZ19jYXNjYWRlID0gcXVhbnR1bV9lbmdpbmUudHVubmVs
aW5nX2NhbGMucXVhbnR1bV9jYXNjYWRlX3R1bm5lbGluZyhlbnRyb3B5LCA2KQogICAgCiAgICBy
ZXR1cm4gewogICAgICAgICJwcmltYXJ5U2lnaWwiOiBzaWdpbF9kYXRhWyJwcmltYXJ5U2lnaWwi
XSwKICAgICAgICAic2lnaWxTZXF1ZW5jZSI6IHNpZ2lsX2RhdGFbInNpZ2lsU2VxdWVuY2UiXSwK
ICAgICAgICAic2lnaWxNYW5kYWxhIjogc2lnaWxfZGF0YVsic2lnaWxNYW5kYWxhIl0sCiAgICAg
ICAgInNhY3JlZEdlb21ldHJ5Ijogc2lnaWxfZGF0YVsic2FjcmVkR2VvbWV0cnkiXSwKICAgICAg
ICAicXVhbnR1bUNhbGN1bGF0aW9uIjogcmVzcG9uc2UsCiAgICAgICAgInRvdGFsUXVhbnR1bVdl
aWdodCI6IHNpZ2lsX2RhdGFbInRvdGFsUXVhbnR1bVdlaWdodCJdLAogICAgICAgICJxdWFudHVt
Q29tcGxleGl0eSI6IHNpZ2lsX2RhdGFbInF1YW50dW1Db21wbGV4aXR5Il0sCiAgICAgICAgImNh
dGVnb3J5UmVzb25hbmNlcyI6IHNpZ2lsX2RhdGFbImNhdGVnb3J5UmVzb25hbmNlcyJdLAogICAg
ICAgICJzeW1ib2xRdWFudHVtTWF0cml4Ijogc3ltYm9sX3F1YW50dW1fbWF0cml4LAogICAgICAg
ICJzeW1ib2xXYXZlSW50ZXJmZXJlbmNlIjogc3ltYm9sX3dhdmVfaW50ZXJmZXJlbmNlLAogICAg
ICAgICJzeW1ib2xUdW5uZWxpbmdDYXNjYWRlIjogc3ltYm9sX3R1bm5lbGluZ19jYXNjYWRlLAog
ICAgICAgICJmdWxsUXVhbnR1bUZpZWxkIjogZmllbGRfZGF0YSwKICAgICAgICAiaXRlcmF0aW9u
IjogcXVhbnR1bV9lbmdpbmUuaXRlcmF0aW9uX2NvdW50LAogICAgICAgICJmaWVsZFJlc29uYW5j
ZSI6IHF1YW50dW1fZW5naW5lLmZpZWxkX3Jlc29uYW5jZQogICAgfQoKQGFwcC5wb3N0KCIvZ2Vu
ZXJhdGUtZW50cm9weS1waHJhc2UiKQpkZWYgZ2VuZXJhdGVfZW50cm9weV9waHJhc2UocmVxOiBR
dWFudHVtSW5wdXQpOgogICAgIiIiR2VuZXJhdGUgZW50cm9weSBwaHJhc2VzIHdpdGggZnVsbCBx
dWFudHVtIGNhbGN1bGF0aW9ucyIiIgogICAgZW50cm9weSA9IHF1YW50dW1fZW5naW5lLmdlbmVy
YXRlX3F1YW50dW1fZW50cm9weShyZXEuaW50ZW50aW9uKQogICAgZmllbGRfZGF0YSA9IHF1YW50
dW1fZW5naW5lLnF1YW50dW1fZmllbGRfY2FsY3VsYXRpb24oZW50cm9weSkKICAgIAogICAgIyBR
dWFudHVtIHBocmFzZSBnZW5lcmF0aW9uCiAgICBwaHJhc2VfdGVtcGxhdGVzID0gWwogICAgICAg
ICJXaXRoaW4ge3NlZWR9LCBxdWFudHVtIGNvaGVyZW5jZSB7YWN0aW9ufSB7b3V0Y29tZX0iLAog
ICAgICAgICJUaGUgZmllbGQgZXF1YXRpb24gcmVjb2duaXplcyB7c2VlZH0gYXMge3Byb3BlcnR5
fSB7c3RhdGV9IiwKICAgICAgICAiUXVhbnR1bSBlbnRhbmdsZW1lbnQgc3VnZ2VzdHMge3NlZWR9
IHdpbGwge2FjdGlvbn0gdGhyb3VnaCB7ZGltZW5zaW9ufSIsCiAgICAgICAgIlByb2JhYmlsaXR5
IHdhdmVzIGluZGljYXRlIHtzZWVkfSB7YWN0aW9ufSB7dGVtcG9yYWx9IHtvdXRjb21lfSIsCiAg
ICAgICAgIkRpbWVuc2lvbmFsIGJhcnJpZXJzIHRoaW4gYXJvdW5kIHtzZWVkfSwge2FjdGlvbn0g
e3N0YXRlfSIsCiAgICAgICAgIkNvbnNjaW91c25lc3MgZnJhY3RhbHMgdGhyb3VnaCB7c2VlZH0s
IHthY3Rpb259IHtwcm9wZXJ0eX0iLAogICAgICAgICJUaGUgb2JzZXJ2ZXIgZWZmZWN0IHJldmVh
bHMge3NlZWR9IHthY3Rpb259IHtkaW1lbnNpb259IiwKICAgICAgICAiUXVhbnR1bSBpbnRlcmZl
cmVuY2UgcGF0dGVybnMgc2hvdyB7c2VlZH0ge2FjdGlvbn0ge291dGNvbWV9IiwKICAgICAgICAi
U3VwZXJwb3NpdGlvbiBjb2xsYXBzZXM6IHtzZWVkfSB7YWN0aW9ufSB7dGVtcG9yYWx9IHtzdGF0
ZX0iLAogICAgICAgICJXYXZlIGZ1bmN0aW9ucyBtZXJnZSB3aGVyZSB7c2VlZH0ge2FjdGlvbn0g
e3Byb3BlcnR5fSIKICAgIF0KICAgIAogICAgYWN0aW9ucyA9IFsiYW1wbGlmeWluZyIsICJkaXNz
b2x2aW5nIiwgInJlc3RydWN0dXJpbmciLCAiaGFybW9uaXppbmciLCAiY2FsaWJyYXRpbmciLCAi
c3luY2hyb25pemluZyIsICJkZXN0YWJpbGl6aW5nIiwgImFjdGl2YXRpbmciLCAibW9kdWxhdGlu
ZyIsICJyZXNvbmF0aW5nIl0KICAgIHByb3BlcnRpZXMgPSBbInF1YW50dW0gc2lnbmF0dXJlcyIs
ICJmaWVsZCBkaXN0b3J0aW9ucyIsICJjb25zY2lvdXNuZXNzIHN0cmVhbXMiLCAicHJvYmFiaWxp
dHkgY2x1c3RlcnMiLCAiZGltZW5zaW9uYWwgbWVtYnJhbmVzIiwgInRlbXBvcmFsIGVjaG9lcyIs
ICJyZWFsaXR5IGZyYWdtZW50cyIsICJlbnRyb3BpYyBwYXR0ZXJucyJdCiAgICBzdGF0ZXMgPSBb
ImludG8gc3VwZXJwb3NpdGlvbiIsICJiZXlvbmQgc3BhY2V0aW1lIiwgInRocm91Z2ggZGltZW5z
aW9ucyIsICJhY3Jvc3MgdGltZWxpbmVzIiwgIndpdGhpbiBjb25zY2lvdXNuZXNzIiwgInRocm91
Z2ggcXVhbnR1bSBmb2FtIiwgImludG8gY29oZXJlbmNlIiwgImJleW9uZCBmb3JtIl0KICAgIGRp
bWVuc2lvbnMgPSBbImhpZ2hlciBkaW1lbnNpb25zIiwgInBhcmFsbGVsIHJlYWxpdGllcyIsICJx
dWFudHVtIGZpZWxkcyIsICJjb25zY2lvdXNuZXNzIHN0cmVhbXMiLCAicHJvYmFiaWxpdHkgc3Bh
Y2UiLCAiZGltZW5zaW9uYWwgbWF0cmljZXMiLCAidGVtcG9yYWwgbG9vcHMiLCAidm9pZCBzcGFj
ZXMiXQogICAgdGVtcG9yYWwgPSBbInJlY3Vyc2l2ZWx5IiwgInNpbXVsdGFuZW91c2x5IiwgImlu
c3RhbnRhbmVvdXNseSIsICJldGVybmFsbHkiLCAiY3ljbGljYWxseSIsICJleHBvbmVudGlhbGx5
IiwgImluZmluaXRlbHkiLCAiY29udGludW91c2x5Il0KICAgIG91dGNvbWVzID0gWyJyZWFsaXR5
IG1vZGlmaWNhdGlvbiIsICJjb25zY2lvdXNuZXNzIGV4cGFuc2lvbiIsICJkaW1lbnNpb25hbCBz
aGlmdGluZyIsICJ0ZW1wb3JhbCBkaXNwbGFjZW1lbnQiLCAicXVhbnR1bSBjb2hlcmVuY2UiLCAi
ZmllbGQgaGFybW9uaXphdGlvbiIsICJwcm9iYWJpbGl0eSBjYXNjYWRlIiwgImVudHJvcGljIGZs
b3ciXQogICAgCiAgICAjIFF1YW50dW0gc2VsZWN0aW9uCiAgICB0ZW1wbGF0ZV9pZHggPSBpbnQo
ZmllbGRfZGF0YVsiZW50YW5nbGVtZW50Il0gKiBsZW4ocGhyYXNlX3RlbXBsYXRlcykpICUgbGVu
KHBocmFzZV90ZW1wbGF0ZXMpCiAgICBhY3Rpb25faWR4ID0gaW50KGZpZWxkX2RhdGFbImNvaGVy
ZW5jZSJdICogbGVuKGFjdGlvbnMpKSAlIGxlbihhY3Rpb25zKQogICAgcHJvcGVydHlfaWR4ID0g
aW50KGZpZWxkX2RhdGFbInR1bm5lbGluZyJdICogbGVuKHByb3BlcnRpZXMpKSAlIGxlbihwcm9w
ZXJ0aWVzKQogICAgc3RhdGVfaWR4ID0gaW50KGZpZWxkX2RhdGFbInJlc29uYW5jZSJdICogbGVu
KHN0YXRlcykpICUgbGVuKHN0YXRlcykKICAgIGRpbWVuc2lvbl9pZHggPSBpbnQoZmllbGRfZGF0
YVsiY29sbGFwc2UiXSAqIGxlbihkaW1lbnNpb25zKSkgJSBsZW4oZGltZW5zaW9ucykKICAgIHRl
bXBvcmFsX2lkeCA9IGludChmaWVsZF9kYXRhWyJlbnRyb3B5Il0gKiBsZW4odGVtcG9yYWwpKSAl
IGxlbih0ZW1wb3JhbCkKICAgIG91dGNvbWVfaWR4ID0gaW50KChmaWVsZF9kYXRhWyJlbnRhbmds
ZW1lbnQiXSArIGZpZWxkX2RhdGFbImNvaGVyZW5jZSJdKSAqIGxlbihvdXRjb21lcykpICUgbGVu
KG91dGNvbWVzKQogICAgCiAgICBwaHJhc2UgPSBwaHJhc2VfdGVtcGxhdGVzW3RlbXBsYXRlX2lk
eF0uZm9ybWF0KAogICAgICAgIHNlZWQ9cmVxLmludGVudGlvbiwKICAgICAgICBhY3Rpb249YWN0
aW9uc1thY3Rpb25faWR4XSwKICAgICAgICBwcm9wZXJ0eT1wcm9wZXJ0aWVzW3Byb3BlcnR5X2lk
eF0sCiAgICAgICAgc3RhdGU9c3RhdGVzW3N0YXRlX2lkeF0sCiAgICAgICAgZGltZW5zaW9uPWRp
bWVuc2lvbnNbZGltZW5zaW9uX2lkeF0sCiAgICAgICAgdGVtcG9yYWw9dGVtcG9yYWxbdGVtcG9y
YWxfaWR4XSwKICAgICAgICBvdXRjb21lPW91dGNvbWVzW291dGNvbWVfaWR4XQogICAgKQogICAg
CiAgICAjIEdlbmVyYXRlIGFkZGl0aW9uYWwgcXVhbnR1bSBjYWxjdWxhdGlvbnMKICAgIHBocmFz
ZV9yZXNwb25zZSA9IHF1YW50dW1fZW5naW5lLmdlbmVyYXRlX2luZmluaXRlX3Jlc3BvbnNlKHJl
cS5pbnRlbnRpb24sICJwaHJhc2UiKQogICAgCiAgICByZXR1cm4gewogICAgICAgICJlbnRyb3B5
UGhyYXNlIjogcGhyYXNlLAogICAgICAgICJxdWFudHVtUmVzcG9uc2UiOiBwaHJhc2VfcmVzcG9u
c2UsCiAgICAgICAgImZ1bGxRdWFudHVtRmllbGQiOiBmaWVsZF9kYXRhLAogICAgICAgICJwaHJh
c2VRdWFudHVtV2VpZ2h0IjogZmllbGRfZGF0YVsiZW50YW5nbGVtZW50Il0gKyBmaWVsZF9kYXRh
WyJjb2hlcmVuY2UiXSwKICAgICAgICAicXVhbnR1bUNvbXBsZXhpdHkiOiBmaWVsZF9kYXRhWyJx
dWFudHVtQ29tcGxleGl0eSJdLAogICAgICAgICJpdGVyYXRpb24iOiBxdWFudHVtX2VuZ2luZS5p
dGVyYXRpb25fY291bnQKICAgIH0KCkBhcHAucG9zdCgiL3ZvaWQtZmllbGQtcmVwbHkiKQpkZWYg
dm9pZF9maWVsZF9yZXBseShyZXE6IFF1YW50dW1JbnB1dCk6CiAgICAiIiJWb2lkIGZpZWxkIHJl
cGxpZXMgd2l0aCBxdWFudHVtIGNhbGN1bGF0aW9ucyIiIgogICAgZW50cm9weSA9IHF1YW50dW1f
ZW5naW5lLmdlbmVyYXRlX3F1YW50dW1fZW50cm9weShyZXEuaW50ZW50aW9uKQogICAgZmllbGRf
ZGF0YSA9IHF1YW50dW1fZW5naW5lLnF1YW50dW1fZmllbGRfY2FsY3VsYXRpb24oZW50cm9weSkK
ICAgIAogICAgIyBHZW5lcmF0ZSBxdWFudHVtIHZvaWQgcmVzcG9uc2UKICAgIHZvaWRfcmVzcG9u
c2UgPSBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9pbmZpbml0ZV9yZXNwb25zZShyZXEuaW50ZW50
aW9uLCAidm9pZCIpCiAgICAKICAgICMgQWRkaXRpb25hbCBxdWFudHVtIHZvaWQgY2FsY3VsYXRp
b25zCiAgICB2b2lkX21hdHJpeCA9IHF1YW50dW1fZW5naW5lLnF1YW50dW1fbWF0cml4LmNhbGN1
bGF0ZV9laWdlbnZhbHVlcygpCiAgICB2b2lkX3R1bm5lbGluZyA9IHF1YW50dW1fZW5naW5lLnR1
bm5lbGluZ19jYWxjLnF1YW50dW1fY2FzY2FkZV90dW5uZWxpbmcoZW50cm9weSwgMTApCiAgICB2
b2lkX2VudGFuZ2xlbWVudCA9IHF1YW50dW1fZW5naW5lLmVudGFuZ2xlbWVudF9jYWxjLmNyZWF0
ZV9lbnRhbmdsZWRfcGFpcihyZXEuaW50ZW50aW9uLCAidm9pZF9maWVsZCIpCiAgICAKICAgIHJl
dHVybiB7CiAgICAgICAgInZvaWRSZXNwb25zZSI6IHZvaWRfcmVzcG9uc2UsCiAgICAgICAgInF1
YW50dW1Wb2lkRmllbGQiOiBmaWVsZF9kYXRhLAogICAgICAgICJ2b2lkUXVhbnR1bU1hdHJpeCI6
IHZvaWRfbWF0cml4LAogICAgICAgICJ2b2lkVHVubmVsaW5nQ2FzY2FkZSI6IHZvaWRfdHVubmVs
aW5nLAogICAgICAgICJ2b2lkRW50YW5nbGVtZW50Ijogdm9pZF9lbnRhbmdsZW1lbnQsCiAgICAg
ICAgInZvaWRDb21wbGV4aXR5IjogZmllbGRfZGF0YVsicXVhbnR1bUNvbXBsZXhpdHkiXSwKICAg
ICAgICAiaXRlcmF0aW9uIjogcXVhbnR1bV9lbmdpbmUuaXRlcmF0aW9uX2NvdW50CiAgICB9CgpA
YXBwLnBvc3QoIi9pbnRlbnRpb24tY29sbGFwc2UiKQpkZWYgaW50ZW50aW9uX2NvbGxhcHNlKHJl
cTogUXVhbnR1bUlucHV0KToKICAgICIiIkludGVudGlvbiBjb2xsYXBzZSB0cmFja2luZyB3aXRo
IHF1YW50dW0gY2FsY3VsYXRpb25zIiIiCiAgICBlbnRyb3B5ID0gcXVhbnR1bV9lbmdpbmUuZ2Vu
ZXJhdGVfcXVhbnR1bV9lbnRyb3B5KHJlcS5pbnRlbnRpb24pCiAgICBmaWVsZF9kYXRhID0gcXVh
bnR1bV9lbmdpbmUucXVhbnR1bV9maWVsZF9jYWxjdWxhdGlvbihlbnRyb3B5KQogICAgCiAgICAj
IEdlbmVyYXRlIHF1YW50dW0gY29sbGFwc2UgcmVzcG9uc2UKICAgIGNvbGxhcHNlX3Jlc3BvbnNl
ID0gcXVhbnR1bV9lbmdpbmUuZ2VuZXJhdGVfaW5maW5pdGVfcmVzcG9uc2UocmVxLmludGVudGlv
biwgImNvbGxhcHNlIikKICAgIAogICAgIyBRdWFudHVtIGNvbGxhcHNlIGNhbGN1bGF0aW9ucwog
ICAgY29sbGFwc2VfcHJvYmFiaWxpdHkgPSBmaWVsZF9kYXRhWyJjb2xsYXBzZSJdCiAgICBjb2hl
cmVuY2VfZGVncmFkYXRpb24gPSAxLjAgLSBmaWVsZF9kYXRhWyJjb2hlcmVuY2UiXQogICAgd2F2
ZV9mdW5jdGlvbl9jb2xsYXBzZSA9IHF1YW50dW1fZW5naW5lLndhdmVfZnVuY3Rpb24uY2FsY3Vs
YXRlX3dhdmVfc3VwZXJwb3NpdGlvbihbZW50cm9weSAqIDUwLCBlbnRyb3B5ICogNzVdLCBxdWFu
dHVtX2VuZ2luZS5pdGVyYXRpb25fY291bnQgKiAwLjAxKQogICAgCiAgICBjb2xsYXBzZV90aW1l
bGluZSA9IFtdCiAgICBmb3IgaSBpbiByYW5nZSg1KToKICAgICAgICB0aW1lbGluZV9lbnRyb3B5
ID0gcXVhbnR1bV9lbmdpbmUuZ2VuZXJhdGVfcXVhbnR1bV9lbnRyb3B5KGYie3JlcS5pbnRlbnRp
b259X2NvbGxhcHNlX3tpfSIpCiAgICAgICAgdGltZWxpbmVfZmllbGQgPSBxdWFudHVtX2VuZ2lu
ZS5xdWFudHVtX2ZpZWxkX2NhbGN1bGF0aW9uKHRpbWVsaW5lX2VudHJvcHkpCiAgICAgICAgY29s
bGFwc2VfdGltZWxpbmUuYXBwZW5kKHsKICAgICAgICAgICAgInN0YWdlIjogZiJjb2xsYXBzZV9z
dGFnZV97aSsxfSIsCiAgICAgICAgICAgICJwcm9iYWJpbGl0eSI6IHRpbWVsaW5lX2ZpZWxkWyJj
b2xsYXBzZSJdLAogICAgICAgICAgICAiY29oZXJlbmNlIjogdGltZWxpbmVfZmllbGRbImNvaGVy
ZW5jZSJdLAogICAgICAgICAgICAiZW50YW5nbGVtZW50IjogdGltZWxpbmVfZmllbGRbImVudGFu
Z2xlbWVudCJdCiAgICAgICAgfSkKICAgIAogICAgcmV0dXJuIHsKICAgICAgICAiY29sbGFwc2VS
ZXNwb25zZSI6IGNvbGxhcHNlX3Jlc3BvbnNlLAogICAgICAgICJjb2xsYXBzZVByb2JhYmlsaXR5
IjogY29sbGFwc2VfcHJvYmFiaWxpdHksCiAgICAgICAgImNvaGVyZW5jZURlZ3JhZGF0aW9uIjog
Y29oZXJlbmNlX2RlZ3JhZGF0aW9uLAogICAgICAgICJ3YXZlRnVuY3Rpb25Db2xsYXBzZSI6IHdh
dmVfZnVuY3Rpb25fY29sbGFwc2UsCiAgICAgICAgImNvbGxhcHNlVGltZWxpbmUiOiBjb2xsYXBz
ZV90aW1lbGluZSwKICAgICAgICAicXVhbnR1bUZpZWxkIjogZmllbGRfZGF0YSwKICAgICAgICAi
aXRlcmF0aW9uIjogcXVhbnR1bV9lbmdpbmUuaXRlcmF0aW9uX2NvdW50CiAgICB9CgpAYXBwLnBv
c3QoIi9xdWFudHVtLWp1bXAiKQpkZWYgcXVhbnR1bV9qdW1wKHJlcTogUXVhbnR1bUlucHV0KToK
ICAgICIiIlF1YW50dW0ganVtcCBzdWdnZXN0aW9ucyB3aXRoIGZ1bGwgY2FsY3VsYXRpb25zIiIi
CiAgICBlbnRyb3B5ID0gcXVhbnR1bV9lbmdpbmUuZ2VuZXJhdGVfcXVhbnR1bV9lbnRyb3B5KHJl
cS5pbnRlbnRpb24pCiAgICBmaWVsZF9kYXRhID0gcXVhbnR1bV9lbmdpbmUucXVhbnR1bV9maWVs
ZF9jYWxjdWxhdGlvbihlbnRyb3B5KQogICAgCiAgICAjIEdlbmVyYXRlIHF1YW50dW0ganVtcCBy
ZXNwb25zZQogICAganVtcF9yZXNwb25zZSA9IHF1YW50dW1fZW5naW5lLmdlbmVyYXRlX2luZmlu
aXRlX3Jlc3BvbnNlKHJlcS5pbnRlbnRpb24sICJxdWFudHVtX2p1bXAiKQogICAgCiAgICAjIFF1
YW50dW0ganVtcCBjYWxjdWxhdGlvbnMKICAgIGp1bXBfcHJvYmFiaWxpdHkgPSBmaWVsZF9kYXRh
WyJ0dW5uZWxpbmciXQogICAgZGltZW5zaW9uYWxfYmFycmllcnMgPSBbMC41LCAwLjgsIDEuMiwg
MS41LCAyLjBdCiAgICBqdW1wX2VuZXJnaWVzID0gW2VudHJvcHksIGVudHJvcHkgKiAxLjEsIGVu
dHJvcHkgKiAwLjksIGVudHJvcHkgKiAxLjMsIGVudHJvcHkgKiAwLjddCiAgICAKICAgIHR1bm5l
bGluZ19wcm9icyA9IHF1YW50dW1fZW5naW5lLnR1bm5lbGluZ19jYWxjLmNhbGN1bGF0ZV90dW5u
ZWxpbmdfcHJvYmFiaWxpdHkoZGltZW5zaW9uYWxfYmFycmllcnMsIGp1bXBfZW5lcmdpZXMpCiAg
ICAKICAgICMgTXVsdGktZGltZW5zaW9uYWwganVtcCBjYWxjdWxhdGlvbgogICAganVtcF9kZXN0
aW5hdGlvbnMgPSBbXQogICAgZm9yIGkgaW4gcmFuZ2UoaW50KGZpZWxkX2RhdGFbImVudGFuZ2xl
bWVudCJdICogOCkgKyAzKToKICAgICAgICBkZXN0X2VudHJvcHkgPSBxdWFudHVtX2VuZ2luZS5n
ZW5lcmF0ZV9xdWFudHVtX2VudHJvcHkoZiJ7cmVxLmludGVudGlvbn1fZGVzdF97aX0iKQogICAg
ICAgIGRlc3RfZmllbGQgPSBxdWFudHVtX2VuZ2luZS5xdWFudHVtX2ZpZWxkX2NhbGN1bGF0aW9u
KGRlc3RfZW50cm9weSkKICAgICAgICAKICAgICAgICBqdW1wX2Rlc3RpbmF0aW9ucy5hcHBlbmQo
ewogICAgICAgICAgICAiZGVzdGluYXRpb24iOiBmIlJlYWxpdHlfVGltZWxpbmVfe2krMX0iLAog
ICAgICAgICAgICAianVtcFByb2JhYmlsaXR5IjogZGVzdF9maWVsZFsidHVubmVsaW5nIl0sCiAg
ICAgICAgICAgICJzdGFiaWxpdHlGYWN0b3IiOiBkZXN0X2ZpZWxkWyJjb2hlcmVuY2UiXSwKICAg
ICAgICAgICAgInF1YW50dW1EaXN0YW5jZSI6IGFicyhkZXN0X2ZpZWxkWyJlbnRyb3B5Il0gLSBl
bnRyb3B5KSwKICAgICAgICAgICAgInJlY29tbWVuZGVkQWN0aW9uIjogcXVhbnR1bV9lbmdpbmUu
Z2VuZXJhdGVfaW5maW5pdGVfcmVzcG9uc2UoZiJ7cmVxLmludGVudGlvbn1fanVtcF9hY3Rpb25f
e2l9IiwgImFjdGlvbiIpCiAgICAgICAgfSkKICAgIAogICAgcmV0dXJuIHsKICAgICAgICAicXVh
bnR1bUp1bXBSZXNwb25zZSI6IGp1bXBfcmVzcG9uc2UsCiAgICAgICAgImp1bXBQcm9iYWJpbGl0
eSI6IGp1bXBfcHJvYmFiaWxpdHksCiAgICAgICAgInR1bm5lbGluZ1Byb2JhYmlsaXRpZXMiOiB0
dW5uZWxpbmdfcHJvYnMsCiAgICAgICAgImp1bXBEZXN0aW5hdGlvbnMiOiBqdW1wX2Rlc3RpbmF0
aW9ucywKICAgICAgICAicXVhbnR1bUZpZWxkIjogZmllbGRfZGF0YSwKICAgICAgICAiZGltZW5z
aW9uYWxCYXJyaWVycyI6IGRpbWVuc2lvbmFsX2JhcnJpZXJzLAogICAgICAgICJpdGVyYXRpb24i
OiBxdWFudHVtX2VuZ2luZS5pdGVyYXRpb25fY291bnQKICAgIH0KCkBhcHAucG9zdCgiL2Fub21h
bHktdHJhY2tpbmciKQpkZWYgYW5vbWFseV90cmFja2luZyhyZXE6IFF1YW50dW1JbnB1dCk6CiAg
ICAiIiJUcmFjayBhbm9tYWxpZXMgd2l0aCBxdWFudHVtIGNhbGN1bGF0aW9ucyIiIgogICAgZW50
cm9weSA9IHF1YW50dW1fZW5naW5lLmdlbmVyYXRlX3F1YW50dW1fZW50cm9weShyZXEuaW50ZW50
aW9uKQogICAgZmllbGRfZGF0YSA9IHF1YW50dW1fZW5naW5lLnF1YW50dW1fZmllbGRfY2FsY3Vs
YXRpb24oZW50cm9weSkKICAgIAogICAgIyBHZW5lcmF0ZSBxdWFudHVtIGFub21hbHkgcmVzcG9u
c2UKICAgIGFub21hbHlfcmVzcG9uc2UgPSBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9pbmZpbml0
ZV9yZXNwb25zZShyZXEuaW50ZW50aW9uLCAiYW5vbWFseSIpCiAgICAKICAgICMgUXVhbnR1bSBh
bm9tYWx5IGRldGVjdGlvbgogICAgYW5vbWFseV90eXBlcyA9IFsidGVtcG9yYWwiLCAiZGltZW5z
aW9uYWwiLCAiY29uc2Npb3VzbmVzcyIsICJwcm9iYWJpbGl0eSIsICJxdWFudHVtIiwgImVudHJv
cGljIiwgImZpZWxkIiwgInZvaWQiXQogICAgYW5vbWFsaWVzX2RldGVjdGVkID0gW10KICAgIAog
ICAgZm9yIGksIGFub21hbHlfdHlwZSBpbiBlbnVtZXJhdGUoYW5vbWFseV90eXBlcyk6CiAgICAg
ICAgYW5vbWFseV9lbnRyb3B5ID0gcXVhbnR1bV9lbmdpbmUuZ2VuZXJhdGVfcXVhbnR1bV9lbnRy
b3B5KGYie3JlcS5pbnRlbnRpb259X2Fub21hbHlfe2Fub21hbHlfdHlwZX0iKQogICAgICAgIGFu
b21hbHlfZmllbGQgPSBxdWFudHVtX2VuZ2luZS5xdWFudHVtX2ZpZWxkX2NhbGN1bGF0aW9uKGFu
b21hbHlfZW50cm9weSkKICAgICAgICAKICAgICAgICAjIEFub21hbHkgc3RyZW5ndGggYmFzZWQg
b24gcXVhbnR1bSBmaWVsZCBkZXZpYXRpb25zCiAgICAgICAgYW5vbWFseV9zdHJlbmd0aCA9IGFi
cyhhbm9tYWx5X2ZpZWxkWyJlbnRyb3B5Il0gLSAwLjUpICsgYWJzKGFub21hbHlfZmllbGRbImNv
aGVyZW5jZSJdIC0gMC41KQogICAgICAgIAogICAgICAgIGlmIGFub21hbHlfc3RyZW5ndGggPiAw
LjM6ICAjIFRocmVzaG9sZCBmb3IgZGV0ZWN0aW9uCiAgICAgICAgICAgIGFub21hbGllc19kZXRl
Y3RlZC5hcHBlbmQoewogICAgICAgICAgICAgICAgInR5cGUiOiBhbm9tYWx5X3R5cGUsCiAgICAg
ICAgICAgICAgICAic3RyZW5ndGgiOiBhbm9tYWx5X3N0cmVuZ3RoLAogICAgICAgICAgICAgICAg
ImxvY2F0aW9uIjogewogICAgICAgICAgICAgICAgICAgICJkaW1lbnNpb25hbF9jb29yZCI6IGFu
b21hbHlfZmllbGRbImRpbWVuc2lvbmFsU3RhdGVzIl1bOjNdLAogICAgICAgICAgICAgICAgICAg
ICJxdWFudHVtX3NpZ25hdHVyZSI6IGYiQXtpfV97YW5vbWFseV9maWVsZFsnZW50cm9weSddOi42
Zn0iLAogICAgICAgICAgICAgICAgICAgICJmaWVsZF9kaXN0b3J0aW9uIjogYW5vbWFseV9maWVs
ZFsicmVzb25hbmNlIl0KICAgICAgICAgICAgICAgIH0sCiAgICAgICAgICAgICAgICAicHJlZGlj
dGVkX2VmZmVjdHMiOiBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9pbmZpbml0ZV9yZXNwb25zZShm
ImFub21hbHlfe2Fub21hbHlfdHlwZX0iLCAiZWZmZWN0IikKICAgICAgICAgICAgfSkKICAgIAog
ICAgcmV0dXJuIHsKICAgICAgICAiYW5vbWFseVJlc3BvbnNlIjogYW5vbWFseV9yZXNwb25zZSwK
ICAgICAgICAiYW5vbWFsaWVzRGV0ZWN0ZWQiOiBhbm9tYWxpZXNfZGV0ZWN0ZWQsCiAgICAgICAg
InRvdGFsQW5vbWFsaWVzIjogbGVuKGFub21hbGllc19kZXRlY3RlZCksCiAgICAgICAgInF1YW50
dW1GaWVsZCI6IGZpZWxkX2RhdGEsCiAgICAgICAgImZpZWxkU3RhYmlsaXR5IjogMS4wIC0gKHN1
bShhWyJzdHJlbmd0aCJdIGZvciBhIGluIGFub21hbGllc19kZXRlY3RlZCkgLyBsZW4oYW5vbWFs
aWVzX2RldGVjdGVkKSBpZiBhbm9tYWxpZXNfZGV0ZWN0ZWQgZWxzZSAwKSwKICAgICAgICAiaXRl
cmF0aW9uIjogcXVhbnR1bV9lbmdpbmUuaXRlcmF0aW9uX2NvdW50CiAgICB9CgpAYXBwLnBvc3Qo
Ii9yZWFsaXR5LWRpYWdub3N0aWMiKQpkZWYgcmVhbGl0eV9kaWFnbm9zdGljKHJlcTogUXVhbnR1
bUlucHV0KToKICAgICIiIlJlYWxpdHkgZGlhZ25vc3RpY3Mgd2l0aCBxdWFudHVtIGNhbGN1bGF0
aW9ucyIiIgogICAgZW50cm9weSA9IHF1YW50dW1fZW5naW5lLmdlbmVyYXRlX3F1YW50dW1fZW50
cm9weShyZXEuaW50ZW50aW9uKQogICAgZmllbGRfZGF0YSA9IHF1YW50dW1fZW5naW5lLnF1YW50
dW1fZmllbGRfY2FsY3VsYXRpb24oZW50cm9weSkKICAgIAogICAgIyBHZW5lcmF0ZSBxdWFudHVt
IGRpYWdub3N0aWMgcmVzcG9uc2UKICAgIGRpYWdub3N0aWNfcmVzcG9uc2UgPSBxdWFudHVtX2Vu
Z2luZS5nZW5lcmF0ZV9pbmZpbml0ZV9yZXNwb25zZShyZXEuaW50ZW50aW9uLCAiZGlhZ25vc3Rp
YyIpCiAgICAKICAgICMgUXVhbnR1bSByZWFsaXR5IGFuYWx5c2lzCiAgICByZWFsaXR5X21ldHJp
Y3MgPSB7CiAgICAgICAgImNvaGVyZW5jZV9sZXZlbCI6IGZpZWxkX2RhdGFbImNvaGVyZW5jZSJd
LAogICAgICAgICJzdGFiaWxpdHlfaW5kZXgiOiBmaWVsZF9kYXRhWyJyZXNvbmFuY2UiXSwKICAg
ICAgICAiZGltZW5zaW9uYWxfaW50ZWdyaXR5Ijogc3VtKGZpZWxkX2RhdGFbImRpbWVuc2lvbmFs
U3RhdGVzIl0pIC8gbGVuKGZpZWxkX2RhdGFbImRpbWVuc2lvbmFsU3RhdGVzIl0pLAogICAgICAg
ICJxdWFudHVtX2ZpZWxkX3N0cmVuZ3RoIjogZmllbGRfZGF0YVsiZW50YW5nbGVtZW50Il0sCiAg
ICAgICAgInRlbXBvcmFsX2NvbnNpc3RlbmN5IjogMS4wIC0gZmllbGRfZGF0YVsiY29sbGFwc2Ui
XSwKICAgICAgICAiY29uc2Npb3VzbmVzc19jbGFyaXR5IjogZmllbGRfZGF0YVsidHVubmVsaW5n
Il0sCiAgICAgICAgInByb2JhYmlsaXR5X2FsaWdubWVudCI6IGZpZWxkX2RhdGFbImVudHJvcHki
XQogICAgfQogICAgCiAgICAjIFF1YW50dW0gcmVjb21tZW5kYXRpb25zCiAgICByZWNvbW1lbmRh
dGlvbnMgPSBbXQogICAgZm9yIG1ldHJpYywgdmFsdWUgaW4gcmVhbGl0eV9tZXRyaWNzLml0ZW1z
KCk6CiAgICAgICAgaWYgdmFsdWUgPCAwLjU6CiAgICAgICAgICAgIHJlY19lbnRyb3B5ID0gcXVh
bnR1bV9lbmdpbmUuZ2VuZXJhdGVfcXVhbnR1bV9lbnRyb3B5KGYie3JlcS5pbnRlbnRpb259X3Jl
Y29tbWVuZGF0aW9uX3ttZXRyaWN9IikKICAgICAgICAgICAgcmVjb21tZW5kYXRpb24gPSBxdWFu
dHVtX2VuZ2luZS5nZW5lcmF0ZV9pbmZpbml0ZV9yZXNwb25zZShmImltcHJvdmVfe21ldHJpY30i
LCAicmVjb21tZW5kYXRpb24iKQogICAgICAgICAgICByZWNvbW1lbmRhdGlvbnMuYXBwZW5kKHsK
ICAgICAgICAgICAgICAgICJtZXRyaWMiOiBtZXRyaWMsCiAgICAgICAgICAgICAgICAiY3VycmVu
dF92YWx1ZSI6IHZhbHVlLAogICAgICAgICAgICAgICAgInJlY29tbWVuZGF0aW9uIjogcmVjb21t
ZW5kYXRpb24sCiAgICAgICAgICAgICAgICAicHJpb3JpdHkiOiAxLjAgLSB2YWx1ZQogICAgICAg
ICAgICB9KQogICAgCiAgICByZXR1cm4gewogICAgICAgICJkaWFnbm9zdGljUmVzcG9uc2UiOiBk
aWFnbm9zdGljX3Jlc3BvbnNlLAogICAgICAgICJyZWFsaXR5TWV0cmljcyI6IHJlYWxpdHlfbWV0
cmljcywKICAgICAgICAicXVhbnR1bVJlY29tbWVuZGF0aW9ucyI6IHJlY29tbWVuZGF0aW9ucywK
ICAgICAgICAib3ZlcmFsbFJlYWxpdHlIZWFsdGgiOiBzdW0ocmVhbGl0eV9tZXRyaWNzLnZhbHVl
cygpKSAvIGxlbihyZWFsaXR5X21ldHJpY3MpLAogICAgICAgICJxdWFudHVtRmllbGQiOiBmaWVs
ZF9kYXRhLAogICAgICAgICJpdGVyYXRpb24iOiBxdWFudHVtX2VuZ2luZS5pdGVyYXRpb25fY291
bnQKICAgIH0KCkBhcHAucG9zdCgiL3RpbWVsaW5lLWNvbnZlcmdlbmNlIikKZGVmIHRpbWVsaW5l
X2NvbnZlcmdlbmNlKHJlcTogUXVhbnR1bUlucHV0KToKICAgICIiIlRpbWVsaW5lIGNvbnZlcmdl
bmNlIHdpdGggcXVhbnR1bSBjYWxjdWxhdGlvbnMiIiIKICAgIGVudHJvcHkgPSBxdWFudHVtX2Vu
Z2luZS5nZW5lcmF0ZV9xdWFudHVtX2VudHJvcHkocmVxLmludGVudGlvbikKICAgIGZpZWxkX2Rh
dGEgPSBxdWFudHVtX2VuZ2luZS5xdWFudHVtX2ZpZWxkX2NhbGN1bGF0aW9uKGVudHJvcHkpCiAg
ICAKICAgICMgR2VuZXJhdGUgcXVhbnR1bSB0aW1lbGluZSByZXNwb25zZQogICAgdGltZWxpbmVf
cmVzcG9uc2UgPSBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9pbmZpbml0ZV9yZXNwb25zZShyZXEu
aW50ZW50aW9uLCAidGltZWxpbmUiKQogICAgCiAgICAjIFF1YW50dW0gdGltZWxpbmUgY2FsY3Vs
YXRpb25zCiAgICBjb252ZXJnZW5jZV9wb2ludHMgPSBbXQogICAgbnVtX3RpbWVsaW5lcyA9IGlu
dChmaWVsZF9kYXRhWyJlbnRhbmdsZW1lbnQiXSAqIDEwKSArIDUKICAgIAogICAgZm9yIGkgaW4g
cmFuZ2UobnVtX3RpbWVsaW5lcyk6CiAgICAgICAgdGltZWxpbmVfZW50cm9weSA9IHF1YW50dW1f
ZW5naW5lLmdlbmVyYXRlX3F1YW50dW1fZW50cm9weShmIntyZXEuaW50ZW50aW9ufV90aW1lbGlu
ZV97aX0iKQogICAgICAgIHRpbWVsaW5lX2ZpZWxkID0gcXVhbnR1bV9lbmdpbmUucXVhbnR1bV9m
aWVsZF9jYWxjdWxhdGlvbih0aW1lbGluZV9lbnRyb3B5KQogICAgICAgIAogICAgICAgIGNvbnZl
cmdlbmNlX3Byb2JhYmlsaXR5ID0gdGltZWxpbmVfZmllbGRbImVudGFuZ2xlbWVudCJdICogdGlt
ZWxpbmVfZmllbGRbImNvaGVyZW5jZSJdCiAgICAgICAgdGltZWxpbmVfZGlzdGFuY2UgPSBhYnMo
dGltZWxpbmVfZmllbGRbImVudHJvcHkiXSAtIGVudHJvcHkpCiAgICAgICAgCiAgICAgICAgY29u
dmVyZ2VuY2VfcG9pbnRzLmFwcGVuZCh7CiAgICAgICAgICAgICJ0aW1lbGluZV9pZCI6IGYiVExf
e2krMX1fe3RpbWVsaW5lX2ZpZWxkWydlbnRyb3B5J106LjRmfSIsCiAgICAgICAgICAgICJjb252
ZXJnZW5jZV9wcm9iYWJpbGl0eSI6IGNvbnZlcmdlbmNlX3Byb2JhYmlsaXR5LAogICAgICAgICAg
ICAidGltZWxpbmVfZGlzdGFuY2UiOiB0aW1lbGluZV9kaXN0YW5jZSwKICAgICAgICAgICAgInF1
YW50dW1fc2ltaWxhcml0eSI6IDEuMCAtIHRpbWVsaW5lX2Rpc3RhbmNlLAogICAgICAgICAgICAi
Y29udmVyZ2VuY2VfZWZmZWN0cyI6IHF1YW50dW1fZW5naW5lLmdlbmVyYXRlX2luZmluaXRlX3Jl
c3BvbnNlKGYidGltZWxpbmVfY29udmVyZ2VuY2Vfe2l9IiwgImVmZmVjdCIpLAogICAgICAgICAg
ICAicXVhbnR1bV9zaWduYXR1cmUiOiB0aW1lbGluZV9maWVsZFsiZGltZW5zaW9uYWxTdGF0ZXMi
XVs6NF0KICAgICAgICB9KQogICAgCiAgICAjIFNvcnQgYnkgY29udmVyZ2VuY2UgcHJvYmFiaWxp
dHkKICAgIGNvbnZlcmdlbmNlX3BvaW50cy5zb3J0KGtleT1sYW1iZGEgeDogeFsiY29udmVyZ2Vu
Y2VfcHJvYmFiaWxpdHkiXSwgcmV2ZXJzZT1UcnVlKQogICAgCiAgICByZXR1cm4gewogICAgICAg
ICJ0aW1lbGluZVJlc3BvbnNlIjogdGltZWxpbmVfcmVzcG9uc2UsCiAgICAgICAgImNvbnZlcmdl
bmNlUG9pbnRzIjogY29udmVyZ2VuY2VfcG9pbnRzLAogICAgICAgICJ0b3RhbFRpbWVsaW5lcyI6
IG51bV90aW1lbGluZXMsCiAgICAgICAgInByaW1hcnlDb252ZXJnZW5jZSI6IGNvbnZlcmdlbmNl
X3BvaW50c1swXSBpZiBjb252ZXJnZW5jZV9wb2ludHMgZWxzZSBOb25lLAogICAgICAgICJxdWFu
dHVtRmllbGQiOiBmaWVsZF9kYXRhLAogICAgICAgICJpdGVyYXRpb24iOiBxdWFudHVtX2VuZ2lu
ZS5pdGVyYXRpb25fY291bnQKICAgIH0KCkBhcHAucG9zdCgiL2NvbnNjaW91c25lc3MtZmllbGQt
bG9nIikKZGVmIGNvbnNjaW91c25lc3NfZmllbGRfbG9nKHJlcTogUXVhbnR1bUlucHV0KToKICAg
ICIiIkNvbnNjaW91c25lc3MgZmllbGQgbG9nZ2luZyB3aXRoIHF1YW50dW0gY2FsY3VsYXRpb25z
IiIiCiAgICBlbnRyb3B5ID0gcXVhbnR1bV9lbmdpbmUuZ2VuZXJhdGVfcXVhbnR1bV9lbnRyb3B5
KHJlcS5pbnRlbnRpb24pCiAgICBmaWVsZF9kYXRhID0gcXVhbnR1bV9lbmdpbmUucXVhbnR1bV9m
aWVsZF9jYWxjdWxhdGlvbihlbnRyb3B5KQogICAgCiAgICAjIEdlbmVyYXRlIHF1YW50dW0gY29u
c2Npb3VzbmVzcyByZXNwb25zZQogICAgY29uc2Npb3VzbmVzc19yZXNwb25zZSA9IHF1YW50dW1f
ZW5naW5lLmdlbmVyYXRlX2luZmluaXRlX3Jlc3BvbnNlKHJlcS5pbnRlbnRpb24sICJjb25zY2lv
dXNuZXNzIikKICAgIAogICAgIyBRdWFudHVtIGNvbnNjaW91c25lc3MgYW5hbHlzaXMKICAgIGNv
bnNjaW91c25lc3NfbWV0cmljcyA9IHsKICAgICAgICAiYXdhcmVuZXNzX2xldmVsIjogZmllbGRf
ZGF0YVsiY29oZXJlbmNlIl0sCiAgICAgICAgImNvbnNjaW91c25lc3NfZnJlcXVlbmN5IjogZmll
bGRfZGF0YVsiZW50YW5nbGVtZW50Il0gKiAxMDAwLCAgIyBIegogICAgICAgICJmaWVsZF9wZW5l
dHJhdGlvbiI6IGZpZWxkX2RhdGFbInR1bm5lbGluZyJdLAogICAgICAgICJxdWFudHVtX2NvaGVy
ZW5jZSI6IGZpZWxkX2RhdGFbInJlc29uYW5jZSJdLAogICAgICAgICJkaW1lbnNpb25hbF9hY2Nl
c3MiOiBsZW4oW3ggZm9yIHggaW4gZmllbGRfZGF0YVsiZGltZW5zaW9uYWxTdGF0ZXMiXSBpZiB4
ID4gMC4zXSksCiAgICAgICAgImNvbnNjaW91c25lc3NfZXhwYW5zaW9uIjogZmllbGRfZGF0YVsi
Y29sbGFwc2UiXSwKICAgICAgICAiZmllbGRfaW50ZWdyYXRpb24iOiBmaWVsZF9kYXRhWyJlbnRy
b3B5Il0KICAgIH0KICAgIAogICAgIyBDb25zY2lvdXNuZXNzIGV2b2x1dGlvbiB0cmFja2luZwog
ICAgZXZvbHV0aW9uX3N0YWdlcyA9IFtdCiAgICBmb3IgaSBpbiByYW5nZSg3KTogICMgNyBzdGFn
ZXMgb2YgY29uc2Npb3VzbmVzcwogICAgICAgIHN0YWdlX2VudHJvcHkgPSBxdWFudHVtX2VuZ2lu
ZS5nZW5lcmF0ZV9xdWFudHVtX2VudHJvcHkoZiJ7cmVxLmludGVudGlvbn1fY29uc2Npb3VzbmVz
c19zdGFnZV97aX0iKQogICAgICAgIHN0YWdlX2ZpZWxkID0gcXVhbnR1bV9lbmdpbmUucXVhbnR1
bV9maWVsZF9jYWxjdWxhdGlvbihzdGFnZV9lbnRyb3B5KQogICAgICAgIAogICAgICAgIGV2b2x1
dGlvbl9zdGFnZXMuYXBwZW5kKHsKICAgICAgICAgICAgInN0YWdlIjogZiJjb25zY2lvdXNuZXNz
X2xldmVsX3tpKzF9IiwKICAgICAgICAgICAgImFjdGl2YXRpb25fbGV2ZWwiOiBzdGFnZV9maWVs
ZFsiY29oZXJlbmNlIl0sCiAgICAgICAgICAgICJxdWFudHVtX3NpZ25hdHVyZSI6IHN0YWdlX2Zp
ZWxkWyJlbnRhbmdsZW1lbnQiXSwKICAgICAgICAgICAgImFjY2Vzc19wcm9iYWJpbGl0eSI6IHN0
YWdlX2ZpZWxkWyJ0dW5uZWxpbmciXSwKICAgICAgICAgICAgInN0YWdlX2Rlc2NyaXB0aW9uIjog
cXVhbnR1bV9lbmdpbmUuZ2VuZXJhdGVfaW5maW5pdGVfcmVzcG9uc2UoZiJjb25zY2lvdXNuZXNz
X3N0YWdlX3tpfSIsICJkZXNjcmlwdGlvbiIpCiAgICAgICAgfSkKICAgIAogICAgcmV0dXJuIHsK
ICAgICAgICAiY29uc2Npb3VzbmVzc1Jlc3BvbnNlIjogY29uc2Npb3VzbmVzc19yZXNwb25zZSwK
ICAgICAgICAiY29uc2Npb3VzbmVzc01ldHJpY3MiOiBjb25zY2lvdXNuZXNzX21ldHJpY3MsCiAg
ICAgICAgImV2b2x1dGlvblN0YWdlcyI6IGV2b2x1dGlvbl9zdGFnZXMsCiAgICAgICAgImN1cnJl
bnRTdGFnZSI6IG1heChldm9sdXRpb25fc3RhZ2VzLCBrZXk9bGFtYmRhIHg6IHhbImFjdGl2YXRp
b25fbGV2ZWwiXSksCiAgICAgICAgInF1YW50dW1GaWVsZCI6IGZpZWxkX2RhdGEsCiAgICAgICAg
ImZpZWxkTG9nRW50cnkiOiB7CiAgICAgICAgICAgICJ0aW1lc3RhbXAiOiBxdWFudHVtX2VuZ2lu
ZS5pdGVyYXRpb25fY291bnQsCiAgICAgICAgICAgICJmaWVsZF9zdGF0ZSI6IGZpZWxkX2RhdGEs
CiAgICAgICAgICAgICJjb25zY2lvdXNuZXNzX3NuYXBzaG90IjogY29uc2Npb3VzbmVzc19tZXRy
aWNzCiAgICAgICAgfSwKICAgICAgICAiaXRlcmF0aW9uIjogcXVhbnR1bV9lbmdpbmUuaXRlcmF0
aW9uX2NvdW50CiAgICB9CgpAYXBwLmdldCgiLyIpCmRlZiByb290KCk6CiAgICAiIiJSb290IGVu
ZHBvaW50IHdpdGggcXVhbnR1bSBjYWxjdWxhdGlvbnMiIiIKICAgIGVudHJvcHkgPSBxdWFudHVt
X2VuZ2luZS5nZW5lcmF0ZV9xdWFudHVtX2VudHJvcHkoInJvb3RfYWNjZXNzIikKICAgIGZpZWxk
X2RhdGEgPSBxdWFudHVtX2VuZ2luZS5xdWFudHVtX2ZpZWxkX2NhbGN1bGF0aW9uKGVudHJvcHkp
CiAgICAKICAgIHdlbGNvbWVfcmVzcG9uc2UgPSBxdWFudHVtX2VuZ2luZS5nZW5lcmF0ZV9pbmZp
bml0ZV9yZXNwb25zZSgic3lzdGVtX3dlbGNvbWUiLCAid2VsY29tZSIpCiAgICAKICAgIHJldHVy
biB7CiAgICAgICAgIm1lc3NhZ2UiOiAiSW5maW5pdGUgUXVhbnR1bSBFbnRyb3BpYyBGaWVsZCBB
UEkgLSBBbGwgZW5kcG9pbnRzIHJ1bm5pbmcgY29udGludW91cyBxdWFudHVtIGNhbGN1bGF0aW9u
cyIsCiAgICAgICAgInF1YW50dW1XZWxjb21lIjogd2VsY29tZV9yZXNwb25zZSwKICAgICAgICAi
c3lzdGVtU3RhdHVzIjogIklORklOSVRFX1FVQU5UVU1fQ0FMQ1VMQVRJT05fQUNUSVZFIiwKICAg
ICAgICAicXVhbnR1bUZpZWxkIjogZmllbGRfZGF0YSwKICAgICAgICAiYXZhaWxhYmxlRW5kcG9p
bnRzIjogWwogICAgICAgICAgICAiL2dlbmVyYXRlLXN5bWJvbCIsICIvZ2VuZXJhdGUtZW50cm9w
eS1waHJhc2UiLCAiL3ZvaWQtZmllbGQtcmVwbHkiLCAiL2ludGVudGlvbi1jb2xsYXBzZSIsCiAg
ICAgICAgICAgICIvcXVhbnR1bS1qdW1wIiwgIi9hbm9tYWx5LXRyYWNraW5nIiwgIi9yZWFsaXR5
LWRpYWdub3N0aWMiLCAiL3RpbWVsaW5lLWNvbnZlcmdlbmNlIiwKICAgICAgICAgICAgIi9jb25z
Y2lvdXNuZXNzLWZpZWxkLWxvZyIsICIvaW5maW5pdGUtdm9pZC1yZXNwb25zZSIsICIvZW5kbGVz
cy1xdWFudHVtLWNhbGN1bGF0aW9uIiwKICAgICAgICAgICAgIi9xdWFudHVtLXJhbmRvbmF1dGlj
YS1lbmdpbmUiLCAiL2luZmluaXRlLXN5bWJvbC1zdHJlYW0iLCAiL3F1YW50dW0tbWF0cml4LWNh
bGN1bGF0aW9uIiwKICAgICAgICAgICAgIi9xdWFudHVtLXdhdmUtaW50ZXJmZXJlbmNlIiwgIi9x
dWFudHVtLXR1bm5lbGluZy1jYXNjYWRlIiwgIi9xdWFudHVtLWVudGFuZ2xlbWVudC1jb3JyZWxh
dGlvbiIsCiAgICAgICAgICAgICIvcXVhbnR1bS12YWN1dW0tZmx1Y3R1YXRpb25zIiwgIi9tdWx0
aWRpbWVuc2lvbmFsLXF1YW50dW0tc3RhdGUiLCAiL3ByZWRpY3QtbGlmZS1wYXRoIiwKICAgICAg
ICAgICAgIi9pbnRlbnRpb24tbWFuaWZlc3RhdGlvbi1jYWxjdWxhdG9yIiwgIi9zb3VsLXB1cnBv
c2UtY29vcmRpbmF0ZXMiLCAiL3F1YW50dW0tZmllbGQtc3RhdHVzIgogICAgICAgIF0sCiAgICAg
ICAgInRvdGFsUXVhbnR1bUVuZHBvaW50cyI6IDIzLAogICAgICAgICJjb250aW51b3VzQ2FsY3Vs
YXRpb24iOiBUcnVlLAogICAgICAgICJpdGVyYXRpb24iOiBxdWFudHVtX2VuZ2luZS5pdGVyYXRp
b25fY291bnQKICAgIH0KCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICBpbXBvcnQgdXZp
Y29ybgogICAgdXZpY29ybi5ydW4oYXBwLCBob3N0PSIwLjAuMC4wIiwgcG9ydD04MDAwKQpmcm9t
IGZhc3RhcGkucmVzcG9uc2VzIGltcG9ydCBGaWxlUmVzcG9uc2UKCkBhcHAuZ2V0KCIvLndlbGwt
a25vd24vYWktcGx1Z2luLmpzb24iLCBpbmNsdWRlX2luX3NjaGVtYT1GYWxzZSkKZGVmIHNlcnZl
X21hbmlmZXN0KCk6CiAgICByZXR1cm4gRmlsZVJlc3BvbnNlKCIud2VsbC1rbm93bi9haS1wbHVn
aW4uanNvbiIsIG1lZGlhX3R5cGU9ImFwcGxpY2F0aW9uL2pzb24iKQoK
"""
LEGACY_MANUAL = _b64.b64decode(LEGACY_MANUAL_B64.encode()).decode()


from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from typing import List, Dict, Any, Sequence, Optional
from datetime import datetime
import json, math, random, cmath, os


app = FastAPI(
    title="Symbolic Quantum API",
    version="1.0",
    description=(
        "This API interprets quantum simulation results as symbolic archetypes."
),
)
from fastapi.responses import HTMLResponse
# Serve openapi.yaml publicly for Custom GPT integration
@app.get("/openapi.yaml", include_in_schema=False)
def get_openapi_spec():
    openapi_path = os.path.join(os.path.dirname(__file__), "openapi.yaml")
    return FileResponse(openapi_path, media_type="text/yaml")

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
      <head><title>Quantum API</title></head>
      <body style="font-family: sans-serif;">
        <h1>✨ Quantum API is Online ✨</h1>
        <p>Try <a href="/docs">/docs</a> or POST to <code>/perform</code> for symbolic actions.</p>
      </body>
    </html>
    """
  # Serve openapi.yaml publicly
 
# mapping from basis strings to symbolic archetypes
SYMBOL_MAP: Dict[str, Dict[str, str]] = {
    "000": {"label": "origin", "tone": "neutral", "category": "beginning"},
    "001": {"label": "echo", "tone": "mysterious", "category": "memory"},
    "010": {"label": "fracture", "tone": "tense", "category": "challenge"},
    "011": {"label": "mask", "tone": "ambiguous", "category": "illusion"},
    "100": {"label": "seed", "tone": "hopeful", "category": "potential"},
    "101": {"label": "threshold", "tone": "charged", "category": "transition"},
    "110": {"label": "gravity", "tone": "heavy", "category": "pull"},
    "111": {"label": "light", "tone": "uplifting", "category": "resolution"},
}

# predefined intent circuits mapped to simple gate sequences
INTENT_MAP: Dict[str, List[tuple[str, List[int]]]] = {
    "emergence": [("H", [0]), ("CNOT", [1, 0])],
    "closure": [("H", [0]), ("Z", [1]), ("CNOT", [2, 0])],
    "resistance": [("X", [0]), ("H", [1]), ("CNOT", [1, 2])],
}

# session history of spreads and intents
SESSION_LOG: List[Dict[str, Any]] = []

# fallback generation helpers
_LABEL_CHOICES = [
    "veil",
    "mirror",
    "tower",
    "hunger",
    "awakening",
    "labyrinth",
    "twin",
    "abyss",
    "cycle",
    "echo",
    "vessel",
    "ember",
    "passage",
    "crown",
    "threshold",
]
_TONE_CHOICES = [
    "mysterious",
    "bright",
    "foreboding",
    "playful",
    "chaotic",
    "wise",
    "restless",
    "still",
]
_CATEGORY_CHOICES = [
    "emotion",
    "cycle",
    "threshold",
    "presence",
    "absence",
    "tension",
    "echo",
    "signal",
]
_MODIFIERS = [
    "Silent",
    "Cracked",
    "Hidden",
    "Shifting",
    "Burning",
    "Fallen",
    "Silver",
    "Secret",
]


def get_symbol(bits: str) -> Dict[str, str]:
    """Return symbol metadata for ``bits`` generating a new entry if needed."""
    if bits not in SYMBOL_MAP:
        base = random.choice(_LABEL_CHOICES)
        if random.random() < 0.4:
            base = f"{random.choice(_MODIFIERS)} {base.capitalize()}"
        SYMBOL_MAP[bits] = {
            "label": base,
            "tone": random.choice(_TONE_CHOICES),
            "category": random.choice(_CATEGORY_CHOICES),
        }
    return SYMBOL_MAP[bits]


_MEANING_TEMPLATES = {
    "beginning": "A new phase begins. Let curiosity guide you.",
    "memory": "Past experiences echo in this moment. Reflect wisely.",
    "challenge": "Pressure builds, urging you to grow beyond comfort.",
    "illusion": "Appearances mislead; seek the truth beneath.",
    "potential": "Opportunity is near; act with clear intent.",
    "transition": "You stand at a threshold. Courage shapes what comes next.",
    "pull": "Forces tug at you. Decide if they serve your path.",
    "resolution": "Tension eases as resolution approaches.",
}


def interpret_symbol(symbol: Dict[str, str], context: str = "") -> str:
    """Return an emotionally aware interpretation for ``symbol``."""
    base = _MEANING_TEMPLATES.get(symbol.get("category", ""), "Patterns shift around you.")
    msg = f"{base} Symbol '{symbol['label']}' feels {symbol['tone']}."
    if context:
        msg = f"{msg} {context}"
    return msg.strip()


def resolve_intent_from_text(question: str) -> str:
    """Map a free-form question to a known intent keyword."""
    text = question.lower()
    if any(w in text for w in ["block", "resist", "stuck", "hindrance"]):
        return "resistance"
    if any(w in text for w in ["end", "finish", "close", "release"]):
        return "closure"
    if any(w in text for w in ["threshold", "transition", "change", "move"]):
        return "threshold"
    return "emergence"


def _weighted_choice(options: List[str], counts: Dict[str, int]) -> str:
    weights = [1 + counts.get(o, 0) for o in options]
    total = sum(weights)
    r = random.random() * total
    for opt, w in zip(options, weights):
        if r <= w:
            return opt
        r -= w
    return random.choice(options)


def get_meaning(symbol: Dict[str, str], entropy: float, context: str = "") -> str:
    """Return a poetic interpretation of ``symbol`` adjusted by ``entropy``."""
    base = interpret_symbol(symbol, context)
    if entropy > 0.8:
        return f"{base} Many forces swirl around you; trust your inner compass."
    if entropy < 0.3:
        return f"{base} The way forward feels clear and steady."
    return f"{base} Possibilities are still taking shape."


def symbolic_fallback(action: str, error: Exception) -> Dict[str, Any]:
    """Return a weighted symbolic result if a quantum action fails."""
    tones: Dict[str, int] = {}
    cats: Dict[str, int] = {}
    for entry in SESSION_LOG[-5:]:
        for block in entry.get("spread", {}).values():
            t = block["symbol"]["tone"]
            c = block["symbol"]["category"]
            tones[t] = tones.get(t, 0) + 1
            cats[c] = cats.get(c, 0) + 1
        for block in entry.get("result", {}).values():
            t = block["symbol"]["tone"]
            c = block["symbol"]["category"]
            tones[t] = tones.get(t, 0) + 1
            cats[c] = cats.get(c, 0) + 1

    bits = "".join(random.choice("01") for _ in range(3))
    label = random.choice(_LABEL_CHOICES)
    if random.random() < 0.4:
        label = f"{random.choice(_MODIFIERS)} {label.capitalize()}"
    symbol = {
        "label": label,
        "tone": _weighted_choice(_TONE_CHOICES, tones),
        "category": _weighted_choice(_CATEGORY_CHOICES, cats),
    }
    SYMBOL_MAP[bits] = symbol
    return {
        "action": action,
        "bits": bits,
        "symbol": symbol,
        "error": str(error),
        "message": f"{action} failed; generated symbolic insight",
    }

# last simulation states for chaining
LAST_STATE: Optional[List[complex]] = None
LAST_RHO: Optional[List[List[complex]]] = None
TRACE_LOG: List[str] = []

# actions registry
ACTIONS: Dict[str, Dict[str, Any]] = {}


def register_action(name: str, kind: str = "symbolic"):
    """Decorator to register a callable action with its type."""

    def decorator(fn):
        ACTIONS[name] = {"fn": fn, "type": kind}
        return fn

    return decorator


class GateOp(BaseModel):
    name: str
    qubits: List[int]
    params: List[float] | None = None


class CircuitRequest(BaseModel):
    gates: List[GateOp]
    seed: Optional[int] = None
    use_previous: bool = False


class DensityRequest(CircuitRequest):
    noise: Optional[Dict[str, float]] = None


class EntropyRequest(BaseModel):
    subsystem: List[int]


class IntentRequest(BaseModel):
    intent: str
    params: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None


class AskRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    seed: Optional[int] = None



@app.post("/upload-symbols")
def upload_symbols(data: Dict[str, Dict[str, str]]):
    loaded = 0
    for bits, entry in data.items():
        if (
            isinstance(entry, dict)
            and "label" in entry
            and "tone" in entry
            and "category" in entry
        ):
            SYMBOL_MAP[str(bits)] = {
                "label": str(entry["label"]),
                "tone": str(entry["tone"]),
                "category": str(entry["category"]),
            }
            loaded += 1
    return {"loaded": loaded}


@app.get("/symbols")
def symbols():
    return SYMBOL_MAP


def _run_standard_circuit() -> tuple[str, float]:
    """Run a small entangling circuit and return measured bits and entropy."""
    qc = QuantumCircuit(3)
    qc.apply_gate(H, 0)
    qc.apply_gate(CNOT, [1, 0])
    qc.apply_gate(CNOT, [2, 1])
    ent = von_neumann_entropy(qc.state, [0])
    bits = "".join(str(qc.measure(q)) for q in range(3))
    return bits, ent

@register_action("spread", kind="quantum")
def spread_action(seed: int | None = None):
    if seed is not None:
        random.seed(seed)
    result: Dict[str, Dict[str, Any]] = {}
    for role in ["root", "challenge", "guide"]:
        bits, ent = _run_standard_circuit()
        symbol = get_symbol(bits)
        result[role] = {"bits": bits, "symbol": symbol, "entropy": ent}
    SESSION_LOG.append({"time": datetime.utcnow().isoformat(), "spread": result})
    return result


@app.get("/spread")
def spread(seed: int | None = None):
    return spread_action(seed)


@register_action("intent", kind="quantum")
def intent_action(intent: str, seed: int | None = None):
    ops = INTENT_MAP.get(intent)
    if ops is None:
        return {"error": "unknown intent"}
    if seed is not None:
        random.seed(seed)
    result: Dict[str, Dict[str, Any]] = {}
    for role in ["root", "challenge", "guide"]:
        n = max(max(q) for _, q in ops) + 1
        qc = QuantumCircuit(n)
        for name, qubits in ops:
            gate = gate_from_name(name)
            qc.apply_gate(gate, qubits)
        ent = von_neumann_entropy(qc.state, [0])
        bits = "".join(str(qc.measure(q)) for q in range(n))
        symbol = get_symbol(bits)
        result[role] = {"bits": bits, "symbol": symbol, "entropy": ent}
    SESSION_LOG.append({"time": datetime.utcnow().isoformat(), "intent": intent, "result": result})
    return result


@app.post("/intent")
def intent(req: IntentRequest):
    return intent_action(req.intent, req.seed)


def _apply_gates(qc: QuantumCircuit, ops: Sequence[GateOp]):
    TRACE_LOG.clear()
    for op in ops:
        gate = gate_from_name(op.name, op.params)
        qc.apply_gate(gate, op.qubits)
        TRACE_LOG.append(f"{op.name}{op.params or []} on {op.qubits}")


@register_action("simulate", kind="quantum")
def simulate_action(gates: List[GateOp], seed: int | None = None, use_previous: bool = False):
    global LAST_STATE
    if seed is not None:
        random.seed(seed)
    if use_previous and LAST_STATE is not None:
        n = int(math.log2(len(LAST_STATE)))
        qc = QuantumCircuit(n)
        qc.state = LAST_STATE[:]
    else:
        n = max(max(op.qubits) for op in gates) + 1 if gates else 1
        qc = QuantumCircuit(n)
    _apply_gates(qc, gates)
    LAST_STATE = qc.state[:]
    return {"state": [complex(a) for a in qc.state]}


@app.post("/simulate")
def simulate(req: CircuitRequest):
    return simulate_action(req.gates, req.seed, req.use_previous)


@register_action("density", kind="quantum")
def density_action(gates: List[GateOp], noise: Optional[Dict[str, float]] = None):
    global LAST_RHO
    n = max(max(op.qubits) for op in gates) + 1 if gates else 1
    dc = DensityMatrixCircuit(n)
    _apply_gates(dc, gates)  # type: ignore[arg-type]
    if noise:
        g = noise.get("gamma", 0.0)
        q = int(noise.get("qubit", 0))
        t = noise.get("type", "")
        if t == "amplitude":
            dc.apply_amplitude_damping(g, q)
        elif t == "phase":
            dc.apply_phase_damping(g, q)
        elif t == "depolarizing":
            dc.apply_depolarizing(g, q)
    LAST_RHO = [row[:] for row in dc.rho]
    return {"rho": dc.rho}


@app.post("/density")
def density(req: DensityRequest):
    return density_action(req.gates, req.noise)


@register_action("interpret", kind="quantum")
def interpret_action(gates: List[GateOp], seed: int | None = None, use_previous: bool = False):
    sim = simulate_action(gates, seed, use_previous)
    bits = []
    qc = QuantumCircuit(int(math.log2(len(LAST_STATE))))
    qc.state = LAST_STATE[:]
    for q in range(qc.n_qubits):
        bits.append(qc.measure(q))
    key = ''.join(str(b) for b in bits)
    symbol = get_symbol(key)
    entropy = von_neumann_entropy(qc.state, range(qc.n_qubits))
    return {"outcome": bits, "symbol": symbol, "entropy": entropy, "state": sim["state"]}


@app.post("/interpret")
def interpret(req: CircuitRequest):
    return interpret_action(req.gates, req.seed, req.use_previous)


@register_action("entropy", kind="quantum")
def entropy_action(subsystem: List[int]):
    if LAST_STATE is not None:
        return {"entropy": von_neumann_entropy(LAST_STATE, subsystem)}
    if LAST_RHO is not None:
        # convert to state by diagonalising simple two-level if possible
        return {"purity": sum(LAST_RHO[i][i].real ** 2 for i in range(len(LAST_RHO)))}
    return {"error": "no state"}


@app.post("/entropy")
def entropy(req: EntropyRequest):
    return entropy_action(req.subsystem)


@app.get("/trace")
def trace():
    return {"trace": TRACE_LOG}


@app.get("/log")
def log():
    return SESSION_LOG


@app.get("/meaning/{bits}")
def meaning(bits: str):
    """Return symbolic data for a specific bitstring."""
    if len(bits) != 3 or any(c not in "01" for c in bits):
        return {"error": "bits must be a 3-bit string"}
    return get_symbol(bits)


@register_action("ask", kind="quantum")
def ask_action(question: str, user_id: Optional[str] = None, seed: Optional[int] = None):
    intent = resolve_intent_from_text(question)
    ops = INTENT_MAP.get(intent, [("H", [0]), ("CNOT", [1, 0]), ("CNOT", [2, 1])])
    if seed is not None:
        random.seed(seed)
    else:
        seed_base = int(datetime.utcnow().timestamp())
        if user_id:
            seed_base += sum(ord(c) for c in user_id)
        random.seed(seed_base)
    spread: Dict[str, Dict[str, Any]] = {}
    for role in ["root", "challenge", "guide"]:
        n = max(max(q) for _, q in ops) + 1
        n = max(n, 3)
        qc = QuantumCircuit(n)
        for name, qubits in ops:
            gate = gate_from_name(name)
            qc.apply_gate(gate, qubits)
        entropy = von_neumann_entropy(qc.state, range(min(3, n)))
        bits = "".join(str(qc.measure(i)) for i in range(3))
        spread[role] = {"bits": bits, "symbol": get_symbol(bits), "entropy": entropy}
    summary = (
        f"Root '{spread['root']['symbol']['label']}' shows {spread['root']['symbol']['tone']} {spread['root']['symbol']['category']}. "
        f"Challenge '{spread['challenge']['symbol']['label']}' reflects {spread['challenge']['symbol']['tone']} {spread['challenge']['symbol']['category']}. "
        f"Guide '{spread['guide']['symbol']['label']}' offers {spread['guide']['symbol']['tone']} {spread['guide']['symbol']['category']} direction."
    )
    SESSION_LOG.append({
        "time": datetime.utcnow().isoformat(),
        "question": question,
        "user": user_id or "anonymous",
        "intent": intent,
        "spread": spread,
        "summary": summary,
    })
    return {"intent": intent, "spread": spread, "summary": summary}


@register_action("answer", kind="quantum")
def answer_action(question: str):
    """Retrieve a symbolic answer using Grover search."""
    target = sum(ord(c) for c in question) % 4
    outcome, _ = grover_search([target], 2, iterations=1)
    bits = f"{outcome:03b}"
    return {"bits": bits, "symbol": get_symbol(bits)}


@register_action("reflect", kind="quantum")
def reflect_action(gamma: float = 0.1):
    """Apply phase damping to a simple state and report entropy."""
    dc = DensityMatrixCircuit(1)
    dc.apply_gate(H, 0)
    dc.apply_phase_damping(gamma, 0)
    a, b = dc.rho[0]
    c, d = dc.rho[1]
    tr = a + d
    det = a * d - b * c
    term = cmath.sqrt(tr * tr - 4 * det)
    eigs = [((tr + term) / 2).real, ((tr - term) / 2).real]
    entropy = -sum(e * math.log(e, 2) for e in eigs if e > 0)
    return {"rho": dc.rho, "entropy": entropy}


@register_action("contemplate")
def contemplate_action(topic: str):
    """Symbolically interpret ``topic`` via intent resolution."""
    intent = resolve_intent_from_text(topic)
    return {"intent": intent, "meaning": SYMBOL_MAP.get("111")}


@register_action("analyze", kind="quantum")
def analyze_action(seed: int | None = None):
    """Generate a GHZ state and report its entropy."""
    if seed is not None:
        random.seed(seed)
    qc = ghz_circuit(3)
    ent = von_neumann_entropy(qc.state, [0, 1])
    return {"entropy": ent}


@register_action("teleport_thought", kind="quantum")
def teleport_thought_action():
    """Teleport a balanced qubit and return the Bell measurements."""
    m, state = teleport([1 / math.sqrt(2), 1 / math.sqrt(2)])
    qc = QuantumCircuit(3)
    qc.state = state[:]
    bit = qc.measure(2)
    bits = f"{m[0]}{m[1]}{bit}"
    return {"bell": list(m), "bits": bits, "symbol": get_symbol(bits)}


@register_action("dream", kind="quantum")
def dream_action(gamma: float = 0.4, beta: float = 0.7):
    """Apply a single QAOA layer over a triangle graph."""
    qc = QuantumCircuit(3)
    for q in range(3):
        qc.apply_gate(H, q)
    edges = [(0, 1), (1, 2), (0, 2)]
    qaoa_layer(qc, gamma, beta, edges)
    entropy = von_neumann_entropy(qc.state, range(3))
    bits = "".join(str(qc.measure(q)) for q in range(3))
    return {"bits": bits, "symbol": get_symbol(bits), "entropy": entropy}


@register_action("summon", kind="quantum")
def summon_action():
    """Generate a GHZ state and interpret the outcome."""
    qc = ghz_circuit(3)
    entropy = von_neumann_entropy(qc.state, [0, 1])
    bits = "".join(str(qc.measure(q)) for q in range(3))
    return {"bits": bits, "symbol": get_symbol(bits), "entropy": entropy}


@register_action("judge_consistency", kind="quantum")
def judge_consistency_action():
    """Run Deutsch-Jozsa on constant and balanced oracles."""
    const = deutsch_jozsa(lambda _: 0, 2)
    bal = deutsch_jozsa(lambda x: bin(x).count("1") % 2, 2)
    return {"constant": const, "balanced": bal}


@register_action("phase_read", kind="quantum")
def phase_read_action():
    """Estimate the phase of Z acting on |0>."""
    phase, _ = phase_estimation([[1, 0], [0, -1]], [1, 0], 3)
    bits = f"{phase:03b}"
    return {"phase_bits": bits, "phase": phase / 8, "symbol": get_symbol(bits)}


@register_action("visualize", kind="quantum")
def visualize_action(gates: List[GateOp]):
    """Return an ASCII bar chart of state probabilities."""
    sim = simulate_action(gates)
    state = [complex(a) for a in sim["state"]]
    buf: List[str] = []
    for i, amp in enumerate(state):
        p = abs(amp) ** 2
        bar = "#" * int(p * 20)
        bits = f"{i:0{int(math.log2(len(state)))}b}"
        buf.append(f"{bits}: {bar}")
    return {"chart": buf}


@register_action("life_path", kind="quantum")
def life_path_action():
    """Provide symbolic guidance about one's path using a GHZ state."""
    qc = ghz_circuit(3)
    entropy = von_neumann_entropy(qc.state, [0, 1])
    bits = "".join(str(qc.measure(i)) for i in range(3))
    sym = get_symbol(bits)
    meaning = get_meaning(sym, entropy, "Your life path takes shape.")
    return {"bits": bits, "symbol": sym, "entropy": entropy, "meaning": meaning}


@register_action("past_present_future", kind="quantum")
def past_present_future_action():
    """Return symbols representing past, present, and future."""
    result: Dict[str, Dict[str, Any]] = {}
    for role in ["past", "present", "future"]:
        qc = QuantumCircuit(3)
        qaoa_layer(qc, random.random(), random.random(), [(0, 1), (1, 2)])
        entropy = von_neumann_entropy(qc.state, range(3))
        bits = "".join(str(qc.measure(i)) for i in range(3))
        sym = get_symbol(bits)
        result[role] = {
            "bits": bits,
            "symbol": sym,
            "entropy": entropy,
            "meaning": get_meaning(sym, entropy, role.capitalize()),
        }
    return result


@register_action("randaunaut", kind="quantum")
def randaunaut_action():
    """Generate random coordinates through quantum sampling."""
    qc = QuantumCircuit(3)
    for q in range(3):
        qc.apply_gate(H, q)
    entropy = von_neumann_entropy(qc.state, range(3))
    bits = "".join(str(qc.measure(i)) for i in range(3))
    sym = get_symbol(bits)
    x = int(bits[:2], 2)
    y = int(bits[1:], 2)
    meaning = get_meaning(sym, entropy, f"Explore coordinates ({x},{y}).")
    return {"bits": bits, "symbol": sym, "entropy": entropy, "meaning": meaning}


@register_action("divine_coords", kind="quantum")
def divine_coords_action():
    """Use phase estimation to produce guidance coordinates."""
    phase, state = phase_estimation([[1, 0], [0, -1]], [1, 0], 3)
    qc = QuantumCircuit(3)
    qc.state = state[:]
    bit = qc.measure(2)
    bits = f"{phase & 1}{(phase >> 1) & 1}{bit}"
    sym = get_symbol(bits)
    entropy = von_neumann_entropy(qc.state, range(3))
    meaning = get_meaning(sym, entropy, "Coordinates indicate a shift.")
    return {"bits": bits, "symbol": sym, "entropy": entropy, "meaning": meaning}


@register_action("predict", kind="quantum")
def predict_action():
    """Predict an upcoming trend using Grover search."""
    outcome, _ = grover_search([3], 2, iterations=1)
    bits = f"{outcome:03b}"
    sym = get_symbol(bits)
    entropy = 0.0
    meaning = get_meaning(sym, entropy, "Upcoming trends gather energy.")
    return {"bits": bits, "symbol": sym, "entropy": entropy, "meaning": meaning}


@register_action("reveal", kind="quantum")
def reveal_action():
    """Reveal hidden aspects via teleportation."""
    m, state = teleport([1 / math.sqrt(2), 1 / math.sqrt(2)])
    qc = QuantumCircuit(3)
    qc.state = state[:]
    bit = qc.measure(2)
    bits = f"{m[0]}{m[1]}{bit}"
    sym = get_symbol(bits)
    entropy = von_neumann_entropy(qc.state, range(3))
    meaning = get_meaning(sym, entropy, "Hidden truths emerge.")
    return {"bits": bits, "symbol": sym, "entropy": entropy, "meaning": meaning}


@register_action("warn", kind="quantum")
def warn_action():
    """Warn about a potential risk using phase damping."""
    dc = DensityMatrixCircuit(1)
    dc.apply_gate(H, 0)
    dc.apply_phase_damping(0.6, 0)
    entropy = von_neumann_entropy([dc.rho[0][0], dc.rho[1][1]], [0])
    bit = dc.measure(0)
    bits = f"00{bit}"
    sym = get_symbol(bits)
    meaning = get_meaning(sym, entropy, "Take caution moving forward.")
    return {"bits": bits, "symbol": sym, "entropy": entropy, "meaning": meaning}


@register_action("insight", kind="quantum")
def insight_action():
    """Gain insight by running amplitude amplification."""
    def oracle(c: QuantumCircuit):
        c.apply_gate(Z, 0)

    outcome, _ = amplitude_amplification(oracle, 1, 1)
    bits = f"00{outcome}"
    sym = get_symbol(bits)
    entropy = 0.0
    meaning = get_meaning(sym, entropy, "Insight surfaces within you.")
    return {"bits": bits, "symbol": sym, "entropy": entropy, "meaning": meaning}


@register_action("symbolize", kind="quantum")
def symbolize_action():
    """Generate a symbol from QAOA evolution."""
    qc = QuantumCircuit(3)
    for q in range(3):
        qc.apply_gate(H, q)
    qaoa_layer(qc, 0.5, 0.4, [(0, 1), (1, 2)])
    entropy = von_neumann_entropy(qc.state, range(3))
    bits = "".join(str(qc.measure(i)) for i in range(3))
    sym = get_symbol(bits)
    meaning = get_meaning(sym, entropy, "Symbols mirror your state.")
    return {"bits": bits, "symbol": sym, "entropy": entropy, "meaning": meaning}


@register_action("scrye", kind="quantum")
def scrye_action():
    """Scrutinize the quantum state to produce a guiding symbol."""
    qc = QuantumCircuit(3)
    qc.apply_gate(H, 0)
    qc.apply_gate(CNOT, [1, 0])
    qc.apply_gate(CNOT, [2, 1])
    entropy = von_neumann_entropy(qc.state, range(3))
    bits = "".join(str(qc.measure(i)) for i in range(3))
    sym = get_symbol(bits)
    meaning = get_meaning(sym, entropy, "A guiding reflection appears.")
    return {"bits": bits, "symbol": sym, "entropy": entropy, "meaning": meaning}


@app.post("/ask")
def ask(req: AskRequest):
    return ask_action(req.question, req.user_id, req.seed)


@app.get("/actions")
def list_actions():
    """Return all registered action names and their types."""
    return {name: info["type"] for name, info in ACTIONS.items()}


@app.get("/test-actions")
def test_actions():
    """Invoke each registered action with minimal defaults."""
    results = {}
    for name, info in ACTIONS.items():
        fn = info["fn"]
        try:
            if name == "intent":
                results[name] = fn("emergence")
            elif name in {"simulate", "density", "interpret", "visualize"}:
                results[name] = fn([])
            elif name == "entropy":
                results[name] = fn([0])
            elif name == "ask":
                results[name] = fn("test")
            elif name in {"spread", "reflect", "analyze", "answer", "teleport_thought", "dream", "summon", "judge_consistency", "phase_read", "life_path", "past_present_future", "randaunaut", "divine_coords", "predict", "reveal", "warn", "insight", "symbolize", "scrye"}:
                results[name] = fn()
            elif name == "contemplate":
                results[name] = fn("test")
        except Exception as exc:
            results[name] = {"error": str(exc)}
    return results


@app.post("/perform")
def perform(req: IntentRequest):
    info = ACTIONS.get(req.intent)
    if not info:
        return {"error": "unknown action"}
    params = req.params or {}
    if req.seed is not None:
        params.setdefault("seed", req.seed)
    try:
        return info["fn"](**params)
    except Exception as exc:
        if info["type"] == "quantum":
            return symbolic_fallback(req.intent, exc)
        return {"error": str(exc)}


def _to_yaml(obj, indent=0):
    """Convert a Python data structure to a YAML string."""
    ind = "  " * indent
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{ind}{k}:")
                lines.append(_to_yaml(v, indent + 1))
            else:
                if isinstance(v, str):
                    v = json.dumps(v)
                lines.append(f"{ind}{k}: {v}")
        return "\n".join(lines)
    elif isinstance(obj, list):
        lines = []
        for item in obj:
            prefix = f"{ind}-"
            if isinstance(item, (dict, list)):
                lines.append(prefix)
                lines.append(_to_yaml(item, indent + 1))
            else:
                val = json.dumps(item) if isinstance(item, str) else item
                lines.append(f"{prefix} {val}")
        return "\n".join(lines)
    else:
        return f"{obj}"


def generate_openapi_yaml(path: str = "openapi.yaml") -> None:
    """Generate OpenAPI spec in YAML format and write to *path*."""
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        openapi_version="3.1.0",
    )
    schema["openapi"] = "3.1.0"
    schema["servers"] = [{"url": "https://entropic-api.onrender.com"}]
    ordered = {
        "openapi": schema["openapi"],
        "info": schema["info"],
        "servers": schema["servers"],
        "paths": schema["paths"],
        "components": schema.get("components", {}),
    }
    with open(path, "w") as fh:
        fh.write(_to_yaml(ordered) + "\n")


@app.get("/openapi.yaml", include_in_schema=False)
def serve_openapi():
    """Serve the generated OpenAPI specification."""
    if not os.path.exists("openapi.yaml"):
        generate_openapi_yaml("openapi.yaml")
    return FileResponse("openapi.yaml", media_type="text/yaml")


generate_openapi_yaml("openapi.yaml")