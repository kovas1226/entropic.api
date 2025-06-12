from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Sequence, Optional
from datetime import datetime
import json, math, random

from .quantum_sim import (
    QuantumCircuit,
    DensityMatrixCircuit,
    gate_from_name,
    von_neumann_entropy,
    depolarizing_channel,
    amplitude_damping,
    phase_damping,
    H,
    CNOT,
)

app = FastAPI(title="Symbolic Quantum API", version="1.0")

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

# last simulation states for chaining
LAST_STATE: Optional[List[complex]] = None
LAST_RHO: Optional[List[List[complex]]] = None
TRACE_LOG: List[str] = []


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
    seed: Optional[int] = None


class AskRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    seed: Optional[int] = None


@app.post("/upload-symbols")
def upload_symbols(file: UploadFile = File(...)):
    data = json.load(file.file)
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


@app.get("/spread")
def spread(seed: int | None = None):
    if seed is not None:
        random.seed(seed)
    result: Dict[str, Dict[str, Any]] = {}
    for role in ["root", "challenge", "guide"]:
        bits, ent = _run_standard_circuit()
        symbol = get_symbol(bits)
        result[role] = {"bits": bits, "symbol": symbol, "entropy": ent}
    SESSION_LOG.append({"time": datetime.utcnow().isoformat(), "spread": result})
    return result


@app.post("/intent")
def intent(req: IntentRequest):
    ops = INTENT_MAP.get(req.intent)
    if ops is None:
        return {"error": "unknown intent"}
    if req.seed is not None:
        random.seed(req.seed)
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
    SESSION_LOG.append({"time": datetime.utcnow().isoformat(), "intent": req.intent, "result": result})
    return result


def _apply_gates(qc: QuantumCircuit, ops: Sequence[GateOp]):
    TRACE_LOG.clear()
    for op in ops:
        gate = gate_from_name(op.name, op.params)
        qc.apply_gate(gate, op.qubits)
        TRACE_LOG.append(f"{op.name}{op.params or []} on {op.qubits}")


@app.post("/simulate")
def simulate(req: CircuitRequest):
    global LAST_STATE
    if req.seed is not None:
        random.seed(req.seed)
    if req.use_previous and LAST_STATE is not None:
        n = int(math.log2(len(LAST_STATE)))
        qc = QuantumCircuit(n)
        qc.state = LAST_STATE[:]
    else:
        n = max(max(op.qubits) for op in req.gates) + 1 if req.gates else 1
        qc = QuantumCircuit(n)
    _apply_gates(qc, req.gates)
    LAST_STATE = qc.state[:]
    return {"state": [complex(a) for a in qc.state]}


@app.post("/density")
def density(req: DensityRequest):
    global LAST_RHO
    n = max(max(op.qubits) for op in req.gates) + 1 if req.gates else 1
    dc = DensityMatrixCircuit(n)
    _apply_gates(dc, req.gates)  # type: ignore[arg-type]
    if req.noise:
        g = req.noise.get("gamma", 0.0)
        q = int(req.noise.get("qubit", 0))
        t = req.noise.get("type", "")
        if t == "amplitude":
            dc.apply_amplitude_damping(g, q)
        elif t == "phase":
            dc.apply_phase_damping(g, q)
        elif t == "depolarizing":
            dc.apply_depolarizing(g, q)
    LAST_RHO = [row[:] for row in dc.rho]
    return {"rho": dc.rho}


@app.post("/interpret")
def interpret(req: CircuitRequest):
    sim = simulate(req)
    bits = []
    qc = QuantumCircuit(int(math.log2(len(LAST_STATE))))
    qc.state = LAST_STATE[:]
    for q in range(qc.n_qubits):
        bits.append(qc.measure(q))
    key = ''.join(str(b) for b in bits)
    symbol = get_symbol(key)
    entropy = von_neumann_entropy(qc.state, range(qc.n_qubits))
    return {"outcome": bits, "symbol": symbol, "entropy": entropy, "state": sim["state"]}


@app.post("/entropy")
def entropy(req: EntropyRequest):
    if LAST_STATE is not None:
        return {"entropy": von_neumann_entropy(LAST_STATE, req.subsystem)}
    if LAST_RHO is not None:
        # convert to state by diagonalising simple two-level if possible
        return {"purity": sum(LAST_RHO[i][i].real ** 2 for i in range(len(LAST_RHO)))}
    return {"error": "no state"}


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


@app.post("/ask")
def ask(req: AskRequest):
    intent = resolve_intent_from_text(req.question)
    ops = INTENT_MAP.get(intent, [("H", [0]), ("CNOT", [1, 0]), ("CNOT", [2, 1])])
    if req.seed is not None:
        random.seed(req.seed)
    else:
        seed_base = int(datetime.utcnow().timestamp())
        if req.user_id:
            seed_base += sum(ord(c) for c in req.user_id)
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
        "question": req.question,
        "user": req.user_id or "anonymous",
        "intent": intent,
        "spread": spread,
        "summary": summary,
    })
    return {"intent": intent, "spread": spread, "summary": summary}