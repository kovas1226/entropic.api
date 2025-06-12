from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Sequence, Optional
from datetime import datetime
import json, math, random, cmath

from .quantum_sim import (
    QuantumCircuit,
    DensityMatrixCircuit,
    gate_from_name,
    von_neumann_entropy,
    depolarizing_channel,
    amplitude_damping,
    phase_damping,
    teleport,
    qaoa_layer,
    deutsch_jozsa,
    phase_estimation,
    H,
    CNOT,
    ghz_circuit,
    grover_search,
    visualize_probabilities,
)

app = FastAPI(title="Symbolic Quantum API", version="1.0")
@app.get("/")
def root():
    return {"status": "Quantum API is online."}

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


def _weighted_choice(options: List[str], counts: Dict[str, int]) -> str:
    weights = [1 + counts.get(o, 0) for o in options]
    total = sum(weights)
    r = random.random() * total
    for opt, w in zip(options, weights):
        if r <= w:
            return opt
        r -= w
    return random.choice(options)


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
            elif name in {"spread", "reflect", "analyze", "answer", "teleport_thought", "dream", "summon", "judge_consistency", "phase_read"}:
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
