import hashlib
import math
import random
from typing import Tuple, Optional, Dict

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse

# ---------------------------------------------------------------------------
# Quantum seeding utilities
# ---------------------------------------------------------------------------

def quantum_seed(question: str) -> Tuple[int, float]:
    """Derive a deterministic seed and entropy from *question* using
    Grover-style amplification and a simple QFT phase mix."""

    digest = hashlib.sha256(question.encode()).digest()
    target = digest[0] % 8
    # uniform superposition over three qubits
    amps = [1 / math.sqrt(8) for _ in range(8)]
    # Grover amplify the target state twice
    for _ in range(2):
        amps[target] *= -1  # oracle
        avg = sum(amps) / 8
        amps = [2 * avg - a for a in amps]  # diffusion
    # apply a crude QFT phase rotation
    phases = [complex(math.cos(2 * math.pi * i / 8), math.sin(2 * math.pi * i / 8)) for i in range(8)]
    amps = [a * phases[i] for i, a in enumerate(amps)]
    probs = [abs(a) ** 2 for a in amps]
    # measure deterministically based on digest-derived randomness
    rnd = random.Random(digest)
    r = rnd.random()
    cumulative = 0.0
    outcome = 0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            outcome = i
            break
    bits = f"{outcome:03b}"
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return int(bits, 2), entropy

# ---------------------------------------------------------------------------
# FateOracle implementation
# ---------------------------------------------------------------------------

class FateOracle:
    """Deterministic oracle that crafts responses from quantum-derived seeds."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed or random.getrandbits(16)
        self.entropy = (self.seed % 1000) / 1000
        self.state = ""
        self._rng = random.Random(self.seed)
        self.summary = self._generate_fate_summary()

    def _generate_fate_summary(self) -> str:
        moods = ["steady", "restless", "bright", "dim", "open", "tense"]
        aspects = ["work", "relationships", "creativity", "health", "goals"]
        return f"Things feel {self._rng.choice(moods)} around your {self._rng.choice(aspects)}."

    def _entropy_tone(self) -> str:
        if self.entropy > 0.7:
            return "Many possibilities are unfolding."
        if self.entropy < 0.3:
            return "The direction ahead looks clear."
        return "Some uncertainty remains, yet patterns are forming."

    def generate_prompt(self, question: str) -> str:
        tone = self._entropy_tone()
        keywords = [
            "shift",
            "cycle",
            "mirror",
            "signal",
            "pulse",
            "drift",
            "spark",
            "flow",
        ]
        cue = keywords[self.seed % len(keywords)]
        return (
            "You are an emotionally intelligent, intuitive AI.\n"
            "You interpret fate from quantum-derived entropy.\n"
            f"Fate Summary: {self.summary}\n"
            f"Entropy: {self.entropy:.2f} -> {tone}\n"
            f"Key focus: {cue}\n"
            f"User asked: \"{question}\"\n"
            "Respond like a friend. Be natural, specific, emotionally deep, and non-mystic."
        )

    def simulate_gpt_completion(self, prompt: str) -> str:
        """Simulate the GPT completion that would be generated from *prompt*."""
        return f"[GPT would answer: {prompt[:60]}...]"

    def ask(self, question: str) -> str:
        self.seed, self.entropy = quantum_seed(question)
        self._rng.seed(self.seed)
        self.state = question
        prompt = self.generate_prompt(question)
        return self.simulate_gpt_completion(prompt)

    def predict(self, topic: str) -> str:
        return self.ask(f"What is likely to happen regarding {topic}?")

    def reveal(self, subject: str) -> str:
        return self.ask(f"Reveal hidden dynamics about {subject}.")

    def expand(self, question: str) -> str:
        return self.ask(f"{question} Please elaborate further.")

# ---------------------------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Fate Oracle API", version="1.0")

class OracleRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class OracleResponse(BaseModel):
    answer: str

oracle_cache: Dict[str, FateOracle] = {}

@app.post("/oracle", response_model=OracleResponse)
def oracle_endpoint(req: OracleRequest) -> OracleResponse:
    bits, entropy = quantum_seed(req.question)
    oracle = oracle_cache.setdefault(req.session_id or str(bits), FateOracle(seed=bits))
    answer = oracle.ask(req.question)
    return OracleResponse(answer=answer)

@app.get("/debug_seed", include_in_schema=False)
def debug_seed(q: str) -> Dict[str, float]:
    bits, entropy = quantum_seed(q)
    return {"bits": bits, "entropy": entropy}

@app.get("/openapi.yaml", include_in_schema=False)
def serve_openapi() -> FileResponse:
    return FileResponse("openapi.yaml", media_type="text/yaml")
