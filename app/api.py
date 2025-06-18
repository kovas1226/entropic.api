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
    """Return an integer seed and entropy derived from *question*.

    The function hashes the question to derive deterministic phases which are
    used in a tiny quantum-inspired routine. The routine prepares three
    single-qubit states with angles based on the hash, computes measurement
    probabilities, and samples bits. The Shannon entropy of each qubit's
    probability distribution is averaged to obtain a final entropy value.
    """

    digest = hashlib.sha256(question.encode()).digest()
    rnd = random.Random(digest)
    bits = ""
    entropies = []
    for i in range(3):
        angle = digest[i] / 255 * math.pi
        p1 = math.sin(angle / 2) ** 2
        bit = "1" if rnd.random() < p1 else "0"
        bits += bit
        if 0 < p1 < 1:
            entropies.append(-(p1 * math.log2(p1) + (1 - p1) * math.log2(1 - p1)))
        else:
            entropies.append(0.0)
    entropy = sum(entropies) / len(entropies)
    return int(bits, 2), entropy

# ---------------------------------------------------------------------------
# FateOracle implementation
# ---------------------------------------------------------------------------

class FateOracle:
    """Simple deterministic oracle seeded by an integer."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed or random.getrandbits(16)
        self._rng = random.Random(self.seed)
        self.entropy = (self.seed % 1000) / 1000
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
        return (
            "You are an emotionally intelligent, intuitive AI.\n"
            "You interpret fate from quantum-derived entropy.\n"
            f"Fate Summary: {self.summary}\n"
            f"Entropy: {self.entropy:.2f} -> {tone}\n"
            f"User asked: \"{question}\"\n"
            "Respond like a friend. Be natural, specific, emotionally deep, and non-mystic."
        )

    def call_gpt(self, prompt: str) -> str:
        """Placeholder for real GPT integration."""
        return f"[GPT would answer: {prompt[:60]}...]"

    def ask(self, question: str) -> str:
        prompt = self.generate_prompt(question)
        return self.call_gpt(prompt)

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
