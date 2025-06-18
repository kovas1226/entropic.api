import random
import re
from typing import Optional

CITIES = [
    "Chicago",
    "New York",
    "Los Angeles",
    "Austin",
    "Seattle",
    "Denver",
    "Boston",
    "Miami",
]

MOVE_WORDS = ["live", "move", "relocate", "travel", "go", "stay"]


def get_location_from_entropy(bits: str, question: str) -> Optional[str]:
    """Return a city suggestion if the question seems location-related."""
    text = question.lower()
    if not any(word in text for word in MOVE_WORDS):
        return None
    seed = int(bits, 2)
    rng = random.Random(seed)
    return rng.choice(CITIES)
