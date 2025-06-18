# Entropic Psychic API

This project exposes a symbolic quantum API using FastAPI. Quantum
measurements seed a small "FateOracle" that generates grounded guidance in a
natural, conversational tone.  Custom GPTs interact with the service through a
single endpoint, `/oracle`, which accepts a question and returns a plain text
answer.

## Features
- Pure Python quantum simulator (`app/quantum_sim.py`)
- Symbolic interpretation layer with predefined and uploaded symbols
- Single `/oracle` endpoint for questions
- OpenAPI 3.1 schema available at `/openapi.yaml`
- Minimal dependencies

## Setup
1. Install dependencies:
   ```bash
 pip install -r requirements.txt
  ```
2. Start the API locally:
  ```bash
  uvicorn app.api:app --reload --port 8000
  ```
3. Ask the oracle from the command line:
   ```bash
   python -m app.cli
   ```
   Then type questions interactively.
3. Visit `http://localhost:8000/docs` to explore the API.

## Deployment
The included `render.yaml` file configures the service for Render.com. When
deployed, the API will be available at
`https://entropic-api.onrender.com`.

## Testing
Run the unit tests with `pytest`:
```bash
pytest -q
```
Some tests require `httpx`; if not installed, tests will be skipped.
