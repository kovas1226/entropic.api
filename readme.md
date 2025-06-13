# Entropic Psychic API

This project exposes a symbolic quantum API using FastAPI. It interprets the
measurements of a lightweight quantum simulator as archetypal symbols. The
service can be used by custom GPT models to generate grounded insights.

## Features
- Pure Python quantum simulator (`app/quantum_sim.py`)
- Symbolic interpretation layer with predefined and uploaded symbols
- Endpoints for circuit simulation, symbolic spreads, and intent-based actions
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