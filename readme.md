# Entropic Psychic API

This project exposes a symbolic quantum API using FastAPI. It interprets the
measurements of a lightweight quantum simulator as archetypal symbols. The
service can be used by custom GPT models or plugins to generate grounded, eerily specific insights.

## Features
- Pure Python quantum simulator (all in `app/api.py`)
- Symbolic interpretation layer with predefined and user-uploaded symbols
- Human-like `/predict-life` endpoint for endless, conversational life readings
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
3. Visit [http://localhost:8000/docs](http://localhost:8000/docs) to explore the API.

## Key Endpoints

- `/predict-life`  
  Generate a deep, psychic, conversational prediction based on quantum-symbolic context.  
  Accepts detailed questions and circuit/symbol parameters.  
  Designed for use with custom GPTs or as a plug-and-play life advice API.

- `/spread`, `/intent`, `/simulate`, `/density`, `/interpret`, etc.  
  Lower-level endpoints for quantum and symbolic experiments.

- `/symbols`  
  View or upload custom archetypes.

See the OpenAPI spec (`/openai.yaml`) for the full list.

## Deployment
The included `render.yaml` file configures the service for Render.com. When
deployed, the API will be available at
`https://entropic-api.onrender.com`.

## Testing
Run the unit tests with `pytest`:
```bash
pytest -q
```
Some tests require `httpx`; if it isn't installed the tests will be skipped.
Install dependencies from `requirements.txt` first to ensure `httpx` is
available. You can also run `./scripts/run_tests.sh` to automatically install
dependencies and execute the test suite.


---

**For custom GPT integration:**  
Point your GPT or plugin manifest at `/openai.yaml` and use `/predict-life` for uncannily specific, friend-like readings.