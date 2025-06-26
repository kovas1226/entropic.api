#!/bin/bash
set -e

# Install dependencies
pip install -r requirements.txt

# Run pytest
pytest -q