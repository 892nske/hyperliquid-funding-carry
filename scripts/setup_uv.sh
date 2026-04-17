#!/usr/bin/env bash
set -euo pipefail

uv python install 3.12
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
