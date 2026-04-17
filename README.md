# hyperliquid-funding-carry

Research and backtesting scaffold for **Hyperliquid Strategy 1: Funding Carry**.

This repository is intentionally scoped to:
- historical data ingestion
- feature engineering
- signal generation
- bar-based backtesting
- performance evaluation

It is intentionally **not** scoped to live trading yet.

## Recommended stack
- Python 3.12
- `uv` for environment and dependency management
- `pytest` for tests
- `ruff` for lint/format
- `mypy` for type checks

## Quick start
```bash
uv python install 3.12
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

## Proposed package layout
See [`docs/project_structure.md`](docs/project_structure.md).

## First implementation targets
1. Historical funding and asset context loaders
2. Funding carry feature builder
3. Signal engine for entry/exit conditions
4. Bar-based simulator
5. Performance report generation
