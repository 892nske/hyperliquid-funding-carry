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

## Week2 improvements
- explicit hourly funding event timing in UTC
- 2-leg spot/perp ledger with separated PnL decomposition
- time-aware entry/exit windows around funding events
- artifact saving for backtests and sweeps
- parameter sweep runner for research comparisons

## Usage
Run the sample end-to-end funding carry backtest with:

```bash
uv run python -m hl_funding_carry backtest --config configs/funding_carry.base.yaml
```

The old week1 form still works:

```bash
uv run python -m hl_funding_carry --config configs/funding_carry.base.yaml
```

Run a parameter sweep with:

```bash
uv run python -m hl_funding_carry sweep --config configs/funding_carry.base.yaml --grid configs/funding_carry.sweep.yaml
```

Use `--output-dir` to control where artifacts are saved.

## Artifacts
Backtest and sweep outputs are saved under `artifacts/` by default.

Single backtest output includes:
- `summary.csv`
- `equity_curve.csv`
- `trades.csv`
- `ledger.csv`
- `params.json`

Sweep output includes:
- top-level `summary.csv` and `summary.parquet`
- per-run subdirectories under `runs/`

## Accounting assumptions
- timestamps are treated as UTC internally
- funding pnl is only accrued on explicit funding event timestamps
- `spot_perp` mode records separate spot and perp legs for research
- margin, borrowing, liquidation, and live execution are still out of scope

## Proposed package layout
See [`docs/project_structure.md`](docs/project_structure.md).

## First implementation targets
1. Historical funding and asset context loaders
2. Funding carry feature builder
3. Signal engine for entry/exit conditions
4. Bar-based simulator
5. Performance report generation
