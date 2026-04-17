# AGENTS.md

## Purpose
This repository is for **Strategy 1: Funding Carry** on Hyperliquid.
The current scope is **research and backtesting only**. Do **not** implement live order placement, wallet signing, or production trading flows unless explicitly requested.

## Repository goals
1. Build a clean Python package for research and backtesting.
2. Keep market data ingestion, feature engineering, strategy logic, and simulation separated.
3. Prefer simple, testable code over clever abstractions.
4. Optimize for reproducibility and offline analysis before any execution logic.

## Working rules for Codex / Codes CLI
- Read `README.md` and `docs/project_structure.md` before editing.
- Keep code under `src/hl_funding_carry/`.
- Keep tests under `tests/`.
- Put one concept per module where practical.
- Prefer typed Python and small pure functions.
- When adding dependencies, update `pyproject.toml` and document the reason in the relevant PR/commit/task note.
- Do not introduce broker/exchange SDK dependencies unless the task explicitly requires them.
- Do not add notebook-only logic to the package source.

## Initial boundaries
Allowed now:
- data models
- loaders for historical snapshots and CSV/Parquet inputs
- feature engineering
- strategy state and signal generation
- bar-based backtest simulation
- evaluation metrics
- tests and fixtures

Not allowed now:
- live trading
- private key handling
- wallet signing
- websocket execution bots
- deployment infra

## Preferred commands
Use `uv` for environment and command execution.

Setup:
- `uv python install 3.12`
- `uv sync`

Quality checks:
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run pytest`
- `uv run mypy src`

## Coding conventions
- Python 3.12 target.
- Use `src/` layout.
- Use `pathlib` instead of string paths.
- Use `pydantic` or dataclasses for structured configs/data where useful.
- Use pandas/numpy for research paths, but keep transformation steps explicit and easy to test.
- Keep side effects at the edges; strategy logic should be deterministic from inputs.

## Backtest assumptions
- Backtests must separate `price_pnl`, `funding_pnl`, `fees`, and `slippage`.
- All timestamps should be treated as UTC internally.
- Persist intermediate outputs to `data/` only when the task explicitly asks for generated artifacts.
- Default to conservative execution assumptions.

## When extending this repo
If a future task adds other strategies, create parallel modules rather than mixing multi-strategy logic into the Funding Carry package.
