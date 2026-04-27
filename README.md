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

## Week3 improvements
- switchable execution models: `next_open`, `twap_5m`, `vwap_1m`
- execution-aware fee / slippage / timing drag tracking
- execution attribution outputs in artifacts
- report regeneration from an existing artifact run directory

## Week4 improvements
- Hyperliquid real-data ingestion and normalization for research batch workflows
- processed dataset validation with UTC / gap / duplicate checks
- real-data compatible backtest path using the existing common schema
- walk-forward runner with fold artifacts and selected parameter tracking
- raw / processed data layout for reproducible offline experiments

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

Regenerate attribution files from an existing run with:

```bash
uv run python -m hl_funding_carry report --input-dir artifacts/<run_dir>
```

Use `--output-dir` to control where artifacts are saved.

Ingest Hyperliquid-style real data fixture into `data/raw/real` and `data/processed` with:

```bash
uv run python -m hl_funding_carry ingest --config configs/funding_carry.ingest.yaml
```

Validate a processed dataset directory with:

```bash
uv run python -m hl_funding_carry validate-data --input-dir data/processed/hyperliquid/BTC/20260101T000000Z_20260102T050000Z
```

Run a backtest from the processed real-data fixture with:

```bash
uv run python -m hl_funding_carry backtest --config configs/funding_carry.real.yaml
```

Run a walk-forward study with:

```bash
uv run python -m hl_funding_carry walkforward --config configs/funding_carry.walkforward.yaml
```

## Artifacts
Backtest and sweep outputs are saved under `artifacts/` by default.

Single backtest output includes:
- `summary.csv`
- `equity_curve.csv`
- `trades.csv`
- `ledger.csv`
- `pnl_attribution.csv`
- `execution_summary.csv`
- `trade_attribution.csv`
- `params.json`

Sweep output includes:
- top-level `summary.csv` and `summary.parquet`
- per-run subdirectories under `runs/`

Walk-forward output includes:
- `walkforward_summary.csv`
- `fold_results.csv`
- `selected_params.csv`
- `data_validation_summary.csv`
- per-fold backtest artifact directories under `folds/`

Processed real-data directories also save:
- `candles.parquet` or `.csv`
- `asset_context.parquet` or `.csv`
- `funding_inputs.parquet` or `.csv`
- `funding_history.parquet` or `.csv`
- `data_validation_summary.csv`

Raw Hyperliquid ingestion directories save the fetched vendor-shaped payloads separately from normalized outputs.

## Raw / Processed layout
Example layout after `ingest`:

```text
data/
├── raw/
│   └── real/
│       └── hyperliquid/
│           └── BTC/
│               └── 20260101T000000Z_20260102T050000Z/
│                   ├── candles_raw.parquet
│                   ├── asset_context_raw.parquet
│                   ├── funding_history_raw.parquet
│                   └── predicted_funding_raw.parquet
└── processed/
    └── hyperliquid/
        └── BTC/
            └── 20260101T000000Z_20260102T050000Z/
                ├── candles.parquet
                ├── asset_context.parquet
                ├── funding_inputs.parquet
                ├── funding_history.parquet
                └── data_validation_summary.csv
```

`sample` configs keep using `data/raw/sample_*.csv`. `real` configs point at a processed directory and reuse the same backtest schema.

## Data validation
`validate-data` and `ingest` emit a `data_validation_summary.csv` with:
- `row_count`
- `missing_ratio`
- `duplicate_count`
- `non_monotonic_count`
- `gap_count`
- `max_gap_minutes`
- `min_timestamp`
- `max_timestamp`

This is a lightweight research quality check, not a full vendor reconciliation report.

## Execution model assumptions
- `next_open`: current ledger row timestamp at the next signal bar is the benchmark execution price
- `twap_5m`: 5m intrabar typical price average over the post-entry execution window; if unavailable, 1m proxy, then `next_open`
- `vwap_1m`: 1m close-volume weighted average over the post-entry execution window; if volume is unusable, TWAP fallback
- `timing_drag` is tracked separately from `slippage` as the deviation from the `next_open` benchmark implied by the execution model

## Attribution outputs
- `pnl_attribution.csv`: run / symbol / execution model level decomposition
- `execution_summary.csv`: execution model level cost / fallback summary
- `trade_attribution.csv`: per-trade decomposition for downstream analysis

## Accounting assumptions
- timestamps are treated as UTC internally
- funding pnl is only accrued on explicit funding event timestamps
- `spot_perp` mode records separate spot and perp legs for research
- margin, borrowing, liquidation, and live execution are still out of scope

## Real-data ingestion assumptions
- real-data ingestion is batch-oriented and offline-first
- Hyperliquid source handling is split into fetch, normalize, validate, and save steps
- local dump files and processed directories are first-class inputs; API fetching is adapter-scoped rather than embedded in strategy code
- if predicted funding is unavailable, processed `funding_inputs` fall back to `current_funding` so the common schema remains backtestable without future leakage

## Walk-forward assumptions
- walk-forward uses the existing strategy and execution stack unchanged
- `train_window`, `test_window`, and `step_size` are configured as pandas-style durations such as `12h` or `7D`
- train folds use simple grid search over the configured parameter set and select one metric, currently `total_return` or `sharpe_like`

## Known constraints
- historical asset-context ingestion still depends on having a Hyperliquid-compatible dump or archive; the public API path alone is not enough for full historical reconstruction
- execution intrabar data is optional in processed real-data runs; without it, `twap_5m` / `vwap_1m` fall back toward the week3 proxy logic
- predicted funding support is optional and may be approximated with `current_funding` when only realized history is available
- this repository still does not implement live trading, order routing, authentication, or wallet signing

## Proposed package layout
See [`docs/project_structure.md`](docs/project_structure.md).

## First implementation targets
1. Historical funding and asset context loaders
2. Funding carry feature builder
3. Signal engine for entry/exit conditions
4. Bar-based simulator
5. Performance report generation
