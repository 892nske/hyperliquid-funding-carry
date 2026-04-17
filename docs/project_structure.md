# Project structure

## Principles
- Keep research code importable and testable.
- Separate **data ingestion**, **features**, **strategy**, and **simulation**.
- Optimize for backtesting first.

## Directory tree
```text
hyperliquid-funding-carry/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .python-version
├── configs/
│   └── funding_carry.base.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── features/
│   └── backtests/
├── docs/
│   └── project_structure.md
├── notebooks/
├── scripts/
│   └── setup_uv.sh
├── src/
│   └── hl_funding_carry/
│       ├── __init__.py
│       ├── settings.py
│       ├── types.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── loaders.py
│       ├── features/
│       │   ├── __init__.py
│       │   └── funding.py
│       ├── strategies/
│       │   ├── __init__.py
│       │   └── funding_carry.py
│       └── backtest/
│           ├── __init__.py
│           ├── simulator.py
│           └── metrics.py
└── tests/
    ├── conftest.py
    ├── test_features.py
    └── test_strategy.py
```

## Responsibility by module
### `data/loaders.py`
Historical data loading utilities for:
- candles
- funding history
- asset contexts
- optional spot/perp merged inputs

### `features/funding.py`
Feature engineering for:
- predicted/current funding
- basis
- funding z-score
- open interest deltas
- spread filters

### `strategies/funding_carry.py`
Core strategy logic:
- entry eligibility
- position sizing hints
- exit rules
- state transitions

### `backtest/simulator.py`
Bar-based execution model for:
- target position transitions
- fees/slippage
- funding accrual
- holding period limits

### `backtest/metrics.py`
Performance metrics and decomposition:
- total pnl
- price pnl
- funding pnl
- fees
- slippage
- drawdown
- turnover

## Data policy
- Keep raw vendor/API dumps under `data/raw/`.
- Keep cleaned merged research tables under `data/interim/`.
- Keep engineered feature tables under `data/features/`.
- Keep experiment outputs under `data/backtests/`.
