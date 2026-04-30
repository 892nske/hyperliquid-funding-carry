from __future__ import annotations

from hl_funding_carry.backtest.execution import resolve_execution_fill
from hl_funding_carry.data.loaders import load_execution_inputs, load_processed_execution_inputs
from hl_funding_carry.settings import load_config


def test_execution_model_switch_changes_fill_prices(config_path):
    config = load_config(config_path)
    execution_inputs = load_execution_inputs(
        config.data.execution_5m_path,
        config.data.execution_1m_path,
    )
    timestamp = "2026-01-01T00:00:00Z"
    benchmark_price = 100.01
    next_open_fill = resolve_execution_fill(
        "next_open",
        config.execution,
        execution_inputs,
        "BTC",
        timestamp,
        benchmark_price,
    )
    twap_fill = resolve_execution_fill(
        "twap_5m",
        config.execution,
        execution_inputs,
        "BTC",
        timestamp,
        benchmark_price,
    )
    vwap_fill = resolve_execution_fill(
        "vwap_1m",
        config.execution,
        execution_inputs,
        "BTC",
        timestamp,
        benchmark_price,
    )
    assert next_open_fill.fill_price != twap_fill.fill_price
    assert twap_fill.fill_price != vwap_fill.fill_price


def test_vwap_falls_back_without_volume(config_path):
    config = load_config(config_path)
    execution_inputs = load_execution_inputs(
        config.data.execution_5m_path,
        config.data.execution_1m_path,
    )
    execution_inputs["1m"].loc[:, "volume"] = 0.0
    fill = resolve_execution_fill(
        "vwap_1m",
        config.execution,
        execution_inputs,
        "BTC",
        "2026-01-01T00:05:00Z",
        100.11,
    )
    assert fill.fallback == "proxy_twap_no_volume"


def test_real_data_execution_inputs_support_model_comparison(multi_config_path):
    config = load_config(multi_config_path)
    assert config.data.processed_dir is not None
    execution_inputs = load_processed_execution_inputs(config.data.processed_dir, recursive=True)
    benchmark_price = 200.02

    next_open_fill = resolve_execution_fill(
        "next_open",
        config.execution,
        execution_inputs,
        "ETH",
        "2026-01-01T00:00:00Z",
        benchmark_price,
    )
    twap_fill = resolve_execution_fill(
        "twap_5m",
        config.execution,
        execution_inputs,
        "ETH",
        "2026-01-01T00:00:00Z",
        benchmark_price,
    )
    vwap_fill = resolve_execution_fill(
        "vwap_1m",
        config.execution,
        execution_inputs,
        "ETH",
        "2026-01-01T00:00:00Z",
        benchmark_price,
    )
    assert next_open_fill.fill_price != twap_fill.fill_price
    assert twap_fill.fill_price != vwap_fill.fill_price
