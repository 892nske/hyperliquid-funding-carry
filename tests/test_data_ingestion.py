from __future__ import annotations

from hl_funding_carry.data.ingestion import ingest_hyperliquid_batch, ingest_hyperliquid_bulk
from hl_funding_carry.data.loaders import (
    load_processed_execution_inputs,
    load_processed_research_dataset,
)
from hl_funding_carry.data.validation import (
    summarize_validation_report,
    validate_processed_directory,
)
from hl_funding_carry.settings import load_ingest_config


def test_hyperliquid_ingest_normalizes_to_common_schema(tmp_path, ingest_config_path):
    ingest_config = load_ingest_config(ingest_config_path)
    ingest_config.raw_output_dir = tmp_path / "raw"
    ingest_config.processed_output_dir = tmp_path / "processed"
    result = ingest_hyperliquid_batch(ingest_config)

    dataset = load_processed_research_dataset(result.processed_dir)
    assert {"timestamp", "symbol", "mark_price", "oracle_price", "pred_funding_1h"}.issubset(
        dataset.columns,
    )
    assert str(dataset["timestamp"].dt.tz) == "UTC"
    assert dataset["symbol"].eq("BTC").all()


def test_validation_detects_missing_duplicate_and_gap():
    import pandas as pd

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T03:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["BTC", "BTC", "BTC"],
            "value": [1.0, None, 3.0],
        },
    )
    report = summarize_validation_report(df, "toy")

    assert report.duplicate_count >= 1
    assert report.missing_ratio > 0.0
    assert report.gap_count >= 1


def test_validate_processed_directory_returns_summary(real_config_path):
    from hl_funding_carry.settings import load_config

    config = load_config(real_config_path)
    assert config.data.processed_dir is not None
    summary_df = validate_processed_directory(config.data.processed_dir)
    assert {"dataset", "row_count", "missing_ratio", "duplicate_count"}.issubset(
        summary_df.columns,
    )


def test_bulk_ingest_handles_multiple_symbols_and_chunks(tmp_path, bulk_ingest_config_path):
    ingest_config = load_ingest_config(bulk_ingest_config_path)
    ingest_config.raw_output_dir = tmp_path / "raw"
    ingest_config.processed_output_dir = tmp_path / "processed"
    result = ingest_hyperliquid_bulk(ingest_config)

    assert not result.batch_summary.empty
    assert result.batch_summary["symbol"].nunique() == 3
    assert len(result.batch_summary) == 6


def test_processed_loader_reads_execution_inputs_from_recursive_root(multi_config_path):
    from hl_funding_carry.settings import load_config

    config = load_config(multi_config_path)
    assert config.data.processed_dir is not None
    dataset = load_processed_research_dataset(
        config.data.processed_dir,
        recursive=True,
        symbols=config.strategy.symbols,
    )
    execution_inputs = load_processed_execution_inputs(config.data.processed_dir, recursive=True)

    assert dataset["symbol"].nunique() == 2
    assert "1m" in execution_inputs
    assert "5m" in execution_inputs
