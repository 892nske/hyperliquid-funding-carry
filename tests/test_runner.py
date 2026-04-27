from __future__ import annotations

from hl_funding_carry.experiments.runner import run_backtest, run_sweep
from hl_funding_carry.settings import CONFIG_DIR, load_config, load_sweep_grid


def test_sweep_generates_multiple_runs(tmp_path):
    config = load_config(CONFIG_DIR / "funding_carry.base.yaml")
    grid = load_sweep_grid(CONFIG_DIR / "funding_carry.sweep.yaml")
    summary_df = run_sweep(config, grid, output_dir=tmp_path)

    assert len(summary_df) > 1
    assert summary_df["run_id"].nunique() == len(summary_df)
    assert (tmp_path).exists()


def test_backtest_saves_artifacts(tmp_path):
    config = load_config(CONFIG_DIR / "funding_carry.base.yaml")
    result = run_backtest(config, output_dir=tmp_path, run_id="test_run")

    assert result.artifact_dir is not None
    assert (result.artifact_dir / "summary.csv").exists()
    assert (result.artifact_dir / "equity_curve.csv").exists()
    assert (result.artifact_dir / "trades.csv").exists()
    assert (result.artifact_dir / "ledger.csv").exists()
    assert (result.artifact_dir / "params.json").exists()
