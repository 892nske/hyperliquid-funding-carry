from __future__ import annotations

from hl_funding_carry.experiments.runner import run_backtest, run_sweep, run_walkforward
from hl_funding_carry.settings import CONFIG_DIR, load_config, load_sweep_grid


def test_sweep_generates_multiple_runs_with_execution_models(tmp_path):
    config = load_config(CONFIG_DIR / "funding_carry.base.yaml")
    grid = load_sweep_grid(CONFIG_DIR / "funding_carry.sweep.yaml")
    summary_df = run_sweep(config, grid, output_dir=tmp_path)

    assert len(summary_df) > 1
    assert "execution_model" in summary_df.columns
    assert summary_df["run_id"].nunique() == len(summary_df)


def test_backtest_saves_artifacts(tmp_path):
    config = load_config(CONFIG_DIR / "funding_carry.base.yaml")
    result = run_backtest(config, output_dir=tmp_path, run_id="test_run")

    assert result.artifact_dir is not None
    assert (result.artifact_dir / "summary.csv").exists()
    assert (result.artifact_dir / "equity_curve.csv").exists()
    assert (result.artifact_dir / "trades.csv").exists()
    assert (result.artifact_dir / "ledger.csv").exists()
    assert (result.artifact_dir / "pnl_attribution.csv").exists()
    assert (result.artifact_dir / "execution_summary.csv").exists()
    assert (result.artifact_dir / "trade_attribution.csv").exists()
    assert (result.artifact_dir / "params.json").exists()


def test_walkforward_generates_fold_outputs(tmp_path):
    config = load_config(CONFIG_DIR / "funding_carry.walkforward.yaml")
    fold_results = run_walkforward(config, output_dir=tmp_path)

    assert not fold_results.empty
    walkforward_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert walkforward_dirs
    walkforward_dir = walkforward_dirs[0]
    assert (walkforward_dir / "walkforward_summary.csv").exists()
    assert (walkforward_dir / "fold_results.csv").exists()
    assert (walkforward_dir / "selected_params.csv").exists()
