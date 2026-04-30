from __future__ import annotations

from pathlib import Path

from hl_funding_carry.__main__ import main
from hl_funding_carry.settings import CONFIG_DIR


def test_cli_backtest_runs(tmp_path, capsys):
    main(
        [
            "backtest",
            "--config",
            str(CONFIG_DIR / "funding_carry.base.yaml"),
            "--output-dir",
            str(tmp_path),
        ],
    )
    captured = capsys.readouterr()
    assert "Funding Carry backtest summary" in captured.out


def test_cli_sweep_runs(tmp_path, capsys):
    main(
        [
            "sweep",
            "--config",
            str(CONFIG_DIR / "funding_carry.base.yaml"),
            "--grid",
            str(CONFIG_DIR / "funding_carry.sweep.yaml"),
            "--output-dir",
            str(tmp_path),
        ],
    )
    captured = capsys.readouterr()
    assert "Funding Carry sweep summary" in captured.out


def test_cli_report_runs(tmp_path, capsys):
    main(
        [
            "backtest",
            "--config",
            str(CONFIG_DIR / "funding_carry.base.yaml"),
            "--output-dir",
            str(tmp_path),
        ],
    )
    run_dirs = [path for path in Path(tmp_path).iterdir() if path.is_dir()]
    assert run_dirs
    main(["report", "--input-dir", str(run_dirs[0])])
    captured = capsys.readouterr()
    assert "Funding Carry attribution report regenerated" in captured.out


def test_cli_ingest_and_validate_data_run(tmp_path, capsys):
    main(
        [
            "ingest",
            "--config",
            str(CONFIG_DIR / "funding_carry.ingest.yaml"),
            "--output-dir",
            str(tmp_path / "raw"),
            "--processed-dir",
            str(tmp_path / "processed"),
        ],
    )
    captured = capsys.readouterr()
    assert "Funding Carry ingest completed" in captured.out

    processed_roots = list((tmp_path / "processed").glob("*/*/*"))
    assert processed_roots
    main(["validate-data", "--input-dir", str(processed_roots[0])])
    captured = capsys.readouterr()
    assert "Funding Carry data validation summary" in captured.out


def test_cli_walkforward_runs(tmp_path, capsys):
    main(
        [
            "walkforward",
            "--config",
            str(CONFIG_DIR / "funding_carry.walkforward.yaml"),
            "--output-dir",
            str(tmp_path),
        ],
    )
    captured = capsys.readouterr()
    assert "Funding Carry walk-forward summary" in captured.out


def test_cli_bulk_ingest_runs(tmp_path, capsys):
    main(
        [
            "ingest",
            "--config",
            str(CONFIG_DIR / "funding_carry.bulk.yaml"),
            "--output-dir",
            str(tmp_path / "raw"),
            "--processed-dir",
            str(tmp_path / "processed"),
        ],
    )
    captured = capsys.readouterr()
    assert "Funding Carry bulk ingest completed" in captured.out


def test_cli_multi_backtest_and_walkforward_run(tmp_path, capsys):
    main(
        [
            "backtest",
            "--config",
            str(CONFIG_DIR / "funding_carry.multi.yaml"),
            "--output-dir",
            str(tmp_path / "multi"),
        ],
    )
    captured = capsys.readouterr()
    assert "Funding Carry backtest summary" in captured.out

    main(
        [
            "walkforward",
            "--config",
            str(CONFIG_DIR / "funding_carry.multi_walkforward.yaml"),
            "--output-dir",
            str(tmp_path / "multi_wf"),
        ],
    )
    captured = capsys.readouterr()
    assert "Funding Carry walk-forward summary" in captured.out
