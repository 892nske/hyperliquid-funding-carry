from __future__ import annotations

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
