from __future__ import annotations

import pytest

from hl_funding_carry.features.funding import build_funding_features


def test_build_funding_features_adds_expected_columns(sample_dataset):
    out = build_funding_features(sample_dataset)
    expected_columns = {
        "basis",
        "basis_bps",
        "basis_change_1h",
        "oi_change_1h",
        "oi_change_4h",
        "funding_z_24h",
        "carry_score",
    }
    assert expected_columns.issubset(out.columns)
    assert out.loc[0, "basis"] == pytest.approx(0.002)


def test_funding_zscore_uses_only_past_data(sample_dataset):
    out = build_funding_features(sample_dataset)
    modified = sample_dataset.copy()
    modified.loc[29, "current_funding"] = 0.001
    modified_out = build_funding_features(modified)
    assert out.loc[23, "funding_z_24h"] == modified_out.loc[23, "funding_z_24h"]
