from hl_funding_carry.features.funding import build_funding_features


def test_build_funding_features_adds_expected_columns(funding_input):
    out = build_funding_features(funding_input)
    assert "basis" in out.columns
    assert "oi_change_1h" in out.columns
    assert "funding_z_24h" in out.columns
