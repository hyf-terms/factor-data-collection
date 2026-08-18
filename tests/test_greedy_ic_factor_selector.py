import pandas as pd

from greedy_ic_factor_selector import greedy_select, load_daily_ic


def test_greedy_orders_by_abs_ic_and_blocks_high_correlation():
    dates = pd.date_range("2024-01-01", periods=40)
    base = pd.Series([0.041 + (-1) ** i * (0.003 + i / 100000) for i in range(40)])
    daily = pd.concat(
        [
            pd.DataFrame({"TRADE_DATE": dates, "factor": "best", "ic": base}),
            pd.DataFrame({"TRADE_DATE": dates, "factor": "duplicate", "ic": base * 0.99}),
            pd.DataFrame({"TRADE_DATE": dates, "factor": "diversifier", "ic": 0.036 + (pd.Series(range(40)) % 5 - 2) / 1000}),
        ],
        ignore_index=True,
    )
    decisions, correlation, selected_daily = greedy_select(
        daily, ic_threshold=0.035, correlation_threshold=0.85
    )
    selected = decisions.loc[decisions.selected, "factor"].tolist()
    assert "best" in selected
    assert "duplicate" not in selected
    assert "diversifier" in selected
    assert correlation.loc["best", "duplicate"] > 0.85
    assert selected_daily.columns.tolist() == selected


def test_threshold_is_absolute():
    dates = pd.date_range("2024-01-01", periods=40)
    daily = pd.DataFrame({"TRADE_DATE": dates, "factor": "negative", "ic": -0.04})
    decisions, _, _ = greedy_select(daily, ic_threshold=0.035)
    assert decisions.loc[0, "selected"]


def test_loads_factors_neus_wide_version_format(tmp_path):
    path = tmp_path / "daily_ic.parquet"
    pd.DataFrame(
        {
            "TRADE_DATE": pd.date_range("2024-01-01", periods=2),
            "factor": ["alpha", "alpha"],
            "raw_ic": [0.01, 0.02],
            "neutral_ic": [0.03, 0.04],
            "neutral_n": [100, 100],
        }
    ).to_parquet(path, index=False)
    loaded = load_daily_ic([path])
    assert loaded["factor"].tolist() == ["alpha", "alpha"]
    assert loaded["ic"].tolist() == [0.03, 0.04]
