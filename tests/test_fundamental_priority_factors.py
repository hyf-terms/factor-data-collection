from __future__ import annotations

import numpy as np
import pandas as pd

from fundamental_priority_factors import (
    build_accrual_quality_events,
    build_cfo_sue_events,
    build_operating_profit_events,
    build_ttm_metric,
)


def _quarterly(column: str, values: list[float]) -> pd.DataFrame:
    quarter_index = np.arange(8001, 8001 + len(values))
    return pd.DataFrame(
        {
            "SECURITY_ID": 1,
            "QUARTER_INDEX": quarter_index,
            "FISCAL_YEAR": (quarter_index - 1) // 4,
            "FISCAL_QUARTER": (quarter_index - 1) % 4 + 1,
            "END_DATE": pd.date_range(
                "2010-03-31",
                periods=len(values),
                freq="QE",
            ),
            "EVENT_TIME": pd.date_range(
                "2010-04-20 18:00:00",
                periods=len(values),
                freq="QE",
            ),
            column: values,
        }
    )


def _balances(count: int) -> pd.DataFrame:
    quarter_index = np.arange(8001, 8001 + count)
    return pd.DataFrame(
        {
            "SECURITY_ID": 1,
            "QUARTER_INDEX": quarter_index,
            "T_ASSETS": np.full(count, 100.0),
            "BALANCE_EVENT_TIME": pd.date_range(
                "2010-04-20 18:00:00",
                periods=count,
                freq="QE",
            ),
        }
    )


def test_ttm_requires_four_consecutive_quarters() -> None:
    quarterly = _quarterly("OPERATE_PROFIT", [1, 2, 3, 4, 5, 6])
    output = build_ttm_metric(quarterly, "OPERATE_PROFIT")
    assert output["TTM_OPERATE_PROFIT"].tolist() == [10.0, 14.0, 18.0]

    missing = quarterly.loc[quarterly["QUARTER_INDEX"].ne(8003)]
    output_missing = build_ttm_metric(missing, "OPERATE_PROFIT")
    assert output_missing["QUARTER_INDEX"].tolist() == []


def test_operating_growth_and_acceleration() -> None:
    values = [10.0] * 4 + [12.0] * 4 + [15.0] * 4
    quarterly = _quarterly("OPERATE_PROFIT", values)
    growth, acceleration = build_operating_profit_events(
        quarterly,
        _balances(len(values)),
    )
    # At 8008: (48 - 40) / 100 = 0.08.
    row = growth.loc[growth["QUARTER_INDEX"].eq(8008)].iloc[0]
    assert np.isclose(row["operating_profit_growth"], 0.08)
    # Growth at 8009 is (51 - 42) / 100 = 0.09, so acceleration is 0.01.
    row_acc = acceleration.loc[
        acceleration["QUARTER_INDEX"].eq(8009)
    ].iloc[0]
    assert np.isclose(row_acc["operating_profit_acceleration"], 0.01)


def test_cfo_sue_uses_only_preceding_surprises() -> None:
    # Seasonal changes are 1..12; at least eight prior surprises are needed.
    values = [10.0, 10.0, 10.0, 10.0]
    for seasonal_change in range(1, 13):
        values.append(values[-4] + float(seasonal_change))
    quarterly = _quarterly("N_CF_OPERATE_A", values)
    output = build_cfo_sue_events(
        quarterly,
        _balances(len(quarterly)),
    )
    assert output["QUARTER_INDEX"].min() == 8013
    expected = 9.0 / np.std(np.arange(1.0, 9.0), ddof=1)
    first = output.iloc[0]
    assert np.isclose(first["cfo_sue"], expected)


def test_accrual_quality_is_cash_minus_income_over_average_assets() -> None:
    net_income = _quarterly("N_INCOME", [2.0] * 8)
    cfo = _quarterly("N_CF_OPERATE_A", [3.0] * 8)
    output = build_accrual_quality_events(
        net_income,
        cfo,
        _balances(8),
    )
    # TTM CFO - TTM income = 12 - 8 = 4; average assets = 100.
    assert np.allclose(output["accrual_quality"], 0.04)
