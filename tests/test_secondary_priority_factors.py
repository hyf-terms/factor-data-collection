from __future__ import annotations

import numpy as np
import pandas as pd

from secondary_priority_factors import (
    build_noa_quality_events,
    build_ttm_growth_events,
    calculate_composite_scores,
)


def _assets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SECURITY_ID": [1, 1, 1],
            "QUARTER_INDEX": [8000, 8004, 8008],
            "FISCAL_YEAR": [1999, 2000, 2001],
            "FISCAL_QUARTER": [4, 4, 4],
            "END_DATE": pd.to_datetime(
                ["1999-12-31", "2000-12-31", "2001-12-31"]
            ),
            "T_ASSETS": [80.0, 100.0, 110.0],
            "BALANCE_EVENT_TIME": pd.to_datetime(
                [
                    "2000-04-20 18:00",
                    "2001-04-20 18:00",
                    "2002-04-20 18:00",
                ]
            ),
        }
    )


def _ttm(column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SECURITY_ID": [1, 1],
            "QUARTER_INDEX": [8004, 8008],
            "FISCAL_YEAR": [2000, 2001],
            "FISCAL_QUARTER": [4, 4],
            "END_DATE": pd.to_datetime(["2000-12-31", "2001-12-31"]),
            f"TTM_{column}": [10.0, 15.0],
            f"TTM_{column}_EVENT_TIME": pd.to_datetime(
                ["2001-04-20 18:00", "2002-04-20 18:00"]
            ),
        }
    )


def test_ttm_growth_is_scaled_by_prior_assets() -> None:
    output = build_ttm_growth_events(
        _ttm("REVENUE"),
        _assets(),
        column="REVENUE",
        factor="revenue_growth",
    )
    assert len(output) == 1
    assert np.isclose(output.iloc[0]["revenue_growth"], 0.05)


def test_noa_quality_formula() -> None:
    common = {
        "ID": [1, 2],
        "SECURITY_ID": [1, 1],
        "ACT_PUBTIME": pd.to_datetime(
            ["2020-04-20 18:00", "2021-04-20 18:00"]
        ),
        "END_DATE": pd.to_datetime(["2019-12-31", "2020-12-31"]),
        "END_DATE_REP": pd.to_datetime(["2019-12-31", "2020-12-31"]),
        "REPORT_TYPE": ["A", "A"],
        "FISCAL_PERIOD": [12, 12],
        "MERGED_FLAG": ["1", "1"],
        "IS_CURRENT_PERIOD": [True, True],
        "INDUSTRY_CATEGORY": ["一般工商业", "一般工商业"],
        "T_ASSETS": [100.0, 110.0],
        "CASH_C_EQUIV": [10.0, 10.0],
        "T_LIAB": [50.0, 55.0],
        "ST_BORR": [5.0, 5.0],
        "NCL_WITHIN_1_Y": [0.0, 0.0],
        "LT_BORR": [10.0, 10.0],
        "BOND_PAYABLE": [0.0, 0.0],
    }
    output = build_noa_quality_events(pd.DataFrame(common))
    # Current OA=100, operating liabilities=40, NOA=60; / lag assets 100.
    assert np.isclose(output.iloc[0]["noa_quality"], -0.60)


def _component_panel() -> pd.DataFrame:
    rows = []
    for security_id in (1, 2, 3):
        row = {
            "TRADE_DATE": pd.Timestamp("2024-01-02"),
            "SECURITY_ID": security_id,
            "operating_profit_growth": float(security_id),
            "cfo_sue": float(security_id),
            "accrual_quality": float(security_id),
            "asset_growth": float(security_id),
            "investment_to_assets": float(security_id),
            "cfo_to_assets": float(security_id),
            "gross_margin_change": float(security_id),
            "asset_turnover_change": float(security_id),
            "noa_quality": float(security_id),
            "nonrecurring_quality": float(security_id),
            "revenue_growth": float(security_id),
            "deducted_profit_growth": float(security_id),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def test_composites_use_cross_sectional_percentile_ranks() -> None:
    output = calculate_composite_scores(_component_panel())
    for factor in [
        "profitability_quality_score",
        "management_mispricing_score",
        "accrual_nonrecurring_score",
        "fundamental_momentum",
    ]:
        assert np.allclose(output[factor], [1 / 3, 2 / 3, 1.0])


def test_minimum_component_rule() -> None:
    panel = _component_panel()
    panel.loc[0, ["cfo_to_assets", "gross_margin_change"]] = np.nan
    output = calculate_composite_scores(panel)
    assert np.isnan(output.loc[0, "profitability_quality_score"])
