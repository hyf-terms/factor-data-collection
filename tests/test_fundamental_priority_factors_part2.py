from __future__ import annotations

import numpy as np
import pandas as pd

from fundamental_priority_factors_part2 import (
    build_abnormal_growth_events,
    build_asset_growth_events,
    build_investment_events,
)


def _balance() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [1, 2],
            "SECURITY_ID": [1, 1],
            "ACT_PUBTIME": pd.to_datetime(
                ["2020-04-20 18:00", "2021-04-20 18:00"]
            ),
            "END_DATE": pd.to_datetime(["2019-12-31", "2020-12-31"]),
            "END_DATE_REP": pd.to_datetime(
                ["2019-12-31", "2020-12-31"]
            ),
            "REPORT_TYPE": ["A", "A"],
            "FISCAL_PERIOD": [12, 12],
            "MERGED_FLAG": ["1", "1"],
            "IS_CURRENT_PERIOD": [True, True],
            "INDUSTRY_CATEGORY": ["一般工商业", "一般工商业"],
            "T_ASSETS": [100.0, 110.0],
            "FIXED_ASSETS_TOTAL": [20.0, 25.0],
            "INVENTORIES": [10.0, 12.0],
            "AR": [10.0, 12.0],
        }
    )


def _ttm(column: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SECURITY_ID": [1, 1],
            "QUARTER_INDEX": [8080, 8084],
            f"TTM_{column}": values,
            f"TTM_{column}_EVENT_TIME": pd.to_datetime(
                ["2020-04-20 18:00", "2021-04-20 18:00"]
            ),
        }
    )


def test_asset_growth_direction() -> None:
    output = build_asset_growth_events(_balance())
    assert len(output) == 1
    assert np.isclose(output.iloc[0]["asset_growth"], -0.10)


def test_investment_to_assets_direction() -> None:
    output = build_investment_events(_balance())
    # -[(25 - 20) + (12 - 10)] / 100 = -0.07.
    assert np.isclose(output.iloc[0]["investment_to_assets"], -0.07)


def test_receivable_abnormal_growth() -> None:
    output = build_abnormal_growth_events(
        _balance(),
        _ttm("REVENUE", [100.0, 110.0]),
        balance_column="AR",
        flow_column="REVENUE",
        factor="receivable_abnormal_growth",
    )
    # Revenue grows 10%, receivables 20%: factor = -10%.
    assert np.isclose(
        output.iloc[0]["receivable_abnormal_growth"],
        -0.10,
    )


def test_inventory_abnormal_growth() -> None:
    output = build_abnormal_growth_events(
        _balance(),
        _ttm("COGS", [60.0, 66.0]),
        balance_column="INVENTORIES",
        flow_column="COGS",
        factor="inventory_abnormal_growth",
    )
    # COGS grows 10%, inventory 20%: factor = -10%.
    assert np.isclose(
        output.iloc[0]["inventory_abnormal_growth"],
        -0.10,
    )
