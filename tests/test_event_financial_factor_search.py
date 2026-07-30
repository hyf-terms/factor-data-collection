import pandas as pd

from event_financial_factor_search import (
    build_candidates_for_slice,
)


def test_event_masks_and_high_ivol_selection() -> None:
    date = pd.Timestamp("2025-04-30")
    keys = pd.DataFrame(
        {
            "TRADE_DATE": [date] * 3,
            "SECURITY_ID": [1, 2, 3],
        }
    )
    factors = keys.assign(
        quarterly_f_score=[2.0, 5.0, 8.0],
        operating_profit_acceleration=[-1.0, 0.0, 1.0],
        asset_growth=[0.3, 0.2, 0.1],
        profitability_quality_score=[0.2, 0.5, 0.8],
    )
    barra = keys.assign(residual_volatility=[-1.0, 0.0, 1.0])
    pead = keys.assign(
        pead_value=[-2.0, 0.0, 2.0],
        pead_quarter=[1, 1, 1],
        pead_age=[10, 10, 10],
    )
    deducted = keys.assign(
        deducted_value=[-1.0, 0.0, 1.0],
        deducted_quarter=[1, 1, 1],
        deducted_age=[10, 10, 10],
    )

    result = build_candidates_for_slice(
        factors,
        barra,
        pead,
        deducted,
    ).set_index("SECURITY_ID")

    assert result["pead_sue_q1_80d"].notna().all()
    assert result["deducted_sue_q1_80d"].notna().all()
    assert result["q1_financial_composite"].notna().all()
    assert result["pead_sue_high_ivol30_80d"].notna().sum() == 1
    assert result.loc[3, "pead_sue_high_ivol30_80d"] == 2.0
