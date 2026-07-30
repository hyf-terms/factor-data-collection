import pandas as pd

from literature_financial_factor_search import (
    CANDIDATE_COLUMNS,
    FACTOR_INPUTS,
    build_candidates_for_slice,
)


def test_literature_candidates_respect_event_windows() -> None:
    date = pd.Timestamp("2025-04-30")
    keys = pd.DataFrame(
        {
            "TRADE_DATE": [date] * 3,
            "SECURITY_ID": [1, 2, 3],
        }
    )
    factors = keys.copy()
    for index, column in enumerate(FACTOR_INPUTS):
        factors[column] = [index + 1.0, index + 2.0, index + 3.0]

    mapped = {}
    for index, prefix in enumerate(
        ["pead", "deducted", "revenue", "operate", "gross", "pead_ma4"]
    ):
        mapped[prefix] = keys.assign(
            **{
                f"{prefix}_value": [
                    -1.0 - index,
                    0.0,
                    1.0 + index,
                ],
                f"{prefix}_quarter": [1, 1, 1],
                f"{prefix}_age": [10, 10, 10],
            }
        )

    result = build_candidates_for_slice(factors, mapped)
    assert set(CANDIDATE_COLUMNS).issubset(result.columns)
    assert result["q1_joint_earnings_revenue"].notna().all()
    assert result["q1_financial_60d"].notna().all()
    assert result["q1_all_profit_surprises"].notna().all()

    mapped["revenue"]["revenue_age"] = 100
    expired = build_candidates_for_slice(factors, mapped)
    assert expired["revenue_sue_60d"].isna().all()
    assert expired["q1_joint_earnings_revenue"].isna().all()
