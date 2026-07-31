import pandas as pd
import numpy as np

from unreplicated_financial_factor_search import (
    CANDIDATE_COLUMNS,
    build_candidates_for_slice,
)


def test_new_candidates_respect_q1_and_age_windows() -> None:
    keys = pd.DataFrame(
        {
            "TRADE_DATE": [pd.Timestamp("2025-04-30")] * 3,
            "SECURITY_ID": [1, 2, 3],
        }
    )
    prefixes = [
        "margin",
        "turnover",
        "noa",
        "cfoa",
        "revenue_growth",
        "deducted_growth",
        "nonrecurring",
        "rd",
        "capex",
        "earnings_stability",
        "sales_stability",
    ]
    mapped = {}
    for index, prefix in enumerate(prefixes):
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

    result = build_candidates_for_slice(keys, mapped)
    assert set(CANDIDATE_COLUMNS).issubset(result)
    assert result["q1_dupont_improvement_80d"].notna().all()
    assert result["q1_quality_efficiency_80d"].notna().all()
    assert result["q1_intangible_growth_quality_80d"].notna().all()
    assert result.loc[2, "q1_dupont_improvement_80d"] == 1.0

    mapped["margin"]["margin_age"] = 121
    expired = build_candidates_for_slice(keys, mapped)
    assert expired["gross_margin_change_120d"].isna().all()
    assert expired["dupont_improvement_120d"].isna().all()
    assert expired["q1_dupont_improvement_80d"].isna().all()


def test_stability_metrics_keep_higher_is_better_direction() -> None:
    keys = pd.DataFrame(
        {
            "TRADE_DATE": [pd.Timestamp("2025-04-30")] * 3,
            "SECURITY_ID": [1, 2, 3],
        }
    )
    prefixes = [
        "margin",
        "turnover",
        "noa",
        "cfoa",
        "revenue_growth",
        "deducted_growth",
        "nonrecurring",
        "rd",
        "capex",
        "earnings_stability",
        "sales_stability",
    ]
    mapped = {
        prefix: keys.assign(
            **{
                f"{prefix}_value": [1.0, 2.0, 3.0],
                f"{prefix}_quarter": [1, 1, 1],
                f"{prefix}_age": [10, 10, 10],
            }
        )
        for prefix in prefixes
    }
    result = build_candidates_for_slice(keys, mapped)
    assert np.allclose(
        result["earnings_stability_120d"],
        [1.02, 2.0, 2.98],
    )
    assert np.allclose(
        result["sales_growth_stability_120d"],
        [1.02, 2.0, 2.98],
    )
