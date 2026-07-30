"""Small deterministic checks for the CH-3/CH-4 portfolio arithmetic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ch_factor_models import (
    BuildConfig,
    build_factor_returns,
)


def synthetic_assignments() -> pd.DataFrame:
    ep_returns = {
        "SV": 0.06,
        "SM": 0.05,
        "SG": 0.04,
        "BV": 0.03,
        "BM": 0.02,
        "BG": 0.01,
    }
    turnover_returns = {
        "SP": 0.07,
        "SN": 0.05,
        "SO": 0.03,
        "BP": 0.04,
        "BN": 0.02,
        "BO": 0.00,
    }
    records = []
    date = pd.Timestamp("2024-02-29")
    security_id = 1
    for ep_name, ep_return in ep_returns.items():
        for turnover_name, turnover_return in turnover_returns.items():
            for _ in range(2):
                records.append(
                    {
                        "SECURITY_ID": security_id,
                        "RETURN_MONTH_END_DATE": date,
                        "EP_PORTFOLIO": ep_name,
                        "TURNOVER_PORTFOLIO": turnover_name,
                        "MONTH_RETURN": (
                            ep_return + turnover_return
                        )
                        / 2.0,
                        "FORMATION_MARKET_VALUE_A": 1.0,
                    }
                )
                security_id += 1
    return pd.DataFrame(records)


def test_factor_arithmetic() -> None:
    assignments = synthetic_assignments()
    config = BuildConfig(
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-12-31"),
        annual_risk_free=0.0,
        min_portfolio_size=1,
    )
    ch3, ch4, all_factors = build_factor_returns(assignments, config)
    assert len(ch3) == 1
    assert len(ch4) == 1
    assert np.isfinite(all_factors.loc[0, "SMB_EP"])
    assert np.isfinite(all_factors.loc[0, "SMB_TURNOVER"])
    assert np.isclose(
        all_factors.loc[0, "SMB_CH4"],
        np.mean(
            [
                all_factors.loc[0, "SMB_EP"],
                all_factors.loc[0, "SMB_TURNOVER"],
            ]
        ),
    )
    assert ch4.loc[0, "PMO"] > 0


if __name__ == "__main__":
    test_factor_arithmetic()
    print("CH-3/CH-4 synthetic checks passed")
