import unittest

import numpy as np
import pandas as pd

from dense_literature_profitability_factors import (
    CANDIDATE_COLUMNS,
    calculate_factor_events,
)


class DenseLiteratureProfitabilityTests(unittest.TestCase):
    def test_formulas_and_candidate_set(self) -> None:
        quarters = np.arange(8001, 8009)
        years = ((quarters - 1) // 4).astype("int16")
        fiscal_quarters = ((quarters - 1) % 4 + 1).astype("int8")
        times = pd.date_range("2020-04-01", periods=8, freq="90D")
        values = {
            "N_INCOME_ATTR_P": [8, 9, 10, 11, 12, 13, 14, 15],
            "N_CF_OPERATE_A": [10, 11, 12, 13, 15, 16, 17, 18],
            "REVENUE": [100, 110, 120, 130, 140, 150, 160, 170],
            "COGS": [60, 65, 70, 75, 80, 85, 90, 95],
            "SELL_EXP": [5] * 8,
            "ADMIN_EXP": [4] * 8,
            "INT_EXP_FINAN_EXP": [1] * 8,
        }
        flows = {}
        for field, field_values in values.items():
            flows[field] = pd.DataFrame(
                {
                    "SECURITY_ID": 1,
                    "FISCAL_YEAR": years,
                    "FISCAL_QUARTER": fiscal_quarters,
                    "QUARTER_INDEX": quarters,
                    "END_DATE": times,
                    "EVENT_TIME": times,
                    field: field_values,
                }
            )
        balance = pd.DataFrame(
            {
                "SECURITY_ID": 1,
                "FISCAL_YEAR": years,
                "FISCAL_QUARTER": fiscal_quarters,
                "QUARTER_INDEX": quarters,
                "END_DATE": times,
                "BALANCE_EVENT_TIME": times,
                "INDUSTRY_CATEGORY": "制造业",
                "T_ASSETS": [100, 105, 110, 115, 120, 125, 130, 135],
                "T_EQUITY_ATTR_P": [50, 52, 54, 56, 58, 60, 62, 64],
                "AR": [10, 11, 12, 13, 14, 15, 16, 17],
                "INVENTORIES": [20, 21, 22, 23, 24, 25, 26, 27],
                "PREPAYMENT": [2] * 8,
                "DEFER_REVENUE": [1, 1, 1, 1, 2, 2, 2, 2],
                "AP": [15, 16, 17, 18, 20, 21, 22, 23],
                "ACCRUED_EXP": [3, 3, 3, 3, 4, 4, 4, 4],
            }
        )
        events = calculate_factor_events(flows, balance)
        self.assertEqual(set(events["factor"]), set(CANDIDATE_COLUMNS))
        qroe = events.loc[events["factor"].eq("dense_lit_hxz_qroe"), "value"].iloc[0]
        self.assertAlmostEqual(qroe, 9 / 50)
        ttm_gp = events.loc[events["factor"].eq("dense_lit_ttm_gp_assets"), "value"].iloc[0]
        # The first TTM observation with a one-year-lag balance is quarter
        # 8005, so its rolling window contains quarters 8002--8005.
        self.assertAlmostEqual(ttm_gp, ((110 + 120 + 130 + 140) - (65 + 70 + 75 + 80)) / 120)


if __name__ == "__main__":
    unittest.main()
