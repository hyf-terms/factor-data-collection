import unittest

import numpy as np
import pandas as pd

from all_quarter_raw_factor_search import (
    CANDIDATE_COLUMNS,
    SIGNAL_COLUMNS,
    build_candidates_for_slice,
    calculate_event_signals,
)


class AllQuarterRawFactorSearchTest(unittest.TestCase):
    def test_raw_growth_and_ratio_formulas(self):
        row = {
            "REVENUE": 120.0,
            "COGS": 60.0,
            "N_INCOME_ATTR_P": 12.0,
            "R_D_EXP": 6.0,
            "N_CF_OPERATE_A": 18.0,
            "C_FR_SALE_G_S": 126.0,
            "PUR_FIX_ASSETS_OTH": 3.0,
            "PRIOR_REVENUE": 100.0,
            "PRIOR_COGS": 60.0,
            "PRIOR_N_INCOME_ATTR_P": 10.0,
            "PRIOR_R_D_EXP": 5.0,
            "PRIOR_N_CF_OPERATE_A": 15.0,
            "PRIOR_C_FR_SALE_G_S": 105.0,
            "PRIOR_PUR_FIX_ASSETS_OTH": 2.0,
        }
        result = calculate_event_signals(pd.DataFrame([row])).iloc[0]
        self.assertAlmostEqual(result["REVENUE_GROWTH"], 0.2)
        self.assertAlmostEqual(result["PROFIT_GROWTH"], 0.2)
        self.assertAlmostEqual(result["GROSS_MARGIN"], 0.5)
        self.assertAlmostEqual(result["CASH_MARGIN"], 0.15)

    def test_candidates_keep_metrics_without_percentile_rank(self):
        rows = []
        values = [0.10, 0.20, 0.30]
        for security_id, profit_growth in enumerate(values, start=1):
            row = {
                "TRADE_DATE": pd.Timestamp("2024-04-22"),
                "SECURITY_ID": security_id,
                "AVAILABLE_DATE": pd.Timestamp("2024-04-22"),
                "FISCAL_QUARTER": 2,
                "QUARTER_INDEX": 2024 * 4 + 2,
                "EVENT_AGE": 0,
            }
            row.update({signal: 0.2 for signal in SIGNAL_COLUMNS})
            row["PROFIT_GROWTH"] = profit_growth
            rows.append(row)
        result = build_candidates_for_slice(pd.DataFrame(rows))
        self.assertEqual(
            result.columns.tolist(),
            ["TRADE_DATE", "SECURITY_ID", *CANDIDATE_COLUMNS],
        )
        middle = result.loc[
            result["SECURITY_ID"].eq(2),
            "allq_raw_metric_profit_growth_60d",
        ].iloc[0]
        self.assertAlmostEqual(float(middle), 0.20, places=6)
        self.assertNotAlmostEqual(float(middle), 2 / 3, places=3)
        self.assertTrue(
            result["q2_raw_metric_profit_growth_60d"].notna().all()
        )
        self.assertTrue(
            result["q1_raw_metric_profit_growth_60d"].isna().all()
        )
        self.assertTrue(
            np.isfinite(
                result[
                    "allq_raw_metric_growth_cash_breadth_60d"
                ].to_numpy()
            ).all()
        )

    def test_latest_quarter_candidates_extend_availability_to_120_days(self):
        row = {
            "TRADE_DATE": pd.Timestamp("2024-08-20"),
            "SECURITY_ID": 1,
            "AVAILABLE_DATE": pd.Timestamp("2024-04-22"),
            "FISCAL_QUARTER": 1,
            "QUARTER_INDEX": 2024 * 4 + 1,
            "EVENT_AGE": 80,
        }
        row.update({signal: 0.2 for signal in SIGNAL_COLUMNS})
        result = build_candidates_for_slice(pd.DataFrame([row]))
        self.assertTrue(
            np.isnan(result["q1_raw_metric_gross_profit_growth_60d"].iloc[0])
        )
        self.assertAlmostEqual(
            float(
                result[
                    "latestq_raw_metric_gross_profit_growth_120d"
                ].iloc[0]
            ),
            0.2,
        )


if __name__ == "__main__":
    unittest.main()
