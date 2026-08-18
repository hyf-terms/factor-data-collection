import unittest

import numpy as np
import pandas as pd

from dense_q1_gross_profit_factors import (
    build_grid_for_slice,
    candidate_specs,
    robust_daily_zscore,
    select_parameters,
)


class DenseQ1GrossProfitFactorsTest(unittest.TestCase):
    def test_dense_grid_has_no_missing_values(self):
        frame = pd.DataFrame(
            {
                "TRADE_DATE": pd.to_datetime(["2024-04-25"] * 4),
                "SECURITY_ID": [1, 2, 3, 4],
                "FISCAL_QUARTER": [1, 1, 2, np.nan],
                "EVENT_AGE": [1, 20, 3, np.nan],
                "GP_GROWTH": [0.1, 0.2, -0.1, np.nan],
                "GP_ACCELERATION": [0.05, 0.10, -0.05, np.nan],
                "GP_SUE": [1.0, 2.0, -1.0, np.nan],
            }
        )
        result = build_grid_for_slice(frame, candidate_specs())
        factor_columns = result.columns.difference(
            ["TRADE_DATE", "SECURITY_ID"]
        )
        self.assertFalse(result[factor_columns].isna().any().any())
        self.assertTrue((result.loc[result.SECURITY_ID.eq(4), factor_columns] == 0).all().all())

    def test_robust_zscore_maps_missing_to_neutral(self):
        frame = pd.DataFrame(
            {
                "TRADE_DATE": pd.to_datetime(["2024-01-02"] * 5),
                "x": [1.0, 2.0, 3.0, 1000.0, np.nan],
            }
        )
        result = robust_daily_zscore(frame, ["x"])
        self.assertEqual(float(result.iloc[-1, 0]), 0.0)
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_parameter_selection_ignores_holdout_advantage(self):
        specs = pd.DataFrame(
            [
                {"factor": "a", "version": "v", "q1_weight": 0.25, "half_life": np.nan},
                {"factor": "b", "version": "v", "q1_weight": 1.0, "half_life": np.nan},
            ]
        )
        rows = []
        for year in range(2017, 2023):
            rows.extend(
                [
                    {"TRADE_DATE": pd.Timestamp(year, 6, 1), "factor": "a", "neutral_ic": 0.03},
                    {"TRADE_DATE": pd.Timestamp(year, 6, 1), "factor": "b", "neutral_ic": 0.01},
                ]
            )
        rows.extend(
            [
                {"TRADE_DATE": pd.Timestamp("2025-06-01"), "factor": "a", "neutral_ic": -0.50},
                {"TRADE_DATE": pd.Timestamp("2025-06-01"), "factor": "b", "neutral_ic": 0.90},
            ]
        )
        selected, _ = select_parameters(pd.DataFrame(rows), specs)
        self.assertEqual(selected.iloc[0]["factor"], "a")


if __name__ == "__main__":
    unittest.main()
