import unittest

import numpy as np
import pandas as pd

from first_round_factor_optimization import build_candidates, grid_specs


class FirstRoundOptimizationTest(unittest.TestCase):
    def test_grid_is_dense_and_has_expected_size(self):
        rows = []
        for security_id in range(1, 11):
            rows.append(
                {
                    "TRADE_DATE": pd.Timestamp("2024-05-06"),
                    "SECURITY_ID": security_id,
                    "EVENT_AGE": security_id,
                    "GROSS_PROFIT_YOY": float(security_id),
                    "N_CF_OPA_NIA": float(11 - security_id),
                    "NI_ATTR_P_YOY": float(security_id - 4),
                    "N_CF_OPA_YOY": float(security_id + 2),
                    "dense_q1_earnings_sue_ensemble": (
                        np.nan if security_id == 10 else float(security_id - 5)
                    ),
                    "INDUSTRY": "A" if security_id <= 5 else "B",
                }
            )
        result = build_candidates(pd.DataFrame(rows))
        values = result.drop(columns=["TRADE_DATE", "SECURITY_ID"])
        self.assertEqual(values.shape[1], 11)
        self.assertFalse(values.isna().any().any())
        self.assertEqual(len(grid_specs()), 11)


if __name__ == "__main__":
    unittest.main()
