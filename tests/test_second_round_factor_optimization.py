import unittest

import numpy as np
import pandas as pd

from second_round_factor_optimization import build_candidates, grid_specs


class SecondRoundOptimizationTest(unittest.TestCase):
    def test_standardized_candidate_grid_is_dense(self):
        rows = []
        for security_id in range(1, 21):
            row = {
                "TRADE_DATE": pd.Timestamp("2024-05-06"),
                "SECURITY_ID": security_id,
                "INDUSTRY": "A" if security_id <= 10 else "B",
            }
            for index, field in enumerate(
                [
                    "GROSS_PROFIT_YOY", "NP_MARGIN_YOY", "P_COST_EXP",
                    "PERIOD_EXP_TR", "AR_REC_R", "AR_R", "N_CF_OPA_YOY",
                    "dense_q1_earnings_sue_ensemble", "dense_q1_gross_profit_sue",
                ]
            ):
                row[field] = np.nan if security_id == 20 else float(security_id + index)
            rows.append(row)
        result = build_candidates(pd.DataFrame(rows))
        values = result.drop(columns=["TRADE_DATE", "SECURITY_ID"])
        self.assertEqual(values.shape[1], len(grid_specs()))
        self.assertFalse(values.isna().any().any())


if __name__ == "__main__":
    unittest.main()
