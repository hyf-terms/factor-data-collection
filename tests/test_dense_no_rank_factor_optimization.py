import unittest

import numpy as np
import pandas as pd

from dense_no_rank_factor_optimization import build_grid_for_slice, grid_specs


class DenseNoRankOptimizationTest(unittest.TestCase):
    def test_all_grid_values_are_dense(self):
        rows = []
        for security_id in range(1, 6):
            row = {
                "TRADE_DATE": pd.Timestamp("2024-05-06"),
                "SECURITY_ID": security_id,
                "RAW_QUARTER": 1 if security_id < 3 else 2,
                "RAW_AGE": security_id,
            }
            for signal in [
                "PROFIT_GROWTH",
                "GROWTH_CASH_BREADTH",
                "MARGIN_CASH_IMPROVEMENT",
                "RD_GROWTH_EFFICIENCY",
                "GROSS_PROFIT_GROWTH",
            ]:
                row[signal] = float(security_id) if security_id < 5 else np.nan
            for signal in [
                "NET_INCOME_SUE",
                "DEDUCTED_INCOME_SUE",
                "OPERATING_PROFIT_SUE",
                "GROSS_PROFIT_SUE",
            ]:
                row[signal] = float(security_id - 2) if security_id < 5 else np.nan
                row[f"{signal}_QUARTER"] = row["RAW_QUARTER"]
                row[f"{signal}_AGE"] = row["RAW_AGE"]
            rows.append(row)
        result = build_grid_for_slice(pd.DataFrame(rows), grid_specs())
        values = result.drop(columns=["TRADE_DATE", "SECURITY_ID"])
        self.assertFalse(values.isna().any().any())
        self.assertEqual(values.shape[1], 40)


if __name__ == "__main__":
    unittest.main()
