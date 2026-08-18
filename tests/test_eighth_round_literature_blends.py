import unittest

import numpy as np
import pandas as pd

from eighth_round_literature_blends import build_candidates, candidate_names


class EighthRoundLiteratureBlendTests(unittest.TestCase):
    def test_candidates_are_dense_and_have_expected_columns(self) -> None:
        rows = []
        for date in pd.to_datetime(["2024-01-02", "2024-01-03"]):
            for security_id in range(1, 21):
                rows.append(
                    {
                        "TRADE_DATE": date,
                        "SECURITY_ID": security_id,
                        "optimized_interaction": float(security_id),
                        "r8_ab_sales_sga_gap": np.log1p(security_id) + date.day * 0.01,
                        "r8_ab_sales_receivable_gap": security_id ** 0.5 + (security_id % 3) * 0.2,
                    }
                )
        result = build_candidates(pd.DataFrame(rows))
        self.assertEqual(result.columns.tolist()[2:], candidate_names())
        self.assertFalse(result[candidate_names()].isna().any().any())
        self.assertGreater(result["r8_anchor_ab_sga_w05"].std(), 0)


if __name__ == "__main__":
    unittest.main()
