import unittest

import numpy as np
import pandas as pd

from seventh_round_literature_blends import build_candidates, candidate_names


class SeventhRoundLiteratureBlendTests(unittest.TestCase):
    def test_all_fixed_blends_are_dense(self) -> None:
        rows = []
        for date in pd.to_datetime(["2024-01-02", "2024-01-03"]):
            for security_id in range(1, 9):
                rows.append(
                    {
                        "TRADE_DATE": date,
                        "SECURITY_ID": security_id,
                        "optimized_interaction": security_id + (date.day - 2) * 0.1,
                        "dense_lit_q_gp_assets": np.log1p(security_id),
                        "dense_lit_q_op_assets": security_id ** 0.5 + (security_id % 2) * 0.2,
                    }
                )
        result = build_candidates(pd.DataFrame(rows))
        self.assertEqual(result.columns.tolist()[2:], candidate_names())
        self.assertFalse(result[candidate_names()].isna().any().any())
        self.assertEqual(len(candidate_names()), 17)


if __name__ == "__main__":
    unittest.main()
