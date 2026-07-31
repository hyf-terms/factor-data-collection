import unittest

import numpy as np
import pandas as pd

from factors_neus_only import (
    KEYS,
    _residualize_one_factor,
    residualize_factor_matrix_by_date,
)


class FactorsNeusBatchTest(unittest.TestCase):
    def test_batch_residuals_match_single_factor_results(self):
        rng = np.random.default_rng(7)
        rows = []
        exposure_cols = ["x1", "x2"]
        factor_cols = ["f1", "f2", "f3"]
        for date in pd.to_datetime(["2024-01-02", "2024-01-03"]):
            for security_id in range(12):
                x1, x2 = rng.normal(size=2)
                rows.append(
                    {
                        "TRADE_DATE": date,
                        "SECURITY_ID": security_id,
                        "x1": x1,
                        "x2": x2,
                        "f1": 2 * x1 - x2 + rng.normal(),
                        "f2": -x1 + 3 * x2 + rng.normal(),
                        "f3": np.nan
                        if security_id < 2
                        else x1 + x2 + rng.normal(),
                    }
                )
        frame = pd.DataFrame(rows)
        batch, invalid, moment = residualize_factor_matrix_by_date(
            frame, factor_cols, exposure_cols, min_cross_section=5
        )
        self.assertEqual(invalid, [])
        self.assertLess(moment, 1e-10)
        for factor in factor_cols:
            single, _ = _residualize_one_factor(
                frame, factor, exposure_cols, min_cross_section=5
            )
            compared = batch[KEYS + [factor]].dropna().merge(
                single,
                on=KEYS,
                suffixes=("_batch", "_single"),
                validate="one_to_one",
            )
            np.testing.assert_allclose(
                compared[f"{factor}_batch"],
                compared[f"{factor}_single"],
                atol=1e-10,
                rtol=1e-10,
            )


if __name__ == "__main__":
    unittest.main()
