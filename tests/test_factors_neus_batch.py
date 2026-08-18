import unittest
from inspect import signature
from unittest.mock import patch

import numpy as np
import pandas as pd

from factors_neus_only import (
    KEYS,
    mad_shrink_factors_by_date,
    _residualize_one_factor,
    residualize_factor_matrix_by_date,
    summarize_factor_quality,
)
import factors_neus_only as legacy_tester
import factors_neus_only2 as strict_tester


class FactorsNeusBatchTest(unittest.TestCase):
    def test_entry_points_have_distinct_default_policies(self):
        self.assertEqual(
            signature(strict_tester.run_factor_test_pipeline)
            .parameters["cross_section_policy"]
            .default,
            "strict",
        )
        with patch.object(
            legacy_tester,
            "_strict_run_factor_test_pipeline",
            return_value={"ok": True},
        ) as runner:
            legacy_tester.run_factor_test_pipeline("f", "b", "l", "o")
        self.assertEqual(runner.call_args.kwargs["cross_section_policy"], "skip")

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

    def test_constant_date_is_skipped_without_deleting_other_dates(self):
        rows = []
        for date, constant in [
            (pd.Timestamp("2024-01-02"), True),
            (pd.Timestamp("2024-01-03"), False),
        ]:
            for security_id in range(10):
                rows.append(
                    {
                        "TRADE_DATE": date,
                        "SECURITY_ID": security_id,
                        "x1": float(security_id % 3),
                        "factor": 1.0 if constant else float(security_id),
                    }
                )
        frame = pd.DataFrame(rows)
        result, invalid, _ = residualize_factor_matrix_by_date(
            frame,
            ["factor"],
            ["x1"],
            min_cross_section=5,
        )
        self.assertEqual(invalid, [])
        self.assertFalse(
            result.loc[result["TRADE_DATE"].eq(pd.Timestamp("2024-01-02")), "factor"]
            .notna()
            .any()
        )
        self.assertTrue(
            result.loc[result["TRADE_DATE"].eq(pd.Timestamp("2024-01-03")), "factor"]
            .notna()
            .all()
        )

    def test_mad_and_quality_diagnostics_preserve_sparse_factor(self):
        frame = pd.DataFrame(
            {
                "TRADE_DATE": pd.to_datetime(
                    ["2024-01-02"] * 4 + ["2024-01-03"] * 4
                ),
                "SECURITY_ID": list(range(4)) * 2,
                "factor": [np.nan] * 4 + [1.0, 2.0, 3.0, 100.0],
            }
        )
        processed = mad_shrink_factors_by_date(frame, ["factor"])
        self.assertEqual(processed["factor"].notna().sum(), 4)
        self.assertLess(processed["factor"].max(), 100.0)
        quality = summarize_factor_quality(frame, ["factor"], 3, 2024).iloc[0]
        self.assertEqual(int(quality["all_nan_days"]), 1)
        self.assertEqual(int(quality["valid_variance_days"]), 1)
        self.assertEqual(int(quality["incomplete_days"]), 1)
        self.assertFalse(bool(quality["quant_usable"]))

    def test_dense_varying_factor_is_quant_usable(self):
        frame = pd.DataFrame(
            {
                "TRADE_DATE": pd.to_datetime(
                    ["2024-01-02"] * 4 + ["2024-01-03"] * 4
                ),
                "SECURITY_ID": list(range(4)) * 2,
                "factor": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        quality = summarize_factor_quality(frame, ["factor"], 3, 2024).iloc[0]
        self.assertEqual(int(quality["missing_rows"]), 0)
        self.assertEqual(int(quality["low_variance_days"]), 0)
        self.assertTrue(bool(quality["quant_usable"]))


if __name__ == "__main__":
    unittest.main()
