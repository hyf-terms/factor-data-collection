import numpy as np
import pandas as pd
import unittest

from third_round_new_factor_optimization import (
    PANEL_FIELDS,
    VENDOR_FIELDS,
    _daily_orthogonal_residual,
    build_candidates,
    build_incremental_blends,
    candidate_names,
)


class ThirdRoundTests(unittest.TestCase):
    def test_orthogonal_residual_has_zero_daily_covariance(self) -> None:
        dates = pd.Series(pd.to_datetime(["2024-01-02"] * 5 + ["2024-01-03"] * 5))
        anchor = pd.Series(list(range(5)) * 2, dtype=float)
        signal = 2 * anchor + pd.Series([1, -1, 0, 1, -1] * 2, dtype=float)
        residual = _daily_orthogonal_residual(signal, anchor, dates)
        for _, positions in dates.groupby(dates).groups.items():
            self.assertLess(
                abs(np.cov(residual.loc[positions], anchor.loc[positions], ddof=0)[0, 1]),
                1e-10,
            )

    def test_candidates_are_dense_and_include_incremental_versions(self) -> None:
        rows = []
        for date in pd.to_datetime(["2024-01-02", "2024-01-03"]):
            for security in range(1, 8):
                row = {"TRADE_DATE": date, "SECURITY_ID": security}
                for offset, field in enumerate(VENDOR_FIELDS + PANEL_FIELDS, start=1):
                    row[field] = security * offset + ((security + offset) % 3) ** 2 + (date.day % 2)
                rows.append(row)
        result = build_candidates(pd.DataFrame(rows))
        self.assertTrue(set(candidate_names()).issubset(result.columns))
        self.assertFalse(result[candidate_names()].isna().any().any())
        self.assertTrue((result.groupby("TRADE_DATE")[candidate_names()[:13]].nunique() > 1).all().all())

    def test_incremental_blends_use_requested_weights(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keys = {"TRADE_DATE": pd.to_datetime(["2024-01-02"]), "SECURITY_ID": [1]}
            pd.DataFrame({**keys, "optimized_interaction": [2.0]}).to_parquet(root / "r2.parquet")
            pd.DataFrame({**keys, "r3_margin_efficiency_incremental": [4.0]}).to_parquet(root / "r3.parquet")
            build_incremental_blends(root / "r2.parquet", root / "r3.parquet", root / "out.parquet")
            result = pd.read_parquet(root / "out.parquet")
            self.assertAlmostEqual(result.loc[0, "r3_interaction_margin_w05"], 2.1)
            self.assertAlmostEqual(result.loc[0, "r3_interaction_margin_w20"], 2.4)
