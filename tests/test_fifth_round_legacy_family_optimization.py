import unittest

import pandas as pd

from fifth_round_legacy_family_optimization import (
    EXISTING_SIX,
    LEGACY_COMPONENTS,
    FULL_RAW_SIGNALS,
    SUE_SIGNALS,
    build_candidates,
    candidate_names,
    repair_financial_candidates,
)


class FifthRoundTests(unittest.TestCase):
    def test_dense_families_and_existing_six_share_one_panel(self) -> None:
        rows = []
        fields = [*FULL_RAW_SIGNALS, *SUE_SIGNALS, *LEGACY_COMPONENTS, *EXISTING_SIX]
        for date in pd.to_datetime(["2024-01-02", "2024-01-03"]):
            for security in range(1, 9):
                row = {"TRADE_DATE": date, "SECURITY_ID": security}
                for offset, field in enumerate(fields, start=1):
                    row[field] = security * offset + ((security + offset) % 4) ** 2
                rows.append(row)
        result = build_candidates(pd.DataFrame(rows))
        self.assertEqual(result.columns.tolist(), ["TRADE_DATE", "SECURITY_ID", *candidate_names()])
        self.assertFalse(result[candidate_names()].isna().any().any())
        self.assertTrue(set(EXISTING_SIX).issubset(result.columns))

    def test_financial_repair_uses_dense_anchor(self) -> None:
        import tempfile
        from pathlib import Path
        from fifth_round_legacy_family_optimization import REJECTED_FINANCIAL

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"TRADE_DATE": [pd.Timestamp("2024-01-02")], "SECURITY_ID": [1], "dense_q1_net_income_sue": [2.0]}
            row.update({factor: [1.0] for factor in REJECTED_FINANCIAL})
            pd.DataFrame(row).to_parquet(root / "grid.parquet")
            repair_financial_candidates(root / "grid.parquet", root / "out.parquet")
            result = pd.read_parquet(root / "out.parquet")
            self.assertAlmostEqual(result.loc[0, "dense_allq_financial_composite_anchor10"], 1.1)


if __name__ == "__main__":
    unittest.main()
