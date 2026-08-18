import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eighth_round_literature_factors import (
    CANDIDATE_COLUMNS,
    _safe_ratio,
    fill_after_test,
)


class EighthRoundLiteratureFactorTests(unittest.TestCase):
    def test_safe_ratio_uses_absolute_denominator(self) -> None:
        numerator = pd.Series([2.0, -2.0, 1.0])
        denominator = pd.Series([-4.0, 4.0, 0.0])
        result = _safe_ratio(numerator, denominator, absolute_denominator=True)
        self.assertAlmostEqual(result.iloc[0], 0.5)
        self.assertAlmostEqual(result.iloc[1], -0.5)
        self.assertTrue(np.isnan(result.iloc[2]))

    def test_fill_requires_complete_sparse_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for date in pd.to_datetime(["2024-01-02", "2024-01-03"]):
                for security_id in (1, 2, 3):
                    row = {"TRADE_DATE": date, "SECURITY_ID": security_id}
                    for position, factor in enumerate(CANDIDATE_COLUMNS):
                        row[factor] = np.nan if security_id == 3 else float(position + security_id)
                    rows.append(row)
            sparse = root / "sparse.parquet"
            pd.DataFrame(rows).to_parquet(sparse, index=False)
            summary = root / "ic.csv"
            pd.DataFrame({"factor": CANDIDATE_COLUMNS}).to_csv(summary, index=False)
            output = root / "filled.parquet"
            report = root / "report.csv"
            fill_after_test(sparse, summary, output, report)
            result = pd.read_parquet(output)
            self.assertFalse(result[CANDIDATE_COLUMNS].isna().any().any())
            self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()
