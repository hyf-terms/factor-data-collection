import unittest

import pandas as pd

from fourth_round_full_quarter_factors import (
    ANCHOR,
    PANEL_FIELDS,
    RAW_SIGNALS,
    VENDOR_FIELDS,
    build_candidates,
    candidate_names,
)


class FourthRoundTests(unittest.TestCase):
    def test_all_four_groups_are_dense_and_have_incremental_versions(self) -> None:
        rows = []
        fields = RAW_SIGNALS + VENDOR_FIELDS + PANEL_FIELDS + [ANCHOR]
        for date in pd.to_datetime(["2024-01-02", "2024-01-03"]):
            for security in range(1, 9):
                row = {"TRADE_DATE": date, "SECURITY_ID": security}
                for offset, field in enumerate(fields, start=1):
                    row[field] = security * offset + ((security + offset) % 4) ** 2
                rows.append(row)
        result = build_candidates(pd.DataFrame(rows))
        self.assertTrue(set(candidate_names()).issubset(result.columns))
        self.assertFalse(result[candidate_names()].isna().any().any())
        for family in ["cost_stickiness", "operating_leverage", "cfo_surprise", "sales_collection"]:
            self.assertIn(f"r4_{family}_incremental", result)


if __name__ == "__main__":
    unittest.main()
