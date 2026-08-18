import unittest

import pandas as pd

from sixth_round_controlled_composites import ANCHOR, COMPONENTS, build_candidates, candidate_names


class SixthRoundTests(unittest.TestCase):
    def test_all_controlled_composites_are_dense(self) -> None:
        rows = []
        for date in pd.to_datetime(["2024-01-02", "2024-01-03"]):
            for security in range(1, 9):
                row = {"TRADE_DATE": date, "SECURITY_ID": security}
                for offset, field in enumerate([*COMPONENTS.values(), ANCHOR], start=1):
                    row[field] = security * offset + ((security + offset) % 4) ** 2
                rows.append(row)
        equal = {field: 0.2 for field in COMPONENTS.values()}
        parameters = {"weights": {
            "ridge025": equal, "ridge025_equal50": equal,
            "ridge100": equal, "ridge100_equal50": equal,
        }}
        result = build_candidates(pd.DataFrame(rows), parameters)
        self.assertEqual(result.columns.tolist(), ["TRADE_DATE", "SECURITY_ID", *candidate_names()])
        self.assertFalse(result[candidate_names()].isna().any().any())


if __name__ == "__main__":
    unittest.main()
