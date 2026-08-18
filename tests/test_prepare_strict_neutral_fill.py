import unittest

import numpy as np
import pandas as pd

from prepare_strict_neutral_fill import fill_daily_neutral


class StrictNeutralFillTest(unittest.TestCase):
    def test_missing_values_receive_daily_median(self):
        frame = pd.DataFrame(
            {
                "TRADE_DATE": pd.to_datetime(["2024-01-02"] * 4),
                "factor": [1.0, 2.0, 4.0, np.nan],
            }
        )
        result, missing = fill_daily_neutral(frame, "factor")
        self.assertEqual(missing, 1)
        self.assertEqual(float(result.iloc[-1]), 2.0)

    def test_all_missing_date_is_rejected(self):
        frame = pd.DataFrame(
            {
                "TRADE_DATE": pd.to_datetime(["2024-01-02"] * 3),
                "factor": [np.nan, np.nan, np.nan],
            }
        )
        with self.assertRaisesRegex(ValueError, "整日全空"):
            fill_daily_neutral(frame, "factor")


if __name__ == "__main__":
    unittest.main()
