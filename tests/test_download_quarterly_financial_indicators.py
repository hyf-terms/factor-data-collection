import tempfile
import unittest
from pathlib import Path

import pandas as pd

import download_quarterly_financial_indicators as downloader


class QuarterlyFinancialIndicatorDownloadTest(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame(
            {
                "SECURITY_ID": [1001, 1002],
                "ID": [11, 12],
                "PARTY_ID": [101, 102],
                "PUBLISH_DATE": ["2024-04-20", "2024-04-21"],
                "END_DATE_REP": ["2024-03-31", "2024-03-31"],
                "END_DATE": ["2024-03-31", "2024-03-31"],
                "IS_NEW": [1, 1],
                "REPORT_TYPE": ["Q1", "Q1"],
                "MERGED_FLAG": ["1", "1"],
                "FISCAL_PERIOD": [3, 3],
                "UPDATE_TIME": [
                    "2024-04-20 08:00:00",
                    "2024-04-21 08:00:00",
                ],
                "GROSS_MARGIN": [20.0, 30.0],
                "NI_ATTR_P_CUT_YOY": [10.0, 15.0],
            }
        )

    def test_date_chunks_are_non_overlapping(self):
        chunks = list(
            downloader._date_chunks(
                "2024-01-01",
                "2024-01-10",
                days=4,
            )
        )
        self.assertEqual(
            chunks,
            [
                (
                    pd.Timestamp("2024-01-01"),
                    pd.Timestamp("2024-01-04"),
                ),
                (
                    pd.Timestamp("2024-01-05"),
                    pd.Timestamp("2024-01-08"),
                ),
                (
                    pd.Timestamp("2024-01-09"),
                    pd.Timestamp("2024-01-10"),
                ),
            ],
        )

    def test_date_chunks_do_not_cross_year_partition(self):
        chunks = list(
            downloader._date_chunks(
                "2024-12-30",
                "2025-01-03",
                days=4,
            )
        )
        self.assertEqual(
            chunks,
            [
                (
                    pd.Timestamp("2024-12-30"),
                    pd.Timestamp("2024-12-31"),
                ),
                (
                    pd.Timestamp("2025-01-01"),
                    pd.Timestamp("2025-01-03"),
                ),
            ],
        )

    def test_prepare_chunk_types_and_current_period(self):
        result = downloader._prepare_chunk(self.raw)
        self.assertEqual(len(result), 2)
        self.assertTrue(result["IS_CURRENT_PERIOD"].all())
        self.assertEqual(str(result["FISCAL_PERIOD"].dtype), "Int16")
        self.assertEqual(str(result["IS_NEW"].dtype), "Int8")
        self.assertEqual(str(result["GROSS_MARGIN"].dtype), "float64")
        self.assertTrue(
            pd.api.types.is_datetime64_ns_dtype(result["PUBLISH_DATE"])
        )

    def test_duplicate_join_event_is_rejected(self):
        duplicate = pd.concat(
            [self.raw.iloc[[0]], self.raw.iloc[[0]]],
            ignore_index=True,
        )
        duplicate.loc[1, "GROSS_MARGIN"] = 21.0
        with self.assertRaises(ValueError):
            downloader._prepare_chunk(duplicate)

    def test_partition_path_is_year_partitioned(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = downloader._partition_path(
                Path(temporary),
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-04-01"),
            )
        self.assertEqual(target.parent.name, "year=2024")
        self.assertEqual(
            target.name,
            "part-20240101-20240401.parquet",
        )


if __name__ == "__main__":
    unittest.main()
