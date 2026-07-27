import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import download_new_pit as downloader


class DownloadNewPitParquetTest(unittest.TestCase):
    def setUp(self):
        self.spec = downloader.PIT_TABLES[1]
        self.raw = pd.DataFrame(
            {
                "ID": [10, 11],
                "PARTY_ID": [100, 101],
                "SECURITY_ID": [1000, 1001],
                "TICKER_SYMBOL": ["000001", "000002"],
                "EXCHANGE_CD": ["XSHE", "XSHE"],
                "PUBLISH_DATE": ["2024-04-20", "2024-04-21"],
                "ACT_PUBTIME": ["2024-04-20 08:00", "2024-04-21 20:00"],
                "END_DATE": ["2023-12-31", "2023-12-31"],
                "END_DATE_REP": ["2023-12-31", "2023-12-31"],
                "UPDATE_TIME": ["2024-04-20 09:00", "2024-04-21 21:00"],
                "REPORT_TYPE": ["A", "A"],
                "FISCAL_PERIOD": [12, 12],
                "MERGED_FLAG": ["1", "1"],
                "REVENUE": [200.0, 300.0],
                "COGS": [120.0, 180.0],
            }
        )

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow"),
        "需要pyarrow执行Parquet读写集成测试",
    )
    def test_partition_round_trip_and_resume(self):
        prepared = downloader._prepare_chunk(self.raw, self.spec)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(downloader, "_read_chunk", return_value=prepared):
                first = downloader.download_one_table(
                    conn=object(),
                    output_dir=root,
                    spec=self.spec,
                    start_date="2024-01-01",
                    end_date="2024-12-31",
                    resume=True,
                )
            self.assertEqual(first["rows"], 2)
            self.assertEqual(first["files"], 1)

            loaded = downloader.read_pit_dataset(
                "new_pit_income",
                output_dir=root,
            )
            self.assertEqual(len(loaded), 2)
            self.assertIn("ACT_PUBTIME", loaded.columns)
            self.assertTrue(loaded["IS_CURRENT_PERIOD"].all())

            with patch.object(downloader, "_read_chunk") as read_mock:
                second = downloader.download_one_table(
                    conn=object(),
                    output_dir=root,
                    spec=self.spec,
                    start_date="2024-01-01",
                    end_date="2024-12-31",
                    resume=True,
                )
            read_mock.assert_not_called()
            self.assertEqual(second["rows"], 0)

    def test_resume_stops_before_database_query(self):
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(
                    downloader,
                    "_last_stored_date",
                    return_value=pd.Timestamp("2024-12-31"),
                ),
                patch.object(downloader, "_read_chunk") as read_mock,
            ):
                result = downloader.download_one_table(
                    conn=object(),
                    output_dir=temporary,
                    spec=self.spec,
                    start_date="2024-01-01",
                    end_date="2024-12-31",
                    resume=True,
                )
        read_mock.assert_not_called()
        self.assertEqual(result["rows"], 0)

    def test_duplicate_pit_event_is_rejected(self):
        duplicate = pd.concat([self.raw.iloc[[0]], self.raw.iloc[[0]]])
        with self.assertRaises(ValueError):
            downloader._prepare_chunk(duplicate, self.spec)


if __name__ == "__main__":
    unittest.main()
