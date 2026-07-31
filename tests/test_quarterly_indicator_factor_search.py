import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quarterly_indicator_factor_search import (
    CANDIDATE_COLUMNS,
    HYBRID_BASES,
    VALUE_FIELDS,
    build_candidates_for_slice,
    load_indicator_events,
    prepare_available_events,
)


def _event(
    security_id: int,
    event_id: int,
    publish_date: str,
    end_date: str,
    report_type: str,
) -> dict:
    row = {
        "SECURITY_ID": security_id,
        "ID": event_id,
        "PUBLISH_DATE": publish_date,
        "END_DATE_REP": end_date,
        "END_DATE": end_date,
        "UPDATE_TIME": publish_date,
        "REPORT_TYPE": report_type,
        "FISCAL_PERIOD": 3,
    }
    row.update({field: 1.0 for field in VALUE_FIELDS})
    return row


class QuarterlyIndicatorFactorSearchTest(unittest.TestCase):
    def test_events_start_next_day_and_reject_old_revision(self):
        raw = pd.DataFrame(
            [
                _event(1, 1, "2024-04-20", "2024-03-31", "Q1"),
                _event(1, 2, "2024-08-20", "2024-06-30", "S1"),
                _event(1, 3, "2024-09-01", "2024-03-31", "Q1"),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.parquet"
            raw.to_parquet(path, index=False)
            events = load_indicator_events(path)
        calendar = pd.DatetimeIndex(
            ["2024-04-22", "2024-04-23", "2024-08-21", "2024-09-02"]
        )
        prepared = prepare_available_events(events, calendar)
        self.assertEqual(
            prepared["AVAILABLE_DATE"].min(),
            pd.Timestamp("2024-04-22"),
        )
        self.assertEqual(
            set(prepared["QUARTER_INDEX"]),
            {2024 * 4 + 1, 2024 * 4 + 2},
        )
        self.assertNotIn(
            pd.Timestamp("2024-09-02"),
            set(prepared["AVAILABLE_DATE"]),
        )

    def test_candidate_builder_returns_all_columns(self):
        rows = []
        for date in pd.to_datetime(["2024-04-22", "2024-04-23"]):
            for security_id, scale in [(1, 1.0), (2, 2.0), (3, 3.0)]:
                row = {
                    "TRADE_DATE": date,
                    "SECURITY_ID": security_id,
                    "AVAILABLE_DATE": pd.Timestamp("2024-04-22"),
                    "FISCAL_QUARTER": 1,
                    "QUARTER_INDEX": 2024 * 4 + 1,
                    "EVENT_AGE": 0 if date.day == 22 else 1,
                }
                row.update({field: scale for field in VALUE_FIELDS})
                row.update({field: scale for field in HYBRID_BASES})
                rows.append(row)
        result = build_candidates_for_slice(pd.DataFrame(rows))
        self.assertEqual(
            result.columns.tolist(),
            ["TRADE_DATE", "SECURITY_ID", *CANDIDATE_COLUMNS],
        )
        gated = [
            "q1_joint_surprise_cash_confirmed_60d",
            "q1_all_profit_cash_confirmed_60d",
        ]
        complete = result[
            [column for column in CANDIDATE_COLUMNS if column not in gated]
        ].notna().all()
        self.assertTrue(
            complete.all(),
            msg=f"Missing candidates: {complete.index[~complete].tolist()}",
        )
        self.assertTrue(result[gated].notna().any().all())
        self.assertEqual(
            result.duplicated(["TRADE_DATE", "SECURITY_ID"]).sum(),
            0,
        )


if __name__ == "__main__":
    unittest.main()
