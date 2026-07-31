import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from raw_q1_minimal_factor_search import (
    CANDIDATE_COLUMNS,
    EVENT_KEYS,
    SIGNAL_COLUMNS,
    SOURCE_FIELDS,
    attach_latest_visible_prior_year,
    build_candidates_for_slice,
    calculate_event_signals,
    load_exact_q1_events,
)


def _statement_row(
    security_id: int,
    end_date: str,
    event_time: str,
    fields: list[str],
    scale: float,
) -> dict:
    row = {
        "SECURITY_ID": security_id,
        "ACT_PUBTIME": event_time,
        "END_DATE_REP": end_date,
        "END_DATE": end_date,
        "REPORT_TYPE": "Q1",
        "IS_CURRENT_PERIOD": True,
    }
    row.update({field: scale for field in fields})
    return row


class RawQ1MinimalFactorSearchTest(unittest.TestCase):
    def test_exact_join_and_visible_prior_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, fields in SOURCE_FIELDS.items():
                rows = []
                for end_date, event_time, scale in [
                    ("2023-03-31", "2023-04-20 18:00:00", 10.0),
                    ("2023-03-31", "2024-02-01 18:00:00", 20.0),
                    ("2023-03-31", "2024-06-01 18:00:00", 30.0),
                    ("2024-03-31", "2024-04-20 18:00:00", 40.0),
                ]:
                    rows.append(
                        _statement_row(
                            1, end_date, event_time, fields, scale
                        )
                    )
                pd.DataFrame(rows).to_parquet(
                    root / f"new_pit_{name}", index=False
                )
            events = load_exact_q1_events(root)
        paired = attach_latest_visible_prior_year(events)
        current = paired.loc[
            paired["END_DATE"].eq(pd.Timestamp("2024-03-31"))
        ].iloc[0]
        self.assertEqual(current["PRIOR_REVENUE"], 20.0)
        self.assertEqual(
            current["PRIOR_EVENT_TIME"],
            pd.Timestamp("2024-02-01 18:00:00"),
        )

    def test_event_signal_formulas(self):
        row = {
            "REVENUE": 120.0,
            "COGS": 60.0,
            "N_INCOME_ATTR_P": 12.0,
            "R_D_EXP": 6.0,
            "N_CF_OPERATE_A": 18.0,
            "C_FR_SALE_G_S": 126.0,
            "PUR_FIX_ASSETS_OTH": 3.0,
            "T_ASSETS": 200.0,
            "AR": 20.0,
            "INVENTORIES": 30.0,
            "AP": 10.0,
            "PRIOR_REVENUE": 100.0,
            "PRIOR_COGS": 60.0,
            "PRIOR_N_INCOME_ATTR_P": 10.0,
            "PRIOR_R_D_EXP": 5.0,
            "PRIOR_N_CF_OPERATE_A": 15.0,
            "PRIOR_C_FR_SALE_G_S": 105.0,
            "PRIOR_PUR_FIX_ASSETS_OTH": 2.0,
            "PRIOR_T_ASSETS": 180.0,
            "PRIOR_AR": 15.0,
            "PRIOR_INVENTORIES": 25.0,
            "PRIOR_AP": 8.0,
        }
        result = calculate_event_signals(pd.DataFrame([row])).iloc[0]
        self.assertAlmostEqual(result["GROSS_MARGIN"], 0.5)
        self.assertAlmostEqual(result["GROSS_MARGIN_CHANGE"], 0.1)
        self.assertAlmostEqual(result["REVENUE_GROWTH"], 0.2)
        self.assertAlmostEqual(result["PROFIT_GROWTH"], 0.2)
        self.assertAlmostEqual(result["CASH_PROFIT_CONVERSION"], 1.5)

    def test_candidate_builder_returns_expected_schema(self):
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
                row.update(
                    {
                        field: scale + index / 10
                        for index, field in enumerate(SIGNAL_COLUMNS)
                    }
                )
                rows.append(row)
        result = build_candidates_for_slice(pd.DataFrame(rows))
        self.assertEqual(
            result.columns.tolist(),
            ["TRADE_DATE", "SECURITY_ID", *CANDIDATE_COLUMNS],
        )
        self.assertTrue(
            np.isfinite(result[CANDIDATE_COLUMNS].to_numpy()).all()
        )
        self.assertEqual(
            result.duplicated(["TRADE_DATE", "SECURITY_ID"]).sum(), 0
        )


if __name__ == "__main__":
    unittest.main()
