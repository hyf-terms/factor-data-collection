import unittest

import pandas as pd

from gross_profitability_factor import (
    assign_available_trade_date,
    build_daily_gross_profitability,
    build_gross_profitability_events,
)


class GrossProfitabilityFactorTest(unittest.TestCase):
    def setUp(self):
        common = {
            "SECURITY_ID": [1, 2],
            "END_DATE": ["2023-12-31", "2023-12-31"],
            "END_DATE_REP": ["2023-12-31", "2023-12-31"],
            "REPORT_TYPE": ["A", "A"],
            "MERGED_FLAG": ["1", "1"],
            "IS_CURRENT_PERIOD": [True, True],
        }
        self.income = pd.DataFrame(
            common
            | {
                "ACT_PUBTIME": ["2024-04-29 08:00", "2024-04-29 20:00"],
                "REVENUE": [200.0, 120.0],
                "COGS": [120.0, 90.0],
            }
        )
        self.balance = pd.DataFrame(
            common
            | {
                "ACT_PUBTIME": ["2024-04-29 08:00", "2024-04-29 20:00"],
                "T_ASSETS": [400.0, 300.0],
            }
        )

    def test_paper_formula(self):
        result = build_gross_profitability_events(self.income, self.balance)
        values = result.set_index("SECURITY_ID")["GROSS_PROFITABILITY"]
        self.assertAlmostEqual(values.loc[1], 0.20)
        self.assertAlmostEqual(values.loc[2], 0.10)

    def test_after_close_moves_to_next_trade_day(self):
        events = build_gross_profitability_events(self.income, self.balance)
        result = assign_available_trade_date(
            events,
            ["2024-04-29", "2024-04-30", "2024-05-06"],
        ).set_index("SECURITY_ID")
        self.assertEqual(result.loc[1, "AVAILABLE_DATE"], pd.Timestamp("2024-04-29"))
        self.assertEqual(result.loc[2, "AVAILABLE_DATE"], pd.Timestamp("2024-04-30"))

    def test_daily_panel_does_not_look_ahead(self):
        universe = pd.DataFrame(
            {
                "TRADE_DATE": [
                    "2024-04-29",
                    "2024-04-29",
                    "2024-04-30",
                    "2024-04-30",
                ],
                "SECURITY_ID": [1, 2, 1, 2],
            }
        )
        result = build_daily_gross_profitability(
            self.income,
            self.balance,
            universe,
            winsor_limits=None,
        ).set_index(["TRADE_DATE", "SECURITY_ID"])
        self.assertAlmostEqual(
            result.loc[(pd.Timestamp("2024-04-29"), 1), "GROSS_PROFITABILITY"],
            0.20,
        )
        self.assertTrue(
            pd.isna(
                result.loc[
                    (pd.Timestamp("2024-04-29"), 2),
                    "GROSS_PROFITABILITY",
                ]
            )
        )
        self.assertAlmostEqual(
            result.loc[(pd.Timestamp("2024-04-30"), 2), "GROSS_PROFITABILITY"],
            0.10,
        )

    def test_old_report_revision_does_not_replace_new_report(self):
        income = pd.concat(
            [
                self.income.iloc[[0]],
                self.income.iloc[[0]].assign(
                    END_DATE="2024-12-31",
                    END_DATE_REP="2024-12-31",
                    ACT_PUBTIME="2025-04-20 08:00",
                    REVENUE=300.0,
                    COGS=150.0,
                ),
                self.income.iloc[[0]].assign(
                    ACT_PUBTIME="2025-05-01 08:00",
                    REVENUE=240.0,
                    COGS=120.0,
                ),
            ],
            ignore_index=True,
        )
        balance = pd.concat(
            [
                self.balance.iloc[[0]],
                self.balance.iloc[[0]].assign(
                    END_DATE="2024-12-31",
                    END_DATE_REP="2024-12-31",
                    ACT_PUBTIME="2025-04-20 08:00",
                    T_ASSETS=500.0,
                ),
                self.balance.iloc[[0]].assign(
                    ACT_PUBTIME="2025-05-01 08:00",
                    T_ASSETS=400.0,
                ),
            ],
            ignore_index=True,
        )
        universe = pd.DataFrame(
            {
                "TRADE_DATE": ["2025-04-21", "2025-05-06"],
                "SECURITY_ID": [1, 1],
            }
        )
        result = build_daily_gross_profitability(
            income,
            balance,
            universe,
            winsor_limits=None,
        )
        self.assertEqual(
            result.iloc[-1]["END_DATE"],
            pd.Timestamp("2024-12-31"),
        )
        self.assertAlmostEqual(result.iloc[-1]["GROSS_PROFITABILITY"], 0.30)


if __name__ == "__main__":
    unittest.main()
