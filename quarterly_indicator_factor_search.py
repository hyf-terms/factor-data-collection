"""Build strict-PIT factor candidates from DataYes quarterly indicators.

The source is ``fdmt_main_data_q_pit`` exported by
``download_quarterly_financial_indicators.py``.  Candidate generation never
reads labels or future returns.  Because the source has a publication date
but no publication time, every event becomes usable on the next trading day.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from event_financial_factor_search import KEYS, _normalize_panel
from pead_sue_factor import assign_available_trade_date


BASE_DIR = Path(__file__).resolve().parent
FULL_WINDOW = 120
Q1_WINDOW = 60

VALUE_FIELDS = [
    "N_CF_OPA_NIA",
    "N_CF_OPA_R",
    "CFSGS_R",
    "N_CF_OPA_NIA_YOY",
    "N_CF_OPA_NIA_QOQ",
    "NI_ATTR_P_CUT_YOY",
    "NI_ATTR_P_CUT_QOQ",
    "NI_CUT_NI",
    "GROSS_MARGIN",
    "NP_MARGIN",
    "ROE",
    "ROA",
    "GROSS_PROFIT_YOY",
    "REVENUE_YOY",
    "COGS_YOY",
    "N_CF_OPA_YOY",
    "C_FR_SALE_G_S_YOY",
    "R_D_EXP_TR",
    "R_D_EXP_YOY",
    "AR_R",
    "AR_REC_R",
    "NR_AR_R",
    "N_CF_OPA_CL",
    "N_CF_OPA_LIAB",
    "N_CF_OPA_ID",
    "N_CF_OPA_ND",
    "OPA_P_TR",
    "VAL_CHG_P_TR",
]

HYBRID_BASES = [
    "q1_joint_earnings_revenue",
    "q1_all_profit_surprises",
    "q1_financial_60d",
]

CANDIDATE_COLUMNS = [
    "cash_profit_conversion_120d",
    "cash_margin_120d",
    "deducted_profit_share_120d",
    "cash_earnings_growth_gap_120d",
    "cash_quality_composite_120d",
    "growth_cash_breadth_120d",
    "growth_consistency_120d",
    "gross_profit_cash_confirmation_120d",
    "cost_discipline_growth_120d",
    "receivable_collection_quality_120d",
    "cash_debt_coverage_120d",
    "profit_composition_quality_120d",
    "rd_growth_efficiency_120d",
    "margin_cash_improvement_120d",
    "q1_cash_profit_conversion_60d",
    "q1_cash_quality_composite_60d",
    "q1_growth_cash_breadth_60d",
    "q1_growth_consistency_60d",
    "q1_gross_profit_cash_confirmation_60d",
    "q1_cost_discipline_growth_60d",
    "q1_receivable_collection_quality_60d",
    "q1_cash_debt_coverage_60d",
    "q1_profit_composition_quality_60d",
    "q1_rd_growth_efficiency_60d",
    "q1_margin_cash_improvement_60d",
    "q1_joint_surprise_cash_quality_60d",
    "q1_all_profit_cash_quality_60d",
    "q1_financial_cash_breadth_60d",
    "q1_joint_surprise_growth_cash_60d",
    "q1_all_profit_growth_consistency_60d",
    "q1_joint_surprise_cash_confirmed_60d",
    "q1_all_profit_cash_confirmed_60d",
]


def load_indicator_events(dataset: Path) -> pd.DataFrame:
    columns = [
        "SECURITY_ID",
        "ID",
        "PUBLISH_DATE",
        "END_DATE_REP",
        "END_DATE",
        "UPDATE_TIME",
        "REPORT_TYPE",
        "FISCAL_PERIOD",
        *VALUE_FIELDS,
    ]
    events = pd.read_parquet(dataset, columns=columns)
    for column in [
        "PUBLISH_DATE",
        "END_DATE_REP",
        "END_DATE",
        "UPDATE_TIME",
    ]:
        events[column] = pd.to_datetime(events[column], errors="coerce")
    events = events.loc[
        events["END_DATE"].eq(events["END_DATE_REP"])
    ].copy()
    quarter_map = {"Q1": 1, "S1": 2, "Q3": 3, "A": 4}
    events["FISCAL_QUARTER"] = (
        events["REPORT_TYPE"].astype("string").map(quarter_map)
    )
    events["FISCAL_YEAR"] = events["END_DATE"].dt.year
    events["QUARTER_INDEX"] = (
        events["FISCAL_YEAR"] * 4 + events["FISCAL_QUARTER"]
    )
    events["EVENT_TIME"] = (
        events["PUBLISH_DATE"].dt.normalize()
        + pd.to_timedelta(86_399, unit="s")
    )
    events = events.dropna(
        subset=[
            "SECURITY_ID",
            "ID",
            "EVENT_TIME",
            "FISCAL_QUARTER",
            "QUARTER_INDEX",
        ]
    ).copy()
    events["SECURITY_ID"] = pd.to_numeric(
        events["SECURITY_ID"], errors="raise"
    ).astype("int64")
    events["ID"] = pd.to_numeric(events["ID"], errors="raise").astype(
        "int64"
    )
    events["FISCAL_QUARTER"] = events["FISCAL_QUARTER"].astype("int8")
    events["QUARTER_INDEX"] = events["QUARTER_INDEX"].astype("int32")
    for column in VALUE_FIELDS:
        events[column] = pd.to_numeric(events[column], errors="coerce")

    # Several revisions can share one publication day.  Keep the last
    # database revision for the same security and fiscal quarter that day.
    events = events.sort_values(
        [
            "SECURITY_ID",
            "PUBLISH_DATE",
            "QUARTER_INDEX",
            "UPDATE_TIME",
            "ID",
        ],
        na_position="first",
    )
    events = events.drop_duplicates(
        [
            "SECURITY_ID",
            "PUBLISH_DATE",
            "QUARTER_INDEX",
        ],
        keep="last",
    )
    return events.reset_index(drop=True)


def prepare_available_events(
    events: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Assign next-trading-day availability and reject stale revisions."""
    available = assign_available_trade_date(events, calendar)
    available = available.sort_values(
        [
            "SECURITY_ID",
            "AVAILABLE_DATE",
            "EVENT_TIME",
            "QUARTER_INDEX",
            "ID",
        ]
    )
    newest = available.groupby(
        "SECURITY_ID", sort=False
    )["QUARTER_INDEX"].cummax()
    clean = available.loc[available["QUARTER_INDEX"].eq(newest)].copy()
    clean = clean.drop_duplicates(
        ["SECURITY_ID", "AVAILABLE_DATE"],
        keep="last",
    )
    return clean[
        [
            "SECURITY_ID",
            "AVAILABLE_DATE",
            "FISCAL_QUARTER",
            "QUARTER_INDEX",
            *VALUE_FIELDS,
        ]
    ].reset_index(drop=True)


def map_events_to_panel(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    calendar_ordinal: pd.Series,
) -> pd.DataFrame:
    left = panel.sort_values(["TRADE_DATE", "SECURITY_ID"])
    right = events.sort_values(["AVAILABLE_DATE", "SECURITY_ID"])
    joined = pd.merge_asof(
        left,
        right,
        by="SECURITY_ID",
        left_on="TRADE_DATE",
        right_on="AVAILABLE_DATE",
        direction="backward",
    )
    joined["EVENT_AGE"] = (
        joined["TRADE_DATE"].map(calendar_ordinal)
        - joined["AVAILABLE_DATE"].map(calendar_ordinal)
    )
    return joined.sort_values(KEYS).reset_index(drop=True)


def _winsorize_daily(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    values = data[columns].apply(pd.to_numeric, errors="coerce")
    grouped = values.groupby(data["TRADE_DATE"], sort=False)
    lower = grouped.transform(lambda group: group.quantile(0.01))
    upper = grouped.transform(lambda group: group.quantile(0.99))
    return values.clip(lower=lower, upper=upper)


def _rank_daily(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return data.groupby("TRADE_DATE", sort=False)[columns].rank(
        method="average",
        pct=True,
    )


def _mean_complete(ranks: pd.DataFrame, columns: list[str]) -> pd.Series:
    return ranks[columns].mean(axis=1, skipna=False)


def build_candidates_for_slice(mapped: pd.DataFrame) -> pd.DataFrame:
    data = mapped.copy()
    data[VALUE_FIELDS] = _winsorize_daily(data, VALUE_FIELDS)
    data["CASH_EARNINGS_GROWTH_GAP"] = (
        data["N_CF_OPA_YOY"] - data["NI_ATTR_P_CUT_YOY"]
    )
    data["COST_DISCIPLINE"] = (
        data["REVENUE_YOY"] - data["COGS_YOY"]
    )
    data["NEG_ABS_VALUE_CHANGE"] = -data["VAL_CHG_P_TR"].abs()
    derived = [
        "CASH_EARNINGS_GROWTH_GAP",
        "COST_DISCIPLINE",
        "NEG_ABS_VALUE_CHANGE",
    ]
    rank_columns = VALUE_FIELDS + derived + HYBRID_BASES
    ranks = _rank_daily(data, rank_columns)

    cash_quality = _mean_complete(
        ranks,
        ["N_CF_OPA_NIA", "N_CF_OPA_R", "CFSGS_R", "NI_CUT_NI"],
    )
    growth_fields = [
        "REVENUE_YOY",
        "NI_ATTR_P_CUT_YOY",
        "N_CF_OPA_YOY",
        "C_FR_SALE_G_S_YOY",
    ]
    growth_cash = _mean_complete(ranks, growth_fields)
    growth_consistency = (
        ranks[growth_fields].mean(axis=1, skipna=False)
        - ranks[growth_fields].std(axis=1, ddof=0, skipna=False)
    )
    gross_cash = _mean_complete(
        ranks,
        [
            "GROSS_PROFIT_YOY",
            "N_CF_OPA_YOY",
            "C_FR_SALE_G_S_YOY",
        ],
    )
    cost_discipline = _mean_complete(
        ranks,
        ["COST_DISCIPLINE", "GROSS_PROFIT_YOY"],
    )
    receivable = pd.concat(
        [
            ranks["AR_REC_R"],
            ranks["CFSGS_R"],
            1.0 - ranks["AR_R"],
            1.0 - ranks["NR_AR_R"],
        ],
        axis=1,
    ).mean(axis=1, skipna=False)
    cash_debt = _mean_complete(
        ranks,
        [
            "N_CF_OPA_CL",
            "N_CF_OPA_LIAB",
            "N_CF_OPA_ID",
            "N_CF_OPA_ND",
        ],
    )
    profit_composition = _mean_complete(
        ranks,
        ["OPA_P_TR", "NEG_ABS_VALUE_CHANGE", "NI_CUT_NI"],
    )
    rd_efficiency = _mean_complete(
        ranks,
        ["R_D_EXP_YOY", "REVENUE_YOY", "NI_ATTR_P_CUT_YOY"],
    )
    margin_cash = _mean_complete(
        ranks,
        [
            "GROSS_PROFIT_YOY",
            "N_CF_OPA_NIA_YOY",
            "NI_ATTR_P_CUT_YOY",
        ],
    )

    full_mask = data["EVENT_AGE"].ge(0) & data["EVENT_AGE"].lt(
        FULL_WINDOW
    )
    q1_mask = (
        data["EVENT_AGE"].ge(0)
        & data["EVENT_AGE"].lt(Q1_WINDOW)
        & data["FISCAL_QUARTER"].eq(1)
    )
    result = data[KEYS].copy()
    result["cash_profit_conversion_120d"] = data["N_CF_OPA_NIA"].where(
        full_mask
    )
    result["cash_margin_120d"] = data["N_CF_OPA_R"].where(full_mask)
    result["deducted_profit_share_120d"] = data["NI_CUT_NI"].where(
        full_mask
    )
    result["cash_earnings_growth_gap_120d"] = data[
        "CASH_EARNINGS_GROWTH_GAP"
    ].where(full_mask)

    full_composites = {
        "cash_quality_composite_120d": cash_quality,
        "growth_cash_breadth_120d": growth_cash,
        "growth_consistency_120d": growth_consistency,
        "gross_profit_cash_confirmation_120d": gross_cash,
        "cost_discipline_growth_120d": cost_discipline,
        "receivable_collection_quality_120d": receivable,
        "cash_debt_coverage_120d": cash_debt,
        "profit_composition_quality_120d": profit_composition,
        "rd_growth_efficiency_120d": rd_efficiency,
        "margin_cash_improvement_120d": margin_cash,
    }
    for name, values in full_composites.items():
        result[name] = values.where(full_mask)

    q1_composites = {
        "q1_cash_profit_conversion_60d": ranks["N_CF_OPA_NIA"],
        "q1_cash_quality_composite_60d": cash_quality,
        "q1_growth_cash_breadth_60d": growth_cash,
        "q1_growth_consistency_60d": growth_consistency,
        "q1_gross_profit_cash_confirmation_60d": gross_cash,
        "q1_cost_discipline_growth_60d": cost_discipline,
        "q1_receivable_collection_quality_60d": receivable,
        "q1_cash_debt_coverage_60d": cash_debt,
        "q1_profit_composition_quality_60d": profit_composition,
        "q1_rd_growth_efficiency_60d": rd_efficiency,
        "q1_margin_cash_improvement_60d": margin_cash,
    }
    for name, values in q1_composites.items():
        result[name] = values.where(q1_mask)

    joint = ranks["q1_joint_earnings_revenue"]
    all_profit = ranks["q1_all_profit_surprises"]
    financial = ranks["q1_financial_60d"]
    hybrid = {
        "q1_joint_surprise_cash_quality_60d": (
            0.60 * joint + 0.40 * cash_quality
        ),
        "q1_all_profit_cash_quality_60d": (
            0.60 * all_profit + 0.40 * cash_quality
        ),
        "q1_financial_cash_breadth_60d": (
            0.50 * financial + 0.50 * growth_cash
        ),
        "q1_joint_surprise_growth_cash_60d": (
            0.50 * joint + 0.50 * growth_cash
        ),
        "q1_all_profit_growth_consistency_60d": (
            0.60 * all_profit + 0.40 * growth_consistency
        ),
        "q1_joint_surprise_cash_confirmed_60d": joint.where(
            cash_quality.ge(0.50)
        ),
        "q1_all_profit_cash_confirmed_60d": all_profit.where(
            cash_quality.ge(0.50)
        ),
    }
    for name, values in hybrid.items():
        result[name] = values.where(q1_mask)

    for column in CANDIDATE_COLUMNS:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        ).astype("float32")
    return result[KEYS + CANDIDATE_COLUMNS]


def generate_candidates(
    factor_path: Path,
    indicator_dataset: Path,
    output_path: Path,
) -> None:
    dates = pd.read_parquet(factor_path, columns=["TRADE_DATE"])[
        "TRADE_DATE"
    ]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique())
    calendar = calendar.normalize().drop_duplicates().sort_values()
    calendar_ordinal = pd.Series(
        np.arange(len(calendar), dtype=np.int16),
        index=calendar,
    )
    raw_events = load_indicator_events(indicator_dataset)
    events = prepare_available_events(raw_events, calendar)
    print(
        f"Prepared {len(events):,} usable PIT events from "
        f"{len(raw_events):,} current-period source rows"
    )

    temporary = output_path.with_suffix(".tmp.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for year in sorted(set(calendar.year)):
            filters = [
                ("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)),
                ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31)),
            ]
            panel = pd.read_parquet(
                factor_path,
                columns=KEYS + HYBRID_BASES,
                filters=filters,
            )
            panel = _normalize_panel(panel)
            mapped = map_events_to_panel(
                panel,
                events,
                calendar_ordinal,
            )
            candidates = build_candidates_for_slice(mapped)
            table = pa.Table.from_pandas(candidates, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                )
            writer.write_table(table)
            counts = (
                candidates[CANDIDATE_COLUMNS].notna().sum().to_dict()
            )
            print(f"{year}: rows={len(candidates):,}; non-null={counts}")
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("No quarterly indicator candidates were generated")
    os.replace(temporary, output_path)
    print(f"Wrote candidates to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factors",
        type=Path,
        default=BASE_DIR / "factors.parquet",
    )
    parser.add_argument(
        "--indicator-dataset",
        type=Path,
        default=(
            BASE_DIR
            / "data"
            / "quarterly_financial_indicators"
            / "quarterly_financial_indicator_pit"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            BASE_DIR
            / "factor_components"
            / "quarterly_indicator_candidates.parquet"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate_candidates(
        arguments.factors.resolve(),
        arguments.indicator_dataset.resolve(),
        arguments.output.resolve(),
    )
