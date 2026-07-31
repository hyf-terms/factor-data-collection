"""Mine Q1 factors directly from a small set of strict-PIT raw fields.

The module intentionally does not read labels.  It exact-joins the income,
cash-flow and balance-sheet PIT revisions by publication timestamp, matches
each event to the latest prior-year Q1 revision that was already public, and
carries the signal for at most 60 trading days.
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
from quarterly_indicator_factor_search import (
    _mean_complete,
    _rank_daily,
    _winsorize_daily,
    map_events_to_panel,
)


BASE_DIR = Path(__file__).resolve().parent
Q1_WINDOW = 60
EVENT_KEYS = ["SECURITY_ID", "END_DATE", "ACT_PUBTIME"]

SOURCE_FIELDS = {
    "income": [
        "REVENUE",
        "COGS",
        "N_INCOME_ATTR_P",
        "R_D_EXP",
    ],
    "cashflow": [
        "N_CF_OPERATE_A",
        "C_FR_SALE_G_S",
        "PUR_FIX_ASSETS_OTH",
    ],
    "balance": [
        "T_ASSETS",
        "AR",
        "INVENTORIES",
        "AP",
    ],
}
RAW_FIELDS = [
    field for fields in SOURCE_FIELDS.values() for field in fields
]

SIGNAL_COLUMNS = [
    "GROSS_MARGIN",
    "GROSS_MARGIN_CHANGE",
    "CASH_MARGIN",
    "CASH_PROFIT_CONVERSION",
    "COLLECTION_RATIO",
    "ACCRUAL_QUALITY",
    "WORKING_CAPITAL_PRESSURE",
    "LOW_CAPEX",
    "ASSET_TURNOVER",
    "REVENUE_GROWTH",
    "COGS_GROWTH",
    "PROFIT_GROWTH",
    "CFO_GROWTH",
    "SALES_CASH_GROWTH",
    "GROSS_PROFIT_GROWTH",
    "RD_GROWTH",
]

CANDIDATE_COLUMNS = [
    "q1_raw_gross_margin_60d",
    "q1_raw_gross_margin_change_60d",
    "q1_raw_cash_margin_60d",
    "q1_raw_cash_profit_conversion_60d",
    "q1_raw_collection_ratio_60d",
    "q1_raw_accrual_quality_60d",
    "q1_raw_working_capital_pressure_60d",
    "q1_raw_low_capex_60d",
    "q1_raw_asset_turnover_60d",
    "q1_raw_revenue_growth_60d",
    "q1_raw_cogs_growth_60d",
    "q1_raw_profit_growth_60d",
    "q1_raw_cfo_growth_60d",
    "q1_raw_sales_cash_growth_60d",
    "q1_raw_gross_profit_growth_60d",
    "q1_raw_rd_growth_60d",
    "q1_raw_growth_cash_breadth_60d",
    "q1_raw_growth_consistency_60d",
    "q1_raw_margin_cash_improvement_60d",
    "q1_raw_cost_discipline_growth_60d",
    "q1_raw_profit_cash_confirmation_60d",
    "q1_raw_revenue_cash_confirmation_60d",
    "q1_raw_quality_growth_60d",
    "q1_raw_low_investment_growth_60d",
    "q1_raw_rd_growth_efficiency_60d",
    "q1_raw_cash_quality_composite_60d",
]


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    absolute_denominator: bool = False,
) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    if absolute_denominator:
        denominator = denominator.abs()
    valid = denominator.abs().gt(1e-12)
    return numerator.div(denominator.where(valid)).replace(
        [np.inf, -np.inf], np.nan
    )


def _load_one_q1_dataset(
    dataset: Path,
    value_fields: list[str],
) -> pd.DataFrame:
    columns = [
        "SECURITY_ID",
        "ACT_PUBTIME",
        "END_DATE_REP",
        "END_DATE",
        "REPORT_TYPE",
        "IS_CURRENT_PERIOD",
        *value_fields,
    ]
    data = pd.read_parquet(dataset, columns=columns)
    for column in ["ACT_PUBTIME", "END_DATE_REP", "END_DATE"]:
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data = data.loc[
        data["IS_CURRENT_PERIOD"].fillna(False).astype(bool)
        & data["REPORT_TYPE"].astype("string").eq("Q1")
        & data["END_DATE"].dt.month.eq(3)
        & data["END_DATE"].eq(data["END_DATE_REP"])
    ].copy()
    data["SECURITY_ID"] = pd.to_numeric(
        data["SECURITY_ID"], errors="coerce"
    )
    data = data.dropna(subset=EVENT_KEYS)
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    for column in value_fields:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    duplicate = data.duplicated(EVENT_KEYS, keep=False)
    if duplicate.any():
        raise ValueError(
            f"{dataset} contains {int(duplicate.sum()):,} duplicate "
            "security-quarter-publication rows"
        )
    return data[EVENT_KEYS + value_fields]


def load_exact_q1_events(data_root: Path) -> pd.DataFrame:
    """Load and exact-join the three raw Q1 PIT statement datasets."""
    paths = {
        "income": data_root / "new_pit_income",
        "cashflow": data_root / "new_pit_cashflow",
        "balance": data_root / "new_pit_balance",
    }
    frames = {
        name: _load_one_q1_dataset(paths[name], fields)
        for name, fields in SOURCE_FIELDS.items()
    }
    events = frames["income"].merge(
        frames["cashflow"],
        on=EVENT_KEYS,
        how="inner",
        validate="one_to_one",
    )
    events = events.merge(
        frames["balance"],
        on=EVENT_KEYS,
        how="inner",
        validate="one_to_one",
    )
    events["EVENT_TIME"] = events["ACT_PUBTIME"]
    events["FISCAL_QUARTER"] = np.int8(1)
    events["QUARTER_INDEX"] = (
        events["END_DATE"].dt.year * 4 + 1
    ).astype("int32")
    return events.sort_values(
        ["SECURITY_ID", "EVENT_TIME", "QUARTER_INDEX"]
    ).reset_index(drop=True)


def attach_latest_visible_prior_year(events: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest prior-Q1 revision visible at each current event."""
    current = events.copy().reset_index(drop=True)
    current["EVENT_ROW_ID"] = np.arange(len(current), dtype=np.int64)
    current["LOOKBACK_QUARTER_INDEX"] = current["QUARTER_INDEX"] - 4

    prior = events[
        ["SECURITY_ID", "QUARTER_INDEX", "EVENT_TIME", *RAW_FIELDS]
    ].copy()
    prior = prior.rename(
        columns={
            "QUARTER_INDEX": "LOOKBACK_QUARTER_INDEX",
            "EVENT_TIME": "PRIOR_EVENT_TIME",
            **{field: f"PRIOR_{field}" for field in RAW_FIELDS},
        }
    )
    pairs = current[
        [
            "EVENT_ROW_ID",
            "SECURITY_ID",
            "LOOKBACK_QUARTER_INDEX",
            "EVENT_TIME",
        ]
    ].merge(
        prior,
        on=["SECURITY_ID", "LOOKBACK_QUARTER_INDEX"],
        how="left",
    )
    pairs = pairs.loc[
        pairs["PRIOR_EVENT_TIME"].isna()
        | pairs["PRIOR_EVENT_TIME"].le(pairs["EVENT_TIME"])
    ]
    pairs = pairs.sort_values(
        ["EVENT_ROW_ID", "PRIOR_EVENT_TIME"],
        na_position="first",
    ).drop_duplicates("EVENT_ROW_ID", keep="last")
    prior_columns = [
        "EVENT_ROW_ID",
        "PRIOR_EVENT_TIME",
        *[f"PRIOR_{field}" for field in RAW_FIELDS],
    ]
    result = current.merge(
        pairs[prior_columns],
        on="EVENT_ROW_ID",
        how="left",
        validate="one_to_one",
    )
    return result.drop(
        columns=["EVENT_ROW_ID", "LOOKBACK_QUARTER_INDEX"]
    )


def calculate_event_signals(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate low-input raw metrics and PIT-safe year-on-year changes."""
    data = events.copy()
    gross_profit = data["REVENUE"] - data["COGS"]
    prior_gross_profit = (
        data["PRIOR_REVENUE"] - data["PRIOR_COGS"]
    )
    gross_margin = _safe_divide(gross_profit, data["REVENUE"])
    prior_gross_margin = _safe_divide(
        prior_gross_profit, data["PRIOR_REVENUE"]
    )
    data["GROSS_MARGIN"] = gross_margin
    data["GROSS_MARGIN_CHANGE"] = gross_margin - prior_gross_margin
    data["CASH_MARGIN"] = _safe_divide(
        data["N_CF_OPERATE_A"], data["REVENUE"]
    )
    data["CASH_PROFIT_CONVERSION"] = _safe_divide(
        data["N_CF_OPERATE_A"],
        data["N_INCOME_ATTR_P"],
        absolute_denominator=True,
    )
    data["COLLECTION_RATIO"] = _safe_divide(
        data["C_FR_SALE_G_S"], data["REVENUE"]
    )
    data["ACCRUAL_QUALITY"] = _safe_divide(
        data["N_CF_OPERATE_A"] - data["N_INCOME_ATTR_P"],
        data["T_ASSETS"],
        absolute_denominator=True,
    )
    working_capital = data["AR"] + data["INVENTORIES"] - data["AP"]
    data["WORKING_CAPITAL_PRESSURE"] = -_safe_divide(
        working_capital,
        data["T_ASSETS"],
        absolute_denominator=True,
    )
    data["LOW_CAPEX"] = -_safe_divide(
        data["PUR_FIX_ASSETS_OTH"],
        data["T_ASSETS"],
        absolute_denominator=True,
    )
    data["ASSET_TURNOVER"] = _safe_divide(
        data["REVENUE"],
        data["T_ASSETS"],
        absolute_denominator=True,
    )

    growth_specs = {
        "REVENUE_GROWTH": ("REVENUE", "PRIOR_REVENUE"),
        "COGS_GROWTH": ("COGS", "PRIOR_COGS"),
        "PROFIT_GROWTH": (
            "N_INCOME_ATTR_P",
            "PRIOR_N_INCOME_ATTR_P",
        ),
        "CFO_GROWTH": ("N_CF_OPERATE_A", "PRIOR_N_CF_OPERATE_A"),
        "SALES_CASH_GROWTH": (
            "C_FR_SALE_G_S",
            "PRIOR_C_FR_SALE_G_S",
        ),
        "RD_GROWTH": ("R_D_EXP", "PRIOR_R_D_EXP"),
    }
    for output, (current, prior) in growth_specs.items():
        data[output] = _safe_divide(
            data[current] - data[prior],
            data[prior],
            absolute_denominator=True,
        )
    data["GROSS_PROFIT_GROWTH"] = _safe_divide(
        gross_profit - prior_gross_profit,
        prior_gross_profit,
        absolute_denominator=True,
    )
    data[SIGNAL_COLUMNS] = data[SIGNAL_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )
    return data


def prepare_available_events(
    events: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Assign availability and prevent late old revisions taking over."""
    available = assign_available_trade_date(events, calendar)
    available = available.sort_values(
        [
            "SECURITY_ID",
            "AVAILABLE_DATE",
            "EVENT_TIME",
            "QUARTER_INDEX",
        ]
    )
    newest = available.groupby(
        "SECURITY_ID", sort=False
    )["QUARTER_INDEX"].cummax()
    clean = available.loc[available["QUARTER_INDEX"].eq(newest)].copy()
    clean = clean.drop_duplicates(
        ["SECURITY_ID", "AVAILABLE_DATE"], keep="last"
    )
    return clean[
        [
            "SECURITY_ID",
            "AVAILABLE_DATE",
            "FISCAL_QUARTER",
            "QUARTER_INDEX",
            *SIGNAL_COLUMNS,
        ]
    ].reset_index(drop=True)


def build_candidates_for_slice(mapped: pd.DataFrame) -> pd.DataFrame:
    """Winsorize/rank raw signals and create low-field composites."""
    data = mapped.copy()
    data[SIGNAL_COLUMNS] = _winsorize_daily(data, SIGNAL_COLUMNS)
    data["COST_DISCIPLINE"] = (
        data["REVENUE_GROWTH"] - data["COGS_GROWTH"]
    )
    rank_columns = [*SIGNAL_COLUMNS, "COST_DISCIPLINE"]
    ranks = _rank_daily(data, rank_columns)

    growth_fields = [
        "REVENUE_GROWTH",
        "PROFIT_GROWTH",
        "CFO_GROWTH",
        "SALES_CASH_GROWTH",
    ]
    growth_breadth = _mean_complete(ranks, growth_fields)
    growth_consistency = (
        ranks[growth_fields].mean(axis=1, skipna=False)
        - ranks[growth_fields].std(axis=1, ddof=0, skipna=False)
    )
    composites = {
        "q1_raw_growth_cash_breadth_60d": growth_breadth,
        "q1_raw_growth_consistency_60d": growth_consistency,
        "q1_raw_margin_cash_improvement_60d": _mean_complete(
            ranks,
            ["GROSS_MARGIN_CHANGE", "PROFIT_GROWTH", "CFO_GROWTH"],
        ),
        "q1_raw_cost_discipline_growth_60d": _mean_complete(
            ranks,
            ["COST_DISCIPLINE", "GROSS_PROFIT_GROWTH"],
        ),
        "q1_raw_profit_cash_confirmation_60d": _mean_complete(
            ranks,
            ["PROFIT_GROWTH", "CFO_GROWTH", "CASH_PROFIT_CONVERSION"],
        ),
        "q1_raw_revenue_cash_confirmation_60d": _mean_complete(
            ranks,
            ["REVENUE_GROWTH", "SALES_CASH_GROWTH", "COLLECTION_RATIO"],
        ),
        "q1_raw_quality_growth_60d": _mean_complete(
            ranks,
            ["ACCRUAL_QUALITY", "PROFIT_GROWTH", "CFO_GROWTH"],
        ),
        "q1_raw_low_investment_growth_60d": _mean_complete(
            ranks,
            ["LOW_CAPEX", "PROFIT_GROWTH"],
        ),
        "q1_raw_rd_growth_efficiency_60d": _mean_complete(
            ranks,
            ["RD_GROWTH", "REVENUE_GROWTH", "PROFIT_GROWTH"],
        ),
        "q1_raw_cash_quality_composite_60d": _mean_complete(
            ranks,
            ["CASH_PROFIT_CONVERSION", "CASH_MARGIN", "COLLECTION_RATIO"],
        ),
    }
    direct = {
        f"q1_raw_{signal.lower()}_60d": ranks[signal]
        for signal in SIGNAL_COLUMNS
    }
    valid = (
        data["EVENT_AGE"].ge(0)
        & data["EVENT_AGE"].lt(Q1_WINDOW)
        & data["FISCAL_QUARTER"].eq(1)
    )
    result = data[KEYS].copy()
    for name, values in {**direct, **composites}.items():
        result[name] = values.where(valid)
    missing = sorted(set(CANDIDATE_COLUMNS).difference(result.columns))
    extra = sorted(
        set(result.columns).difference(KEYS + CANDIDATE_COLUMNS)
    )
    if missing or extra:
        raise RuntimeError(
            f"Candidate schema mismatch; missing={missing}, extra={extra}"
        )
    for column in CANDIDATE_COLUMNS:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        ).astype("float32")
    return result[KEYS + CANDIDATE_COLUMNS]


def generate_candidates(
    factor_path: Path,
    data_root: Path,
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
    raw = load_exact_q1_events(data_root)
    paired = attach_latest_visible_prior_year(raw)
    signaled = calculate_event_signals(paired)
    events = prepare_available_events(signaled, calendar)
    print(
        f"Prepared {len(events):,} usable Q1 PIT events from "
        f"{len(raw):,} exact three-statement revisions"
    )

    temporary = output_path.with_suffix(".tmp.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for year in sorted(set(calendar.year)):
            panel = pd.read_parquet(
                factor_path,
                columns=KEYS,
                filters=[
                    ("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)),
                    ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31)),
                ],
            )
            panel = _normalize_panel(panel)
            mapped = map_events_to_panel(
                panel, events, calendar_ordinal
            )
            candidates = build_candidates_for_slice(mapped)
            table = pa.Table.from_pandas(
                candidates, preserve_index=False
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, table.schema, compression="zstd"
                )
            writer.write_table(table)
            non_null = candidates[CANDIDATE_COLUMNS].notna().sum()
            print(
                f"{year}: rows={len(candidates):,}; "
                f"non-null range={int(non_null.min()):,}-"
                f"{int(non_null.max()):,}"
            )
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("No raw Q1 candidates were generated")
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
        "--data-root",
        type=Path,
        default=BASE_DIR / "data" / "new_pit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            BASE_DIR
            / "factor_components"
            / "raw_q1_minimal_candidates.parquet"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate_candidates(
        arguments.factors.resolve(),
        arguments.data_root.resolve(),
        arguments.output.resolve(),
    )
