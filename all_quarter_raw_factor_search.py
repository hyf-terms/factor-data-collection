"""Build all-quarter factors directly from strict-PIT raw statement fields.

Unlike the earlier Q1 ranked candidates, this module converts cumulative
flows into standalone Q1-Q4 values and keeps winsorized raw ratios/growth
metrics without cross-sectional percentile ranking.  Candidate generation
never reads labels.
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
from quarterly_f_score import COMMON_COLUMNS, build_standalone_quarterly_metric
from quarterly_indicator_factor_search import _winsorize_daily, map_events_to_panel
from raw_q1_minimal_factor_search import _safe_divide


BASE_DIR = Path(__file__).resolve().parent
EVENT_WINDOW = 60
LATEST_QUARTER_WINDOW = 120
INCOME_FIELDS = ["REVENUE", "COGS", "N_INCOME_ATTR_P", "R_D_EXP"]
CASHFLOW_FIELDS = [
    "N_CF_OPERATE_A",
    "C_FR_SALE_G_S",
    "PUR_FIX_ASSETS_OTH",
]
RAW_FIELDS = [*INCOME_FIELDS, *CASHFLOW_FIELDS]

SIGNAL_COLUMNS = [
    "GROSS_MARGIN",
    "GROSS_MARGIN_CHANGE",
    "CASH_MARGIN",
    "CASH_PROFIT_CONVERSION",
    "COLLECTION_RATIO",
    "LOW_CAPEX_TO_REVENUE",
    "REVENUE_GROWTH",
    "COGS_GROWTH",
    "PROFIT_GROWTH",
    "CFO_GROWTH",
    "SALES_CASH_GROWTH",
    "GROSS_PROFIT_GROWTH",
    "RD_GROWTH",
]

DIRECT_CANDIDATES = {
    signal: f"allq_raw_metric_{signal.lower()}_60d"
    for signal in SIGNAL_COLUMNS
}
COMPOSITE_CANDIDATES = [
    "allq_raw_metric_growth_cash_breadth_60d",
    "allq_raw_metric_growth_consistency_60d",
    "allq_raw_metric_margin_cash_improvement_60d",
    "allq_raw_metric_cost_discipline_growth_60d",
    "allq_raw_metric_gross_cash_confirmation_60d",
    "allq_raw_metric_rd_growth_efficiency_60d",
]
QUARTER_SIGNALS = [
    "PROFIT_GROWTH",
    "GROSS_PROFIT_GROWTH",
    "GROWTH_CASH_BREADTH",
    "MARGIN_CASH_IMPROVEMENT",
    "GROWTH_CONSISTENCY",
    "RD_GROWTH_EFFICIENCY",
]
QUARTER_CANDIDATES = [
    f"q{quarter}_raw_metric_{signal.lower()}_60d"
    for quarter in range(1, 5)
    for signal in QUARTER_SIGNALS
]
OPTIMIZED_CANDIDATES = [
    "latestq_raw_metric_gross_profit_growth_120d",
    "latestq_raw_metric_rd_growth_efficiency_120d",
]
CANDIDATE_COLUMNS = [
    *DIRECT_CANDIDATES.values(),
    *COMPOSITE_CANDIDATES,
    *QUARTER_CANDIDATES,
    *OPTIMIZED_CANDIDATES,
]


def _load_statement(dataset: Path, fields: list[str]) -> pd.DataFrame:
    columns = [*COMMON_COLUMNS, *fields]
    data = pd.read_parquet(dataset, columns=columns)
    duplicate_columns = data.columns[data.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"Duplicate source columns: {duplicate_columns}")
    return data


def _standalone_metrics(
    statement: pd.DataFrame,
    fields: list[str],
    statement_name: str,
) -> dict[str, pd.DataFrame]:
    return {
        field: build_standalone_quarterly_metric(
            statement,
            field,
            name=f"{statement_name}:{field}",
        )
        for field in fields
    }


def load_all_quarter_events(data_root: Path) -> pd.DataFrame:
    """Load raw statements and align standalone quarterly flow metrics."""
    income = _load_statement(
        data_root / "new_pit_income", INCOME_FIELDS
    )
    cashflow = _load_statement(
        data_root / "new_pit_cashflow", CASHFLOW_FIELDS
    )
    frames = {
        **_standalone_metrics(income, INCOME_FIELDS, "income"),
        **_standalone_metrics(cashflow, CASHFLOW_FIELDS, "cashflow"),
    }
    keys = ["SECURITY_ID", "QUARTER_INDEX"]
    base = frames["REVENUE"][
        keys
        + [
            "FISCAL_YEAR",
            "FISCAL_QUARTER",
            "END_DATE",
            "EVENT_TIME",
            "REVENUE",
        ]
    ].rename(columns={"EVENT_TIME": "REVENUE_EVENT_TIME"})
    for field in RAW_FIELDS[1:]:
        metric = frames[field][keys + ["EVENT_TIME", field]].rename(
            columns={"EVENT_TIME": f"{field}_EVENT_TIME"}
        )
        base = base.merge(
            metric,
            on=keys,
            how="left",
            validate="one_to_one",
        )
    event_columns = [f"{field}_EVENT_TIME" for field in RAW_FIELDS]
    base["EVENT_TIME"] = base[event_columns].max(axis=1)
    return base.sort_values(
        ["SECURITY_ID", "QUARTER_INDEX"]
    ).reset_index(drop=True)


def attach_prior_year(events: pd.DataFrame) -> pd.DataFrame:
    current = events.copy()
    prior = events[["SECURITY_ID", "QUARTER_INDEX", *RAW_FIELDS]].copy()
    prior["QUARTER_INDEX"] += 4
    prior = prior.rename(
        columns={field: f"PRIOR_{field}" for field in RAW_FIELDS}
    )
    return current.merge(
        prior,
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="left",
        validate="one_to_one",
    )


def calculate_event_signals(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate raw-field ratios and same-quarter growth without ranks."""
    data = events.copy()
    gross_profit = data["REVENUE"] - data["COGS"]
    prior_gross_profit = data["PRIOR_REVENUE"] - data["PRIOR_COGS"]
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
    data["LOW_CAPEX_TO_REVENUE"] = -_safe_divide(
        data["PUR_FIX_ASSETS_OTH"],
        data["REVENUE"],
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
    return data.replace([np.inf, -np.inf], np.nan)


def prepare_available_events(
    events: pd.DataFrame, calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    available = assign_available_trade_date(events, calendar)
    available = available.sort_values(
        ["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"]
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


def _mean_complete(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    return data[columns].mean(axis=1, skipna=False)


def build_candidates_for_slice(mapped: pd.DataFrame) -> pd.DataFrame:
    """Build winsorized raw metrics and composites without percentile ranks."""
    data = mapped.copy()
    data[SIGNAL_COLUMNS] = _winsorize_daily(data, SIGNAL_COLUMNS)
    data["COST_DISCIPLINE"] = (
        data["REVENUE_GROWTH"] - data["COGS_GROWTH"]
    )
    growth_fields = [
        "REVENUE_GROWTH",
        "PROFIT_GROWTH",
        "CFO_GROWTH",
        "SALES_CASH_GROWTH",
    ]
    composites = {
        "GROWTH_CASH_BREADTH": _mean_complete(data, growth_fields),
        "GROWTH_CONSISTENCY": (
            data[growth_fields].mean(axis=1, skipna=False)
            - data[growth_fields].std(axis=1, ddof=0, skipna=False)
        ),
        "MARGIN_CASH_IMPROVEMENT": _mean_complete(
            data,
            ["GROSS_MARGIN_CHANGE", "PROFIT_GROWTH", "CFO_GROWTH"],
        ),
        "COST_DISCIPLINE_GROWTH": _mean_complete(
            data, ["COST_DISCIPLINE", "GROSS_PROFIT_GROWTH"]
        ),
        "GROSS_CASH_CONFIRMATION": _mean_complete(
            data,
            [
                "GROSS_PROFIT_GROWTH",
                "CFO_GROWTH",
                "SALES_CASH_GROWTH",
            ],
        ),
        "RD_GROWTH_EFFICIENCY": _mean_complete(
            data, ["RD_GROWTH", "REVENUE_GROWTH", "PROFIT_GROWTH"]
        ),
    }
    valid = data["EVENT_AGE"].ge(0) & data["EVENT_AGE"].lt(EVENT_WINDOW)
    latest_quarter_valid = data["EVENT_AGE"].ge(0) & data["EVENT_AGE"].lt(
        LATEST_QUARTER_WINDOW
    )
    result = data[KEYS].copy()
    for signal, candidate in DIRECT_CANDIDATES.items():
        result[candidate] = data[signal].where(valid)
    for name in COMPOSITE_CANDIDATES:
        signal = name.removeprefix("allq_raw_metric_").removesuffix(
            "_60d"
        ).upper()
        result[name] = composites[signal].where(valid)
    for quarter in range(1, 5):
        quarter_mask = valid & data["FISCAL_QUARTER"].eq(quarter)
        for signal in QUARTER_SIGNALS:
            name = f"q{quarter}_raw_metric_{signal.lower()}_60d"
            values = (
                data[signal]
                if signal in data.columns
                else composites[signal]
            )
            result[name] = values.where(quarter_mask)
    # Dense alternatives to the Q1-only event factors.  They use the newest
    # disclosed standalone quarter (Q1-Q4) and remain valid for 120 trading
    # days, which removes the long all-NaN gaps without injecting artificial
    # jitter into a naturally stepwise fundamental signal.
    result["latestq_raw_metric_gross_profit_growth_120d"] = data[
        "GROSS_PROFIT_GROWTH"
    ].where(latest_quarter_valid)
    result["latestq_raw_metric_rd_growth_efficiency_120d"] = composites[
        "RD_GROWTH_EFFICIENCY"
    ].where(latest_quarter_valid)
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
        np.arange(len(calendar), dtype=np.int16), index=calendar
    )
    raw = load_all_quarter_events(data_root)
    events = calculate_event_signals(attach_prior_year(raw))
    events = prepare_available_events(events, calendar)
    print(f"Prepared {len(events):,} all-quarter PIT events")

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
            mapped = map_events_to_panel(panel, events, calendar_ordinal)
            candidates = build_candidates_for_slice(mapped)
            table = pa.Table.from_pandas(candidates, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, table.schema, compression="zstd"
                )
            writer.write_table(table)
            counts = candidates[CANDIDATE_COLUMNS].notna().sum()
            print(
                f"{year}: rows={len(candidates):,}; non-null "
                f"range={int(counts.min()):,}-{int(counts.max()):,}"
            )
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("No all-quarter candidates were generated")
    os.replace(temporary, output_path)
    print(f"Wrote candidates to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factors", type=Path, default=BASE_DIR / "factors.parquet"
    )
    parser.add_argument(
        "--data-root", type=Path, default=BASE_DIR / "data" / "new_pit"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            BASE_DIR
            / "factor_components"
            / "all_quarter_raw_candidates.parquet"
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
