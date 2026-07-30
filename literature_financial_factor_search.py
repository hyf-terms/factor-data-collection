"""Build additional literature-motivated PIT financial factor candidates.

Implemented families:

* standardized unexpected revenue (SUR), following the time-series surprise
  idea in Jegadeesh and Livnat (2006);
* operating-profit and gross-profit surprises;
* joint earnings/revenue surprise signals;
* four-quarter earnings-surprise persistence;
* constrained parameter variants of the Q1 financial composite.

No label or future-return data are read by this program.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from event_financial_factor_search import (
    KEYS,
    _map_events,
    _normalize_panel,
    _prepare_events,
    build_deducted_sue_events,
)
from pead_sue_factor import build_sue_events
from quarterly_f_score import (
    COMMON_COLUMNS,
    build_standalone_quarterly_metric,
)


BASE_DIR = Path(__file__).resolve().parent
FACTOR_INPUTS = [
    "quarterly_f_score",
    "operating_profit_acceleration",
    "asset_growth",
    "profitability_quality_score",
    "cfo_sue",
    "accrual_quality",
    "gross_profitability",
]
CANDIDATE_COLUMNS = [
    "revenue_sue_60d",
    "operating_profit_sue_60d",
    "gross_profit_sue_60d",
    "joint_earnings_revenue_60d",
    "joint_profit_surprises_60d",
    "pead_sue_ma4_120d",
    "q1_revenue_sue_80d",
    "q1_operating_profit_sue_80d",
    "q1_joint_earnings_revenue",
    "q1_all_profit_surprises",
    "q1_cash_quality",
    "q1_fscore_quality",
    "q1_financial_pead2",
    "q1_financial_no_asset",
    "q1_financial_no_quality",
    "q1_financial_median",
    "q1_financial_60d",
]


def _metric_sue_events(
    statement: pd.DataFrame,
    value_column: str,
    name: str,
) -> pd.DataFrame:
    quarterly = build_standalone_quarterly_metric(
        statement,
        value_column,
        name=name,
    ).rename(
        columns={
            value_column: "QUARTERLY_EARNINGS",
            f"{value_column}_SOURCE": "SOURCE",
        }
    )
    return build_sue_events(quarterly)


def build_income_surprise_events(
    income: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build revenue, operating-profit and gross-profit SUE events."""
    revenue_quarterly = build_standalone_quarterly_metric(
        income,
        "REVENUE",
        name="利润表PIT",
    )
    operate_quarterly = build_standalone_quarterly_metric(
        income,
        "OPERATE_PROFIT",
        name="利润表PIT",
    )
    cogs_quarterly = build_standalone_quarterly_metric(
        income,
        "COGS",
        name="利润表PIT",
    )

    revenue_sue = build_sue_events(
        revenue_quarterly.rename(
            columns={
                "REVENUE": "QUARTERLY_EARNINGS",
                "REVENUE_SOURCE": "SOURCE",
            }
        )
    )
    operate_sue = build_sue_events(
        operate_quarterly.rename(
            columns={
                "OPERATE_PROFIT": "QUARTERLY_EARNINGS",
                "OPERATE_PROFIT_SOURCE": "SOURCE",
            }
        )
    )

    event_keys = [
        "SECURITY_ID",
        "FISCAL_YEAR",
        "FISCAL_QUARTER",
        "QUARTER_INDEX",
    ]
    gross = pd.merge(
        revenue_quarterly,
        cogs_quarterly,
        on=event_keys,
        how="inner",
        suffixes=("_REVENUE", "_COGS"),
        validate="one_to_one",
    )
    gross_quarterly = gross[event_keys].copy()
    gross_quarterly["END_DATE"] = pd.concat(
        [
            pd.to_datetime(gross["END_DATE_REVENUE"], errors="coerce"),
            pd.to_datetime(gross["END_DATE_COGS"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)
    gross_quarterly["EVENT_TIME"] = pd.concat(
        [
            pd.to_datetime(gross["EVENT_TIME_REVENUE"], errors="coerce"),
            pd.to_datetime(gross["EVENT_TIME_COGS"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)
    gross_quarterly["QUARTERLY_EARNINGS"] = (
        gross["REVENUE"] - gross["COGS"]
    )
    gross_quarterly["SOURCE"] = "REVENUE_MINUS_COGS"
    gross_sue = build_sue_events(gross_quarterly)
    return {
        "revenue": revenue_sue,
        "operate": operate_sue,
        "gross": gross_sue,
    }


def build_sue_persistence_events(
    events: pd.DataFrame,
    window: int = 4,
) -> pd.DataFrame:
    """Average contiguous current and prior SUE events, without look-ahead."""
    pieces: list[pd.DataFrame] = []
    for _, group in events.groupby("SECURITY_ID", sort=False):
        data = group.sort_values("QUARTER_INDEX").copy()
        contiguous = (
            data["QUARTER_INDEX"]
            - data["QUARTER_INDEX"].shift(window - 1)
        ).eq(window - 1)
        data["SUE_MA4"] = (
            data["SUE_RAW"].rolling(window, min_periods=window).mean()
        ).where(contiguous)
        prior_times = pd.concat(
            [
                pd.to_datetime(data["EVENT_TIME"], errors="coerce").shift(i)
                for i in range(window)
            ],
            axis=1,
        ).max(axis=1)
        data["EVENT_TIME"] = prior_times
        pieces.append(data.dropna(subset=["SUE_MA4"]))
    return pd.concat(pieces, ignore_index=True)


def _all_present_mean(
    ranks: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    return ranks[columns].mean(axis=1, skipna=False)


def _weighted_all_present_mean(
    ranks: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    columns = list(weights)
    complete = ranks[columns].notna().all(axis=1)
    numerator = sum(ranks[column] * weight for column, weight in weights.items())
    return (numerator / sum(weights.values())).where(complete)


def _same_quarter_mask(
    data: pd.DataFrame,
    prefixes: list[str],
    quarter: int | None,
    max_age: int,
) -> pd.Series:
    mask = pd.Series(True, index=data.index)
    reference = data[f"{prefixes[0]}_quarter"]
    for prefix in prefixes:
        mask &= data[f"{prefix}_age"].ge(0)
        mask &= data[f"{prefix}_age"].lt(max_age)
        mask &= data[f"{prefix}_quarter"].eq(reference)
        if quarter is not None:
            mask &= data[f"{prefix}_quarter"].eq(quarter)
    return mask


def build_candidates_for_slice(
    factors: pd.DataFrame,
    mapped: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    data = factors.copy()
    for prefix, frame in mapped.items():
        data = pd.merge(
            data,
            frame,
            on=KEYS,
            how="left",
            validate="one_to_one",
        )

    rank_columns = [
        "pead_value",
        "deducted_value",
        "revenue_value",
        "operate_value",
        "gross_value",
        *FACTOR_INPUTS,
    ]
    ranks = data.groupby("TRADE_DATE", sort=False)[rank_columns].rank(
        pct=True
    )
    ranks["asset_growth"] = 1.0 - ranks["asset_growth"]

    result = data[KEYS].copy()
    revenue_60 = _same_quarter_mask(data, ["revenue"], None, 60)
    operate_60 = _same_quarter_mask(data, ["operate"], None, 60)
    gross_60 = _same_quarter_mask(data, ["gross"], None, 60)
    earnings_revenue_60 = _same_quarter_mask(
        data, ["pead", "revenue"], None, 60
    )
    profits_60 = _same_quarter_mask(
        data,
        ["pead", "deducted", "operate", "gross"],
        None,
        60,
    )
    pead_ma4_120 = _same_quarter_mask(data, ["pead_ma4"], None, 120)

    result["revenue_sue_60d"] = data["revenue_value"].where(revenue_60)
    result["operating_profit_sue_60d"] = data["operate_value"].where(
        operate_60
    )
    result["gross_profit_sue_60d"] = data["gross_value"].where(gross_60)
    result["joint_earnings_revenue_60d"] = _all_present_mean(
        ranks, ["pead_value", "revenue_value"]
    ).where(earnings_revenue_60)
    result["joint_profit_surprises_60d"] = _all_present_mean(
        ranks,
        ["pead_value", "deducted_value", "operate_value", "gross_value"],
    ).where(profits_60)
    result["pead_sue_ma4_120d"] = data["pead_ma4_value"].where(pead_ma4_120)

    q1_revenue = _same_quarter_mask(data, ["revenue"], 1, 80)
    q1_operate = _same_quarter_mask(data, ["operate"], 1, 80)
    q1_earnings_revenue = _same_quarter_mask(
        data, ["pead", "revenue"], 1, 80
    )
    q1_profit_surprises = _same_quarter_mask(
        data,
        ["pead", "deducted", "revenue", "operate", "gross"],
        1,
        80,
    )
    q1_dual = _same_quarter_mask(data, ["pead", "deducted"], 1, 80)
    q1_dual_60 = _same_quarter_mask(
        data, ["pead", "deducted"], 1, 60
    )

    result["q1_revenue_sue_80d"] = data["revenue_value"].where(q1_revenue)
    result["q1_operating_profit_sue_80d"] = data["operate_value"].where(
        q1_operate
    )
    result["q1_joint_earnings_revenue"] = _all_present_mean(
        ranks, ["pead_value", "revenue_value"]
    ).where(q1_earnings_revenue)
    result["q1_all_profit_surprises"] = _all_present_mean(
        ranks,
        [
            "pead_value",
            "deducted_value",
            "revenue_value",
            "operate_value",
            "gross_value",
        ],
    ).where(q1_profit_surprises)
    result["q1_cash_quality"] = _all_present_mean(
        ranks,
        [
            "deducted_value",
            "operate_value",
            "cfo_sue",
            "accrual_quality",
            "gross_profitability",
        ],
    ).where(q1_dual)
    result["q1_fscore_quality"] = _all_present_mean(
        ranks,
        [
            "quarterly_f_score",
            "operating_profit_acceleration",
            "asset_growth",
            "profitability_quality_score",
            "gross_profitability",
        ],
    ).where(q1_dual)

    original = [
        "pead_value",
        "deducted_value",
        "quarterly_f_score",
        "operating_profit_acceleration",
        "asset_growth",
        "profitability_quality_score",
    ]
    result["q1_financial_pead2"] = _weighted_all_present_mean(
        ranks,
        {
            "pead_value": 2.0,
            "deducted_value": 2.0,
            "quarterly_f_score": 1.0,
            "operating_profit_acceleration": 1.0,
            "asset_growth": 1.0,
            "profitability_quality_score": 1.0,
        },
    ).where(q1_dual)
    result["q1_financial_no_asset"] = _all_present_mean(
        ranks,
        [column for column in original if column != "asset_growth"],
    ).where(q1_dual)
    result["q1_financial_no_quality"] = _all_present_mean(
        ranks,
        [
            column
            for column in original
            if column != "profitability_quality_score"
        ],
    ).where(q1_dual)
    result["q1_financial_median"] = ranks[original].median(
        axis=1, skipna=False
    ).where(q1_dual)
    result["q1_financial_60d"] = _all_present_mean(
        ranks, original
    ).where(q1_dual_60)

    for column in CANDIDATE_COLUMNS:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        ).astype("float32")
    return result[KEYS + CANDIDATE_COLUMNS]


def generate_candidates(
    factor_path: Path,
    income_path: Path,
    indicator_path: Path,
    pead_event_path: Path,
    output_path: Path,
) -> None:
    factor_schema = pq.read_schema(factor_path)
    missing = sorted(set(KEYS + FACTOR_INPUTS).difference(factor_schema.names))
    if missing:
        raise KeyError(f"factors.parquet缺少字段: {missing}")

    dates = pd.read_parquet(factor_path, columns=["TRADE_DATE"])[
        "TRADE_DATE"
    ]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique())
    calendar = calendar.normalize().drop_duplicates().sort_values()
    calendar_ordinal = pd.Series(
        np.arange(len(calendar), dtype=np.int16),
        index=calendar,
    )

    pead_events = pd.read_parquet(pead_event_path)
    indicator = pd.read_parquet(indicator_path)
    deducted_events = build_deducted_sue_events(indicator)
    income = pd.read_parquet(
        income_path,
        columns=COMMON_COLUMNS + ["REVENUE", "OPERATE_PROFIT", "COGS"],
    )
    surprises = build_income_surprise_events(income)
    pead_ma4 = build_sue_persistence_events(pead_events)

    event_specs = {
        "pead": (pead_events, "SUE_RAW"),
        "deducted": (deducted_events, "SUE_RAW"),
        "revenue": (surprises["revenue"], "SUE_RAW"),
        "operate": (surprises["operate"], "SUE_RAW"),
        "gross": (surprises["gross"], "SUE_RAW"),
        "pead_ma4": (pead_ma4, "SUE_MA4"),
    }
    event_groups = {
        prefix: _prepare_events(events, calendar, value_column=value)
        for prefix, (events, value) in event_specs.items()
    }

    temporary = output_path.with_suffix(".tmp.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for year in sorted(set(calendar.year)):
            start = pd.Timestamp(year=year, month=1, day=1)
            end = pd.Timestamp(year=year, month=12, day=31)
            filters = [
                ("TRADE_DATE", ">=", start),
                ("TRADE_DATE", "<=", end),
            ]
            factors = pd.read_parquet(
                factor_path,
                columns=KEYS + FACTOR_INPUTS,
                filters=filters,
            )
            factors = _normalize_panel(factors)
            mapped = {
                prefix: _map_events(
                    factors[KEYS],
                    groups,
                    calendar_ordinal,
                    event_specs[prefix][1],
                    prefix,
                )
                for prefix, groups in event_groups.items()
            }
            candidates = build_candidates_for_slice(factors, mapped)
            table = pa.Table.from_pandas(candidates, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                )
            writer.write_table(table)
            counts = candidates[CANDIDATE_COLUMNS].notna().sum().to_dict()
            print(f"{year}: rows={len(candidates):,}; non-null={counts}")
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("没有生成候选数据")
    os.replace(temporary, output_path)
    print(f"Wrote candidates to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factors", type=Path, default=BASE_DIR / "factors.parquet"
    )
    parser.add_argument(
        "--income",
        type=Path,
        default=BASE_DIR / "data" / "new_pit" / "new_pit_income",
    )
    parser.add_argument(
        "--indicator",
        type=Path,
        default=BASE_DIR / "data" / "ch_models" / "earnings_pit",
    )
    parser.add_argument(
        "--pead-events",
        type=Path,
        default=BASE_DIR / "factor_components" / "pead_sue_events.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            BASE_DIR
            / "factor_components"
            / "literature_financial_candidates.parquet"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_candidates(
        args.factors.resolve(),
        args.income.resolve(),
        args.indicator.resolve(),
        args.pead_events.resolve(),
        args.output.resolve(),
    )
