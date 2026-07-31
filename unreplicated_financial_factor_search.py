"""Build a new batch of literature-based PIT financial candidates.

This module tests standalone accounting signals that had previously appeared
only as components of local composite scores.  It never reads labels or
future returns.  Every quarterly event is mapped from its first usable trade
date, and stale events are bounded by an explicit trading-day window.
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
    _normalize_panel,
)
from pead_sue_factor import assign_available_trade_date


BASE_DIR = Path(__file__).resolve().parent
FULL_WINDOW = 120
Q1_WINDOW = 80

EVENT_SPECS = {
    "margin": ("gross_margin_change_events.parquet", "gross_margin_change"),
    "turnover": (
        "asset_turnover_change_events.parquet",
        "asset_turnover_change",
    ),
    "noa": ("noa_quality_events.parquet", "noa_quality"),
    "cfoa": ("cfo_to_assets_events.parquet", "cfo_to_assets"),
    "revenue_growth": ("revenue_growth_events.parquet", "revenue_growth"),
    "deducted_growth": (
        "deducted_profit_growth_events.parquet",
        "deducted_profit_growth",
    ),
    "nonrecurring": (
        "nonrecurring_quality_events.parquet",
        "nonrecurring_quality",
    ),
}

MOHANRAM_SPECS = {
    "rd": "RD_INTENSITY",
    "capex": "CAPEX_INTENSITY",
    "earnings_stability": "EARNINGS_STABILITY",
    "sales_stability": "SALES_GROWTH_STABILITY",
}

CANDIDATE_COLUMNS = [
    "gross_margin_change_120d",
    "asset_turnover_change_120d",
    "dupont_improvement_120d",
    "net_operating_assets_quality_120d",
    "operating_cashflow_profitability_120d",
    "revenue_growth_120d",
    "deducted_profit_growth_120d",
    "nonrecurring_quality_120d",
    "rd_intensity_120d",
    "capex_intensity_120d",
    "earnings_stability_120d",
    "sales_growth_stability_120d",
    "quality_efficiency_120d",
    "intangible_growth_quality_120d",
    "q1_gross_margin_change_80d",
    "q1_asset_turnover_change_80d",
    "q1_dupont_improvement_80d",
    "q1_noa_quality_80d",
    "q1_cfo_profitability_80d",
    "q1_rd_intensity_80d",
    "q1_quality_efficiency_80d",
    "q1_intangible_growth_quality_80d",
]


def _prepare_mohanram_events(path: Path) -> dict[str, pd.DataFrame]:
    columns = [
        "SECURITY_ID",
        "FISCAL_YEAR",
        "FISCAL_QUARTER",
        "QUARTER_INDEX",
        "EVENT_TIME",
        "RD_INTENSITY",
        "CAPEX_INTENSITY",
        "VAR_ROA",
        "VAR_SALES_GROWTH",
    ]
    events = pd.read_parquet(path, columns=columns)
    events["EARNINGS_STABILITY"] = -pd.to_numeric(
        events["VAR_ROA"], errors="coerce"
    )
    events["SALES_GROWTH_STABILITY"] = -pd.to_numeric(
        events["VAR_SALES_GROWTH"], errors="coerce"
    )
    return {
        prefix: events[
            [
                "SECURITY_ID",
                "FISCAL_YEAR",
                "FISCAL_QUARTER",
                "QUARTER_INDEX",
                "EVENT_TIME",
                value,
            ]
        ].dropna(subset=[value])
        for prefix, value in MOHANRAM_SPECS.items()
    }


def load_event_specs(component_dir: Path) -> dict[str, tuple[pd.DataFrame, str]]:
    specs: dict[str, tuple[pd.DataFrame, str]] = {}
    for prefix, (filename, value) in EVENT_SPECS.items():
        specs[prefix] = (pd.read_parquet(component_dir / filename), value)
    mohanram = _prepare_mohanram_events(
        component_dir / "mohanram_g_score_events.parquet"
    )
    for prefix, value in MOHANRAM_SPECS.items():
        specs[prefix] = (mohanram[prefix], value)
    return specs


def _prepare_event_table_vectorized(
    events: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    value_column: str,
) -> pd.DataFrame:
    required = {
        "SECURITY_ID",
        "QUARTER_INDEX",
        "FISCAL_QUARTER",
        "EVENT_TIME",
        value_column,
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise KeyError(f"Event data missing columns: {missing}")
    available = assign_available_trade_date(events, calendar)
    available = available.sort_values(
        ["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"]
    )
    newest = available.groupby(
        "SECURITY_ID", sort=False
    )["QUARTER_INDEX"].cummax()
    clean = available.loc[available["QUARTER_INDEX"].eq(newest)]
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
            value_column,
        ]
    ]


def _map_events_vectorized(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    calendar_ordinal: pd.Series,
    value_column: str,
    output_prefix: str,
) -> pd.DataFrame:
    """Vectorized equivalent of the per-security PIT as-of mapping."""
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
    joined[f"{output_prefix}_age"] = (
        joined["TRADE_DATE"].map(calendar_ordinal)
        - joined["AVAILABLE_DATE"].map(calendar_ordinal)
    )
    joined = joined.rename(
        columns={
            value_column: f"{output_prefix}_value",
            "FISCAL_QUARTER": f"{output_prefix}_quarter",
        }
    )
    return joined[
        KEYS
        + [
            f"{output_prefix}_value",
            f"{output_prefix}_quarter",
            f"{output_prefix}_age",
        ]
    ].sort_values(KEYS)


def _winsorized_daily_values(
    data: pd.DataFrame,
    columns: list[str],
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    result = data[columns].apply(pd.to_numeric, errors="coerce")
    grouped = result.groupby(data["TRADE_DATE"], sort=False)
    lows = grouped.transform(lambda values: values.quantile(lower))
    highs = grouped.transform(lambda values: values.quantile(upper))
    return result.clip(lower=lows, upper=highs)


def _same_quarter_mask(
    data: pd.DataFrame,
    prefixes: list[str],
    quarter: int | None,
    max_age: int,
) -> pd.Series:
    reference = data[f"{prefixes[0]}_quarter"]
    mask = pd.Series(True, index=data.index)
    for prefix in prefixes:
        mask &= data[f"{prefix}_age"].ge(0)
        mask &= data[f"{prefix}_age"].lt(max_age)
        mask &= data[f"{prefix}_quarter"].eq(reference)
        if quarter is not None:
            mask &= data[f"{prefix}_quarter"].eq(quarter)
    return mask


def _complete_rank_mean(
    ranks: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    return ranks[columns].mean(axis=1, skipna=False)


def build_candidates_for_slice(
    panel: pd.DataFrame,
    mapped: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    data = panel[KEYS].copy()
    for prefix, frame in mapped.items():
        data = data.merge(frame, on=KEYS, how="left", validate="one_to_one")

    prefixes = list(mapped)
    value_columns = [f"{prefix}_value" for prefix in prefixes]
    winsorized = _winsorized_daily_values(data, value_columns)
    for column in value_columns:
        data[column] = winsorized[column]
    ranks = data.groupby("TRADE_DATE", sort=False)[value_columns].rank(
        method="average",
        pct=True,
    )

    full_masks = {
        prefix: _same_quarter_mask(data, [prefix], None, FULL_WINDOW)
        for prefix in prefixes
    }
    q1_masks = {
        prefix: _same_quarter_mask(data, [prefix], 1, Q1_WINDOW)
        for prefix in prefixes
    }

    result = data[KEYS].copy()
    direct = {
        "gross_margin_change_120d": "margin",
        "asset_turnover_change_120d": "turnover",
        "net_operating_assets_quality_120d": "noa",
        "operating_cashflow_profitability_120d": "cfoa",
        "revenue_growth_120d": "revenue_growth",
        "deducted_profit_growth_120d": "deducted_growth",
        "nonrecurring_quality_120d": "nonrecurring",
        "rd_intensity_120d": "rd",
        "capex_intensity_120d": "capex",
        "earnings_stability_120d": "earnings_stability",
        "sales_growth_stability_120d": "sales_stability",
    }
    for candidate, prefix in direct.items():
        result[candidate] = data[f"{prefix}_value"].where(full_masks[prefix])

    dupont = ["margin", "turnover"]
    quality = ["margin", "turnover", "noa", "cfoa"]
    intangible = [
        "rd",
        "revenue_growth",
        "margin",
        "deducted_growth",
    ]
    composite_specs = {
        "dupont_improvement_120d": (dupont, None, FULL_WINDOW),
        "quality_efficiency_120d": (quality, None, FULL_WINDOW),
        "intangible_growth_quality_120d": (
            intangible,
            None,
            FULL_WINDOW,
        ),
        "q1_dupont_improvement_80d": (dupont, 1, Q1_WINDOW),
        "q1_quality_efficiency_80d": (quality, 1, Q1_WINDOW),
        "q1_intangible_growth_quality_80d": (
            intangible,
            1,
            Q1_WINDOW,
        ),
    }
    for candidate, (components, quarter, max_age) in composite_specs.items():
        columns = [f"{prefix}_value" for prefix in components]
        mask = _same_quarter_mask(data, components, quarter, max_age)
        result[candidate] = _complete_rank_mean(ranks, columns).where(mask)

    q1_direct = {
        "q1_gross_margin_change_80d": "margin",
        "q1_asset_turnover_change_80d": "turnover",
        "q1_noa_quality_80d": "noa",
        "q1_cfo_profitability_80d": "cfoa",
        "q1_rd_intensity_80d": "rd",
    }
    for candidate, prefix in q1_direct.items():
        result[candidate] = data[f"{prefix}_value"].where(q1_masks[prefix])

    for column in CANDIDATE_COLUMNS:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        ).astype("float32")
    return result[KEYS + CANDIDATE_COLUMNS]


def generate_candidates(
    factor_path: Path,
    component_dir: Path,
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

    event_specs = load_event_specs(component_dir)
    event_tables = {
        prefix: _prepare_event_table_vectorized(
            events,
            calendar,
            value_column=value,
        )
        for prefix, (events, value) in event_specs.items()
    }

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
                columns=KEYS,
                filters=filters,
            )
            panel = _normalize_panel(panel)
            mapped = {
                prefix: _map_events_vectorized(
                    panel,
                    event_tables[prefix],
                    calendar_ordinal,
                    event_specs[prefix][1],
                    prefix,
                )
                for prefix in event_tables
            }
            candidates = build_candidates_for_slice(panel, mapped)
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
        raise RuntimeError("No candidate data was generated")
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
        "--component-dir",
        type=Path,
        default=BASE_DIR / "factor_components",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            BASE_DIR
            / "factor_components"
            / "unreplicated_financial_candidates.parquet"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_candidates(
        args.factors.resolve(),
        args.component_dir.resolve(),
        args.output.resolve(),
    )
