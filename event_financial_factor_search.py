"""Build event-conditioned financial factor candidates without label leakage.

The candidates combine strictly point-in-time quarterly earnings surprises
with already constructed PIT financial factors.  Candidate construction never
reads ``label.parquet``.  The resulting Parquet is intended to be evaluated by
``factors_neus_only.py``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pead_sue_factor import assign_available_trade_date, build_sue_events
from quarterly_f_score import build_standalone_quarterly_metric


BASE_DIR = Path(__file__).resolve().parent
KEYS = ["TRADE_DATE", "SECURITY_ID"]
BASE_FACTOR_COLUMNS = [
    "quarterly_f_score",
    "operating_profit_acceleration",
    "asset_growth",
    "profitability_quality_score",
]
CANDIDATE_COLUMNS = [
    "pead_sue_q1_80d",
    "pead_sue_high_ivol30_60d",
    "pead_sue_high_ivol30_80d",
    "deducted_sue_q1_80d",
    "q1_dual_sue",
    "q1_financial_composite",
    "event_financial_composite_80d",
    "event_financial_high_ivol30_80d",
]


def _normalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["TRADE_DATE"] = pd.to_datetime(
        result["TRADE_DATE"], errors="coerce"
    ).dt.normalize()
    result["SECURITY_ID"] = pd.to_numeric(
        result["SECURITY_ID"], errors="coerce"
    )
    result = result.dropna(subset=KEYS)
    result["SECURITY_ID"] = result["SECURITY_ID"].astype("int64")
    if result.duplicated(KEYS).any():
        raise ValueError("因子面板存在重复证券-交易日键")
    return result.sort_values(KEYS).reset_index(drop=True)


def _prepare_events(
    events: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    value_column: str,
) -> dict[int, pd.DataFrame]:
    required = {
        "SECURITY_ID",
        "QUARTER_INDEX",
        "FISCAL_QUARTER",
        "EVENT_TIME",
        value_column,
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise KeyError(f"事件数据缺少字段: {missing}")
    available = assign_available_trade_date(events, calendar)
    available = available.sort_values(
        ["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"]
    )
    groups: dict[int, pd.DataFrame] = {}
    for security_id, group in available.groupby("SECURITY_ID", sort=False):
        # A late filing for an old quarter must not replace newer information.
        newest = group["QUARTER_INDEX"].cummax()
        clean = group.loc[group["QUARTER_INDEX"].eq(newest)]
        clean = clean.drop_duplicates("AVAILABLE_DATE", keep="last")
        groups[int(security_id)] = clean[
            [
                "AVAILABLE_DATE",
                "FISCAL_QUARTER",
                "QUARTER_INDEX",
                value_column,
            ]
        ]
    return groups


def _map_events(
    panel: pd.DataFrame,
    event_groups: dict[int, pd.DataFrame],
    calendar_ordinal: pd.Series,
    value_column: str,
    output_prefix: str,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for security_id, stock_days in panel.groupby("SECURITY_ID", sort=False):
        right = event_groups.get(int(security_id))
        left = stock_days.sort_values("TRADE_DATE")
        if right is None or right.empty:
            pieces.append(
                left.assign(
                    **{
                        f"{output_prefix}_value": np.nan,
                        f"{output_prefix}_quarter": np.nan,
                        f"{output_prefix}_age": np.nan,
                    }
                )
            )
            continue
        joined = pd.merge_asof(
            left,
            right,
            left_on="TRADE_DATE",
            right_on="AVAILABLE_DATE",
            direction="backward",
        )
        trade_ordinal = joined["TRADE_DATE"].map(calendar_ordinal)
        event_ordinal = joined["AVAILABLE_DATE"].map(calendar_ordinal)
        joined[f"{output_prefix}_age"] = trade_ordinal - event_ordinal
        joined = joined.rename(
            columns={
                value_column: f"{output_prefix}_value",
                "FISCAL_QUARTER": f"{output_prefix}_quarter",
            }
        )
        pieces.append(
            joined[
                KEYS
                + [
                    f"{output_prefix}_value",
                    f"{output_prefix}_quarter",
                    f"{output_prefix}_age",
                ]
            ]
        )
    return pd.concat(pieces, ignore_index=True).sort_values(KEYS)


def build_deducted_sue_events(indicator: pd.DataFrame) -> pd.DataFrame:
    """Construct SUE from standalone-quarter deducted net income."""
    source = indicator.copy()
    source["IS_CURRENT_PERIOD"] = True
    quarterly = build_standalone_quarterly_metric(
        source,
        "N_INCOME_CUT",
        name="扣非净利润PIT",
    ).rename(
        columns={
            "N_INCOME_CUT": "QUARTERLY_EARNINGS",
            "N_INCOME_CUT_SOURCE": "SOURCE",
        }
    )
    return build_sue_events(quarterly)


def _daily_percentile(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    return frame.groupby("TRADE_DATE", sort=False)[columns].rank(pct=True)


def build_candidates_for_slice(
    factors: pd.DataFrame,
    barra: pd.DataFrame,
    pead_mapped: pd.DataFrame,
    deducted_mapped: pd.DataFrame,
) -> pd.DataFrame:
    data = pd.merge(
        factors,
        barra[KEYS + ["residual_volatility"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    data = pd.merge(
        data,
        pead_mapped,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    data = pd.merge(
        data,
        deducted_mapped,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    pead_80 = data["pead_age"].ge(0) & data["pead_age"].lt(80)
    pead_60 = data["pead_age"].ge(0) & data["pead_age"].lt(60)
    deducted_80 = (
        data["deducted_age"].ge(0) & data["deducted_age"].lt(80)
    )
    pead_q1 = pead_80 & data["pead_quarter"].eq(1)
    deducted_q1 = deducted_80 & data["deducted_quarter"].eq(1)

    eligible_ivol = data["residual_volatility"].where(pead_80)
    ivol_percentile = eligible_ivol.groupby(
        data["TRADE_DATE"], sort=False
    ).rank(pct=True)
    high_ivol = ivol_percentile.ge(0.70)

    result = data[KEYS].copy()
    result["pead_sue_q1_80d"] = data["pead_value"].where(pead_q1)
    result["pead_sue_high_ivol30_60d"] = data["pead_value"].where(
        pead_60 & high_ivol
    )
    result["pead_sue_high_ivol30_80d"] = data["pead_value"].where(
        pead_80 & high_ivol
    )
    result["deducted_sue_q1_80d"] = data["deducted_value"].where(
        deducted_q1
    )

    rank_source = data[
        [
            "TRADE_DATE",
            "pead_value",
            "deducted_value",
            *BASE_FACTOR_COLUMNS,
        ]
    ].copy()
    ranks = _daily_percentile(
        rank_source,
        [
            "pead_value",
            "deducted_value",
            *BASE_FACTOR_COLUMNS,
        ],
    )
    ranks["asset_growth"] = 1.0 - ranks["asset_growth"]

    dual_mask = pead_q1 & deducted_q1
    result["q1_dual_sue"] = ranks[
        ["pead_value", "deducted_value"]
    ].mean(axis=1, skipna=False).where(dual_mask)

    q1_components = [
        "pead_value",
        "deducted_value",
        "quarterly_f_score",
        "operating_profit_acceleration",
        "asset_growth",
        "profitability_quality_score",
    ]
    result["q1_financial_composite"] = ranks[q1_components].mean(
        axis=1, skipna=False
    ).where(dual_mask)

    event_components = [
        "pead_value",
        "quarterly_f_score",
        "operating_profit_acceleration",
        "asset_growth",
        "profitability_quality_score",
    ]
    event_composite = ranks[event_components].mean(axis=1, skipna=False)
    result["event_financial_composite_80d"] = event_composite.where(pead_80)
    result["event_financial_high_ivol30_80d"] = event_composite.where(
        pead_80 & high_ivol
    )

    for column in CANDIDATE_COLUMNS:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        ).astype("float32")
    return result[KEYS + CANDIDATE_COLUMNS]


def generate_candidates(
    factor_path: Path,
    barra_path: Path,
    pead_event_path: Path,
    indicator_path: Path,
    output_path: Path,
) -> None:
    schema = pq.read_schema(factor_path)
    required_factors = set(KEYS + BASE_FACTOR_COLUMNS)
    missing = sorted(required_factors.difference(schema.names))
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
    pead_groups = _prepare_events(
        pead_events, calendar, value_column="SUE_RAW"
    )
    deducted_groups = _prepare_events(
        deducted_events, calendar, value_column="SUE_RAW"
    )

    years = sorted(set(calendar.year))
    temporary = output_path.with_suffix(".tmp.parquet")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for year in years:
            start = pd.Timestamp(year=year, month=1, day=1)
            end = pd.Timestamp(year=year, month=12, day=31)
            filters = [
                ("TRADE_DATE", ">=", start),
                ("TRADE_DATE", "<=", end),
            ]
            factors = pd.read_parquet(
                factor_path,
                columns=KEYS + BASE_FACTOR_COLUMNS,
                filters=filters,
            )
            factors = _normalize_panel(factors)
            barra = pd.read_parquet(
                barra_path,
                columns=KEYS + ["residual_volatility"],
                filters=filters,
            )
            barra = _normalize_panel(barra)
            pead_mapped = _map_events(
                factors[KEYS],
                pead_groups,
                calendar_ordinal,
                "SUE_RAW",
                "pead",
            )
            deducted_mapped = _map_events(
                factors[KEYS],
                deducted_groups,
                calendar_ordinal,
                "SUE_RAW",
                "deducted",
            )
            candidates = build_candidates_for_slice(
                factors,
                barra,
                pead_mapped,
                deducted_mapped,
            )
            table = pa.Table.from_pandas(candidates, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                )
            writer.write_table(table)
            counts = candidates[CANDIDATE_COLUMNS].notna().sum()
            print(
                f"{year}: rows={len(candidates):,}; "
                f"non-null={counts.to_dict()}"
            )
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("没有生成任何候选数据")
    os.replace(temporary, output_path)
    print(f"Wrote candidates to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factors", type=Path, default=BASE_DIR / "factors.parquet"
    )
    parser.add_argument(
        "--barra", type=Path, default=BASE_DIR / "barra_diy.parquet"
    )
    parser.add_argument(
        "--pead-events",
        type=Path,
        default=BASE_DIR / "factor_components" / "pead_sue_events.parquet",
    )
    parser.add_argument(
        "--indicator",
        type=Path,
        default=BASE_DIR / "data" / "ch_models" / "earnings_pit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            BASE_DIR
            / "factor_components"
            / "event_financial_candidates.parquet"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_candidates(
        args.factors.resolve(),
        args.barra.resolve(),
        args.pead_events.resolve(),
        args.indicator.resolve(),
        args.output.resolve(),
    )
