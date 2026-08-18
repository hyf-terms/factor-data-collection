"""Round 20: financial-reporting timeliness and revision behavior factors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eighth_round_literature_factors as workflow
from event_financial_factor_search import KEYS, _normalize_panel
from quarterly_f_score import (
    COMMON_COLUMNS,
    FINANCIAL_INDUSTRIES,
    REPORT_QUARTERS,
    REPORT_TYPES,
)


CANDIDATE_COLUMNS = [
    "r20_reporting_timeliness_level",
    "r20_reporting_timeliness_yoy",
    "r20_reporting_timeliness_acceleration",
    "r20_reporting_timeliness_sue4",
    "r20_reporting_interval_regularity",
    "r20_low_revision_count",
    "r20_low_revision_span",
    "r20_first_release_timeliness",
]


def _load_events(pit_dir: Path) -> pd.DataFrame:
    columns = COMMON_COLUMNS + ["INDUSTRY_CATEGORY"]
    data = pd.read_parquet(pit_dir / "new_pit_balance", columns=columns, engine="pyarrow")
    for column in ("ACT_PUBTIME", "END_DATE", "END_DATE_REP"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["END_DATE"] = data["END_DATE"].dt.normalize()
    data["END_DATE_REP"] = data["END_DATE_REP"].dt.normalize()
    data["SECURITY_ID"] = pd.to_numeric(data["SECURITY_ID"], errors="coerce")
    mask = (
        data["MERGED_FLAG"].astype("string").eq("1")
        & data["REPORT_TYPE"].astype("string").isin(REPORT_TYPES)
        & data["END_DATE"].eq(data["END_DATE_REP"])
        & ~data["INDUSTRY_CATEGORY"].isin(FINANCIAL_INDUSTRIES)
    )
    data = data.loc[mask].dropna(subset=["SECURITY_ID", "ACT_PUBTIME", "END_DATE"])
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    data["FISCAL_YEAR"] = data["END_DATE"].dt.year.astype("int16")
    data["FISCAL_QUARTER"] = data["REPORT_TYPE"].map(REPORT_QUARTERS).astype("int8")
    data["QUARTER_INDEX"] = data["FISCAL_YEAR"].astype("int64") * 4 + data["FISCAL_QUARTER"]
    # Repeated downloads share IDs; count only distinct database records.
    data = data.sort_values(["ACT_PUBTIME", "ID"]).drop_duplicates("ID", keep="first")
    keys = ["SECURITY_ID", "QUARTER_INDEX"]
    stats = data.groupby(keys, sort=False).agg(
        FIRST_EVENT_TIME=("ACT_PUBTIME", "min"),
        LAST_EVENT_TIME=("ACT_PUBTIME", "max"),
        END_DATE=("END_DATE", "first"),
        REVISION_COUNT=("ACT_PUBTIME", "nunique"),
    ).reset_index()
    current = data.loc[data["IS_CURRENT_PERIOD"].fillna(False).astype(bool)].copy()
    current = current.sort_values(keys + ["ACT_PUBTIME", "ID"]).drop_duplicates(keys, keep="first")
    current = current[keys + ["ACT_PUBTIME"]].rename(columns={"ACT_PUBTIME": "EVENT_TIME"})
    result = current.merge(stats, on=keys, how="left", validate="one_to_one")
    result["DELAY_DAYS"] = (result["EVENT_TIME"].dt.normalize() - result["END_DATE"]).dt.days
    result["FIRST_DELAY_DAYS"] = (result["FIRST_EVENT_TIME"].dt.normalize() - result["END_DATE"]).dt.days
    result["REVISION_SPAN_DAYS"] = (result["LAST_EVENT_TIME"] - result["FIRST_EVENT_TIME"]).dt.total_seconds() / 86400.0
    return result.sort_values(keys)


def _calculate(events: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for security_id, group in events.groupby("SECURITY_ID", sort=False):
        ordered = group.sort_values("QUARTER_INDEX")
        start, stop = int(ordered["QUARTER_INDEX"].min()), int(ordered["QUARTER_INDEX"].max()) + 1
        indexed = ordered.set_index("QUARTER_INDEX").reindex(pd.RangeIndex(start, stop))
        delay = pd.to_numeric(indexed["DELAY_DAYS"], errors="coerce")
        improvement = delay.shift(4) - delay
        acceleration = improvement - improvement.shift(1)
        scale = improvement.shift(1).rolling(4, min_periods=4).std(ddof=1)
        sue = improvement.div(scale.where(scale.gt(1e-12)))
        event_time = pd.to_datetime(indexed["EVENT_TIME"], errors="coerce")
        interval = event_time.diff().dt.total_seconds().div(86400.0)
        regularity = -(interval - 91.0).abs()
        available = indexed.loc[ordered["QUARTER_INDEX"].to_numpy()].copy()
        available["SECURITY_ID"] = security_id
        available["QUARTER_INDEX"] = ordered["QUARTER_INDEX"].to_numpy()
        available["r20_reporting_timeliness_level"] = -delay.reindex(ordered["QUARTER_INDEX"]).to_numpy()
        available["r20_reporting_timeliness_yoy"] = improvement.reindex(ordered["QUARTER_INDEX"]).to_numpy()
        available["r20_reporting_timeliness_acceleration"] = acceleration.reindex(ordered["QUARTER_INDEX"]).to_numpy()
        available["r20_reporting_timeliness_sue4"] = sue.reindex(ordered["QUARTER_INDEX"]).to_numpy()
        available["r20_reporting_interval_regularity"] = regularity.reindex(ordered["QUARTER_INDEX"]).to_numpy()
        available["r20_low_revision_count"] = -np.log1p(pd.to_numeric(available["REVISION_COUNT"], errors="coerce"))
        available["r20_low_revision_span"] = -pd.to_numeric(available["REVISION_SPAN_DAYS"], errors="coerce")
        available["r20_first_release_timeliness"] = -pd.to_numeric(available["FIRST_DELAY_DAYS"], errors="coerce")
        pieces.append(available[["SECURITY_ID", "QUARTER_INDEX", "EVENT_TIME", *CANDIDATE_COLUMNS]])
    wide = pd.concat(pieces, ignore_index=True)
    long = wide.melt(
        id_vars=["SECURITY_ID", "QUARTER_INDEX", "EVENT_TIME"],
        value_vars=CANDIDATE_COLUMNS,
        var_name="factor", value_name="value",
    )
    return long.replace([np.inf, -np.inf], np.nan).dropna(subset=["EVENT_TIME", "value"])


def generate_sparse(panel: Path, pit_dir: Path, output_dir: Path) -> None:
    dates = pd.read_parquet(panel, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    events = _calculate(_load_events(pit_dir))
    workflow.CANDIDATE_COLUMNS = CANDIDATE_COLUMNS
    wide = workflow.prepare_wide_events(events, calendar)
    chunks, coverage = [], []
    for year in sorted(set(calendar.year)):
        filters = [("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)), ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31))]
        keys = _normalize_panel(pd.read_parquet(panel, columns=KEYS, filters=filters))
        mapped = workflow._map_sparse(keys, wide)
        chunks.append(mapped)
        for factor in CANDIDATE_COLUMNS:
            series = mapped[factor]
            coverage.append({"year": year, "factor": factor, "rows": len(mapped), "observed_rows": int(series.notna().sum()), "missing_rate_before_fill": float(series.isna().mean()), "observed_days": int(mapped.loc[series.notna(), "TRADE_DATE"].nunique())})
        print(f"{year}: {len(mapped):,} rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.concat(chunks, ignore_index=True).sort_values(KEYS)
    result.to_parquet(output_dir / "round20_sparse_before_fill.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_csv(output_dir / "round20_sparse_coverage.csv", index=False, encoding="utf-8-sig")
    (output_dir / "round20_metadata.json").write_text(json.dumps({"stage": "sparse_before_fill", "factors": CANDIDATE_COLUMNS, "profit_amount_inputs": False, "existing_composite_inputs": False}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--pit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate_sparse(args.panel.resolve(), args.pit_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
