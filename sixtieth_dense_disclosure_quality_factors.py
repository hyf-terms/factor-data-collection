"""Round 60: dense continuous financial-disclosure quality states.

This extends (rather than repeats) round 20 timeliness counts and round 27
guidance/express corrections.  It uses the historical versions preserved in
the three PIT statements to measure company-specific delay anomalies,
cross-statement release asynchrony, multi-field revision magnitude, and
four-quarter revision burden.  No return label or percentile rank is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eighth_round_literature_factors as workflow
from event_financial_factor_search import KEYS, _normalize_panel
from quarterly_f_score import COMMON_COLUMNS, FINANCIAL_INDUSTRIES, REPORT_QUARTERS, REPORT_TYPES


TABLE_FIELDS = {
    "balance": ["T_ASSETS", "T_LIAB", "T_EQUITY_ATTR_P", "CASH_C_EQUIV"],
    "income": ["REVENUE", "OPERATE_PROFIT", "N_INCOME_ATTR_P"],
    "cashflow": ["N_CF_OPERATE_A", "N_CF_FR_INVEST_A", "N_CF_FR_FINAN_A"],
}
CANDIDATE_COLUMNS = [
    "r60_delay_vs_own_history",
    "r60_low_delay_abnormality",
    "r60_low_three_statement_async",
    "r60_low_multifield_revision_magnitude",
    "r60_low_revision_burden4q",
    "r60_disclosure_quality_equal4",
]


def _symmetric_change(latest: pd.Series, first: pd.Series) -> pd.Series:
    denominator = latest.abs() + first.abs()
    return (2.0 * (latest - first)).div(denominator.where(denominator.gt(1.0))).clip(-2, 2)


def _load_statement(pit_dir: Path, table: str, fields: list[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(COMMON_COLUMNS + ["INDUSTRY_CATEGORY"] + fields))
    data = pd.read_parquet(pit_dir / f"new_pit_{table}", columns=columns, engine="pyarrow")
    for column in ["ACT_PUBTIME", "END_DATE", "END_DATE_REP"]:
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["END_DATE"] = data["END_DATE"].dt.normalize()
    data["END_DATE_REP"] = data["END_DATE_REP"].dt.normalize()
    data["SECURITY_ID"] = pd.to_numeric(data["SECURITY_ID"], errors="coerce")
    mask = (
        data["MERGED_FLAG"].astype("string").eq("1")
        & data["REPORT_TYPE"].astype("string").isin(REPORT_TYPES)
        & data["IS_CURRENT_PERIOD"].fillna(False).astype(bool)
        & data["END_DATE"].eq(data["END_DATE_REP"])
        & ~data["INDUSTRY_CATEGORY"].isin(FINANCIAL_INDUSTRIES)
    )
    data = data.loc[mask].dropna(subset=["SECURITY_ID", "ACT_PUBTIME", "END_DATE"])
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    data["FISCAL_QUARTER"] = data["REPORT_TYPE"].map(REPORT_QUARTERS)
    data = data.dropna(subset=["FISCAL_QUARTER"])
    data["QUARTER_INDEX"] = data["END_DATE"].dt.year.astype("int64") * 4 + data["FISCAL_QUARTER"].astype("int64")
    for field in fields:
        data[field] = pd.to_numeric(data[field], errors="coerce")
    data = data.sort_values(["SECURITY_ID", "QUARTER_INDEX", "ACT_PUBTIME", "ID"]).drop_duplicates("ID", keep="first")
    key = ["SECURITY_ID", "QUARTER_INDEX"]
    first = data.drop_duplicates(key, keep="first").set_index(key)
    latest = data.drop_duplicates(key, keep="last").set_index(key)
    stats = data.groupby(key, sort=False).agg(
        FIRST_EVENT_TIME=("ACT_PUBTIME", "min"),
        LAST_EVENT_TIME=("ACT_PUBTIME", "max"),
        VERSION_COUNT=("ACT_PUBTIME", "nunique"),
        END_DATE=("END_DATE", "first"),
    )
    changes = pd.DataFrame(index=stats.index)
    for field in fields:
        changes[field] = _symmetric_change(latest[field], first[field]).abs()
    observed = changes.notna().sum(axis=1)
    stats["REVISION_MAGNITUDE"] = changes.mean(axis=1, skipna=True).where(observed.ge(2))
    stats.loc[stats["VERSION_COUNT"].eq(1), "REVISION_MAGNITUDE"] = 0.0
    stats["REVISION_SPAN"] = (
        stats["LAST_EVENT_TIME"] - stats["FIRST_EVENT_TIME"]
    ).dt.total_seconds().div(86400.0).clip(lower=0.0)
    prefix = table.upper()
    return stats.reset_index().rename(columns={
        "FIRST_EVENT_TIME": f"{prefix}_FIRST_TIME",
        "LAST_EVENT_TIME": f"{prefix}_LAST_TIME",
        "VERSION_COUNT": f"{prefix}_VERSION_COUNT",
        "REVISION_MAGNITUDE": f"{prefix}_REVISION_MAGNITUDE",
        "REVISION_SPAN": f"{prefix}_REVISION_SPAN",
        "END_DATE": f"{prefix}_END_DATE",
    })


def build_quarterly_states(pit_dir: Path) -> pd.DataFrame:
    tables = [_load_statement(pit_dir, name, fields) for name, fields in TABLE_FIELDS.items()]
    merged = tables[0]
    for table in tables[1:]:
        merged = merged.merge(table, on=["SECURITY_ID", "QUARTER_INDEX"], how="outer", validate="one_to_one")
    first_times = [f"{name.upper()}_FIRST_TIME" for name in TABLE_FIELDS]
    last_times = [f"{name.upper()}_LAST_TIME" for name in TABLE_FIELDS]
    merged["EVENT_TIME"] = merged[last_times].max(axis=1)
    merged["END_DATE"] = merged[[f"{name.upper()}_END_DATE" for name in TABLE_FIELDS]].bfill(axis=1).iloc[:, 0]
    merged["DELAY_DAYS"] = (merged["EVENT_TIME"].dt.normalize() - merged["END_DATE"]).dt.days
    merged["THREE_STATEMENT_ASYNC"] = (
        merged[first_times].max(axis=1) - merged[first_times].min(axis=1)
    ).dt.total_seconds().div(86400.0).where(merged[first_times].notna().sum(axis=1).eq(3))
    revision_columns = [f"{name.upper()}_REVISION_MAGNITUDE" for name in TABLE_FIELDS]
    merged["MULTIFIELD_REVISION_MAGNITUDE"] = merged[revision_columns].mean(axis=1, skipna=True)
    merged["REVISION_ANY"] = merged[
        [f"{name.upper()}_VERSION_COUNT" for name in TABLE_FIELDS]
    ].max(axis=1).gt(1).astype("float64")

    pieces = []
    for security_id, group in merged.groupby("SECURITY_ID", sort=False):
        ordered = group.sort_values("QUARTER_INDEX")
        full = pd.RangeIndex(int(ordered.QUARTER_INDEX.min()), int(ordered.QUARTER_INDEX.max()) + 1)
        x = ordered.set_index("QUARTER_INDEX").reindex(full)
        delay = pd.to_numeric(x["DELAY_DAYS"], errors="coerce")
        prior_median = delay.shift(1).rolling(8, min_periods=4).median()
        delay_vs_history = prior_median - delay
        delay_abnormality = -(delay - prior_median).abs()
        revision = pd.to_numeric(x["MULTIFIELD_REVISION_MAGNITUDE"], errors="coerce")
        burden = -(revision + 0.25 * pd.to_numeric(x["REVISION_ANY"], errors="coerce")).rolling(4, min_periods=2).mean()
        available = x.loc[ordered.QUARTER_INDEX.to_numpy()].copy()
        available["SECURITY_ID"] = security_id
        available["QUARTER_INDEX"] = ordered.QUARTER_INDEX.to_numpy()
        available["r60_delay_vs_own_history"] = delay_vs_history.loc[ordered.QUARTER_INDEX].to_numpy()
        available["r60_low_delay_abnormality"] = delay_abnormality.loc[ordered.QUARTER_INDEX].to_numpy()
        available["r60_low_three_statement_async"] = -pd.to_numeric(available["THREE_STATEMENT_ASYNC"], errors="coerce")
        available["r60_low_multifield_revision_magnitude"] = -pd.to_numeric(available["MULTIFIELD_REVISION_MAGNITUDE"], errors="coerce")
        available["r60_low_revision_burden4q"] = burden.loc[ordered.QUARTER_INDEX].to_numpy()
        pieces.append(available)
    result = pd.concat(pieces, ignore_index=True)
    quarter = result["QUARTER_INDEX"]
    components = [
        "r60_delay_vs_own_history", "r60_low_delay_abnormality",
        "r60_low_three_statement_async", "r60_low_multifield_revision_magnitude",
    ]
    z = []
    for column in components:
        values = pd.to_numeric(result[column], errors="coerce")
        center = values.groupby(quarter, sort=False).transform("median")
        mad = (values - center).abs().groupby(quarter, sort=False).transform("median")
        standardized = ((values - center) / (1.4826 * mad).where(mad.gt(1e-12))).clip(-8, 8)
        z.append(standardized.rename(column))
    component_z = pd.concat(z, axis=1)
    result["r60_disclosure_quality_equal4"] = component_z.mean(axis=1, skipna=True).where(
        component_z.notna().sum(axis=1).ge(2)
    )
    return result


def calculate_events(states: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for factor in CANDIDATE_COLUMNS:
        event = states[["SECURITY_ID", "QUARTER_INDEX", "EVENT_TIME"]].copy()
        event["factor"] = factor
        event["value"] = pd.to_numeric(states[factor], errors="coerce")
        pieces.append(event.dropna(subset=["EVENT_TIME", "value"]))
    return pd.concat(pieces, ignore_index=True).sort_values(["factor", "SECURITY_ID", "EVENT_TIME", "QUARTER_INDEX"])


def generate_sparse(panel: Path, pit_dir: Path, output_dir: Path) -> None:
    dates = pd.read_parquet(panel, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    states = build_quarterly_states(pit_dir)
    events = calculate_events(states)
    workflow.CANDIDATE_COLUMNS = CANDIDATE_COLUMNS
    wide = workflow.prepare_wide_events(events, calendar)
    chunks, coverage = [], []
    for year in sorted(set(calendar.year)):
        filters = [("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)), ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31))]
        keys = _normalize_panel(pd.read_parquet(panel, columns=KEYS, filters=filters))
        mapped = workflow._map_sparse(keys, wide)
        chunks.append(mapped)
        for factor in CANDIDATE_COLUMNS:
            values = mapped[factor]
            coverage.append({"year": year, "factor": factor, "rows": len(mapped), "observed_rows": int(values.notna().sum()), "missing_rate_before_fill": float(values.isna().mean()), "observed_days": int(mapped.loc[values.notna(), "TRADE_DATE"].nunique())})
        print(f"{year}: {len(mapped):,} rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.concat(chunks, ignore_index=True).sort_values(KEYS)
    result.to_parquet(output_dir / "round60_sparse_before_fill.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_csv(output_dir / "round60_sparse_coverage.csv", index=False, encoding="utf-8-sig")
    states.to_parquet(output_dir / "round60_quarterly_state_audit.parquet", index=False, compression="zstd")
    (output_dir / "round60_metadata.json").write_text(json.dumps({"factors": CANDIDATE_COLUMNS, "uses_rank": False, "uses_label": False, "q1_or_60d_restriction": False}, ensure_ascii=False, indent=2), encoding="utf-8")


def fill_after_test(sparse: Path, sparse_ic: Path, output: Path) -> None:
    tested = set(pd.read_csv(sparse_ic)["factor"].astype(str))
    if not set(CANDIDATE_COLUMNS).issubset(tested):
        raise RuntimeError("all candidates must be sparse-tested before filling")
    data = pd.read_parquet(sparse, columns=KEYS + CANDIDATE_COLUMNS)
    before = data[CANDIDATE_COLUMNS].isna().mean()
    medians = data.groupby("TRADE_DATE", sort=False)[CANDIDATE_COLUMNS].transform("median")
    data[CANDIDATE_COLUMNS] = data[CANDIDATE_COLUMNS].fillna(medians)
    whole_day = data[CANDIDATE_COLUMNS].isna().sum()
    data[CANDIDATE_COLUMNS] = data[CANDIDATE_COLUMNS].fillna(0.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False, compression="zstd")
    pd.DataFrame({"factor": CANDIDATE_COLUMNS, "missing_rate_before_fill": [float(before[f]) for f in CANDIDATE_COLUMNS], "whole_day_neutral_rows": [int(whole_day[f]) for f in CANDIDATE_COLUMNS], "remaining_missing_rows": [int(data[f].isna().sum()) for f in CANDIDATE_COLUMNS]}).to_csv(output.with_suffix(".fill_report.csv"), index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-sparse")
    generate.add_argument("--panel", type=Path, required=True)
    generate.add_argument("--pit-dir", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    fill = sub.add_parser("fill-after-test")
    fill.add_argument("--sparse", type=Path, required=True)
    fill.add_argument("--sparse-ic", type=Path, required=True)
    fill.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate-sparse":
        generate_sparse(args.panel.resolve(), args.pit_dir.resolve(), args.output_dir.resolve())
    else:
        fill_after_test(args.sparse.resolve(), args.sparse_ic.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
