"""Quarterly time-series financial factors (round 15).

Families:
1. Earnings acceleration: quarter-over-quarter change in seasonally
   differenced quarterly growth, following He and Narayanamoorthy (2020).
2. Foster-style forecast errors: rolling AR(1) forecasts of seasonally
   differenced quarterly fundamentals, estimated using prior quarters only.
3. Surprise streaks and breadth: persistence of same-sign surprises and
   agreement across earnings, sales, and operating cash flow.

All flow inputs remain sparse until the diagnostic IC test is complete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eighth_round_literature_factors as workflow
from dense_literature_profitability_factors import _merge_flow_tables
from event_financial_factor_search import KEYS, _normalize_panel
from quarterly_f_score import COMMON_COLUMNS, build_standalone_quarterly_metric
from tenth_round_misstatement_factors import BALANCE_FIELDS, build_balance_events


INCOME_FIELDS = ["N_INCOME_ATTR_P", "OPERATE_PROFIT", "REVENUE", "COGS"]
CASHFLOW_FIELDS = ["N_CF_OPERATE_A"]
CANDIDATE_COLUMNS = [
    "r15_ni_acceleration_assets",
    "r15_op_acceleration_assets",
    "r15_gp_acceleration_assets",
    "r15_revenue_acceleration_assets",
    "r15_cfo_acceleration_assets",
    "r15_profit_acceleration_equal",
    "r15_quality_acceleration_equal",
    "r15_ni_foster_sue",
    "r15_op_foster_sue",
    "r15_gp_foster_sue",
    "r15_ni_streak_sue",
    "r15_op_streak_sue",
    "r15_gp_streak_sue",
    "r15_sue_breadth",
    "r15_acceleration_breadth",
    "r15_ni_positive_growth_acceleration",
    "r15_gp_streak_sue4",
    "r15_gp_streak_sue6",
    "r15_sue_breadth4",
    "r15_sue_breadth6",
]


def _read_inputs(pit_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    income = pd.read_parquet(
        pit_dir / "new_pit_income",
        columns=COMMON_COLUMNS + INCOME_FIELDS,
        engine="pyarrow",
    )
    cashflow = pd.read_parquet(
        pit_dir / "new_pit_cashflow",
        columns=COMMON_COLUMNS + CASHFLOW_FIELDS,
        engine="pyarrow",
    )
    balance_raw = pd.read_parquet(
        pit_dir / "new_pit_balance",
        columns=COMMON_COLUMNS + BALANCE_FIELDS,
        engine="pyarrow",
    )
    flows: dict[str, pd.DataFrame] = {}
    for field in INCOME_FIELDS:
        flows[field] = build_standalone_quarterly_metric(income, field, name="income PIT")
    for field in CASHFLOW_FIELDS:
        flows[field] = build_standalone_quarterly_metric(cashflow, field, name="cashflow PIT")
    return flows, build_balance_events(balance_raw)


def _metric_frame(flows: dict[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    if metric != "GP":
        source = flows[metric][
            ["SECURITY_ID", "QUARTER_INDEX", "EVENT_TIME", metric]
        ].copy()
        return source.rename(columns={"EVENT_TIME": "METRIC_EVENT_TIME", metric: "VALUE"})
    gross = _merge_flow_tables(flows, ["REVENUE", "COGS"], ttm=False)
    result = gross[["SECURITY_ID", "QUARTER_INDEX", "FLOW_EVENT_TIME"]].copy()
    result["VALUE"] = gross["REVENUE"] - gross["COGS"]
    return result.rename(columns={"FLOW_EVENT_TIME": "METRIC_EVENT_TIME"})


def _rolling_ar_forecast_error(values: pd.Series) -> pd.Series:
    """Prior-only rolling AR(1) error for seasonally differenced values."""
    y = values
    dependent = y.shift(1)
    regressor = y.shift(2)
    mean_y = dependent.rolling(12, min_periods=8).mean()
    mean_x = regressor.rolling(12, min_periods=8).mean()
    covariance = (dependent * regressor).rolling(12, min_periods=8).mean() - mean_y * mean_x
    variance = (regressor * regressor).rolling(12, min_periods=8).mean() - mean_x * mean_x
    beta = covariance.div(variance.where(variance.gt(1e-16))).clip(-0.95, 0.95)
    alpha = mean_y - beta * mean_x
    return y - (alpha + beta * y.shift(1))


def _signed_streak(values: pd.Series) -> pd.Series:
    signs = np.sign(values.to_numpy(dtype="float64"))
    result = np.full(len(values), np.nan, dtype="float64")
    previous = np.nan
    length = 0
    for index, sign in enumerate(signs):
        if not np.isfinite(sign) or sign == 0:
            previous = np.nan
            length = 0
            continue
        length = length + 1 if sign == previous else 1
        previous = sign
        result[index] = sign * min(length, 4)
    return pd.Series(result, index=values.index)


def _temporal_metric(
    frame: pd.DataFrame, assets: pd.DataFrame, prefix: str
) -> pd.DataFrame:
    merged = frame.merge(
        assets[["SECURITY_ID", "QUARTER_INDEX", "T_ASSETS"]],
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="left",
        validate="one_to_one",
    )
    pieces: list[pd.DataFrame] = []
    for security_id, group in merged.groupby("SECURITY_ID", sort=False):
        ordered = group.sort_values("QUARTER_INDEX")
        start = int(ordered["QUARTER_INDEX"].min())
        stop = int(ordered["QUARTER_INDEX"].max()) + 1
        full_index = pd.RangeIndex(start, stop)
        indexed = ordered.set_index("QUARTER_INDEX").reindex(full_index)
        value = pd.to_numeric(indexed["VALUE"], errors="coerce")
        asset = pd.to_numeric(indexed["T_ASSETS"], errors="coerce")
        seasonal = value - value.shift(4)
        denominator = asset.shift(1).abs().where(asset.shift(1).abs().gt(1e-12))
        growth = seasonal / denominator
        acceleration = growth - growth.shift(1)
        historical_std = seasonal.shift(1).rolling(8, min_periods=8).std(ddof=1)
        sue = seasonal.div(historical_std.where(historical_std.gt(1e-12)))
        historical_std4 = seasonal.shift(1).rolling(4, min_periods=4).std(ddof=1)
        historical_std6 = seasonal.shift(1).rolling(6, min_periods=6).std(ddof=1)
        sue4 = seasonal.div(historical_std4.where(historical_std4.gt(1e-12)))
        sue6 = seasonal.div(historical_std6.where(historical_std6.gt(1e-12)))
        forecast_error = _rolling_ar_forecast_error(seasonal)
        forecast_scale = forecast_error.shift(1).rolling(8, min_periods=8).std(ddof=1)
        foster_sue = forecast_error.div(forecast_scale.where(forecast_scale.gt(1e-12)))
        streak = _signed_streak(sue)
        streak_sue = sue * (1.0 + 0.25 * (streak.abs() - 1.0).clip(lower=0.0))
        streak4 = _signed_streak(sue4)
        streak6 = _signed_streak(sue6)
        streak_sue4 = sue4 * (1.0 + 0.25 * (streak4.abs() - 1.0).clip(lower=0.0))
        streak_sue6 = sue6 * (1.0 + 0.25 * (streak6.abs() - 1.0).clip(lower=0.0))
        positive_pattern = acceleration * (
            1.0 + 0.5 * (growth.gt(0.0) & growth.shift(1).gt(0.0)).astype("float64")
        )
        available = indexed.loc[ordered["QUARTER_INDEX"].to_numpy()].copy()
        available["SECURITY_ID"] = security_id
        available["QUARTER_INDEX"] = ordered["QUARTER_INDEX"].to_numpy()
        available[f"{prefix}_growth"] = growth.loc[ordered["QUARTER_INDEX"]].to_numpy()
        available[f"{prefix}_acceleration"] = acceleration.loc[ordered["QUARTER_INDEX"]].to_numpy()
        available[f"{prefix}_sue"] = sue.loc[ordered["QUARTER_INDEX"]].to_numpy()
        available[f"{prefix}_sue4"] = sue4.loc[ordered["QUARTER_INDEX"]].to_numpy()
        available[f"{prefix}_sue6"] = sue6.loc[ordered["QUARTER_INDEX"]].to_numpy()
        available[f"{prefix}_foster_sue"] = foster_sue.loc[ordered["QUARTER_INDEX"]].to_numpy()
        available[f"{prefix}_streak_sue"] = streak_sue.loc[ordered["QUARTER_INDEX"]].to_numpy()
        available[f"{prefix}_streak_sue4"] = streak_sue4.loc[
            ordered["QUARTER_INDEX"]
        ].to_numpy()
        available[f"{prefix}_streak_sue6"] = streak_sue6.loc[
            ordered["QUARTER_INDEX"]
        ].to_numpy()
        available[f"{prefix}_positive_acceleration"] = positive_pattern.loc[
            ordered["QUARTER_INDEX"]
        ].to_numpy()
        pieces.append(
            available[
                [
                    "SECURITY_ID",
                    "QUARTER_INDEX",
                    "METRIC_EVENT_TIME",
                    f"{prefix}_growth",
                    f"{prefix}_acceleration",
                    f"{prefix}_sue",
                    f"{prefix}_sue4",
                    f"{prefix}_sue6",
                    f"{prefix}_foster_sue",
                    f"{prefix}_streak_sue",
                    f"{prefix}_streak_sue4",
                    f"{prefix}_streak_sue6",
                    f"{prefix}_positive_acceleration",
                ]
            ]
        )
    return pd.concat(pieces, ignore_index=True)


def _event(frame: pd.DataFrame, factor: str, value: pd.Series) -> pd.DataFrame:
    result = frame[["SECURITY_ID", "QUARTER_INDEX", "EVENT_TIME"]].copy()
    result["factor"] = factor
    result["value"] = pd.to_numeric(value, errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=["EVENT_TIME", "value"])


def calculate_factor_events(
    flows: dict[str, pd.DataFrame], balance: pd.DataFrame
) -> pd.DataFrame:
    assets = balance[["SECURITY_ID", "QUARTER_INDEX", "T_ASSETS"]].copy()
    metric_sources = {
        "ni": "N_INCOME_ATTR_P",
        "op": "OPERATE_PROFIT",
        "gp": "GP",
        "revenue": "REVENUE",
        "cfo": "N_CF_OPERATE_A",
    }
    temporal = {
        prefix: _temporal_metric(_metric_frame(flows, source), assets, prefix)
        for prefix, source in metric_sources.items()
    }
    events: list[pd.DataFrame] = []
    acceleration_map = {
        "ni": "r15_ni_acceleration_assets",
        "op": "r15_op_acceleration_assets",
        "gp": "r15_gp_acceleration_assets",
        "revenue": "r15_revenue_acceleration_assets",
        "cfo": "r15_cfo_acceleration_assets",
    }
    for prefix, factor in acceleration_map.items():
        frame = temporal[prefix].rename(columns={"METRIC_EVENT_TIME": "EVENT_TIME"})
        events.append(_event(frame, factor, frame[f"{prefix}_acceleration"]))
    for prefix, factor in {
        "ni": "r15_ni_foster_sue",
        "op": "r15_op_foster_sue",
        "gp": "r15_gp_foster_sue",
    }.items():
        frame = temporal[prefix].rename(columns={"METRIC_EVENT_TIME": "EVENT_TIME"})
        events.append(_event(frame, factor, frame[f"{prefix}_foster_sue"]))
    for prefix, factor in {
        "ni": "r15_ni_streak_sue",
        "op": "r15_op_streak_sue",
        "gp": "r15_gp_streak_sue",
    }.items():
        frame = temporal[prefix].rename(columns={"METRIC_EVENT_TIME": "EVENT_TIME"})
        events.append(_event(frame, factor, frame[f"{prefix}_streak_sue"]))
    gp_frame = temporal["gp"].rename(columns={"METRIC_EVENT_TIME": "EVENT_TIME"})
    events.extend(
        [
            _event(gp_frame, "r15_gp_streak_sue4", gp_frame["gp_streak_sue4"]),
            _event(gp_frame, "r15_gp_streak_sue6", gp_frame["gp_streak_sue6"]),
        ]
    )

    merged: pd.DataFrame | None = None
    for prefix, frame in temporal.items():
        renamed = frame.rename(columns={
            "METRIC_EVENT_TIME": f"{prefix}_EVENT_TIME",
            f"{prefix}_acceleration": f"{prefix}_accel",
        })
        keep = [
            "SECURITY_ID", "QUARTER_INDEX", f"{prefix}_EVENT_TIME",
            f"{prefix}_accel", f"{prefix}_sue", f"{prefix}_sue4", f"{prefix}_sue6",
        ]
        merged = renamed[keep] if merged is None else merged.merge(
            renamed[keep], on=["SECURITY_ID", "QUARTER_INDEX"], how="outer", validate="one_to_one"
        )
    assert merged is not None
    event_columns = [column for column in merged if column.endswith("EVENT_TIME")]
    merged["EVENT_TIME"] = merged[event_columns].max(axis=1)
    profit_accel = merged[["ni_accel", "op_accel", "gp_accel"]].mean(axis=1, skipna=False)
    quality_accel = merged[["op_accel", "gp_accel", "revenue_accel", "cfo_accel"]].mean(
        axis=1, skipna=False
    )
    sue_breadth = merged[["ni_sue", "op_sue", "gp_sue", "revenue_sue", "cfo_sue"]].mean(
        axis=1, skipna=True
    ).where(merged[["ni_sue", "op_sue", "gp_sue", "revenue_sue", "cfo_sue"]].notna().sum(axis=1).ge(3))
    sue4_columns = ["ni_sue4", "op_sue4", "gp_sue4", "revenue_sue4", "cfo_sue4"]
    sue6_columns = ["ni_sue6", "op_sue6", "gp_sue6", "revenue_sue6", "cfo_sue6"]
    sue_breadth4 = merged[sue4_columns].mean(axis=1, skipna=True).where(
        merged[sue4_columns].notna().sum(axis=1).ge(3)
    )
    sue_breadth6 = merged[sue6_columns].mean(axis=1, skipna=True).where(
        merged[sue6_columns].notna().sum(axis=1).ge(3)
    )
    accel_breadth = merged[
        ["ni_accel", "op_accel", "gp_accel", "revenue_accel", "cfo_accel"]
    ].mean(axis=1, skipna=True).where(
        merged[["ni_accel", "op_accel", "gp_accel", "revenue_accel", "cfo_accel"]]
        .notna().sum(axis=1).ge(3)
    )
    events.extend(
        [
            _event(merged, "r15_profit_acceleration_equal", profit_accel),
            _event(merged, "r15_quality_acceleration_equal", quality_accel),
            _event(merged, "r15_sue_breadth", sue_breadth),
            _event(merged, "r15_sue_breadth4", sue_breadth4),
            _event(merged, "r15_sue_breadth6", sue_breadth6),
            _event(merged, "r15_acceleration_breadth", accel_breadth),
        ]
    )
    ni = temporal["ni"].rename(columns={"METRIC_EVENT_TIME": "EVENT_TIME"})
    events.append(
        _event(
            ni,
            "r15_ni_positive_growth_acceleration",
            ni["ni_positive_acceleration"],
        )
    )
    result = pd.concat(events, ignore_index=True)
    result["QUARTER_INDEX"] = pd.to_numeric(result["QUARTER_INDEX"]).astype("int64")
    return result.sort_values(["factor", "SECURITY_ID", "EVENT_TIME", "QUARTER_INDEX"])


def generate_sparse(panel: Path, pit_dir: Path, output_dir: Path) -> None:
    dates = pd.read_parquet(panel, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    flows, balance = _read_inputs(pit_dir)
    events = calculate_factor_events(flows, balance)
    workflow.CANDIDATE_COLUMNS = CANDIDATE_COLUMNS
    wide = workflow.prepare_wide_events(events, calendar)
    chunks = []
    coverage = []
    for year in sorted(set(calendar.year)):
        filters = [
            ("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)),
            ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31)),
        ]
        keys = _normalize_panel(pd.read_parquet(panel, columns=KEYS, filters=filters))
        mapped = workflow._map_sparse(keys, wide)
        chunks.append(mapped)
        for factor in CANDIDATE_COLUMNS:
            series = mapped[factor]
            coverage.append(
                {
                    "year": year,
                    "factor": factor,
                    "rows": len(mapped),
                    "observed_rows": int(series.notna().sum()),
                    "missing_rate_before_fill": float(series.isna().mean()),
                    "observed_days": int(mapped.loc[series.notna(), "TRADE_DATE"].nunique()),
                }
            )
        print(f"{year}: {len(mapped):,} rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.concat(chunks, ignore_index=True).sort_values(KEYS)
    result.to_parquet(output_dir / "round15_sparse_before_fill.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_csv(
        output_dir / "round15_sparse_coverage.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "round15_metadata.json").write_text(
        json.dumps(
            {
                "stage": "sparse_before_fill",
                "rows": len(result),
                "factors": CANDIDATE_COLUMNS,
                "period_source_zero_fill": False,
                "rolling_models_use_prior_quarters_only": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def fill_after_test(sparse: Path, ic_summary: Path, output: Path, factors: list[str]) -> None:
    tested = set(pd.read_csv(ic_summary)["factor"].astype(str))
    if not set(factors).issubset(tested):
        raise RuntimeError(f"not sparse-tested: {sorted(set(factors) - tested)}")
    data = pd.read_parquet(sparse, columns=KEYS + factors)
    before = data[factors].isna().mean()
    medians = data.groupby("TRADE_DATE", sort=False)[factors].transform("median")
    data[factors] = data[factors].fillna(medians)
    whole_day_neutral_rows = data[factors].isna().sum()
    data[factors] = data[factors].fillna(0.0)
    remaining = data[factors].isna().sum()
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False, compression="zstd")
    pd.DataFrame(
        {
            "factor": factors,
            "missing_rate_before_fill": [before[f] for f in factors],
            "fill_method": "same_date_median_then_zero_for_whole_day_after_sparse_test",
            "whole_day_neutral_rows": [whole_day_neutral_rows[f] for f in factors],
            "remaining_missing_rows": [remaining[f] for f in factors],
        }
    ).to_csv(output.with_suffix(".fill_report.csv"), index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
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
    fill.add_argument("--factor-columns", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate-sparse":
        generate_sparse(args.panel.resolve(), args.pit_dir.resolve(), args.output_dir.resolve())
    else:
        fill_after_test(
            args.sparse.resolve(), args.sparse_ic.resolve(), args.output.resolve(), args.factor_columns
        )


if __name__ == "__main__":
    main()
