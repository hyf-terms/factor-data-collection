"""PIT-safe announcement-age variants of independent accounting surprises.

This module applies only deterministic half-life decay to three literature
signals (core operating-profit surprise, revenue SUE and tax-expense surprise).
It does not use or residualize against any existing factor.  Missing period
observations remain missing in the generated sparse file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from event_financial_factor_search import KEYS, _normalize_panel
from pead_sue_factor import assign_available_trade_date
from eleventh_round_tax_capex_factors import (
    _historical_surprise_std,
    _read_inputs,
    _ratio,
    _with_assets,
    _with_lag4,
)


BASE_DIR = Path(__file__).resolve().parent
BASE_SIGNALS = ["core", "core_sue", "margin_sue", "revenue", "tax"]
CANDIDATE_COLUMNS = [
    "r12_core_profit_surprise_hl20",
    "r12_core_profit_surprise_hl60",
    "r12_core_profit_surprise_hl120",
    "r12_core_profit_sue_hl60",
    "r12_core_profit_sue_hl120",
    "r12_operating_margin_sue_hl60",
    "r12_operating_margin_sue_hl120",
    "r12_revenue_sue_hl20",
    "r12_revenue_sue_hl60",
    "r12_revenue_sue_hl120",
    "r12_tax_expense_surprise_hl60",
    "r12_tax_expense_surprise_hl120",
]


def _event_table(frame: pd.DataFrame, value: pd.Series, event_columns: list[str]) -> pd.DataFrame:
    result = frame[["SECURITY_ID", "QUARTER_INDEX"]].copy()
    result["EVENT_TIME"] = pd.concat(
        [pd.to_datetime(frame[column], errors="coerce") for column in event_columns], axis=1
    ).max(axis=1)
    result["value"] = pd.to_numeric(value, errors="coerce")
    return result.dropna(subset=["SECURITY_ID", "QUARTER_INDEX", "EVENT_TIME", "value"])


def calculate_base_events(pit_dir: Path) -> dict[str, pd.DataFrame]:
    flows, balance = _read_inputs(pit_dir)

    core = _with_assets(_with_lag4(flows, ["OPERATE_PROFIT", "REVENUE"]), balance)
    core_average_assets = (core["T_ASSETS"] + core["L4_T_ASSETS"]) / 2.0
    core_surprise = core["OPERATE_PROFIT"] - core["L4_OPERATE_PROFIT"]
    core_value = _ratio(core_surprise, core_average_assets)
    core_std = _historical_surprise_std(core, core_surprise)
    core_sue_value = core_surprise.div(core_std.where(core_std.gt(0))).replace(
        [np.inf, -np.inf], np.nan
    )
    margin_surprise = _ratio(core["OPERATE_PROFIT"], core["REVENUE"]) - _ratio(
        core["L4_OPERATE_PROFIT"], core["L4_REVENUE"]
    )
    margin_std = _historical_surprise_std(core, margin_surprise)
    margin_sue_value = margin_surprise.div(margin_std.where(margin_std.gt(0))).replace(
        [np.inf, -np.inf], np.nan
    )
    core_events = _event_table(
        core,
        core_value,
        ["FLOW_EVENT_TIME", "L4_FLOW_EVENT_TIME", "BALANCE_EVENT_TIME", "L4_BALANCE_EVENT_TIME"],
    )
    core_sue_events = _event_table(
        core,
        core_sue_value,
        ["FLOW_EVENT_TIME", "L4_FLOW_EVENT_TIME", "BALANCE_EVENT_TIME", "L4_BALANCE_EVENT_TIME"],
    )
    margin_sue_events = _event_table(
        core,
        margin_sue_value,
        ["FLOW_EVENT_TIME", "L4_FLOW_EVENT_TIME", "BALANCE_EVENT_TIME", "L4_BALANCE_EVENT_TIME"],
    )

    revenue = _with_lag4(flows, ["REVENUE"])
    revenue_surprise = revenue["REVENUE"] - revenue["L4_REVENUE"]
    revenue_std = _historical_surprise_std(revenue, revenue_surprise)
    revenue_value = revenue_surprise.div(revenue_std.where(revenue_std.gt(0))).replace(
        [np.inf, -np.inf], np.nan
    )
    revenue_events = _event_table(
        revenue, revenue_value, ["FLOW_EVENT_TIME", "L4_FLOW_EVENT_TIME"]
    )

    tax = _with_assets(_with_lag4(flows, ["INCOME_TAX"]), balance)
    tax_average_assets = (tax["T_ASSETS"] + tax["L4_T_ASSETS"]) / 2.0
    tax_value = _ratio(tax["INCOME_TAX"] - tax["L4_INCOME_TAX"], tax_average_assets)
    tax_events = _event_table(
        tax,
        tax_value,
        ["FLOW_EVENT_TIME", "L4_FLOW_EVENT_TIME", "BALANCE_EVENT_TIME", "L4_BALANCE_EVENT_TIME"],
    )
    return {
        "core": core_events,
        "core_sue": core_sue_events,
        "margin_sue": margin_sue_events,
        "revenue": revenue_events,
        "tax": tax_events,
    }


def prepare_events(
    events: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex
) -> dict[str, pd.DataFrame]:
    prepared: dict[str, pd.DataFrame] = {}
    for name, frame in events.items():
        available = assign_available_trade_date(frame, calendar)
        available = available.sort_values(
            ["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"]
        )
        newest = available.groupby("SECURITY_ID", sort=False)["QUARTER_INDEX"].cummax()
        available = available.loc[available["QUARTER_INDEX"].eq(newest)]
        available = available.drop_duplicates(["SECURITY_ID", "AVAILABLE_DATE"], keep="last")
        prepared[name] = available[
            ["SECURITY_ID", "AVAILABLE_DATE", "value"]
        ].sort_values(["AVAILABLE_DATE", "SECURITY_ID"])
    return prepared


def _map_base(panel: pd.DataFrame, event: pd.DataFrame, name: str) -> pd.DataFrame:
    right = event.rename(
        columns={"AVAILABLE_DATE": f"{name}_AVAILABLE_DATE", "value": f"{name}_value"}
    )
    return pd.merge_asof(
        panel.sort_values(["TRADE_DATE", "SECURITY_ID"]),
        right.sort_values([f"{name}_AVAILABLE_DATE", "SECURITY_ID"]),
        by="SECURITY_ID",
        left_on="TRADE_DATE",
        right_on=f"{name}_AVAILABLE_DATE",
        direction="backward",
    )


def _decayed(value: pd.Series, age_days: pd.Series, half_life: int) -> pd.Series:
    age = pd.to_numeric(age_days, errors="coerce").clip(lower=0)
    return pd.to_numeric(value, errors="coerce") * np.exp(-np.log(2.0) * age / half_life)


def map_year(panel: pd.DataFrame, events: dict[str, pd.DataFrame]) -> pd.DataFrame:
    mapped = panel.copy()
    for name in BASE_SIGNALS:
        mapped = _map_base(mapped, events[name], name)
    ages = {
        name: (mapped["TRADE_DATE"] - mapped[f"{name}_AVAILABLE_DATE"]).dt.days
        for name in BASE_SIGNALS
    }
    mapped["r12_core_profit_surprise_hl20"] = _decayed(mapped["core_value"], ages["core"], 20)
    mapped["r12_core_profit_surprise_hl60"] = _decayed(mapped["core_value"], ages["core"], 60)
    mapped["r12_core_profit_surprise_hl120"] = _decayed(mapped["core_value"], ages["core"], 120)
    mapped["r12_core_profit_sue_hl60"] = _decayed(mapped["core_sue_value"], ages["core_sue"], 60)
    mapped["r12_core_profit_sue_hl120"] = _decayed(mapped["core_sue_value"], ages["core_sue"], 120)
    mapped["r12_operating_margin_sue_hl60"] = _decayed(mapped["margin_sue_value"], ages["margin_sue"], 60)
    mapped["r12_operating_margin_sue_hl120"] = _decayed(mapped["margin_sue_value"], ages["margin_sue"], 120)
    mapped["r12_revenue_sue_hl20"] = _decayed(mapped["revenue_value"], ages["revenue"], 20)
    mapped["r12_revenue_sue_hl60"] = _decayed(mapped["revenue_value"], ages["revenue"], 60)
    mapped["r12_revenue_sue_hl120"] = _decayed(mapped["revenue_value"], ages["revenue"], 120)
    mapped["r12_tax_expense_surprise_hl60"] = _decayed(mapped["tax_value"], ages["tax"], 60)
    mapped["r12_tax_expense_surprise_hl120"] = _decayed(mapped["tax_value"], ages["tax"], 120)
    mapped[CANDIDATE_COLUMNS] = mapped[CANDIDATE_COLUMNS].astype("float32")
    return mapped[KEYS + CANDIDATE_COLUMNS].sort_values(KEYS)


def generate_sparse(panel_path: Path, pit_dir: Path, output_dir: Path) -> None:
    dates = pd.read_parquet(panel_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    events = prepare_events(calculate_base_events(pit_dir), calendar)
    chunks: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    for year in sorted(set(calendar.year)):
        filters = [
            ("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)),
            ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31)),
        ]
        panel = _normalize_panel(pd.read_parquet(panel_path, columns=KEYS, filters=filters))
        mapped = map_year(panel, events)
        chunks.append(mapped)
        for factor in CANDIDATE_COLUMNS:
            series = mapped[factor]
            coverage_rows.append(
                {
                    "year": year,
                    "factor": factor,
                    "rows": len(mapped),
                    "observed_rows": int(series.notna().sum()),
                    "missing_rate_before_fill": float(series.isna().mean()),
                    "observed_days": int(mapped.loc[series.notna(), "TRADE_DATE"].nunique()),
                }
            )
        print(f"{year}: sparse rows={len(mapped):,}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.concat(chunks, ignore_index=True).sort_values(KEYS)
    data.to_parquet(output_dir / "round12_sparse_before_fill.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage_rows).to_csv(
        output_dir / "round12_sparse_coverage.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "round12_sparse_metadata.json").write_text(
        json.dumps(
            {
                "stage": "sparse_before_fill",
                "rows": len(data),
                "factors": CANDIDATE_COLUMNS,
                "period_source_zero_fill": False,
                "daily_cross_section_fill": False,
                "existing_factor_inputs": [],
                "transform": "calendar-day exponential half-life from first tradable announcement date",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def fill_after_test(
    sparse_path: Path,
    sparse_ic_summary: Path,
    output: Path,
    report: Path,
    factors: list[str],
) -> None:
    if not sparse_ic_summary.exists():
        raise FileNotFoundError(f"Sparse IC summary must exist before fill: {sparse_ic_summary}")
    tested = set(pd.read_csv(sparse_ic_summary)["factor"].astype(str))
    unknown = sorted(set(factors).difference(CANDIDATE_COLUMNS))
    untested = sorted(set(factors).difference(tested))
    if unknown or untested:
        raise RuntimeError(f"unknown={unknown}; not tested before fill={untested}")
    data = pd.read_parquet(sparse_path, columns=KEYS + factors)
    before = data[factors].isna().mean()
    medians = data.groupby("TRADE_DATE", sort=False)[factors].transform("median")
    data[factors] = data[factors].fillna(medians)
    remaining = data[factors].isna().sum()
    if remaining.any():
        raise RuntimeError(f"whole-day missing values remain: {remaining[remaining.gt(0)].to_dict()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False, compression="zstd")
    pd.DataFrame(
        {
            "factor": factors,
            "missing_rate_before_fill": [float(before[f]) for f in factors],
            "fill_method": "same_date_cross_section_median_after_sparse_test",
            "remaining_missing_rows": [int(remaining[f]) for f in factors],
        }
    ).to_csv(report, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    root = BASE_DIR / "artifacts" / "round12"
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-sparse")
    generate.add_argument("--panel", type=Path, required=True)
    generate.add_argument("--pit-dir", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    fill = sub.add_parser("fill-after-test")
    fill.add_argument("--sparse", type=Path, default=root / "round12_sparse_before_fill.parquet")
    fill.add_argument("--sparse-ic", type=Path, default=root / "sparse_diagnostic_test" / "ic_summary.csv")
    fill.add_argument("--output", type=Path, default=root / "round12_filled_after_test.parquet")
    fill.add_argument("--report", type=Path, default=root / "round12_fill_report.csv")
    fill.add_argument("--factor-columns", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate-sparse":
        generate_sparse(args.panel.resolve(), args.pit_dir.resolve(), args.output_dir.resolve())
    else:
        fill_after_test(
            args.sparse.resolve(), args.sparse_ic.resolve(), args.output.resolve(),
            args.report.resolve(), args.factor_columns,
        )


if __name__ == "__main__":
    main()
