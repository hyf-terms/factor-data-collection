"""Independent quarterly tax, capital-expenditure and core-earnings signals.

The tax signals follow Thomas and Zhang's seasonal quarterly tax-expense
surprise and its pre-tax-income/effective-tax-rate decomposition.  The capex
and operating signals extend the fundamental-analysis variables of Lev and
Thiagarajan and Abarbanell and Bushee.  No prior factor, percentile rank, or
cross-factor residual is used.
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


BASE_DIR = Path(__file__).resolve().parent
INCOME_FIELDS = [
    "INCOME_TAX",
    "T_PROFIT",
    "N_INCOME_ATTR_P",
    "REVENUE",
    "OPERATE_PROFIT",
    "ASSETS_IMPAIR_LOSS",
    "CREDIT_IMPAIR_LOSS",
]
CASHFLOW_FIELDS = ["PUR_FIX_ASSETS_OTH", "C_PAID_FOR_TAXES"]
CANDIDATE_COLUMNS = [
    "r11_tax_expense_surprise_assets",
    "r11_tax_expense_sue",
    "r11_tax_etr_component_assets",
    "r11_tax_pretax_component_assets",
    "r11_tax_cash_match_improvement",
    "r11_capex_surprise_assets",
    "r11_capex_sue",
    "r11_capex_sales_gap",
    "r11_revenue_sue",
    "r11_low_impairment_surprise_assets",
    "r11_core_profit_surprise_assets",
    "r11_operating_share_improvement",
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


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return workflow._safe_ratio(numerator, denominator, absolute_denominator=True)


def _with_lag4(flows: dict[str, pd.DataFrame], fields: list[str]) -> pd.DataFrame:
    current = _merge_flow_tables(flows, fields, ttm=False)
    return current.merge(
        workflow._lag_table(
            current,
            4,
            "L4",
            ["FLOW_EVENT_TIME", *fields],
        ),
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    )


def _with_assets(frame: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    current = balance[
        ["SECURITY_ID", "QUARTER_INDEX", "BALANCE_EVENT_TIME", "T_ASSETS"]
    ]
    lagged = workflow._lag_table(
        current,
        4,
        "L4",
        ["BALANCE_EVENT_TIME", "T_ASSETS"],
    )
    return frame.merge(
        current,
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    ).merge(
        lagged,
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    )


def _historical_surprise_std(frame: pd.DataFrame, surprise: pd.Series) -> pd.Series:
    work = frame[["SECURITY_ID", "QUARTER_INDEX"]].copy()
    work["surprise"] = pd.to_numeric(surprise, errors="coerce")
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, group in work.groupby("SECURITY_ID", sort=False):
        ordered = group.sort_values("QUARTER_INDEX")
        start = int(ordered["QUARTER_INDEX"].min())
        stop = int(ordered["QUARTER_INDEX"].max()) + 1
        full = ordered.set_index("QUARTER_INDEX")["surprise"].reindex(range(start, stop))
        historical = full.shift(1).rolling(8, min_periods=8).std(ddof=1)
        result.loc[ordered.index] = ordered["QUARTER_INDEX"].map(historical).to_numpy()
    return result


def _event(frame: pd.DataFrame, factor: str, value: pd.Series) -> pd.DataFrame:
    event_columns = [column for column in frame.columns if column.endswith("EVENT_TIME")]
    return workflow._long_event(frame, factor, value, event_columns)


def calculate_factor_events(
    flows: dict[str, pd.DataFrame], balance: pd.DataFrame
) -> pd.DataFrame:
    events: list[pd.DataFrame] = []

    tax = _with_assets(
        _with_lag4(
            flows,
            ["INCOME_TAX", "T_PROFIT", "N_INCOME_ATTR_P", "C_PAID_FOR_TAXES"],
        ),
        balance,
    )
    average_assets = (tax["T_ASSETS"] + tax["L4_T_ASSETS"]) / 2.0
    tax_surprise = tax["INCOME_TAX"] - tax["L4_INCOME_TAX"]
    tax_std = _historical_surprise_std(tax, tax_surprise)
    tax_sue = tax_surprise.div(tax_std.where(tax_std.gt(0))).replace([np.inf, -np.inf], np.nan)
    current_etr = _ratio(tax["INCOME_TAX"], tax["T_PROFIT"])
    lag_etr = _ratio(tax["L4_INCOME_TAX"], tax["L4_T_PROFIT"])
    etr_component = _ratio(tax["T_PROFIT"] * (current_etr - lag_etr), average_assets)
    pretax_component = _ratio(
        lag_etr * (tax["T_PROFIT"] - tax["L4_T_PROFIT"]), average_assets
    )
    current_tax_cash_gap = tax["INCOME_TAX"] - tax["C_PAID_FOR_TAXES"]
    lag_tax_cash_gap = tax["L4_INCOME_TAX"] - tax["L4_C_PAID_FOR_TAXES"]
    tax_cash_match_improvement = -_ratio(
        current_tax_cash_gap - lag_tax_cash_gap, average_assets
    )
    tax_values = {
        "r11_tax_expense_surprise_assets": _ratio(tax_surprise, average_assets),
        "r11_tax_expense_sue": tax_sue,
        "r11_tax_etr_component_assets": etr_component,
        "r11_tax_pretax_component_assets": pretax_component,
        "r11_tax_cash_match_improvement": tax_cash_match_improvement,
    }
    events.extend(_event(tax, name, value) for name, value in tax_values.items())

    capex = _with_assets(
        _with_lag4(flows, ["PUR_FIX_ASSETS_OTH", "REVENUE"]),
        balance,
    )
    capex_average_assets = (capex["T_ASSETS"] + capex["L4_T_ASSETS"]) / 2.0
    capex_surprise = capex["PUR_FIX_ASSETS_OTH"] - capex["L4_PUR_FIX_ASSETS_OTH"]
    capex_std = _historical_surprise_std(capex, capex_surprise)
    capex_sue = capex_surprise.div(capex_std.where(capex_std.gt(0))).replace(
        [np.inf, -np.inf], np.nan
    )
    capex_growth = _ratio(capex_surprise, capex["L4_PUR_FIX_ASSETS_OTH"])
    sales_growth = _ratio(
        capex["REVENUE"] - capex["L4_REVENUE"], capex["L4_REVENUE"]
    )
    events.extend(
        [
            _event(capex, "r11_capex_surprise_assets", _ratio(capex_surprise, capex_average_assets)),
            _event(capex, "r11_capex_sue", capex_sue),
            _event(capex, "r11_capex_sales_gap", capex_growth - sales_growth),
        ]
    )

    revenue = _with_lag4(flows, ["REVENUE"])
    revenue_surprise = revenue["REVENUE"] - revenue["L4_REVENUE"]
    revenue_std = _historical_surprise_std(revenue, revenue_surprise)
    revenue_sue = revenue_surprise.div(revenue_std.where(revenue_std.gt(0))).replace(
        [np.inf, -np.inf], np.nan
    )
    events.append(_event(revenue, "r11_revenue_sue", revenue_sue))

    impairment = _with_assets(
        _with_lag4(flows, ["ASSETS_IMPAIR_LOSS", "CREDIT_IMPAIR_LOSS"]), balance
    )
    current_impairment = impairment[["ASSETS_IMPAIR_LOSS", "CREDIT_IMPAIR_LOSS"]].sum(
        axis=1, min_count=1
    )
    lag_impairment = impairment[
        ["L4_ASSETS_IMPAIR_LOSS", "L4_CREDIT_IMPAIR_LOSS"]
    ].sum(axis=1, min_count=1)
    impairment_average_assets = (
        impairment["T_ASSETS"] + impairment["L4_T_ASSETS"]
    ) / 2.0
    events.append(
        _event(
            impairment,
            "r11_low_impairment_surprise_assets",
            -_ratio(current_impairment - lag_impairment, impairment_average_assets),
        )
    )

    core = _with_assets(
        _with_lag4(flows, ["OPERATE_PROFIT", "T_PROFIT"]), balance
    )
    core_average_assets = (core["T_ASSETS"] + core["L4_T_ASSETS"]) / 2.0
    operating_surprise = core["OPERATE_PROFIT"] - core["L4_OPERATE_PROFIT"]
    current_share = _ratio(core["OPERATE_PROFIT"], core["T_PROFIT"])
    lag_share = _ratio(core["L4_OPERATE_PROFIT"], core["L4_T_PROFIT"])
    events.extend(
        [
            _event(
                core,
                "r11_core_profit_surprise_assets",
                _ratio(operating_surprise, core_average_assets),
            ),
            _event(
                core,
                "r11_operating_share_improvement",
                current_share - lag_share,
            ),
        ]
    )

    result = pd.concat(events, ignore_index=True)
    result["QUARTER_INDEX"] = pd.to_numeric(result["QUARTER_INDEX"]).astype("int64")
    return result.sort_values(["factor", "SECURITY_ID", "EVENT_TIME", "QUARTER_INDEX"])


def generate_sparse(panel_path: Path, pit_dir: Path, output_dir: Path) -> None:
    dates = pd.read_parquet(panel_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    flows, balance = _read_inputs(pit_dir)
    events = calculate_factor_events(flows, balance)
    workflow.CANDIDATE_COLUMNS = CANDIDATE_COLUMNS
    wide = workflow.prepare_wide_events(events, calendar)
    chunks: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    for year in sorted(set(calendar.year)):
        filters = [
            ("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)),
            ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31)),
        ]
        panel = _normalize_panel(pd.read_parquet(panel_path, columns=KEYS, filters=filters))
        mapped = workflow._map_sparse(panel, wide)
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
    data.to_parquet(output_dir / "round11_sparse_before_fill.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage_rows).to_csv(
        output_dir / "round11_sparse_coverage.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "round11_sparse_metadata.json").write_text(
        json.dumps(
            {
                "stage": "sparse_before_fill",
                "rows": len(data),
                "factors": CANDIDATE_COLUMNS,
                "period_source_zero_fill": False,
                "daily_cross_section_fill": False,
                "existing_factor_inputs": [],
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
    root = BASE_DIR / "artifacts" / "round11"
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-sparse")
    generate.add_argument("--panel", type=Path, required=True)
    generate.add_argument("--pit-dir", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    fill = sub.add_parser("fill-after-test")
    fill.add_argument("--sparse", type=Path, default=root / "round11_sparse_before_fill.parquet")
    fill.add_argument("--sparse-ic", type=Path, default=root / "sparse_diagnostic_test" / "ic_summary.csv")
    fill.add_argument("--output", type=Path, default=root / "round11_filled_after_test.parquet")
    fill.add_argument("--report", type=Path, default=root / "round11_fill_report.csv")
    fill.add_argument("--factor-columns", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate-sparse":
        generate_sparse(args.panel.resolve(), args.pit_dir.resolve(), args.output_dir.resolve())
    else:
        fill_after_test(
            args.sparse.resolve(),
            args.sparse_ic.resolve(),
            args.output.resolve(),
            args.report.resolve(),
            args.factor_columns,
        )


if __name__ == "__main__":
    main()
