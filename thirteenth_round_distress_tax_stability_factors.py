"""Independent tax-gap, distress, stability and labor-efficiency factors.

No existing factor, percentile rank or fitted sample weight is used.  Period
fields are converted to standalone quarters/TTM values without zero filling.
The daily panel is emitted with its natural missing values and may be filled
only after a sparse IC diagnostic has been supplied to ``fill-after-test``.
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
from quarterly_f_score import (
    COMMON_COLUMNS,
    FINANCIAL_INDUSTRIES,
    REPORT_QUARTERS,
    REPORT_TYPES,
    build_standalone_quarterly_metric,
)


BASE_DIR = Path(__file__).resolve().parent
INCOME_FIELDS = [
    "T_PROFIT",
    "INCOME_TAX",
    "N_INCOME_ATTR_P",
    "REVENUE",
    "OPERATE_PROFIT",
    "INT_EXP_FINAN_EXP",
]
CASHFLOW_FIELDS = [
    "N_CF_OPERATE_A",
    "C_PAID_FOR_TAXES",
    "C_PAID_TO_FOR_EMPL",
]
BALANCE_FIELDS = [
    "INDUSTRY_CATEGORY",
    "T_ASSETS",
    "T_CA",
    "CASH_C_EQUIV",
    "T_CL",
    "T_LIAB",
    "T_SH_EQUITY",
    "RETAINED_EARNINGS",
    "DEFER_TAX_ASSETS",
    "DEFER_TAX_LIAB",
    "ST_BORR",
    "NCL_WITHIN_1_Y",
    "LT_BORR",
    "BOND_PAYABLE",
]

CANDIDATE_COLUMNS = [
    "r13_btd_tax_expense_quality",
    "r13_btd_cash_tax_quality",
    "r13_low_net_deferred_tax_assets",
    "r13_low_net_deferred_tax_build",
    "r13_altman_zprime",
    "r13_zmijewski_quality",
    "r13_ohlson_accounting_quality",
    "r13_interest_coverage",
    "r13_cash_interest_coverage",
    "r13_low_net_debt_to_cfo",
    "r13_low_cfo_sales_volatility_12q",
    "r13_low_cfo_sales_volatility_20q",
    "r13_low_cfo_equity_volatility_12q",
    "r13_low_cfo_equity_volatility_20q",
    "r13_low_roa_volatility_12q",
    "r13_employee_cash_productivity_gap",
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


def build_balance_events(balance: pd.DataFrame) -> pd.DataFrame:
    data = balance[COMMON_COLUMNS + BALANCE_FIELDS].copy()
    for column in ("ACT_PUBTIME", "END_DATE", "END_DATE_REP"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["END_DATE"] = data["END_DATE"].dt.normalize()
    data["END_DATE_REP"] = data["END_DATE_REP"].dt.normalize()
    data["SECURITY_ID"] = pd.to_numeric(data["SECURITY_ID"], errors="coerce")
    for field in BALANCE_FIELDS:
        if field != "INDUSTRY_CATEGORY":
            data[field] = pd.to_numeric(data[field], errors="coerce")
    for field in (
        "DEFER_TAX_ASSETS",
        "DEFER_TAX_LIAB",
        "ST_BORR",
        "NCL_WITHIN_1_Y",
        "LT_BORR",
        "BOND_PAYABLE",
    ):
        data[field] = data[field].fillna(0.0)
    mask = (
        data["MERGED_FLAG"].astype("string").eq("1")
        & data["REPORT_TYPE"].astype("string").isin(REPORT_TYPES)
        & data["IS_CURRENT_PERIOD"].fillna(False).astype(bool)
        & data["END_DATE"].eq(data["END_DATE_REP"])
        & ~data["INDUSTRY_CATEGORY"].isin(FINANCIAL_INDUSTRIES)
    )
    data = data.loc[mask].dropna(
        subset=["SECURITY_ID", "ACT_PUBTIME", "END_DATE", "T_ASSETS"]
    )
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    data["FISCAL_YEAR"] = data["END_DATE"].dt.year.astype("int16")
    data["FISCAL_QUARTER"] = data["REPORT_TYPE"].map(REPORT_QUARTERS).astype("int8")
    data["QUARTER_INDEX"] = data["FISCAL_YEAR"].astype("int64") * 4 + data["FISCAL_QUARTER"]
    keys = ["SECURITY_ID", "QUARTER_INDEX"]
    data = data.sort_values(keys + ["ACT_PUBTIME", "ID"]).drop_duplicates(keys, keep="first")
    return data[
        keys + ["FISCAL_YEAR", "FISCAL_QUARTER", "END_DATE", "ACT_PUBTIME", *BALANCE_FIELDS]
    ].rename(columns={"ACT_PUBTIME": "BALANCE_EVENT_TIME"})


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return workflow._safe_ratio(numerator, denominator, absolute_denominator=True)


def _lag(frame: pd.DataFrame, quarters: int, prefix: str, columns: list[str]) -> pd.DataFrame:
    return workflow._lag_table(frame, quarters, prefix, columns)


def _event(frame: pd.DataFrame, factor: str, value: pd.Series, times: list[str]) -> pd.DataFrame:
    return workflow._long_event(frame, factor, value, times)


def _rolling_std(
    frame: pd.DataFrame, value: pd.Series, window: int
) -> pd.Series:
    work = frame[["SECURITY_ID", "QUARTER_INDEX"]].copy()
    work["value"] = pd.to_numeric(value, errors="coerce")
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, group in work.groupby("SECURITY_ID", sort=False):
        ordered = group.sort_values("QUARTER_INDEX")
        start = int(ordered["QUARTER_INDEX"].min())
        stop = int(ordered["QUARTER_INDEX"].max()) + 1
        full = ordered.set_index("QUARTER_INDEX")["value"].reindex(range(start, stop))
        volatility = full.rolling(window, min_periods=window).std(ddof=1)
        result.loc[ordered.index] = ordered["QUARTER_INDEX"].map(volatility).to_numpy()
    return result


def calculate_factor_events(
    flows: dict[str, pd.DataFrame], balance: pd.DataFrame
) -> pd.DataFrame:
    balance_values = [field for field in BALANCE_FIELDS if field != "INDUSTRY_CATEGORY"]
    balance_l4 = _lag(balance, 4, "L4", ["BALANCE_EVENT_TIME", *balance_values])
    times = ["FLOW_EVENT_TIME", "L4_FLOW_EVENT_TIME", "BALANCE_EVENT_TIME", "L4_BALANCE_EVENT_TIME"]

    def base(fields: list[str]) -> pd.DataFrame:
        current = _merge_flow_tables(flows, fields, ttm=True)
        values = [f"TTM_{field}" for field in fields]
        current = current.merge(
            _lag(current, 4, "L4", ["FLOW_EVENT_TIME", *values]),
            on=["SECURITY_ID", "QUARTER_INDEX"], how="inner", validate="one_to_one",
        )
        return current.merge(
            balance,
            on=["SECURITY_ID", "FISCAL_YEAR", "FISCAL_QUARTER", "QUARTER_INDEX"],
            how="inner", validate="one_to_one",
        ).merge(
            balance_l4,
            on=["SECURITY_ID", "QUARTER_INDEX"], how="inner", validate="one_to_one",
        )

    events: list[pd.DataFrame] = []

    tax_expense = base(["T_PROFIT", "INCOME_TAX"])
    tax_avg_assets = (tax_expense["T_ASSETS"] + tax_expense["L4_T_ASSETS"]) / 2.0
    tax_btd = _ratio(
        tax_expense["TTM_T_PROFIT"] - tax_expense["TTM_INCOME_TAX"] / 0.25,
        tax_avg_assets,
    )
    events.append(_event(tax_expense, "r13_btd_tax_expense_quality", -tax_btd, times))

    cash_tax = base(["T_PROFIT", "C_PAID_FOR_TAXES"])
    cash_tax_avg_assets = (cash_tax["T_ASSETS"] + cash_tax["L4_T_ASSETS"]) / 2.0
    cash_btd = _ratio(
        cash_tax["TTM_T_PROFIT"] - cash_tax["TTM_C_PAID_FOR_TAXES"] / 0.25,
        cash_tax_avg_assets,
    )
    events.append(_event(cash_tax, "r13_btd_cash_tax_quality", -cash_btd, times))

    deferred = balance.merge(
        balance_l4, on=["SECURITY_ID", "QUARTER_INDEX"], how="inner", validate="one_to_one"
    )
    deferred_times = ["BALANCE_EVENT_TIME", "L4_BALANCE_EVENT_TIME"]
    deferred_avg_assets = (deferred["T_ASSETS"] + deferred["L4_T_ASSETS"]) / 2.0
    net_dta = deferred["DEFER_TAX_ASSETS"] - deferred["DEFER_TAX_LIAB"]
    lag_net_dta = deferred["L4_DEFER_TAX_ASSETS"] - deferred["L4_DEFER_TAX_LIAB"]
    events.extend([
        _event(
            deferred, "r13_low_net_deferred_tax_assets",
            -_ratio(net_dta, deferred["T_ASSETS"]), deferred_times,
        ),
        _event(
            deferred, "r13_low_net_deferred_tax_build",
            -_ratio(net_dta - lag_net_dta, deferred_avg_assets), deferred_times,
        ),
    ])

    altman_data = base(["T_PROFIT", "INT_EXP_FINAN_EXP", "REVENUE"])
    altman_ebit = altman_data["TTM_T_PROFIT"] + altman_data["TTM_INT_EXP_FINAN_EXP"]
    altman = (
        0.717 * _ratio(altman_data["T_CA"] - altman_data["T_CL"], altman_data["T_ASSETS"])
        + 0.847 * _ratio(altman_data["RETAINED_EARNINGS"], altman_data["T_ASSETS"])
        + 3.107 * _ratio(altman_ebit, altman_data["T_ASSETS"])
        + 0.420 * _ratio(altman_data["T_SH_EQUITY"], altman_data["T_LIAB"])
        + 0.998 * _ratio(altman_data["TTM_REVENUE"], altman_data["T_ASSETS"])
    )
    events.append(_event(altman_data, "r13_altman_zprime", altman, times))

    z_data = base(["N_INCOME_ATTR_P"])
    z_avg_assets = (z_data["T_ASSETS"] + z_data["L4_T_ASSETS"]) / 2.0
    z_roa = _ratio(z_data["TTM_N_INCOME_ATTR_P"], z_avg_assets)
    z_risk = (
        -4.3 - 4.5 * z_roa
        + 5.7 * _ratio(z_data["T_LIAB"], z_data["T_ASSETS"])
        - 0.004 * _ratio(z_data["T_CA"], z_data["T_CL"])
    )
    events.append(_event(z_data, "r13_zmijewski_quality", -z_risk, times))

    o_data = base(["N_INCOME_ATTR_P", "N_CF_OPERATE_A"])
    ni = o_data["TTM_N_INCOME_ATTR_P"]
    prior_ni = o_data["L4_TTM_N_INCOME_ATTR_P"]
    leverage = _ratio(o_data["T_LIAB"], o_data["T_ASSETS"])
    oeneg = (o_data["T_LIAB"] > o_data["T_ASSETS"]).astype(float)
    intwo = ((ni < 0) & (prior_ni < 0)).astype(float)
    chin = _ratio(ni - prior_ni, ni.abs() + prior_ni.abs())
    log_assets = np.log(o_data["T_ASSETS"].where(o_data["T_ASSETS"].gt(0)) / 1_000_000.0)
    o_risk = (
        -1.32 - 0.407 * log_assets + 6.03 * leverage
        - 1.43 * _ratio(o_data["T_CA"] - o_data["T_CL"], o_data["T_ASSETS"])
        + 0.076 * _ratio(o_data["T_CL"], o_data["T_CA"])
        - 1.72 * oeneg - 2.37 * _ratio(ni, o_data["T_ASSETS"])
        - 1.83 * _ratio(o_data["TTM_N_CF_OPERATE_A"], o_data["T_LIAB"])
        + 0.285 * intwo - 0.521 * chin
    )
    events.append(_event(o_data, "r13_ohlson_accounting_quality", -o_risk, times))

    interest_data = base(["T_PROFIT", "INT_EXP_FINAN_EXP"])
    interest = interest_data["TTM_INT_EXP_FINAN_EXP"]
    ebit = interest_data["TTM_T_PROFIT"] + interest
    events.append(_event(interest_data, "r13_interest_coverage", _ratio(ebit, interest), times))

    cash_interest_data = base(["N_CF_OPERATE_A", "INT_EXP_FINAN_EXP"])
    events.append(_event(
        cash_interest_data, "r13_cash_interest_coverage",
        _ratio(cash_interest_data["TTM_N_CF_OPERATE_A"], cash_interest_data["TTM_INT_EXP_FINAN_EXP"]), times,
    ))

    debt_data = base(["N_CF_OPERATE_A"])
    debt = debt_data[["ST_BORR", "NCL_WITHIN_1_Y", "LT_BORR", "BOND_PAYABLE"]].sum(axis=1)
    events.append(_event(
        debt_data, "r13_low_net_debt_to_cfo",
        -_ratio(debt - debt_data["CASH_C_EQUIV"], debt_data["TTM_N_CF_OPERATE_A"]), times,
    ))

    cfo_sales_data = base(["N_CF_OPERATE_A", "REVENUE"])
    cfo_sales = _ratio(cfo_sales_data["TTM_N_CF_OPERATE_A"], cfo_sales_data["TTM_REVENUE"])
    events.extend([
        _event(cfo_sales_data, "r13_low_cfo_sales_volatility_12q", -_rolling_std(cfo_sales_data, cfo_sales, 12), times),
        _event(cfo_sales_data, "r13_low_cfo_sales_volatility_20q", -_rolling_std(cfo_sales_data, cfo_sales, 20), times),
    ])

    cfo_equity_data = base(["N_CF_OPERATE_A"])
    cfo_equity = _ratio(cfo_equity_data["TTM_N_CF_OPERATE_A"], cfo_equity_data["T_SH_EQUITY"])
    events.extend([
        _event(cfo_equity_data, "r13_low_cfo_equity_volatility_12q", -_rolling_std(cfo_equity_data, cfo_equity, 12), times),
        _event(cfo_equity_data, "r13_low_cfo_equity_volatility_20q", -_rolling_std(cfo_equity_data, cfo_equity, 20), times),
    ])

    roa_data = base(["N_INCOME_ATTR_P"])
    roa_avg_assets = (roa_data["T_ASSETS"] + roa_data["L4_T_ASSETS"]) / 2.0
    roa_value = _ratio(roa_data["TTM_N_INCOME_ATTR_P"], roa_avg_assets)
    events.append(_event(
        roa_data, "r13_low_roa_volatility_12q", -_rolling_std(roa_data, roa_value, 12), times,
    ))

    employee = base(["REVENUE", "C_PAID_TO_FOR_EMPL"])
    revenue_growth = _ratio(
        employee["TTM_REVENUE"] - employee["L4_TTM_REVENUE"], employee["L4_TTM_REVENUE"]
    )
    employee_growth = _ratio(
        employee["TTM_C_PAID_TO_FOR_EMPL"] - employee["L4_TTM_C_PAID_TO_FOR_EMPL"],
        employee["L4_TTM_C_PAID_TO_FOR_EMPL"],
    )
    events.append(_event(
        employee, "r13_employee_cash_productivity_gap", revenue_growth - employee_growth, times,
    ))

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
    data.to_parquet(output_dir / "round13_sparse_before_fill.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage_rows).to_csv(
        output_dir / "round13_sparse_coverage.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "round13_sparse_metadata.json").write_text(
        json.dumps(
            {
                "stage": "sparse_before_fill",
                "rows": len(data),
                "factors": CANDIDATE_COLUMNS,
                "period_source_zero_fill": False,
                "daily_cross_section_fill": False,
                "existing_factor_inputs": [],
                "tax_rate_assumption": 0.25,
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
    root = BASE_DIR / "artifacts" / "round13"
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-sparse")
    generate.add_argument("--panel", type=Path, required=True)
    generate.add_argument("--pit-dir", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    fill = sub.add_parser("fill-after-test")
    fill.add_argument("--sparse", type=Path, default=root / "round13_sparse_before_fill.parquet")
    fill.add_argument("--sparse-ic", type=Path, default=root / "sparse_diagnostic_test" / "ic_summary.csv")
    fill.add_argument("--output", type=Path, default=root / "round13_filled_after_test.parquet")
    fill.add_argument("--report", type=Path, default=root / "round13_fill_report.csv")
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
