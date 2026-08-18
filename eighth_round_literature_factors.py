"""Build and audit eighth-round dense PIT accounting factors.

The workflow deliberately separates sparse evaluation from neutral filling:

1. ``generate-sparse`` maps only genuinely available PIT observations to the
   complete daily universe and preserves missing values.
2. The sparse file is evaluated with ``factors_neus_only2.py`` using its
   explicit ``skip`` diagnostic policy.
3. ``fill-after-test`` requires that diagnostic output before replacing the
   remaining daily cross-sectional missing values with the same-date median.

No percentile rank is used.  Balance-sheet observations are point-in-time
stocks; income and cash-flow observations are first converted from cumulative
reports to standalone fiscal quarters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dense_literature_profitability_factors import _merge_flow_tables
from event_financial_factor_search import KEYS, _normalize_panel
from pead_sue_factor import assign_available_trade_date
from quarterly_f_score import (
    COMMON_COLUMNS,
    FINANCIAL_INDUSTRIES,
    REPORT_QUARTERS,
    REPORT_TYPES,
    build_standalone_quarterly_metric,
)


BASE_DIR = Path(__file__).resolve().parent
INCOME_FIELDS = [
    "N_INCOME_ATTR_P",
    "REVENUE",
    "COGS",
    "SELL_EXP",
    "ADMIN_EXP",
]
CASHFLOW_FIELDS = ["N_CF_OPERATE_A"]
BALANCE_FIELDS = [
    "INDUSTRY_CATEGORY",
    "T_ASSETS",
    "T_CA",
    "CASH_C_EQUIV",
    "T_CL",
    "ST_BORR",
    "NCL_WITHIN_1_Y",
    "AR",
    "INVENTORIES",
]

CANDIDATE_COLUMNS = [
    "r8_hlvw_q_low_percent_accruals",
    "r8_hlvw_ttm_low_percent_accruals",
    "r8_tz_low_inventory_investment",
    "r8_ab_sales_inventory_gap",
    "r8_ab_sales_receivable_gap",
    "r8_ab_sales_sga_gap",
    "r8_fy_asset_turnover_change",
    "r8_fy_profit_margin_change",
    "r8_rsst_low_working_capital_accruals",
]


def _latest_time(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return pd.concat(
        [pd.to_datetime(frame[column], errors="coerce") for column in columns],
        axis=1,
    ).max(axis=1)


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    absolute_denominator: bool = False,
) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce")
    if absolute_denominator:
        denominator = denominator.abs()
    scale = denominator.abs().median(skipna=True)
    floor = max(float(scale) * 1e-8, 1e-12) if pd.notna(scale) else 1e-12
    valid = denominator.abs().gt(floor) & np.isfinite(denominator)
    return (
        pd.to_numeric(numerator, errors="coerce")
        .div(denominator.where(valid))
        .replace([np.inf, -np.inf], np.nan)
    )


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
    balance = pd.read_parquet(
        pit_dir / "new_pit_balance",
        columns=COMMON_COLUMNS + BALANCE_FIELDS,
        engine="pyarrow",
    )
    flows: dict[str, pd.DataFrame] = {}
    for field in INCOME_FIELDS:
        flows[field] = build_standalone_quarterly_metric(
            income, field, name="利润表PIT"
        )
    for field in CASHFLOW_FIELDS:
        flows[field] = build_standalone_quarterly_metric(
            cashflow, field, name="现金流量表PIT"
        )
    return flows, build_balance_events(balance)


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
    key = ["SECURITY_ID", "QUARTER_INDEX"]
    data = data.sort_values(key + ["ACT_PUBTIME", "ID"]).drop_duplicates(key, keep="first")
    return data[
        key
        + ["FISCAL_YEAR", "FISCAL_QUARTER", "END_DATE", "ACT_PUBTIME", *BALANCE_FIELDS]
    ].rename(columns={"ACT_PUBTIME": "BALANCE_EVENT_TIME"})


def _lag_table(
    frame: pd.DataFrame,
    quarters: int,
    prefix: str,
    columns: list[str],
) -> pd.DataFrame:
    result = frame[["SECURITY_ID", "QUARTER_INDEX", *columns]].copy()
    result["QUARTER_INDEX"] += quarters
    return result.rename(columns={column: f"{prefix}_{column}" for column in columns})


def _long_event(
    frame: pd.DataFrame,
    factor: str,
    values: pd.Series,
    event_columns: list[str],
) -> pd.DataFrame:
    result = frame[["SECURITY_ID", "QUARTER_INDEX"]].copy()
    result["EVENT_TIME"] = _latest_time(frame, event_columns)
    result["factor"] = factor
    result["value"] = pd.to_numeric(values, errors="coerce")
    return result.dropna(subset=["SECURITY_ID", "QUARTER_INDEX", "EVENT_TIME", "value"])


def _flow_with_prior_year(
    flow_tables: dict[str, pd.DataFrame], fields: list[str], *, ttm: bool
) -> pd.DataFrame:
    current = _merge_flow_tables(flow_tables, fields, ttm=ttm)
    value_fields = [f"TTM_{field}" if ttm else field for field in fields]
    lag_columns = ["FLOW_EVENT_TIME", *value_fields]
    return current.merge(
        _lag_table(current, 4, "L4", lag_columns),
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    )


def calculate_factor_events(
    flow_tables: dict[str, pd.DataFrame], balance: pd.DataFrame
) -> pd.DataFrame:
    events: list[pd.DataFrame] = []

    accrual_q = _merge_flow_tables(
        flow_tables, ["N_INCOME_ATTR_P", "N_CF_OPERATE_A"], ttm=False
    )
    q_percent_accrual = _safe_ratio(
        accrual_q["N_CF_OPERATE_A"] - accrual_q["N_INCOME_ATTR_P"],
        accrual_q["N_INCOME_ATTR_P"],
        absolute_denominator=True,
    )
    events.append(
        _long_event(
            accrual_q,
            "r8_hlvw_q_low_percent_accruals",
            q_percent_accrual,
            ["FLOW_EVENT_TIME"],
        )
    )

    accrual_ttm = _merge_flow_tables(
        flow_tables, ["N_INCOME_ATTR_P", "N_CF_OPERATE_A"], ttm=True
    )
    ttm_percent_accrual = _safe_ratio(
        accrual_ttm["TTM_N_CF_OPERATE_A"] - accrual_ttm["TTM_N_INCOME_ATTR_P"],
        accrual_ttm["TTM_N_INCOME_ATTR_P"],
        absolute_denominator=True,
    )
    events.append(
        _long_event(
            accrual_ttm,
            "r8_hlvw_ttm_low_percent_accruals",
            ttm_percent_accrual,
            ["FLOW_EVENT_TIME"],
        )
    )

    balance_l4 = _lag_table(
        balance,
        4,
        "L4",
        ["BALANCE_EVENT_TIME", *[f for f in BALANCE_FIELDS if f != "INDUSTRY_CATEGORY"]],
    )
    bal = balance.merge(
        balance_l4,
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    )
    balance_times = ["BALANCE_EVENT_TIME", "L4_BALANCE_EVENT_TIME"]
    inventory_investment = -_safe_ratio(
        bal["INVENTORIES"] - bal["L4_INVENTORIES"], bal["L4_T_ASSETS"]
    )
    events.append(
        _long_event(
            bal,
            "r8_tz_low_inventory_investment",
            inventory_investment,
            balance_times,
        )
    )

    current_operating_assets = bal["T_CA"] - bal["CASH_C_EQUIV"]
    prior_operating_assets = bal["L4_T_CA"] - bal["L4_CASH_C_EQUIV"]
    current_operating_liabilities = (
        bal["T_CL"] - bal["ST_BORR"] - bal["NCL_WITHIN_1_Y"]
    )
    prior_operating_liabilities = (
        bal["L4_T_CL"] - bal["L4_ST_BORR"] - bal["L4_NCL_WITHIN_1_Y"]
    )
    working_capital_change = (
        current_operating_assets - current_operating_liabilities
        - prior_operating_assets + prior_operating_liabilities
    )
    events.append(
        _long_event(
            bal,
            "r8_rsst_low_working_capital_accruals",
            -_safe_ratio(working_capital_change, bal["L4_T_ASSETS"]),
            balance_times,
        )
    )

    revenue = _flow_with_prior_year(flow_tables, ["REVENUE"], ttm=False)
    revenue = revenue.merge(
        bal[
            [
                "SECURITY_ID", "QUARTER_INDEX", "BALANCE_EVENT_TIME",
                "L4_BALANCE_EVENT_TIME", "AR", "L4_AR", "INVENTORIES",
                "L4_INVENTORIES",
            ]
        ],
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    )
    revenue_times = ["FLOW_EVENT_TIME", "L4_FLOW_EVENT_TIME", "BALANCE_EVENT_TIME", "L4_BALANCE_EVENT_TIME"]
    revenue_growth = _safe_ratio(
        revenue["REVENUE"] - revenue["L4_REVENUE"],
        revenue["L4_REVENUE"],
        absolute_denominator=True,
    )
    inventory_growth = _safe_ratio(
        revenue["INVENTORIES"] - revenue["L4_INVENTORIES"],
        revenue["L4_INVENTORIES"],
        absolute_denominator=True,
    )
    receivable_growth = _safe_ratio(
        revenue["AR"] - revenue["L4_AR"],
        revenue["L4_AR"],
        absolute_denominator=True,
    )
    events.extend(
        [
            _long_event(
                revenue,
                "r8_ab_sales_inventory_gap",
                revenue_growth - inventory_growth,
                revenue_times,
            ),
            _long_event(
                revenue,
                "r8_ab_sales_receivable_gap",
                revenue_growth - receivable_growth,
                revenue_times,
            ),
        ]
    )

    sga = _flow_with_prior_year(flow_tables, ["REVENUE", "SELL_EXP", "ADMIN_EXP"], ttm=False)
    sales_growth = _safe_ratio(
        sga["REVENUE"] - sga["L4_REVENUE"], sga["L4_REVENUE"], absolute_denominator=True
    )
    current_sga = sga["SELL_EXP"] + sga["ADMIN_EXP"]
    prior_sga = sga["L4_SELL_EXP"] + sga["L4_ADMIN_EXP"]
    sga_growth = _safe_ratio(current_sga - prior_sga, prior_sga, absolute_denominator=True)
    events.append(
        _long_event(
            sga,
            "r8_ab_sales_sga_gap",
            sales_growth - sga_growth,
            ["FLOW_EVENT_TIME", "L4_FLOW_EVENT_TIME"],
        )
    )

    profitability = _flow_with_prior_year(
        flow_tables, ["REVENUE", "COGS", "SELL_EXP", "ADMIN_EXP"], ttm=True
    )
    balance_l8 = _lag_table(
        balance,
        8,
        "L8",
        ["BALANCE_EVENT_TIME", "T_ASSETS"],
    )
    profitability = profitability.merge(
        bal[["SECURITY_ID", "QUARTER_INDEX", "BALANCE_EVENT_TIME", "L4_BALANCE_EVENT_TIME", "T_ASSETS", "L4_T_ASSETS"]],
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    ).merge(
        balance_l8,
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    )
    current_op = (
        profitability["TTM_REVENUE"] - profitability["TTM_COGS"]
        - profitability["TTM_SELL_EXP"] - profitability["TTM_ADMIN_EXP"]
    )
    prior_op = (
        profitability["L4_TTM_REVENUE"] - profitability["L4_TTM_COGS"]
        - profitability["L4_TTM_SELL_EXP"] - profitability["L4_TTM_ADMIN_EXP"]
    )
    current_avg_assets = (profitability["T_ASSETS"] + profitability["L4_T_ASSETS"]) / 2
    prior_avg_assets = (profitability["L4_T_ASSETS"] + profitability["L8_T_ASSETS"]) / 2
    current_turnover = _safe_ratio(profitability["TTM_REVENUE"], current_avg_assets)
    prior_turnover = _safe_ratio(profitability["L4_TTM_REVENUE"], prior_avg_assets)
    current_margin = _safe_ratio(current_op, profitability["TTM_REVENUE"])
    prior_margin = _safe_ratio(prior_op, profitability["L4_TTM_REVENUE"])
    profit_times = [
        "FLOW_EVENT_TIME", "L4_FLOW_EVENT_TIME", "BALANCE_EVENT_TIME",
        "L4_BALANCE_EVENT_TIME", "L8_BALANCE_EVENT_TIME",
    ]
    events.extend(
        [
            _long_event(
                profitability,
                "r8_fy_asset_turnover_change",
                current_turnover - prior_turnover,
                profit_times,
            ),
            _long_event(
                profitability,
                "r8_fy_profit_margin_change",
                current_margin - prior_margin,
                profit_times,
            ),
        ]
    )

    result = pd.concat(events, ignore_index=True)
    result["QUARTER_INDEX"] = pd.to_numeric(result["QUARTER_INDEX"]).astype("int64")
    return result.sort_values(["factor", "SECURITY_ID", "EVENT_TIME", "QUARTER_INDEX"])


def prepare_wide_events(events: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for factor, group in events.groupby("factor", sort=False):
        available = assign_available_trade_date(group, calendar)
        available = available.sort_values(
            ["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"]
        )
        newest = available.groupby("SECURITY_ID", sort=False)["QUARTER_INDEX"].cummax()
        available = available.loc[available["QUARTER_INDEX"].eq(newest)]
        available = available.drop_duplicates(["SECURITY_ID", "AVAILABLE_DATE"], keep="last")
        available["factor"] = factor
        pieces.append(available[["SECURITY_ID", "AVAILABLE_DATE", "factor", "value"]])
    long = pd.concat(pieces, ignore_index=True)
    wide = long.pivot_table(
        index=["SECURITY_ID", "AVAILABLE_DATE"],
        columns="factor",
        values="value",
        aggfunc="last",
    ).reset_index()
    for factor in CANDIDATE_COLUMNS:
        if factor not in wide:
            wide[factor] = np.nan
    wide = wide.sort_values(["SECURITY_ID", "AVAILABLE_DATE"])
    wide[CANDIDATE_COLUMNS] = wide.groupby("SECURITY_ID", sort=False)[CANDIDATE_COLUMNS].ffill()
    return wide[["SECURITY_ID", "AVAILABLE_DATE", *CANDIDATE_COLUMNS]]


def _map_sparse(panel: pd.DataFrame, wide_events: pd.DataFrame) -> pd.DataFrame:
    mapped = pd.merge_asof(
        panel.sort_values(["TRADE_DATE", "SECURITY_ID"]),
        wide_events.sort_values(["AVAILABLE_DATE", "SECURITY_ID"]),
        by="SECURITY_ID",
        left_on="TRADE_DATE",
        right_on="AVAILABLE_DATE",
        direction="backward",
    )
    for factor in CANDIDATE_COLUMNS:
        mapped[factor] = pd.to_numeric(mapped[factor], errors="coerce").astype("float32")
    return mapped[KEYS + CANDIDATE_COLUMNS].sort_values(KEYS)


def generate_sparse(
    panel_path: Path,
    pit_dir: Path,
    output: Path,
    coverage_output: Path,
    metadata_output: Path,
) -> None:
    panel_dates = pd.read_parquet(panel_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(panel_dates).unique()).normalize().sort_values()
    flows, balance = _read_inputs(pit_dir)
    events = calculate_factor_events(flows, balance)
    wide = prepare_wide_events(events, calendar)

    chunks: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    for year in sorted(set(calendar.year)):
        filters = [
            ("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)),
            ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31)),
        ]
        panel = _normalize_panel(pd.read_parquet(panel_path, columns=KEYS, filters=filters))
        mapped = _map_sparse(panel, wide)
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
    result = pd.concat(chunks, ignore_index=True).sort_values(KEYS)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(coverage_output, index=False, encoding="utf-8-sig")
    metadata_output.write_text(
        json.dumps(
            {
                "stage": "sparse_before_fill",
                "rows": len(result),
                "factors": CANDIDATE_COLUMNS,
                "period_source_zero_fill": False,
                "daily_cross_section_fill": False,
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
    report_output: Path,
) -> None:
    if not sparse_ic_summary.exists():
        raise FileNotFoundError(
            f"必须先完成未填充版本测试，缺少: {sparse_ic_summary}"
        )
    tested = pd.read_csv(sparse_ic_summary)
    tested_factors = set(tested["factor"].astype(str))
    missing_test = sorted(set(CANDIDATE_COLUMNS).difference(tested_factors))
    if missing_test:
        raise RuntimeError(f"未填充测试缺少因子: {missing_test}")

    data = pd.read_parquet(sparse_path)
    before = data[CANDIDATE_COLUMNS].isna().mean()
    medians = data.groupby("TRADE_DATE", sort=False)[CANDIDATE_COLUMNS].transform("median")
    data[CANDIDATE_COLUMNS] = data[CANDIDATE_COLUMNS].fillna(medians)
    remaining = data[CANDIDATE_COLUMNS].isna().sum()
    if remaining.any():
        bad = remaining.loc[remaining.gt(0)].to_dict()
        raise RuntimeError(f"当日中位数仍无法填充: {bad}")
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False, compression="zstd")
    report = pd.DataFrame(
        {
            "factor": CANDIDATE_COLUMNS,
            "missing_rate_before_fill": [float(before[f]) for f in CANDIDATE_COLUMNS],
            "fill_method": "same_date_cross_section_median_after_sparse_test",
            "remaining_missing_rows": [int(remaining[f]) for f in CANDIDATE_COLUMNS],
        }
    )
    report.to_csv(report_output, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    root = BASE_DIR / "新测试结果" / "第八轮新增文献财务因子"
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-sparse")
    generate.add_argument("--panel", type=Path, default=BASE_DIR / "factors.parquet")
    generate.add_argument("--pit-dir", type=Path, default=BASE_DIR / "data" / "new_pit")
    generate.add_argument("--output", type=Path, default=root / "round8_sparse_before_fill.parquet")
    generate.add_argument("--coverage", type=Path, default=root / "round8_sparse_coverage.csv")
    generate.add_argument("--metadata", type=Path, default=root / "round8_sparse_metadata.json")

    fill = sub.add_parser("fill-after-test")
    fill.add_argument("--sparse", type=Path, default=root / "round8_sparse_before_fill.parquet")
    fill.add_argument("--sparse-ic", type=Path, default=root / "sparse_diagnostic_test" / "ic_summary.csv")
    fill.add_argument("--output", type=Path, default=root / "round8_filled_after_test.parquet")
    fill.add_argument("--report", type=Path, default=root / "round8_fill_report.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate-sparse":
        generate_sparse(
            args.panel.resolve(), args.pit_dir.resolve(), args.output.resolve(),
            args.coverage.resolve(), args.metadata.resolve(),
        )
    else:
        fill_after_test(
            args.sparse.resolve(), args.sparse_ic.resolve(), args.output.resolve(),
            args.report.resolve(),
        )


if __name__ == "__main__":
    main()
