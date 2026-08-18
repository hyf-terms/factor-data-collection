"""Build independent Beneish/Dechow accounting-risk factors from PIT data.

The factors in this module never use an existing factor as an input.  Period
items are first converted to standalone quarters and then to TTM values.  The
workflow deliberately writes a sparse, unfilled daily panel first.  Filling is
available only after an IC summary for that sparse panel has been supplied.
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
INCOME_FIELDS = ["N_INCOME_ATTR_P", "REVENUE", "COGS", "SELL_EXP", "ADMIN_EXP"]
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
    "FIXED_ASSETS_TOTAL",
    "T_LIAB",
    "LT_BORR",
    "BOND_PAYABLE",
    "PAID_IN_CAPITAL",
]

CANDIDATE_COLUMNS = [
    "r10_beneish_quality_m7_cashflow",
    "r10_beneish_low_dsri",
    "r10_beneish_low_gmi",
    "r10_beneish_low_aqi",
    "r10_beneish_low_sgi",
    "r10_beneish_sgai_quality_component",
    "r10_beneish_lvgi_quality_component",
    "r10_beneish_low_tata_cashflow",
    "r10_dechow_quality_fscore7",
    "r10_dechow_low_rsst",
    "r10_dechow_low_receivable_change",
    "r10_dechow_low_inventory_change",
    "r10_dechow_low_soft_assets",
    "r10_dechow_low_cash_sales_change",
    "r10_dechow_roa_change",
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
    balance = pd.read_parquet(
        pit_dir / "new_pit_balance",
        columns=COMMON_COLUMNS + BALANCE_FIELDS,
        engine="pyarrow",
    )
    flows: dict[str, pd.DataFrame] = {}
    for field in INCOME_FIELDS:
        flows[field] = build_standalone_quarterly_metric(income, field, name="income PIT")
    for field in CASHFLOW_FIELDS:
        flows[field] = build_standalone_quarterly_metric(cashflow, field, name="cashflow PIT")
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
    # Undisclosed debt components are economically zero; no period flow is filled here.
    for field in ("ST_BORR", "NCL_WITHIN_1_Y", "LT_BORR", "BOND_PAYABLE"):
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
        keys
        + ["FISCAL_YEAR", "FISCAL_QUARTER", "END_DATE", "ACT_PUBTIME", *BALANCE_FIELDS]
    ].rename(columns={"ACT_PUBTIME": "BALANCE_EVENT_TIME"})


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return workflow._safe_ratio(numerator, denominator, absolute_denominator=True)


def _lag(frame: pd.DataFrame, quarters: int, prefix: str, columns: list[str]) -> pd.DataFrame:
    return workflow._lag_table(frame, quarters, prefix, columns)


def _event(frame: pd.DataFrame, name: str, value: pd.Series, times: list[str]) -> pd.DataFrame:
    return workflow._long_event(frame, name, value, times)


def calculate_factor_events(
    flows: dict[str, pd.DataFrame], balance: pd.DataFrame
) -> pd.DataFrame:
    fields = INCOME_FIELDS + CASHFLOW_FIELDS
    ttm = _merge_flow_tables(flows, fields, ttm=True)
    ttm_values = [f"TTM_{field}" for field in fields]
    ttm = ttm.merge(
        _lag(ttm, 4, "L4", ["FLOW_EVENT_TIME", *ttm_values]),
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    )

    balance_values = [field for field in BALANCE_FIELDS if field != "INDUSTRY_CATEGORY"]
    bal = balance.merge(
        _lag(balance, 4, "L4", ["BALANCE_EVENT_TIME", *balance_values]),
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    ).merge(
        _lag(balance, 8, "L8", ["BALANCE_EVENT_TIME", *balance_values]),
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    )
    data = ttm.merge(
        bal,
        on=["SECURITY_ID", "QUARTER_INDEX"],
        how="inner",
        validate="one_to_one",
    )
    times = [
        "FLOW_EVENT_TIME",
        "L4_FLOW_EVENT_TIME",
        "BALANCE_EVENT_TIME",
        "L4_BALANCE_EVENT_TIME",
        "L8_BALANCE_EVENT_TIME",
    ]

    sales = data["TTM_REVENUE"]
    lag_sales = data["L4_TTM_REVENUE"]
    gross_margin = _ratio(sales - data["TTM_COGS"], sales)
    lag_gross_margin = _ratio(
        lag_sales - data["L4_TTM_COGS"], lag_sales
    )
    dsri = _ratio(_ratio(data["AR"], sales), _ratio(data["L4_AR"], lag_sales))
    gmi = _ratio(lag_gross_margin, gross_margin)
    current_asset_quality = 1.0 - _ratio(
        data["T_CA"] + data["FIXED_ASSETS_TOTAL"], data["T_ASSETS"]
    )
    lag_asset_quality = 1.0 - _ratio(
        data["L4_T_CA"] + data["L4_FIXED_ASSETS_TOTAL"], data["L4_T_ASSETS"]
    )
    aqi = _ratio(current_asset_quality, lag_asset_quality)
    sgi = _ratio(sales, lag_sales)
    sga_ratio = _ratio(data["TTM_SELL_EXP"] + data["TTM_ADMIN_EXP"], sales)
    lag_sga_ratio = _ratio(
        data["L4_TTM_SELL_EXP"] + data["L4_TTM_ADMIN_EXP"], lag_sales
    )
    sgai = _ratio(sga_ratio, lag_sga_ratio)
    leverage = _ratio(data["T_LIAB"], data["T_ASSETS"])
    lag_leverage = _ratio(data["L4_T_LIAB"], data["L4_T_ASSETS"])
    lvgi = _ratio(leverage, lag_leverage)
    tata = _ratio(
        data["TTM_N_INCOME_ATTR_P"] - data["TTM_N_CF_OPERATE_A"],
        data["T_ASSETS"],
    )
    m7_risk = (
        0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        - 0.172 * sgai
        - 0.327 * lvgi
        + 4.697 * tata
    )

    average_assets = (data["T_ASSETS"] + data["L4_T_ASSETS"]) / 2.0
    debt = data[["ST_BORR", "NCL_WITHIN_1_Y", "LT_BORR", "BOND_PAYABLE"]].sum(axis=1)
    lag_debt = data[["L4_ST_BORR", "L4_NCL_WITHIN_1_Y", "L4_LT_BORR", "L4_BOND_PAYABLE"]].sum(axis=1)
    noa = (data["T_ASSETS"] - data["CASH_C_EQUIV"]) - (data["T_LIAB"] - debt)
    lag_noa = (data["L4_T_ASSETS"] - data["L4_CASH_C_EQUIV"]) - (data["L4_T_LIAB"] - lag_debt)
    rsst = _ratio(noa - lag_noa, average_assets)
    ch_rec = _ratio(data["AR"] - data["L4_AR"], average_assets)
    ch_inv = _ratio(data["INVENTORIES"] - data["L4_INVENTORIES"], average_assets)
    soft_assets = _ratio(
        data["T_ASSETS"] - data["CASH_C_EQUIV"] - data["FIXED_ASSETS_TOTAL"],
        data["T_ASSETS"],
    )
    current_cash_sales = sales - (data["AR"] - data["L4_AR"])
    lag_cash_sales = lag_sales - (data["L4_AR"] - data["L8_AR"])
    ch_cash_sales = _ratio(current_cash_sales - lag_cash_sales, lag_cash_sales)
    roa = _ratio(data["TTM_N_INCOME_ATTR_P"], average_assets)
    lag_average_assets = (data["L4_T_ASSETS"] + data["L8_T_ASSETS"]) / 2.0
    lag_roa = _ratio(data["L4_TTM_N_INCOME_ATTR_P"], lag_average_assets)
    ch_roa = roa - lag_roa
    issue = (data["PAID_IN_CAPITAL"] > data["L4_PAID_IN_CAPITAL"] * (1.0 + 1e-8)).astype(float)
    fscore_risk = (
        -7.893
        + 0.790 * rsst
        + 2.518 * ch_rec
        + 1.191 * ch_inv
        + 1.979 * soft_assets
        + 0.171 * ch_cash_sales
        - 0.932 * ch_roa
        + 1.029 * issue
    )

    values = {
        "r10_beneish_quality_m7_cashflow": -m7_risk,
        "r10_beneish_low_dsri": -dsri,
        "r10_beneish_low_gmi": -gmi,
        "r10_beneish_low_aqi": -aqi,
        "r10_beneish_low_sgi": -sgi,
        "r10_beneish_sgai_quality_component": sgai,
        "r10_beneish_lvgi_quality_component": lvgi,
        "r10_beneish_low_tata_cashflow": -tata,
        "r10_dechow_quality_fscore7": -fscore_risk,
        "r10_dechow_low_rsst": -rsst,
        "r10_dechow_low_receivable_change": -ch_rec,
        "r10_dechow_low_inventory_change": -ch_inv,
        "r10_dechow_low_soft_assets": -soft_assets,
        "r10_dechow_low_cash_sales_change": -ch_cash_sales,
        "r10_dechow_roa_change": ch_roa,
    }
    events = [_event(data, name, value, times) for name, value in values.items()]
    result = pd.concat(events, ignore_index=True)
    result["QUARTER_INDEX"] = pd.to_numeric(result["QUARTER_INDEX"]).astype("int64")
    return result.sort_values(["factor", "SECURITY_ID", "EVENT_TIME", "QUARTER_INDEX"])


def generate_sparse(panel_path: Path, pit_dir: Path, output_dir: Path) -> None:
    panel_dates = pd.read_parquet(panel_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(panel_dates).unique()).normalize().sort_values()
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
    result = pd.concat(chunks, ignore_index=True).sort_values(KEYS)
    result.to_parquet(output_dir / "round10_sparse_before_fill.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage_rows).to_csv(
        output_dir / "round10_sparse_coverage.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "round10_sparse_metadata.json").write_text(
        json.dumps(
            {
                "stage": "sparse_before_fill",
                "rows": len(result),
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
    selected: list[str] | None,
) -> None:
    if not sparse_ic_summary.exists():
        raise FileNotFoundError(f"Sparse IC summary must exist before fill: {sparse_ic_summary}")
    tested = set(pd.read_csv(sparse_ic_summary)["factor"].astype(str))
    factors = selected or CANDIDATE_COLUMNS
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
    root = BASE_DIR / "artifacts" / "round10"
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-sparse")
    generate.add_argument("--panel", type=Path, default=BASE_DIR / "factors.parquet")
    generate.add_argument("--pit-dir", type=Path, default=BASE_DIR / "data" / "new_pit")
    generate.add_argument("--output-dir", type=Path, default=root)
    fill = sub.add_parser("fill-after-test")
    fill.add_argument("--sparse", type=Path, default=root / "round10_sparse_before_fill.parquet")
    fill.add_argument("--sparse-ic", type=Path, default=root / "sparse_diagnostic_test" / "ic_summary.csv")
    fill.add_argument("--output", type=Path, default=root / "round10_filled_after_test.parquet")
    fill.add_argument("--report", type=Path, default=root / "round10_fill_report.csv")
    fill.add_argument("--factor-columns", nargs="+")
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
