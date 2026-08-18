"""Round 17: working-capital levels, changes, and temporal surprises.

The candidates are standalone factors and never read an existing best factor.
Period observations stay missing until a sparse diagnostic test is completed.
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


BALANCE_FIELDS = [
    "INDUSTRY_CATEGORY", "T_ASSETS", "CASH_C_EQUIV", "TRADING_FA",
    "AR", "NOTES_RECEIV", "INVENTORIES", "PREPAYMENT", "OTH_RECEIV_TOTAL",
    "AP", "NOTES_PAYABLE", "ADVANCE_RECEIPTS", "CONT_LIAB", "DEFER_REVENUE",
    "T_LIAB", "ST_BORR", "NCL_WITHIN_1_Y", "LT_BORR", "BOND_PAYABLE",
]
INCOME_FIELDS = ["REVENUE", "COGS", "OPERATE_PROFIT"]

CANDIDATE_COLUMNS = [
    "r17_contract_funding_assets",
    "r17_supplier_funding_assets",
    "r17_low_net_working_capital_assets",
    "r17_low_net_operating_assets",
    "r17_cash_conversion_level",
    "r17_operating_asset_turnover",
    "r17_gp_operating_assets",
    "r17_op_operating_assets",
    "r17_contract_funding_change_assets",
    "r17_supplier_funding_change_assets",
    "r17_working_capital_release_assets",
    "r17_working_capital_sales_gap",
    "r17_cash_conversion_improvement",
    "r17_operating_asset_turnover_change",
    "r17_contract_funding_sue4",
    "r17_working_capital_release_sue4",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce").abs()
    return pd.to_numeric(numerator, errors="coerce").div(den.where(den.gt(1e-12)))


def _read_balance(pit_dir: Path) -> pd.DataFrame:
    data = pd.read_parquet(
        pit_dir / "new_pit_balance",
        columns=COMMON_COLUMNS + BALANCE_FIELDS,
        engine="pyarrow",
    )
    for column in ("ACT_PUBTIME", "END_DATE", "END_DATE_REP"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["END_DATE"] = data["END_DATE"].dt.normalize()
    data["END_DATE_REP"] = data["END_DATE_REP"].dt.normalize()
    data["SECURITY_ID"] = pd.to_numeric(data["SECURITY_ID"], errors="coerce")
    for field in BALANCE_FIELDS:
        if field != "INDUSTRY_CATEGORY":
            data[field] = pd.to_numeric(data[field], errors="coerce")
    # Explicitly absent financing/contract items are economically zero.
    zero_fields = [
        "NOTES_RECEIV", "PREPAYMENT", "OTH_RECEIV_TOTAL", "NOTES_PAYABLE",
        "ADVANCE_RECEIPTS", "CONT_LIAB", "DEFER_REVENUE", "TRADING_FA",
        "ST_BORR", "NCL_WITHIN_1_Y", "LT_BORR", "BOND_PAYABLE",
    ]
    data[zero_fields] = data[zero_fields].fillna(0.0)
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
    return data[keys + ["ACT_PUBTIME", *BALANCE_FIELDS]].rename(
        columns={"ACT_PUBTIME": "BALANCE_EVENT_TIME"}
    )


def _read_flows(pit_dir: Path) -> dict[str, pd.DataFrame]:
    raw = pd.read_parquet(
        pit_dir / "new_pit_income",
        columns=COMMON_COLUMNS + INCOME_FIELDS,
        engine="pyarrow",
    )
    return {
        field: build_standalone_quarterly_metric(raw, field, name="income PIT")
        for field in INCOME_FIELDS
    }


def _lag(frame: pd.DataFrame, quarters: int, prefix: str, columns: list[str]) -> pd.DataFrame:
    return workflow._lag_table(frame, quarters, prefix, columns)


def _event(frame: pd.DataFrame, factor: str, value: pd.Series, times: list[str]) -> pd.DataFrame:
    result = frame[["SECURITY_ID", "QUARTER_INDEX"]].copy()
    result["EVENT_TIME"] = frame[times].max(axis=1)
    result["factor"] = factor
    result["value"] = pd.to_numeric(value, errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=["EVENT_TIME", "value"])


def _prior_sue4(frame: pd.DataFrame, value_column: str) -> pd.Series:
    pieces: list[pd.Series] = []
    for _, group in frame.groupby("SECURITY_ID", sort=False):
        ordered = group.sort_values("QUARTER_INDEX")
        start, stop = int(ordered["QUARTER_INDEX"].min()), int(ordered["QUARTER_INDEX"].max()) + 1
        full = pd.to_numeric(
            ordered.set_index("QUARTER_INDEX")[value_column], errors="coerce"
        ).reindex(pd.RangeIndex(start, stop))
        surprise = full - full.shift(4)
        scale = surprise.shift(1).rolling(4, min_periods=4).std(ddof=1)
        sue = surprise.div(scale.where(scale.gt(1e-12)))
        pieces.append(pd.Series(sue.reindex(ordered["QUARTER_INDEX"]).to_numpy(), index=ordered.index))
    return pd.concat(pieces).reindex(frame.index)


def calculate_factor_events(flows: dict[str, pd.DataFrame], balance: pd.DataFrame) -> pd.DataFrame:
    ttm = _merge_flow_tables(flows, INCOME_FIELDS, ttm=True)
    flow_values = [f"TTM_{field}" for field in INCOME_FIELDS]
    ttm = ttm.merge(
        _lag(ttm, 4, "L4", ["FLOW_EVENT_TIME", *flow_values]),
        on=["SECURITY_ID", "QUARTER_INDEX"], how="inner", validate="one_to_one",
    )
    balance_values = [field for field in BALANCE_FIELDS if field != "INDUSTRY_CATEGORY"]
    bal = balance.merge(
        _lag(balance, 4, "L4", ["BALANCE_EVENT_TIME", *balance_values]),
        on=["SECURITY_ID", "QUARTER_INDEX"], how="inner", validate="one_to_one",
    )
    data = ttm.merge(bal, on=["SECURITY_ID", "QUARTER_INDEX"], how="inner", validate="one_to_one")
    times = ["FLOW_EVENT_TIME", "L4_FLOW_EVENT_TIME", "BALANCE_EVENT_TIME", "L4_BALANCE_EVENT_TIME"]

    contract = data["ADVANCE_RECEIPTS"] + data["CONT_LIAB"] + data["DEFER_REVENUE"]
    l4_contract = data["L4_ADVANCE_RECEIPTS"] + data["L4_CONT_LIAB"] + data["L4_DEFER_REVENUE"]
    receivable = data["AR"] + data["NOTES_RECEIV"]
    l4_receivable = data["L4_AR"] + data["L4_NOTES_RECEIV"]
    payable = data["AP"] + data["NOTES_PAYABLE"]
    l4_payable = data["L4_AP"] + data["L4_NOTES_PAYABLE"]
    supplier = payable + contract
    l4_supplier = l4_payable + l4_contract
    nwc = receivable + data["INVENTORIES"] + data["PREPAYMENT"] - supplier
    l4_nwc = l4_receivable + data["L4_INVENTORIES"] + data["L4_PREPAYMENT"] - l4_supplier
    debt = data["ST_BORR"] + data["NCL_WITHIN_1_Y"] + data["LT_BORR"] + data["BOND_PAYABLE"]
    operating_assets = data["T_ASSETS"] - data["CASH_C_EQUIV"] - data["TRADING_FA"]
    operating_liabilities = data["T_LIAB"] - debt
    noa = operating_assets - operating_liabilities
    avg_operating_assets = 0.5 * (
        operating_assets + data["L4_T_ASSETS"] - data["L4_CASH_C_EQUIV"] - data["L4_TRADING_FA"]
    )
    sales, lag_sales = data["TTM_REVENUE"], data["L4_TTM_REVENUE"]
    cogs, lag_cogs = data["TTM_COGS"], data["L4_TTM_COGS"]
    gross_profit = sales - cogs
    conversion = -_safe_ratio(receivable, sales) - _safe_ratio(data["INVENTORIES"], cogs) + _safe_ratio(payable + contract, cogs)
    l4_conversion = -_safe_ratio(l4_receivable, lag_sales) - _safe_ratio(data["L4_INVENTORIES"], lag_cogs) + _safe_ratio(l4_payable + l4_contract, lag_cogs)
    asset_base = data["L4_T_ASSETS"]
    sales_change = sales - lag_sales
    nwc_change = nwc - l4_nwc

    raw_values = {
        "r17_contract_funding_assets": _safe_ratio(contract, data["T_ASSETS"]),
        "r17_supplier_funding_assets": _safe_ratio(supplier, data["T_ASSETS"]),
        "r17_low_net_working_capital_assets": -_safe_ratio(nwc, data["T_ASSETS"]),
        "r17_low_net_operating_assets": -_safe_ratio(noa, data["T_ASSETS"]),
        "r17_cash_conversion_level": conversion,
        "r17_operating_asset_turnover": _safe_ratio(sales, avg_operating_assets),
        "r17_gp_operating_assets": _safe_ratio(gross_profit, avg_operating_assets),
        "r17_op_operating_assets": _safe_ratio(data["TTM_OPERATE_PROFIT"], avg_operating_assets),
        "r17_contract_funding_change_assets": _safe_ratio(contract - l4_contract, asset_base),
        "r17_supplier_funding_change_assets": _safe_ratio(supplier - l4_supplier, asset_base),
        "r17_working_capital_release_assets": -_safe_ratio(nwc_change, asset_base),
        "r17_working_capital_sales_gap": _safe_ratio(sales_change - nwc_change, asset_base),
        "r17_cash_conversion_improvement": conversion - l4_conversion,
        "r17_operating_asset_turnover_change": _safe_ratio(sales, avg_operating_assets) - _safe_ratio(lag_sales, data["L4_T_ASSETS"] - data["L4_CASH_C_EQUIV"] - data["L4_TRADING_FA"]),
    }
    events = [_event(data, factor, value, times) for factor, value in raw_values.items()]

    temporal = data[["SECURITY_ID", "QUARTER_INDEX", *times]].copy()
    temporal["contract_ratio"] = _safe_ratio(contract, data["T_ASSETS"])
    temporal["working_capital_ratio"] = _safe_ratio(nwc, data["T_ASSETS"])
    temporal["contract_sue4"] = _prior_sue4(temporal, "contract_ratio")
    temporal["wc_sue4"] = -_prior_sue4(temporal, "working_capital_ratio")
    events.extend([
        _event(temporal, "r17_contract_funding_sue4", temporal["contract_sue4"], times),
        _event(temporal, "r17_working_capital_release_sue4", temporal["wc_sue4"], times),
    ])
    return pd.concat(events, ignore_index=True).sort_values(["factor", "SECURITY_ID", "EVENT_TIME"])


def generate_sparse(panel: Path, pit_dir: Path, output_dir: Path) -> None:
    dates = pd.read_parquet(panel, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    events = calculate_factor_events(_read_flows(pit_dir), _read_balance(pit_dir))
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
    result.to_parquet(output_dir / "round17_sparse_before_fill.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_csv(output_dir / "round17_sparse_coverage.csv", index=False, encoding="utf-8-sig")
    (output_dir / "round17_metadata.json").write_text(json.dumps({"stage": "sparse_before_fill", "rows": len(result), "factors": CANDIDATE_COLUMNS, "period_source_zero_fill": False, "existing_best_factor_used": False}, ensure_ascii=False, indent=2), encoding="utf-8")


def fill_after_test(sparse: Path, ic_summary: Path, output: Path, factors: list[str]) -> None:
    tested = set(pd.read_csv(ic_summary)["factor"].astype(str))
    if not set(factors).issubset(tested):
        raise RuntimeError(f"not sparse-tested: {sorted(set(factors) - tested)}")
    data = pd.read_parquet(sparse, columns=KEYS + factors)
    before = data[factors].isna().mean()
    data[factors] = data[factors].fillna(data.groupby("TRADE_DATE", sort=False)[factors].transform("median")).fillna(0.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False, compression="zstd")
    pd.DataFrame({"factor": factors, "missing_rate_before_fill": [before[f] for f in factors], "fill_method": "same_date_median_then_zero_for_whole_day_after_sparse_test"}).to_csv(output.with_suffix(".fill_report.csv"), index=False, encoding="utf-8-sig")


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
    fill.add_argument("--factor-columns", nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "generate-sparse":
        generate_sparse(args.panel.resolve(), args.pit_dir.resolve(), args.output_dir.resolve())
    else:
        fill_after_test(args.sparse.resolve(), args.sparse_ic.resolve(), args.output.resolve(), args.factor_columns)


if __name__ == "__main__":
    main()
