"""Round 34: PIT debt-maturity mismatch and refinancing-pressure factors.

Candidates use quarterly balance-sheet stocks and TTM operating cash flow.
No percentile rank, label, or existing successful factor is used.  The script
enforces the period-data workflow: generate and diagnose the unfilled panel
first; only then may missing histories be neutral-filled for strict testing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eighth_round_literature_factors as workflow
import ninth_round_independent_factors as mapping
from dense_literature_profitability_factors import _merge_flow_tables
from event_financial_factor_search import KEYS, _normalize_panel
from quarterly_f_score import (
    COMMON_COLUMNS, FINANCIAL_INDUSTRIES, REPORT_QUARTERS, REPORT_TYPES,
    build_standalone_quarterly_metric,
)


BALANCE_FIELDS = [
    "INDUSTRY_CATEGORY", "T_ASSETS", "CASH_C_EQUIV", "ST_BORR",
    "NCL_WITHIN_1_Y", "LT_BORR", "BOND_PAYABLE", "LEASE_LIAB",
]
FACTOR_COLUMNS = [
    "r34_low_short_debt_assets",
    "r34_low_short_debt_share",
    "r34_cash_short_debt_buffer",
    "r34_cashflow_short_debt_buffer",
    "r34_low_regularized_short_debt_cfo",
    "r34_short_debt_reduction_yoy",
    "r34_maturity_extension_yoy",
]


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce").abs()
    return pd.to_numeric(numerator, errors="coerce").div(den.where(den.gt(1e-12)))


def read_balance(pit_root: Path) -> pd.DataFrame:
    data = pd.read_parquet(
        pit_root / "new_pit_balance",
        columns=COMMON_COLUMNS + BALANCE_FIELDS,
        engine="pyarrow",
    )
    for column in ["ACT_PUBTIME", "END_DATE", "END_DATE_REP"]:
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["END_DATE"] = data["END_DATE"].dt.normalize()
    data["END_DATE_REP"] = data["END_DATE_REP"].dt.normalize()
    data["SECURITY_ID"] = pd.to_numeric(data["SECURITY_ID"], errors="coerce")
    for field in BALANCE_FIELDS[1:]:
        data[field] = pd.to_numeric(data[field], errors="coerce")
    debt_components = ["ST_BORR", "NCL_WITHIN_1_Y", "LT_BORR", "BOND_PAYABLE", "LEASE_LIAB"]
    data[debt_components] = data[debt_components].fillna(0.0)
    mask = (
        data["MERGED_FLAG"].astype("string").eq("1")
        & data["REPORT_TYPE"].astype("string").isin(REPORT_TYPES)
        & data["IS_CURRENT_PERIOD"].fillna(False).astype(bool)
        & data["END_DATE"].eq(data["END_DATE_REP"])
        & ~data["INDUSTRY_CATEGORY"].isin(FINANCIAL_INDUSTRIES)
    )
    data = data.loc[mask].dropna(
        subset=["SECURITY_ID", "ACT_PUBTIME", "END_DATE", "T_ASSETS", "CASH_C_EQUIV"]
    )
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    data["FISCAL_YEAR"] = data["END_DATE"].dt.year.astype("int16")
    data["FISCAL_QUARTER"] = data["REPORT_TYPE"].map(REPORT_QUARTERS).astype("int8")
    data["QUARTER_INDEX"] = data["FISCAL_YEAR"].astype("int64") * 4 + data["FISCAL_QUARTER"]
    keys = ["SECURITY_ID", "QUARTER_INDEX"]
    data = data.sort_values(keys + ["ACT_PUBTIME", "ID"]).drop_duplicates(keys, keep="first")
    return data[keys + ["ACT_PUBTIME", *BALANCE_FIELDS[1:]]].rename(
        columns={"ACT_PUBTIME": "BALANCE_EVENT_TIME"}
    )


def read_ttm_cfo(pit_root: Path) -> pd.DataFrame:
    raw = pd.read_parquet(
        pit_root / "new_pit_cashflow",
        columns=COMMON_COLUMNS + ["N_CF_OPERATE_A"],
        engine="pyarrow",
    )
    flow = build_standalone_quarterly_metric(raw, "N_CF_OPERATE_A", name="cashflow PIT")
    return _merge_flow_tables({"N_CF_OPERATE_A": flow}, ["N_CF_OPERATE_A"], ttm=True)


def event(frame: pd.DataFrame, factor: str, values: pd.Series) -> pd.DataFrame:
    out = frame[["SECURITY_ID", "QUARTER_INDEX"]].copy()
    out["EVENT_TIME"] = frame[["BALANCE_EVENT_TIME", "FLOW_EVENT_TIME"]].max(axis=1)
    out["factor"] = factor
    out["value"] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=["EVENT_TIME", "value"])


def build_events(balance: pd.DataFrame, cfo: pd.DataFrame) -> pd.DataFrame:
    lag_fields = ["BALANCE_EVENT_TIME", "T_ASSETS", "ST_BORR", "NCL_WITHIN_1_Y", "LT_BORR", "BOND_PAYABLE", "LEASE_LIAB"]
    lag = workflow._lag_table(balance, 4, "L4", lag_fields)
    data = balance.merge(lag, on=["SECURITY_ID", "QUARTER_INDEX"], how="inner", validate="one_to_one")
    data = data.merge(
        cfo[["SECURITY_ID", "QUARTER_INDEX", "FLOW_EVENT_TIME", "TTM_N_CF_OPERATE_A"]],
        on=["SECURITY_ID", "QUARTER_INDEX"], how="inner", validate="one_to_one",
    )
    short = data["ST_BORR"] + data["NCL_WITHIN_1_Y"]
    long = data["LT_BORR"] + data["BOND_PAYABLE"] + data["LEASE_LIAB"]
    total = short + long
    l4_short = data["L4_ST_BORR"] + data["L4_NCL_WITHIN_1_Y"]
    l4_long = data["L4_LT_BORR"] + data["L4_BOND_PAYABLE"] + data["L4_LEASE_LIAB"]
    l4_total = l4_short + l4_long
    assets = data["T_ASSETS"].abs()
    cfo_value = data["TTM_N_CF_OPERATE_A"]
    # The 5% asset floor regularizes loss/near-zero CFO firms without using
    # sample-fitted parameters or cross-sectional ranks.
    regularized_cfo = cfo_value.abs() + 0.05 * assets
    values = {
        "r34_low_short_debt_assets": -safe_ratio(short, assets),
        "r34_low_short_debt_share": -safe_ratio(short, total).fillna(0.0),
        "r34_cash_short_debt_buffer": safe_ratio(data["CASH_C_EQUIV"] - short, assets),
        "r34_cashflow_short_debt_buffer": safe_ratio(data["CASH_C_EQUIV"] + cfo_value - short, assets),
        "r34_low_regularized_short_debt_cfo": -safe_ratio(short, regularized_cfo),
        "r34_short_debt_reduction_yoy": -safe_ratio(short - l4_short, data["L4_T_ASSETS"]),
        "r34_maturity_extension_yoy": safe_ratio(long, total).fillna(0.0) - safe_ratio(l4_long, l4_total).fillna(0.0),
    }
    return pd.concat([event(data, name, value.clip(-5, 5)) for name, value in values.items()])


def generate_sparse(panel_path: Path, pit_root: Path, output_dir: Path) -> None:
    dates = pd.read_parquet(panel_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    events = build_events(read_balance(pit_root), read_ttm_cfo(pit_root))
    mapping.CANDIDATE_COLUMNS = FACTOR_COLUMNS
    wide = mapping.prepare_wide_events(events, calendar)
    chunks, coverage = [], []
    for year in sorted(set(calendar.year)):
        filters = [("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)), ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31))]
        panel = _normalize_panel(pd.read_parquet(panel_path, columns=KEYS, filters=filters))
        mapped = mapping._map_sparse(panel, wide)
        chunks.append(mapped)
        for factor in FACTOR_COLUMNS:
            coverage.append({
                "year": year, "factor": factor, "rows": len(mapped),
                "observed_rows": int(mapped[factor].notna().sum()),
                "missing_rate_before_fill": float(mapped[factor].isna().mean()),
            })
        print(f"{year}: {len(mapped):,} rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.concat(chunks, ignore_index=True).sort_values(KEYS)
    data.to_parquet(output_dir / "round34_sparse_before_fill.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_csv(output_dir / "round34_sparse_coverage.csv", index=False, encoding="utf-8-sig")
    events.to_parquet(output_dir / "round34_events.parquet", index=False, compression="zstd")


def fill_after_test(sparse: Path, sparse_ic: Path, output: Path, report: Path) -> None:
    tested = set(pd.read_csv(sparse_ic)["factor"].astype(str))
    if not set(FACTOR_COLUMNS).issubset(tested):
        raise RuntimeError(f"Sparse test missing: {sorted(set(FACTOR_COLUMNS) - tested)}")
    data = pd.read_parquet(sparse)
    before = data[FACTOR_COLUMNS].isna().mean()
    medians = data.groupby("TRADE_DATE", sort=False)[FACTOR_COLUMNS].transform("median")
    data[FACTOR_COLUMNS] = data[FACTOR_COLUMNS].fillna(medians)
    remaining = data[FACTOR_COLUMNS].isna().sum()
    if remaining.any():
        raise RuntimeError(f"Unfilled rows remain: {remaining[remaining.gt(0)].to_dict()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    data[KEYS + FACTOR_COLUMNS].to_parquet(output, index=False, compression="zstd")
    pd.DataFrame({
        "factor": FACTOR_COLUMNS,
        "missing_rate_before_fill": [float(before[x]) for x in FACTOR_COLUMNS],
        "remaining_missing_rows": [int(remaining[x]) for x in FACTOR_COLUMNS],
        "fill_method": "same_date_median_after_sparse_test",
    }).to_csv(report, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-sparse")
    generate.add_argument("--panel", type=Path, required=True)
    generate.add_argument("--pit-root", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    fill = sub.add_parser("fill-after-test")
    fill.add_argument("--sparse", type=Path, required=True)
    fill.add_argument("--sparse-ic", type=Path, required=True)
    fill.add_argument("--output", type=Path, required=True)
    fill.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate-sparse":
        generate_sparse(args.panel, args.pit_root, args.output_dir)
    else:
        fill_after_test(args.sparse, args.sparse_ic, args.output, args.report)


if __name__ == "__main__":
    main()
