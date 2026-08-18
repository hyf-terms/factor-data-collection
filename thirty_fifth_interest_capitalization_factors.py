"""Round 35: financial-expense and interest-capitalization quality factors.

Uses the PIT financial-expense note and matching balance-sheet stocks.  No
rank transform, label, or existing factor enters construction.  Since note
flows are period observations, the unfilled panel must be tested before the
strict complete-universe version can be produced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import ninth_round_independent_factors as mapping
from event_financial_factor_search import KEYS, _normalize_panel
from quarterly_f_score import REPORT_QUARTERS, REPORT_TYPES
from thirty_fourth_debt_maturity_factors import read_balance, safe_ratio
from twenty_third_alternative_event_factors import map_party, read_partitioned, security_mapping


FACTOR_COLUMNS = [
    "r35_low_interest_capitalization_ratio",
    "r35_low_capitalized_interest_assets",
    "r35_low_net_interest_burden_assets",
    "r35_low_financial_expense_assets",
    "r35_low_effective_debt_cost",
    "r35_interest_income_cash_yield",
    "r35_capitalization_discipline_yoy",
    "r35_interest_burden_improvement_yoy",
]


def read_notes(root: Path, pit_root: Path) -> pd.DataFrame:
    columns = [
        "PARTY_ID", "TICKER_SYMBOL", "ACT_PUBTIME", "END_DATE_REP", "END_DATE",
        "FISCAL_PERIOD", "REPORT_TYPE", "MERGED_FLAG", "INT_EXP",
        "INT_EXP_CAPITAL", "INT_INCOME", "N_INT_EXP", "FIN_EXP",
    ]
    data = map_party(read_partitioned(root, "fdmt_is_fin_exp", columns), security_mapping(pit_root)[0])
    for column in ["ACT_PUBTIME", "END_DATE_REP", "END_DATE"]:
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["END_DATE"] = data["END_DATE"].dt.normalize()
    data["END_DATE_REP"] = data["END_DATE_REP"].dt.normalize()
    for column in ["INT_EXP", "INT_EXP_CAPITAL", "INT_INCOME", "N_INT_EXP", "FIN_EXP"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    # Within an existing financial-expense note, an omitted capitalized-interest
    # line denotes no separately reported capitalization; absent company notes
    # remain missing until the post-diagnostic fill stage.
    data["INT_EXP_CAPITAL"] = data["INT_EXP_CAPITAL"].fillna(0.0)
    data["INT_INCOME"] = data["INT_INCOME"].fillna(0.0)
    mask = (
        data["MERGED_FLAG"].astype("string").eq("1")
        & data["REPORT_TYPE"].astype("string").isin(REPORT_TYPES)
        & data["END_DATE"].eq(data["END_DATE_REP"])
    )
    data = data.loc[mask].dropna(subset=["SECURITY_ID", "ACT_PUBTIME", "END_DATE", "N_INT_EXP", "FIN_EXP"])
    data["FISCAL_YEAR"] = data["END_DATE"].dt.year.astype("int16")
    data["FISCAL_QUARTER"] = data["REPORT_TYPE"].map(REPORT_QUARTERS).astype("int8")
    data["QUARTER_INDEX"] = data["FISCAL_YEAR"].astype("int64") * 4 + data["FISCAL_QUARTER"]
    keys = ["SECURITY_ID", "QUARTER_INDEX"]
    return (
        data.sort_values(keys + ["ACT_PUBTIME"])
        .drop_duplicates(keys, keep="first")
        [keys + ["ACT_PUBTIME", "INT_EXP", "INT_EXP_CAPITAL", "INT_INCOME", "N_INT_EXP", "FIN_EXP"]]
        .rename(columns={"ACT_PUBTIME": "NOTE_EVENT_TIME"})
    )


def make_event(data: pd.DataFrame, name: str, value: pd.Series) -> pd.DataFrame:
    out = data[["SECURITY_ID", "QUARTER_INDEX"]].copy()
    out["EVENT_TIME"] = data[["NOTE_EVENT_TIME", "BALANCE_EVENT_TIME"]].max(axis=1)
    out["factor"] = name
    out["value"] = pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=["EVENT_TIME", "value"])


def build_events(notes: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    data = notes.merge(balance, on=["SECURITY_ID", "QUARTER_INDEX"], how="inner", validate="one_to_one")
    short = data["ST_BORR"] + data["NCL_WITHIN_1_Y"]
    long = data["LT_BORR"] + data["BOND_PAYABLE"] + data["LEASE_LIAB"]
    debt = short + long
    gross_interest = data["INT_EXP"].abs().where(data["INT_EXP"].notna(), data["N_INT_EXP"].abs())
    cap_ratio = safe_ratio(data["INT_EXP_CAPITAL"], gross_interest).clip(0, 2).fillna(0.0)
    net_interest_assets = safe_ratio(data["N_INT_EXP"], data["T_ASSETS"])
    values = {
        "r35_low_interest_capitalization_ratio": -cap_ratio,
        "r35_low_capitalized_interest_assets": -safe_ratio(data["INT_EXP_CAPITAL"], data["T_ASSETS"]),
        "r35_low_net_interest_burden_assets": -net_interest_assets,
        "r35_low_financial_expense_assets": -safe_ratio(data["FIN_EXP"], data["T_ASSETS"]),
        "r35_low_effective_debt_cost": -safe_ratio(data["N_INT_EXP"], debt + 0.01 * data["T_ASSETS"].abs()),
        "r35_interest_income_cash_yield": safe_ratio(data["INT_INCOME"], data["CASH_C_EQUIV"].abs() + 0.01 * data["T_ASSETS"].abs()),
    }
    history = data[["SECURITY_ID", "QUARTER_INDEX", "NOTE_EVENT_TIME", "BALANCE_EVENT_TIME"]].copy()
    history["CAP_RATIO"] = cap_ratio
    history["INTEREST_ASSETS"] = net_interest_assets
    lag = history[["SECURITY_ID", "QUARTER_INDEX", "CAP_RATIO", "INTEREST_ASSETS"]].copy()
    lag["QUARTER_INDEX"] += 4
    lag = lag.rename(columns={"CAP_RATIO": "L4_CAP_RATIO", "INTEREST_ASSETS": "L4_INTEREST_ASSETS"})
    history = history.merge(lag, on=["SECURITY_ID", "QUARTER_INDEX"], how="left", validate="one_to_one")
    values["r35_capitalization_discipline_yoy"] = -(history["CAP_RATIO"] - history["L4_CAP_RATIO"])
    values["r35_interest_burden_improvement_yoy"] = -(history["INTEREST_ASSETS"] - history["L4_INTEREST_ASSETS"])
    events = []
    for name, value in values.items():
        source = history if name.endswith("_yoy") else data
        events.append(make_event(source, name, value.clip(-5, 5)))
    return pd.concat(events, ignore_index=True)


def generate_sparse(panel_path: Path, pit_root: Path, alternative_root: Path, output_dir: Path) -> None:
    dates = pd.read_parquet(panel_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    events = build_events(read_notes(alternative_root, pit_root), read_balance(pit_root))
    mapping.CANDIDATE_COLUMNS = FACTOR_COLUMNS
    wide = mapping.prepare_wide_events(events, calendar)
    chunks, coverage = [], []
    for year in sorted(set(calendar.year)):
        filters = [("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)), ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31))]
        panel = _normalize_panel(pd.read_parquet(panel_path, columns=KEYS, filters=filters))
        mapped = mapping._map_sparse(panel, wide)
        chunks.append(mapped)
        for factor in FACTOR_COLUMNS:
            coverage.append({"year": year, "factor": factor, "rows": len(mapped), "observed_rows": int(mapped[factor].notna().sum()), "missing_rate_before_fill": float(mapped[factor].isna().mean())})
        print(f"{year}: {len(mapped):,} rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.concat(chunks, ignore_index=True).sort_values(KEYS)
    result.to_parquet(output_dir / "round35_sparse_before_fill.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_csv(output_dir / "round35_sparse_coverage.csv", index=False, encoding="utf-8-sig")
    events.to_parquet(output_dir / "round35_events.parquet", index=False, compression="zstd")


def fill_after_test(sparse: Path, sparse_ic: Path, output: Path, report: Path) -> None:
    tested = set(pd.read_csv(sparse_ic)["factor"].astype(str))
    if not set(FACTOR_COLUMNS).issubset(tested):
        raise RuntimeError(f"Sparse test missing: {sorted(set(FACTOR_COLUMNS)-tested)}")
    data = pd.read_parquet(sparse)
    before = data[FACTOR_COLUMNS].isna().mean()
    medians = data.groupby("TRADE_DATE", sort=False)[FACTOR_COLUMNS].transform("median")
    data[FACTOR_COLUMNS] = data[FACTOR_COLUMNS].fillna(medians)
    remaining = data[FACTOR_COLUMNS].isna().sum()
    if remaining.any():
        raise RuntimeError(f"Unfilled rows remain: {remaining[remaining.gt(0)].to_dict()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    data[KEYS + FACTOR_COLUMNS].to_parquet(output, index=False, compression="zstd")
    pd.DataFrame({"factor": FACTOR_COLUMNS, "missing_rate_before_fill": [float(before[x]) for x in FACTOR_COLUMNS], "remaining_missing_rows": [int(remaining[x]) for x in FACTOR_COLUMNS], "fill_method": "same_date_median_after_sparse_test"}).to_csv(report, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-sparse")
    generate.add_argument("--panel", type=Path, required=True)
    generate.add_argument("--pit-root", type=Path, required=True)
    generate.add_argument("--alternative-root", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    fill = sub.add_parser("fill-after-test")
    fill.add_argument("--sparse", type=Path, required=True)
    fill.add_argument("--sparse-ic", type=Path, required=True)
    fill.add_argument("--output", type=Path, required=True)
    fill.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate-sparse":
        generate_sparse(args.panel, args.pit_root, args.alternative_root, args.output_dir)
    else:
        fill_after_test(args.sparse, args.sparse_ic, args.output, args.report)


if __name__ == "__main__":
    main()
