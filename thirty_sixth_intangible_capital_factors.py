"""Round 36: literature-defined organization and R&D capital factors.

Organization capital follows Eisfeldt and Papanikolaou (2013): annual SG&A
is accumulated by the perpetual-inventory method with 15% depreciation and
10% steady-state growth.  R&D capital uses the same transparent stock method,
consistent with the common 15% R&D depreciation convention.  Chinese CPI from
the World Bank converts the recursion into current nominal units.

Construction is PIT-safe, uses no ranks, labels, or previously successful
factors, and emits a complete daily panel for strict testing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pead_sue_factor import assign_available_trade_date
from quarterly_f_score import COMMON_COLUMNS, FINANCIAL_INDUSTRIES, _normalize_common
from twenty_third_alternative_event_factors import KEYS, read_partitioned


DELTA = 0.15
STEADY_GROWTH = 0.10
FACTOR_COLUMNS = [
    "r36_org_cap_assets",
    "r36_org_cap_book_equity",
    "r36_rd_exp_book_equity",
    "r36_rd_cap_assets",
    "r36_rd_cap_book_equity",
    "r36_rd_exp_market_equity",
    "r36_rd_cap_market_equity",
    "r36_total_intangible_cap_assets",
    "r36_org_cap_investment_assets",
    "r36_org_cap_investment_book_equity",
    "r36_rd_cap_investment_assets",
]


def annual_field(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    data = _normalize_common(raw, field, name=f"annual {field}")
    data = data.loc[
        data["REPORT_TYPE"].eq("A") & data["FISCAL_PERIOD"].eq(12),
        ["SECURITY_ID", "END_DATE", "ACT_PUBTIME", field],
    ]
    return data.rename(columns={"ACT_PUBTIME": f"{field}_TIME"})


def read_rd_info(alternative_root: Path) -> pd.DataFrame:
    columns = [
        "SECURITY_ID", "SCANNED_TIME", "PUBLISH_DATE", "END_DATE_REP", "END_DATE",
        "REPORT_TYPE", "MERGED_FLAG", "RD_EXP", "RD_CA", "RD",
    ]
    data = read_partitioned(alternative_root, "fdmt_rd_info", columns)
    for column in ["SCANNED_TIME", "PUBLISH_DATE", "END_DATE_REP", "END_DATE"]:
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["END_DATE"] = data["END_DATE"].dt.normalize()
    data["END_DATE_REP"] = data["END_DATE_REP"].dt.normalize()
    data["RD_INFO_TIME"] = data["SCANNED_TIME"].fillna(
        data["PUBLISH_DATE"].dt.normalize() + pd.Timedelta(1439, unit="m")
    )
    for column in ["RD_EXP", "RD_CA", "RD"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["RD_INFO"] = data["RD"].fillna(data[["RD_EXP", "RD_CA"]].sum(axis=1, min_count=1))
    mask = (
        data["REPORT_TYPE"].astype("string").eq("A")
        & data["MERGED_FLAG"].astype("string").eq("1")
        & data["END_DATE"].eq(data["END_DATE_REP"])
    )
    data = data.loc[mask].dropna(subset=["SECURITY_ID", "END_DATE", "RD_INFO_TIME", "RD_INFO"])
    data["SECURITY_ID"] = pd.to_numeric(data["SECURITY_ID"], errors="coerce").astype("int64")
    return (
        data.sort_values(["SECURITY_ID", "END_DATE", "RD_INFO_TIME"])
        .drop_duplicates(["SECURITY_ID", "END_DATE"], keep="first")
        [["SECURITY_ID", "END_DATE", "RD_INFO_TIME", "RD_INFO"]]
    )


def read_annual_income(pit_root: Path, alternative_root: Path) -> pd.DataFrame:
    fields = ["SELL_EXP", "ADMIN_EXP", "R_D_EXP"]
    raw = pd.read_parquet(
        pit_root / "new_pit_income", columns=COMMON_COLUMNS + fields, engine="pyarrow"
    )
    frames = [annual_field(raw, field) for field in fields]
    data = frames[0]
    for frame in frames[1:]:
        data = data.merge(frame, on=["SECURITY_ID", "END_DATE"], how="outer", validate="one_to_one")
    data["SGA"] = data[["SELL_EXP", "ADMIN_EXP"]].sum(axis=1, min_count=1)
    # A valid annual income report with no separate R&D line means zero
    # reported R&D, not an unavailable company observation.
    rd_info = read_rd_info(alternative_root)
    data = data.merge(rd_info, on=["SECURITY_ID", "END_DATE"], how="left", validate="one_to_one")
    data["R_D_EXP"] = data["RD_INFO"].combine_first(data["R_D_EXP"]).fillna(0.0).clip(lower=0.0)
    data["SGA"] = pd.to_numeric(data["SGA"], errors="coerce").clip(lower=0.0)
    time_columns = [f"{field}_TIME" for field in fields]
    data["INCOME_EVENT_TIME"] = data[time_columns + ["RD_INFO_TIME"]].max(axis=1)
    return data[["SECURITY_ID", "END_DATE", "INCOME_EVENT_TIME", "SGA", "R_D_EXP"]].dropna(
        subset=["SECURITY_ID", "END_DATE", "INCOME_EVENT_TIME", "SGA"]
    )


def read_annual_balance(pit_root: Path) -> pd.DataFrame:
    fields = ["INDUSTRY_CATEGORY", "T_ASSETS", "T_EQUITY_ATTR_P"]
    data = pd.read_parquet(
        pit_root / "new_pit_balance", columns=COMMON_COLUMNS + fields, engine="pyarrow"
    )
    for column in ["ACT_PUBTIME", "END_DATE", "END_DATE_REP"]:
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data["END_DATE"] = data["END_DATE"].dt.normalize()
    data["END_DATE_REP"] = data["END_DATE_REP"].dt.normalize()
    for field in ["T_ASSETS", "T_EQUITY_ATTR_P"]:
        data[field] = pd.to_numeric(data[field], errors="coerce")
    mask = (
        data["MERGED_FLAG"].astype("string").eq("1")
        & data["REPORT_TYPE"].astype("string").eq("A")
        & pd.to_numeric(data["FISCAL_PERIOD"], errors="coerce").eq(12)
        & data["IS_CURRENT_PERIOD"].fillna(False).astype(bool)
        & data["END_DATE"].eq(data["END_DATE_REP"])
        & ~data["INDUSTRY_CATEGORY"].isin(FINANCIAL_INDUSTRIES)
    )
    data = data.loc[mask].dropna(
        subset=["SECURITY_ID", "END_DATE", "ACT_PUBTIME", "T_ASSETS", "T_EQUITY_ATTR_P"]
    )
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    keys = ["SECURITY_ID", "END_DATE"]
    return (
        data.sort_values(keys + ["ACT_PUBTIME", "ID"])
        .drop_duplicates(keys, keep="first")
        [keys + ["ACT_PUBTIME", "T_ASSETS", "T_EQUITY_ATTR_P"]]
        .rename(columns={"ACT_PUBTIME": "BALANCE_EVENT_TIME"})
    )


def read_cpi(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        int(row["date"]): float(row["value"])
        for row in payload[1]
        if row.get("value") is not None
    }


def read_market(root: Path) -> pd.DataFrame:
    data = pd.read_parquet(
        root, columns=["TRADE_DATE", "SECURITY_ID", "MARKET_VALUE_A"], engine="pyarrow"
    )
    data["TRADE_DATE"] = pd.to_datetime(data["TRADE_DATE"], errors="coerce").dt.normalize()
    data["SECURITY_ID"] = pd.to_numeric(data["SECURITY_ID"], errors="coerce")
    data["MARKET_VALUE_A"] = pd.to_numeric(data["MARKET_VALUE_A"], errors="coerce")
    return data.dropna().sort_values(["TRADE_DATE", "SECURITY_ID"])


def build_events(
    income: pd.DataFrame, balance: pd.DataFrame, cpi: dict[int, float], market: pd.DataFrame,
) -> pd.DataFrame:
    data = income.merge(balance, on=["SECURITY_ID", "END_DATE"], how="inner", validate="one_to_one")
    data["EVENT_TIME"] = data[["INCOME_EVENT_TIME", "BALANCE_EVENT_TIME"]].max(axis=1)
    data["YEAR"] = data["END_DATE"].dt.year.astype(int)
    output = []
    for _, group in data.groupby("SECURITY_ID", sort=False):
        g = group.sort_values("YEAR").copy()
        org_stock, rd_stock, prior_year = np.nan, np.nan, None
        org_values, rd_values = [], []
        for row in g.itertuples():
            sga, rd, year = float(row.SGA), float(row.R_D_EXP), int(row.YEAR)
            if prior_year is None or year != prior_year + 1 or not np.isfinite(org_stock):
                org_stock = sga / (DELTA + STEADY_GROWTH)
                rd_stock = rd / (DELTA + STEADY_GROWTH)
            else:
                inflation = cpi.get(year, cpi.get(prior_year, 1.0)) / cpi.get(prior_year, cpi.get(year, 1.0))
                org_stock = (1.0 - DELTA) * inflation * org_stock + sga
                rd_stock = (1.0 - DELTA) * inflation * rd_stock + rd
            org_values.append(org_stock)
            rd_values.append(rd_stock)
            prior_year = year
        g["ORG_CAP"] = org_values
        g["RD_CAP"] = rd_values
        output.append(g)
    data = pd.concat(output, ignore_index=True)
    data["MARKET_LOOKBACK_DATE"] = data["EVENT_TIME"].dt.normalize() - pd.Timedelta(1, unit="D")
    data = pd.merge_asof(
        data.sort_values(["MARKET_LOOKBACK_DATE", "SECURITY_ID"]),
        market, by="SECURITY_ID", left_on="MARKET_LOOKBACK_DATE", right_on="TRADE_DATE",
        direction="backward",
    )
    assets = data["T_ASSETS"].abs().where(data["T_ASSETS"].abs().gt(1e-12))
    equity = data["T_EQUITY_ATTR_P"].where(data["T_EQUITY_ATTR_P"].gt(1e-12))
    data["r36_org_cap_assets"] = data["ORG_CAP"] / assets
    data["r36_org_cap_book_equity"] = data["ORG_CAP"] / equity
    data["r36_rd_exp_book_equity"] = data["R_D_EXP"] / equity
    data["r36_rd_cap_assets"] = data["RD_CAP"] / assets
    data["r36_rd_cap_book_equity"] = data["RD_CAP"] / equity
    market_equity = data["MARKET_VALUE_A"].where(data["MARKET_VALUE_A"].gt(1e-12))
    data["r36_rd_exp_market_equity"] = data["R_D_EXP"] / market_equity
    data["r36_rd_cap_market_equity"] = data["RD_CAP"] / market_equity
    data["r36_total_intangible_cap_assets"] = (data["ORG_CAP"] + data["RD_CAP"]) / assets
    data["r36_org_cap_investment_assets"] = data["SGA"] / assets
    data["r36_org_cap_investment_book_equity"] = data["SGA"] / equity
    data["r36_rd_cap_investment_assets"] = data["R_D_EXP"] / assets
    data[FACTOR_COLUMNS] = data[FACTOR_COLUMNS].replace([np.inf, -np.inf], np.nan).clip(-10, 10)
    return data[["SECURITY_ID", "END_DATE", "EVENT_TIME", *FACTOR_COLUMNS]]


def dense_panel(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.DatetimeIndex(panel["TRADE_DATE"].unique()).sort_values()
    available = assign_available_trade_date(events, calendar).rename(columns={"AVAILABLE_DATE": "EVENT_DATE"})
    available = available.sort_values(["EVENT_DATE", "SECURITY_ID", "EVENT_TIME"]).drop_duplicates(
        ["EVENT_DATE", "SECURITY_ID"], keep="last"
    )
    merged = pd.merge_asof(
        panel.sort_values(["TRADE_DATE", "SECURITY_ID"]),
        available[["SECURITY_ID", "EVENT_DATE", *FACTOR_COLUMNS]].sort_values(["EVENT_DATE", "SECURITY_ID"]),
        by="SECURITY_ID", left_on="TRADE_DATE", right_on="EVENT_DATE", direction="backward",
    )
    # Zero is the economically neutral state for no reported intangible
    # investment/history and preserves the complete strict-test universe.
    merged[FACTOR_COLUMNS] = merged[FACTOR_COLUMNS].fillna(0.0).astype("float32")
    return merged[KEYS + FACTOR_COLUMNS].sort_values(KEYS).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pit-root", type=Path, required=True)
    parser.add_argument("--alternative-root", type=Path, required=True)
    parser.add_argument("--market-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--cpi", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel, columns=KEYS).drop_duplicates(KEYS)
    panel["TRADE_DATE"] = pd.to_datetime(panel["TRADE_DATE"]).dt.normalize()
    panel["SECURITY_ID"] = panel["SECURITY_ID"].astype("int64")
    events = build_events(
        read_annual_income(args.pit_root, args.alternative_root),
        read_annual_balance(args.pit_root), read_cpi(args.cpi), read_market(args.market_root),
    )
    factors = dense_panel(panel, events)
    events.to_parquet(output / "round36_intangible_capital_events.parquet", index=False, compression="zstd")
    factors.to_parquet(output / "round36_intangible_capital_factors.parquet", index=False, compression="zstd")
    metadata = {
        "event_rows": len(events), "event_securities": int(events["SECURITY_ID"].nunique()),
        "factor_columns": FACTOR_COLUMNS, "organization_depreciation": DELTA,
        "steady_state_growth": STEADY_GROWTH, "cpi_source": "World Bank FP.CPI.TOTL CHN",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
