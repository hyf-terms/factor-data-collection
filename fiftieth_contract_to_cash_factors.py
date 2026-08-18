"""Round 50: PIT contract-to-cash conversion factors.

The family uses sales cash receipts and receivable/contract-asset claims.  It
contains no profit, margin, earnings-surprise, labels, ranks or existing
factors. Period-flow candidates remain naturally missing until enough quarters
exist and are diagnosed before any cross-sectional filling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dense_literature_profitability_factors import _merge_flow_tables
from event_financial_factor_search import KEYS, _normalize_panel
from pead_sue_factor import assign_available_trade_date
from quarterly_f_score import (
    COMMON_COLUMNS, FINANCIAL_INDUSTRIES, REPORT_QUARTERS, REPORT_TYPES,
    build_standalone_quarterly_metric,
)


FACTORS = [
    "r50_ttm_cash_collection_assets",
    "r50_cash_collection_receivable_productivity",
    "r50_low_receivable_claims_assets",
    "r50_receivable_claims_yoy_release",
    "r50_cash_collection_surplus_yoy",
]
BALANCE_FIELDS = [
    "INDUSTRY_CATEGORY", "T_ASSETS", "AR", "NOTES_RECEIV", "CONT_ASSETS",
]


def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce")
    floor = max(float(den.abs().median(skipna=True)) * 1e-8, 1e-12)
    return pd.to_numeric(numerator, errors="coerce").div(den.where(den.abs().gt(floor))).replace([np.inf, -np.inf], np.nan)


def balance_events(pit_dir: Path) -> pd.DataFrame:
    d = pd.read_parquet(
        pit_dir / "new_pit_balance", columns=COMMON_COLUMNS + BALANCE_FIELDS,
        engine="pyarrow",
    )
    for c in ["ACT_PUBTIME", "END_DATE", "END_DATE_REP"]:
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d["END_DATE"] = d["END_DATE"].dt.normalize()
    d["END_DATE_REP"] = d["END_DATE_REP"].dt.normalize()
    for c in BALANCE_FIELDS[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # Missing notes/contract assets inside an otherwise disclosed statement
    # denote an absent line. AR and total assets remain mandatory.
    d[["NOTES_RECEIV", "CONT_ASSETS"]] = d[["NOTES_RECEIV", "CONT_ASSETS"]].fillna(0.0)
    mask = (
        d["MERGED_FLAG"].astype("string").eq("1")
        & d["REPORT_TYPE"].astype("string").isin(REPORT_TYPES)
        & d["IS_CURRENT_PERIOD"].fillna(False).astype(bool)
        & d["END_DATE"].eq(d["END_DATE_REP"])
        & ~d["INDUSTRY_CATEGORY"].isin(FINANCIAL_INDUSTRIES)
    )
    d = d.loc[mask].dropna(subset=["SECURITY_ID", "ACT_PUBTIME", "END_DATE", "T_ASSETS", "AR"])
    d["SECURITY_ID"] = pd.to_numeric(d["SECURITY_ID"]).astype("int64")
    d["FISCAL_YEAR"] = d["END_DATE"].dt.year.astype("int16")
    d["FISCAL_QUARTER"] = d["REPORT_TYPE"].map(REPORT_QUARTERS).astype("int8")
    d["QUARTER_INDEX"] = d["FISCAL_YEAR"].astype("int64") * 4 + d["FISCAL_QUARTER"]
    d = d.sort_values(["SECURITY_ID", "QUARTER_INDEX", "ACT_PUBTIME", "ID"]).drop_duplicates(["SECURITY_ID", "QUARTER_INDEX"], keep="first")
    d["RECEIVABLE_CLAIMS"] = d["AR"] + d["NOTES_RECEIV"] + d["CONT_ASSETS"]
    return d[["SECURITY_ID", "QUARTER_INDEX", "ACT_PUBTIME", "T_ASSETS", "RECEIVABLE_CLAIMS"]]


def lag_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame[["SECURITY_ID", "QUARTER_INDEX", *columns]].copy()
    out["QUARTER_INDEX"] += 4
    return out.rename(columns={c: f"L4_{c}" for c in columns})


def factor_events(pit_dir: Path) -> pd.DataFrame:
    raw = pd.read_parquet(
        pit_dir / "new_pit_cashflow",
        columns=COMMON_COLUMNS + ["C_FR_SALE_G_S"], engine="pyarrow",
    )
    raw["C_FR_SALE_G_S"] = pd.to_numeric(raw["C_FR_SALE_G_S"], errors="coerce")
    standalone = build_standalone_quarterly_metric(raw, "C_FR_SALE_G_S", name="cashflow PIT")
    ttm = _merge_flow_tables({"C_FR_SALE_G_S": standalone}, ["C_FR_SALE_G_S"], ttm=True)
    balance = balance_events(pit_dir)
    b = balance.merge(
        lag_table(balance, ["ACT_PUBTIME", "T_ASSETS", "RECEIVABLE_CLAIMS"]),
        on=["SECURITY_ID", "QUARTER_INDEX"], how="inner", validate="one_to_one",
    )
    data = ttm.merge(b, on=["SECURITY_ID", "QUARTER_INDEX"], how="inner", validate="one_to_one")
    prior_ttm = lag_table(ttm, ["FLOW_EVENT_TIME", "TTM_C_FR_SALE_G_S"])
    data = data.merge(prior_ttm, on=["SECURITY_ID", "QUARTER_INDEX"], how="left", validate="one_to_one")
    avg_assets = 0.5 * (data["T_ASSETS"] + data["L4_T_ASSETS"])
    avg_claims = 0.5 * (data["RECEIVABLE_CLAIMS"] + data["L4_RECEIVABLE_CLAIMS"])
    claim_change = data["RECEIVABLE_CLAIMS"] - data["L4_RECEIVABLE_CLAIMS"]
    cash_change = data["TTM_C_FR_SALE_G_S"] - data["L4_TTM_C_FR_SALE_G_S"]
    values = {
        FACTORS[0]: ratio(data["TTM_C_FR_SALE_G_S"], avg_assets),
        FACTORS[1]: ratio(data["TTM_C_FR_SALE_G_S"], avg_claims),
        FACTORS[2]: -ratio(data["RECEIVABLE_CLAIMS"], data["T_ASSETS"]),
        FACTORS[3]: -ratio(claim_change, data["L4_T_ASSETS"]),
        FACTORS[4]: ratio(cash_change - claim_change, data["L4_T_ASSETS"]),
    }
    event_time = pd.concat([
        pd.to_datetime(data["FLOW_EVENT_TIME"]), pd.to_datetime(data["ACT_PUBTIME"]),
        pd.to_datetime(data["L4_ACT_PUBTIME"]), pd.to_datetime(data["L4_FLOW_EVENT_TIME"]),
    ], axis=1).max(axis=1)
    pieces = []
    for factor, value in values.items():
        out = data[["SECURITY_ID", "QUARTER_INDEX"]].copy()
        out["EVENT_TIME"] = event_time
        out["factor"] = factor
        out["value"] = pd.to_numeric(value, errors="coerce").clip(-50, 50)
        pieces.append(out.dropna(subset=["EVENT_TIME", "value"]))
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--pit-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--coverage", type=Path, required=True)
    args = p.parse_args()
    dates = pd.read_parquet(args.panel, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    raw = factor_events(args.pit_dir)
    pieces = []
    for factor, group in raw.groupby("factor", sort=False):
        a = assign_available_trade_date(group, calendar).sort_values(["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "QUARTER_INDEX"])
        newest = a.groupby("SECURITY_ID", sort=False)["QUARTER_INDEX"].cummax()
        a = a.loc[a["QUARTER_INDEX"].eq(newest)].drop_duplicates(["SECURITY_ID", "AVAILABLE_DATE"], keep="last")
        pieces.append(a[["SECURITY_ID", "AVAILABLE_DATE", "factor", "value"]])
    wide = pd.concat(pieces).pivot_table(index=["SECURITY_ID", "AVAILABLE_DATE"], columns="factor", values="value", aggfunc="last").reset_index().sort_values(["SECURITY_ID", "AVAILABLE_DATE"])
    wide[FACTORS] = wide.groupby("SECURITY_ID", sort=False)[FACTORS].ffill()
    chunks, coverage = [], []
    for year in sorted(set(calendar.year)):
        panel = _normalize_panel(pd.read_parquet(args.panel, columns=KEYS, filters=[("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)), ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31))]))
        mapped = pd.merge_asof(panel.sort_values(["TRADE_DATE", "SECURITY_ID"]), wide.sort_values(["AVAILABLE_DATE", "SECURITY_ID"]), by="SECURITY_ID", left_on="TRADE_DATE", right_on="AVAILABLE_DATE", direction="backward")[KEYS + FACTORS]
        mapped[FACTORS] = mapped[FACTORS].astype("float32")
        chunks.append(mapped)
        coverage.extend({"year": year, "factor": f, "missing_rate": float(mapped[f].isna().mean())} for f in FACTORS)
        print(year, f"rows={len(mapped):,}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(chunks).sort_values(KEYS).to_parquet(args.output, index=False, compression="zstd")
    pd.DataFrame(coverage).to_csv(args.coverage, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
