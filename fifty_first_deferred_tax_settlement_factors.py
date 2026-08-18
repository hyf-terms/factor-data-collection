"""Round 51: PIT deferred-tax conservatism and cash-tax settlement factors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dense_literature_profitability_factors import _merge_flow_tables
from event_financial_factor_search import KEYS, _normalize_panel
from pead_sue_factor import assign_available_trade_date
from quarterly_f_score import COMMON_COLUMNS, FINANCIAL_INDUSTRIES, REPORT_QUARTERS, REPORT_TYPES, build_standalone_quarterly_metric


FACTORS = [
    "r51_low_net_deferred_tax_asset",
    "r51_net_deferred_tax_asset_yoy_reduction",
    "r51_low_tax_payable_assets",
    "r51_cash_tax_settlement_coverage",
    "r51_tax_balance_sheet_conservatism",
]
FIELDS = ["INDUSTRY_CATEGORY", "T_ASSETS", "DEFER_TAX_ASSETS", "DEFER_TAX_LIAB", "TAXES_PAYABLE"]


def ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    den = pd.to_numeric(b, errors="coerce")
    floor = max(float(den.abs().median(skipna=True)) * 1e-8, 1e-12)
    return pd.to_numeric(a, errors="coerce").div(den.where(den.abs().gt(floor))).replace([np.inf, -np.inf], np.nan)


def balance_events(pit_dir: Path) -> pd.DataFrame:
    d = pd.read_parquet(pit_dir / "new_pit_balance", columns=COMMON_COLUMNS + FIELDS, engine="pyarrow")
    for c in ["ACT_PUBTIME", "END_DATE", "END_DATE_REP"]: d[c] = pd.to_datetime(d[c], errors="coerce")
    d["END_DATE"] = d["END_DATE"].dt.normalize(); d["END_DATE_REP"] = d["END_DATE_REP"].dt.normalize()
    for c in FIELDS[1:]: d[c] = pd.to_numeric(d[c], errors="coerce")
    # Tax lines absent from an otherwise available statement are zero balances.
    d[["DEFER_TAX_ASSETS", "DEFER_TAX_LIAB", "TAXES_PAYABLE"]] = d[["DEFER_TAX_ASSETS", "DEFER_TAX_LIAB", "TAXES_PAYABLE"]].fillna(0.0)
    mask = (d["MERGED_FLAG"].astype("string").eq("1") & d["REPORT_TYPE"].astype("string").isin(REPORT_TYPES)
            & d["IS_CURRENT_PERIOD"].fillna(False).astype(bool) & d["END_DATE"].eq(d["END_DATE_REP"])
            & ~d["INDUSTRY_CATEGORY"].isin(FINANCIAL_INDUSTRIES))
    d = d.loc[mask].dropna(subset=["SECURITY_ID", "ACT_PUBTIME", "END_DATE", "T_ASSETS"])
    d["SECURITY_ID"] = pd.to_numeric(d["SECURITY_ID"]).astype("int64")
    d["QUARTER_INDEX"] = d["END_DATE"].dt.year.astype("int64") * 4 + d["REPORT_TYPE"].map(REPORT_QUARTERS).astype("int8")
    d = d.sort_values(["SECURITY_ID", "QUARTER_INDEX", "ACT_PUBTIME", "ID"]).drop_duplicates(["SECURITY_ID", "QUARTER_INDEX"], keep="first")
    d["NET_DTA"] = d["DEFER_TAX_ASSETS"] - d["DEFER_TAX_LIAB"]
    return d[["SECURITY_ID", "QUARTER_INDEX", "ACT_PUBTIME", "T_ASSETS", "NET_DTA", "TAXES_PAYABLE"]]


def lag4(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    x = frame[["SECURITY_ID", "QUARTER_INDEX", *columns]].copy(); x["QUARTER_INDEX"] += 4
    return x.rename(columns={c: f"L4_{c}" for c in columns})


def events(pit_dir: Path) -> pd.DataFrame:
    bal = balance_events(pit_dir)
    data = bal.merge(lag4(bal, ["ACT_PUBTIME", "T_ASSETS", "NET_DTA", "TAXES_PAYABLE"]), on=["SECURITY_ID", "QUARTER_INDEX"], how="inner", validate="one_to_one")
    raw = pd.read_parquet(pit_dir / "new_pit_cashflow", columns=COMMON_COLUMNS + ["C_PAID_FOR_TAXES"], engine="pyarrow")
    raw["C_PAID_FOR_TAXES"] = pd.to_numeric(raw["C_PAID_FOR_TAXES"], errors="coerce")
    q = build_standalone_quarterly_metric(raw, "C_PAID_FOR_TAXES", name="cashflow PIT")
    ttm = _merge_flow_tables({"C_PAID_FOR_TAXES": q}, ["C_PAID_FOR_TAXES"], ttm=True)
    data = data.merge(ttm, on=["SECURITY_ID", "QUARTER_INDEX"], how="left", validate="one_to_one")
    avg_tax_payable = 0.5 * (data["TAXES_PAYABLE"] + data["L4_TAXES_PAYABLE"])
    low_dta = -ratio(data["NET_DTA"], data["T_ASSETS"])
    dta_reduction = -ratio(data["NET_DTA"] - data["L4_NET_DTA"], data["L4_T_ASSETS"])
    low_tax_payable = -ratio(data["TAXES_PAYABLE"], data["T_ASSETS"])
    settlement = ratio(data["TTM_C_PAID_FOR_TAXES"], avg_tax_payable.abs() + 0.005 * data["L4_T_ASSETS"].abs())
    # Fixed equal economic components after dimensionless scaling by assets.
    conservatism = 0.5 * low_dta + 0.5 * dta_reduction
    values = dict(zip(FACTORS, [low_dta, dta_reduction, low_tax_payable, settlement, conservatism]))
    base_time = pd.concat([pd.to_datetime(data["ACT_PUBTIME"]), pd.to_datetime(data["L4_ACT_PUBTIME"]), pd.to_datetime(data["FLOW_EVENT_TIME"])], axis=1).max(axis=1)
    pieces=[]
    for f,v in values.items():
        o=data[["SECURITY_ID","QUARTER_INDEX"]].copy(); o["EVENT_TIME"]=base_time; o["factor"]=f; o["value"]=pd.to_numeric(v,errors="coerce").clip(-50,50)
        pieces.append(o.dropna(subset=["EVENT_TIME","value"]))
    return pd.concat(pieces,ignore_index=True)


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--panel",type=Path,required=True); p.add_argument("--pit-dir",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--coverage",type=Path,required=True); a=p.parse_args()
    dates=pd.read_parquet(a.panel,columns=["TRADE_DATE"])["TRADE_DATE"]; cal=pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values(); raw=events(a.pit_dir); pieces=[]
    for f,g in raw.groupby("factor",sort=False):
        x=assign_available_trade_date(g,cal).sort_values(["SECURITY_ID","AVAILABLE_DATE","EVENT_TIME","QUARTER_INDEX"]); newest=x.groupby("SECURITY_ID",sort=False)["QUARTER_INDEX"].cummax(); x=x.loc[x["QUARTER_INDEX"].eq(newest)].drop_duplicates(["SECURITY_ID","AVAILABLE_DATE"],keep="last"); pieces.append(x[["SECURITY_ID","AVAILABLE_DATE","factor","value"]])
    wide=pd.concat(pieces).pivot_table(index=["SECURITY_ID","AVAILABLE_DATE"],columns="factor",values="value",aggfunc="last").reset_index().sort_values(["SECURITY_ID","AVAILABLE_DATE"]); wide[FACTORS]=wide.groupby("SECURITY_ID",sort=False)[FACTORS].ffill(); chunks=[]; coverage=[]
    for y in sorted(set(cal.year)):
        panel=_normalize_panel(pd.read_parquet(a.panel,columns=KEYS,filters=[("TRADE_DATE",">=",pd.Timestamp(y,1,1)),("TRADE_DATE","<=",pd.Timestamp(y,12,31))])); mapped=pd.merge_asof(panel.sort_values(["TRADE_DATE","SECURITY_ID"]),wide.sort_values(["AVAILABLE_DATE","SECURITY_ID"]),by="SECURITY_ID",left_on="TRADE_DATE",right_on="AVAILABLE_DATE",direction="backward")[KEYS+FACTORS]; mapped[FACTORS]=mapped[FACTORS].astype("float32"); chunks.append(mapped); coverage.extend({"year":y,"factor":f,"missing_rate":float(mapped[f].isna().mean())} for f in FACTORS); print(y,f"rows={len(mapped):,}",flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True); pd.concat(chunks).sort_values(KEYS).to_parquet(a.output,index=False,compression="zstd"); pd.DataFrame(coverage).to_csv(a.coverage,index=False,encoding="utf-8-sig")


if __name__ == "__main__": main()
