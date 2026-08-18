"""Round 52: supplier finance, provisions, and capex-pipeline realization.

All candidates are PIT accounting structures and exclude profit, margins and
earnings surprises.  Missing optional balance-sheet lines in an otherwise
available statement are interpreted as zero balances.  Capital commitment is
not available locally; the capex-pipeline formulas are explicitly proxies
based on construction-in-progress and cash capex, not mislabeled commitments.
"""

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
    "r52_supplier_finance_assets",
    "r52_supplier_finance_yoy_change",
    "r52_supplier_finance_mix",
    "r52_low_estimated_liability_assets",
    "r52_estimated_liability_yoy_reduction",
    "r52_capex_pipeline_realization",
    "r52_cip_yoy_reduction_assets",
]
BALANCE_FIELDS = [
    "INDUSTRY_CATEGORY", "T_ASSETS", "AP", "NOTES_PAYABLE",
    "OTH_PAYABLE_TOTAL", "ESTIMATED_LIAB", "CIP_TOTAL", "FIXED_ASSETS_TOTAL",
]


def ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    den = pd.to_numeric(b, errors="coerce")
    floor = max(float(den.abs().median(skipna=True)) * 1e-8, 1e-12)
    return pd.to_numeric(a, errors="coerce").div(den.where(den.abs().gt(floor))).replace([np.inf, -np.inf], np.nan)


def balance_events(pit_dir: Path) -> pd.DataFrame:
    d = pd.read_parquet(pit_dir / "new_pit_balance", columns=COMMON_COLUMNS + BALANCE_FIELDS, engine="pyarrow")
    for c in ["ACT_PUBTIME", "END_DATE", "END_DATE_REP"]: d[c] = pd.to_datetime(d[c], errors="coerce")
    d["END_DATE"] = d["END_DATE"].dt.normalize(); d["END_DATE_REP"] = d["END_DATE_REP"].dt.normalize()
    for c in BALANCE_FIELDS[1:]: d[c] = pd.to_numeric(d[c], errors="coerce")
    optional = ["NOTES_PAYABLE", "OTH_PAYABLE_TOTAL", "ESTIMATED_LIAB", "CIP_TOTAL", "FIXED_ASSETS_TOTAL"]
    d[optional] = d[optional].fillna(0.0)
    mask = (d["MERGED_FLAG"].astype("string").eq("1") & d["REPORT_TYPE"].astype("string").isin(REPORT_TYPES)
            & d["IS_CURRENT_PERIOD"].fillna(False).astype(bool) & d["END_DATE"].eq(d["END_DATE_REP"])
            & ~d["INDUSTRY_CATEGORY"].isin(FINANCIAL_INDUSTRIES))
    d = d.loc[mask].dropna(subset=["SECURITY_ID", "ACT_PUBTIME", "END_DATE", "T_ASSETS", "AP"])
    d["SECURITY_ID"] = pd.to_numeric(d["SECURITY_ID"]).astype("int64")
    d["QUARTER_INDEX"] = d["END_DATE"].dt.year.astype("int64") * 4 + d["REPORT_TYPE"].map(REPORT_QUARTERS).astype("int8")
    d = d.sort_values(["SECURITY_ID", "QUARTER_INDEX", "ACT_PUBTIME", "ID"]).drop_duplicates(["SECURITY_ID", "QUARTER_INDEX"], keep="first")
    d["SUPPLIER_FINANCE"] = d["AP"] + d["NOTES_PAYABLE"] + d["OTH_PAYABLE_TOTAL"]
    return d[["SECURITY_ID", "QUARTER_INDEX", "ACT_PUBTIME", "T_ASSETS", "AP", "NOTES_PAYABLE", "SUPPLIER_FINANCE", "ESTIMATED_LIAB", "CIP_TOTAL", "FIXED_ASSETS_TOTAL"]]


def lag4(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x=d[["SECURITY_ID","QUARTER_INDEX",*cols]].copy(); x["QUARTER_INDEX"]+=4
    return x.rename(columns={c:f"L4_{c}" for c in cols})


def events(pit_dir: Path) -> pd.DataFrame:
    bal=balance_events(pit_dir)
    cols=["ACT_PUBTIME","T_ASSETS","AP","NOTES_PAYABLE","SUPPLIER_FINANCE","ESTIMATED_LIAB","CIP_TOTAL","FIXED_ASSETS_TOTAL"]
    d=bal.merge(lag4(bal,cols),on=["SECURITY_ID","QUARTER_INDEX"],how="inner",validate="one_to_one")
    raw=pd.read_parquet(pit_dir/"new_pit_cashflow",columns=COMMON_COLUMNS+["PUR_FIX_ASSETS_OTH"],engine="pyarrow")
    raw["PUR_FIX_ASSETS_OTH"]=pd.to_numeric(raw["PUR_FIX_ASSETS_OTH"],errors="coerce")
    q=build_standalone_quarterly_metric(raw,"PUR_FIX_ASSETS_OTH",name="cashflow PIT")
    ttm=_merge_flow_tables({"PUR_FIX_ASSETS_OTH":q},["PUR_FIX_ASSETS_OTH"],ttm=True)
    d=d.merge(ttm,on=["SECURITY_ID","QUARTER_INDEX"],how="left",validate="one_to_one")
    sf=d["SUPPLIER_FINANCE"]; lsf=d["L4_SUPPLIER_FINANCE"]
    # Notes-payable intensity within core trade credit. Other payables are
    # excluded from the mix denominator because their trade nature varies.
    trade_core=d["AP"]+d["NOTES_PAYABLE"]
    cip_release=d["L4_CIP_TOTAL"]-d["CIP_TOTAL"]
    fixed_addition=(d["FIXED_ASSETS_TOTAL"]-d["L4_FIXED_ASSETS_TOTAL"]).clip(lower=0)
    realization=ratio(cip_release.clip(lower=0)+fixed_addition,d["L4_CIP_TOTAL"].abs()+0.01*d["L4_T_ASSETS"].abs())
    values={
        FACTORS[0]:ratio(sf,d["T_ASSETS"]),
        FACTORS[1]:ratio(sf-lsf,d["L4_T_ASSETS"]),
        FACTORS[2]:ratio(d["NOTES_PAYABLE"],trade_core),
        FACTORS[3]:-ratio(d["ESTIMATED_LIAB"],d["T_ASSETS"]),
        FACTORS[4]:-ratio(d["ESTIMATED_LIAB"]-d["L4_ESTIMATED_LIAB"],d["L4_T_ASSETS"]),
        FACTORS[5]:realization,
        FACTORS[6]:ratio(cip_release,d["L4_T_ASSETS"]),
    }
    times=pd.concat([pd.to_datetime(d["ACT_PUBTIME"]),pd.to_datetime(d["L4_ACT_PUBTIME"]),pd.to_datetime(d["FLOW_EVENT_TIME"])],axis=1).max(axis=1)
    pieces=[]
    for f,v in values.items():
        o=d[["SECURITY_ID","QUARTER_INDEX"]].copy(); o["EVENT_TIME"]=times; o["factor"]=f; o["value"]=pd.to_numeric(v,errors="coerce").clip(-50,50); pieces.append(o.dropna(subset=["EVENT_TIME","value"]))
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


if __name__=="__main__": main()
