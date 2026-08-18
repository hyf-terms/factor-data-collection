"""Round 54: PIT restricted-cash level and release from cash-note details."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from event_financial_factor_search import KEYS, _normalize_panel
from pead_sue_factor import assign_available_trade_date
from twenty_sixth_asset_quality_detail_factors import attach_assets, read_balance_assets
from twenty_third_alternative_event_factors import event_time, map_party, read_partitioned, security_mapping


FACTORS=[
    "r54_low_restricted_cash_share",
    "r54_low_restricted_cash_assets",
    "r54_restricted_cash_yoy_release",
    "r54_unrestricted_cash_assets",
]


def ratio(a:pd.Series,b:pd.Series)->pd.Series:
    den=pd.to_numeric(b,errors="coerce"); return pd.to_numeric(a,errors="coerce").div(den.where(den.abs().gt(1.0))).replace([np.inf,-np.inf],np.nan)


def events(alternative_root:Path,pit_root:Path)->pd.DataFrame:
    pair,_=security_mapping(pit_root); assets=read_balance_assets(pit_root)
    cols=["PARTY_ID","TICKER_SYMBOL","ACT_PUBTIME","END_DATE","END_DATE_REP","ADJUSTED_FLAG","REPORT_TYPE","MERGED_FLAG","TOTAL_MF","TOT_RES_FD"]
    d=map_party(read_partitioned(alternative_root,"fdmt_mf_item",cols),pair)
    d["EVENT_TIME"]=event_time(d,"ACT_PUBTIME"); d["END_DATE"]=pd.to_datetime(d["END_DATE"],errors="coerce").dt.normalize(); d["END_DATE_REP"]=pd.to_datetime(d["END_DATE_REP"],errors="coerce").dt.normalize()
    d["TOTAL_MF"]=pd.to_numeric(d["TOTAL_MF"],errors="coerce"); d["TOT_RES_FD"]=pd.to_numeric(d["TOT_RES_FD"],errors="coerce").fillna(0.0)
    d=d.loc[d["MERGED_FLAG"].astype(str).eq("1") & d["END_DATE"].eq(d["END_DATE_REP"])].dropna(subset=["SECURITY_ID","EVENT_TIME","END_DATE","TOTAL_MF"])
    d=d.sort_values(["SECURITY_ID","END_DATE","EVENT_TIME"]).drop_duplicates(["SECURITY_ID","END_DATE"],keep="first")
    d=attach_assets(d,assets); d["QINDEX"]=d["END_DATE"].dt.year*4+d["END_DATE"].dt.quarter
    d[FACTORS[0]]=-ratio(d["TOT_RES_FD"],d["TOTAL_MF"].abs()).clip(-2,2)
    d[FACTORS[1]]=-ratio(d["TOT_RES_FD"],d["T_ASSETS"].abs()).clip(-2,2)
    d[FACTORS[3]]=ratio(d["TOTAL_MF"]-d["TOT_RES_FD"],d["T_ASSETS"].abs()).clip(-2,2)
    lag=d[["SECURITY_ID","QINDEX","TOT_RES_FD","T_ASSETS"]].copy(); lag["QINDEX"]+=4; lag=lag.rename(columns={"TOT_RES_FD":"L4_RES","T_ASSETS":"L4_ASSETS"})
    d=d.merge(lag,on=["SECURITY_ID","QINDEX"],how="left",validate="one_to_one")
    d[FACTORS[2]]=-ratio(d["TOT_RES_FD"]-d["L4_RES"],d["L4_ASSETS"].abs()).clip(-2,2)
    return d[["SECURITY_ID","QINDEX","EVENT_TIME",*FACTORS]]


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--alternative-root",type=Path,required=True); p.add_argument("--pit-root",type=Path,required=True); p.add_argument("--panel",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--coverage",type=Path,required=True); a=p.parse_args()
    dates=pd.read_parquet(a.panel,columns=["TRADE_DATE"])["TRADE_DATE"]; cal=pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values(); ev=events(a.alternative_root,a.pit_root); pieces=[]
    for f in FACTORS:
        g=ev[["SECURITY_ID","QINDEX","EVENT_TIME",f]].dropna(subset=[f]).rename(columns={f:"value"}); x=assign_available_trade_date(g,cal).sort_values(["SECURITY_ID","AVAILABLE_DATE","EVENT_TIME","QINDEX"]); newest=x.groupby("SECURITY_ID",sort=False)["QINDEX"].cummax(); x=x.loc[x["QINDEX"].eq(newest)].drop_duplicates(["SECURITY_ID","AVAILABLE_DATE"],keep="last"); pieces.append(x[["SECURITY_ID","AVAILABLE_DATE","value"]].rename(columns={"value":f}))
    wide=pieces[0]
    for x in pieces[1:]: wide=wide.merge(x,on=["SECURITY_ID","AVAILABLE_DATE"],how="outer")
    wide=wide.sort_values(["SECURITY_ID","AVAILABLE_DATE"]); wide[FACTORS]=wide.groupby("SECURITY_ID",sort=False)[FACTORS].ffill(); chunks=[]; coverage=[]
    for y in sorted(set(cal.year)):
        panel=_normalize_panel(pd.read_parquet(a.panel,columns=KEYS,filters=[("TRADE_DATE",">=",pd.Timestamp(y,1,1)),("TRADE_DATE","<=",pd.Timestamp(y,12,31))])); mapped=pd.merge_asof(panel.sort_values(["TRADE_DATE","SECURITY_ID"]),wide.sort_values(["AVAILABLE_DATE","SECURITY_ID"]),by="SECURITY_ID",left_on="TRADE_DATE",right_on="AVAILABLE_DATE",direction="backward")[KEYS+FACTORS]; mapped[FACTORS]=mapped[FACTORS].astype("float32"); chunks.append(mapped); coverage.extend({"year":y,"factor":f,"missing_rate":float(mapped[f].isna().mean())} for f in FACTORS); print(y,flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True); pd.concat(chunks).sort_values(KEYS).to_parquet(a.output,index=False,compression="zstd"); pd.DataFrame(coverage).to_csv(a.coverage,index=False,encoding="utf-8-sig")


if __name__=="__main__":main()
