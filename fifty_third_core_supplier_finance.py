"""Round 53: focused PIT core supplier-finance structures.

Core trade credit is accounts payable plus notes payable.  The formulas avoid
other payables, whose supplier-finance interpretation is ambiguous.  Cash paid
for goods is used only as a scale for a financing-intensity specification.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dense_literature_profitability_factors import _merge_flow_tables
from event_financial_factor_search import KEYS, _normalize_panel
from fifty_second_liability_commitment_structure import balance_events, lag4, ratio
from pead_sue_factor import assign_available_trade_date
from quarterly_f_score import COMMON_COLUMNS, build_standalone_quarterly_metric


FACTORS=[
    "r53_core_trade_credit_yoy_change",
    "r53_notes_payable_yoy_change",
    "r53_accounts_payable_yoy_change",
    "r53_notes_substitution_for_ap",
    "r53_trade_credit_change_to_cash_purchases",
]


def events(pit_dir:Path)->pd.DataFrame:
    b=balance_events(pit_dir)
    d=b.merge(lag4(b,["ACT_PUBTIME","T_ASSETS","AP","NOTES_PAYABLE"]),on=["SECURITY_ID","QUARTER_INDEX"],how="inner",validate="one_to_one")
    raw=pd.read_parquet(pit_dir/"new_pit_cashflow",columns=COMMON_COLUMNS+["C_PAID_G_S"],engine="pyarrow")
    raw["C_PAID_G_S"]=pd.to_numeric(raw["C_PAID_G_S"],errors="coerce")
    q=build_standalone_quarterly_metric(raw,"C_PAID_G_S",name="cashflow PIT")
    ttm=_merge_flow_tables({"C_PAID_G_S":q},["C_PAID_G_S"],ttm=True)
    d=d.merge(ttm,on=["SECURITY_ID","QUARTER_INDEX"],how="left",validate="one_to_one")
    dap=d["AP"]-d["L4_AP"]; dnotes=d["NOTES_PAYABLE"]-d["L4_NOTES_PAYABLE"]; dcore=dap+dnotes
    vals=[ratio(dcore,d["L4_T_ASSETS"]),ratio(dnotes,d["L4_T_ASSETS"]),ratio(dap,d["L4_T_ASSETS"]),ratio(dnotes-dap,d["L4_T_ASSETS"]),ratio(dcore,d["TTM_C_PAID_G_S"].abs()+0.01*d["L4_T_ASSETS"].abs())]
    times=pd.concat([pd.to_datetime(d["ACT_PUBTIME"]),pd.to_datetime(d["L4_ACT_PUBTIME"]),pd.to_datetime(d["FLOW_EVENT_TIME"])],axis=1).max(axis=1)
    out=[]
    for f,v in zip(FACTORS,vals):
        x=d[["SECURITY_ID","QUARTER_INDEX"]].copy(); x["EVENT_TIME"]=times; x["factor"]=f; x["value"]=pd.to_numeric(v,errors="coerce").clip(-50,50); out.append(x.dropna(subset=["EVENT_TIME","value"]))
    return pd.concat(out,ignore_index=True)


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--panel",type=Path,required=True); p.add_argument("--pit-dir",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--coverage",type=Path,required=True); a=p.parse_args()
    dates=pd.read_parquet(a.panel,columns=["TRADE_DATE"])["TRADE_DATE"]; cal=pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values(); raw=events(a.pit_dir); pieces=[]
    for f,g in raw.groupby("factor",sort=False):
        x=assign_available_trade_date(g,cal).sort_values(["SECURITY_ID","AVAILABLE_DATE","EVENT_TIME","QUARTER_INDEX"]); newest=x.groupby("SECURITY_ID",sort=False)["QUARTER_INDEX"].cummax(); x=x.loc[x["QUARTER_INDEX"].eq(newest)].drop_duplicates(["SECURITY_ID","AVAILABLE_DATE"],keep="last"); pieces.append(x[["SECURITY_ID","AVAILABLE_DATE","factor","value"]])
    wide=pd.concat(pieces).pivot_table(index=["SECURITY_ID","AVAILABLE_DATE"],columns="factor",values="value",aggfunc="last").reset_index().sort_values(["SECURITY_ID","AVAILABLE_DATE"]); wide[FACTORS]=wide.groupby("SECURITY_ID",sort=False)[FACTORS].ffill(); chunks=[]; coverage=[]
    for y in sorted(set(cal.year)):
        panel=_normalize_panel(pd.read_parquet(a.panel,columns=KEYS,filters=[("TRADE_DATE",">=",pd.Timestamp(y,1,1)),("TRADE_DATE","<=",pd.Timestamp(y,12,31))])); mapped=pd.merge_asof(panel.sort_values(["TRADE_DATE","SECURITY_ID"]),wide.sort_values(["AVAILABLE_DATE","SECURITY_ID"]),by="SECURITY_ID",left_on="TRADE_DATE",right_on="AVAILABLE_DATE",direction="backward")[KEYS+FACTORS]; mapped[FACTORS]=mapped[FACTORS].astype("float32"); chunks.append(mapped); coverage.extend({"year":y,"factor":f,"missing_rate":float(mapped[f].isna().mean())} for f in FACTORS); print(y,flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True); pd.concat(chunks).sort_values(KEYS).to_parquet(a.output,index=False,compression="zstd"); pd.DataFrame(coverage).to_csv(a.coverage,index=False,encoding="utf-8-sig")


if __name__=="__main__":main()
