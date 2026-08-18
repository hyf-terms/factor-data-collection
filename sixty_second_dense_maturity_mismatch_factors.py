"""Round 62: operating-cycle versus short-maturity funding mismatch."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import eighth_round_literature_factors as workflow
from dense_literature_profitability_factors import _merge_flow_tables
from event_financial_factor_search import KEYS,_normalize_panel
from fifteenth_round_temporal_financial_factors import _read_inputs
from quarterly_f_score import build_standalone_quarterly_metric,COMMON_COLUMNS

FACTORS=["r62_low_operating_cycle_funding_gap","r62_liquid_short_debt_coverage","r62_low_cycle_debt_interaction","r62_maturity_quality_equal3"]
def ratio(a,b):
 b=pd.to_numeric(b,errors="coerce"); return pd.to_numeric(a,errors="coerce").div(b.abs().where(b.abs().gt(1e-12))).replace([np.inf,-np.inf],np.nan)
def events(pit):
 flows,bal=_read_inputs(pit)
 income=pd.read_parquet(pit/"new_pit_income",columns=COMMON_COLUMNS+["REVENUE","COGS"],engine="pyarrow")
 for f in ["REVENUE","COGS"]: flows[f]=build_standalone_quarterly_metric(income,f,name="income PIT")
 ttm=_merge_flow_tables(flows,["REVENUE","COGS","N_CF_OPERATE_A"],ttm=True)
 d=bal.merge(ttm,on=["SECURITY_ID","QUARTER_INDEX"],how="inner",validate="one_to_one")
 ap_field=None
 # AP is absent from the compact round-15 balance builder; read it consistently and attach by quarter.
 raw=pd.read_parquet(pit/"new_pit_balance",columns=COMMON_COLUMNS+["AP"],engine="pyarrow"); raw["AP"]=pd.to_numeric(raw.AP,errors="coerce").fillna(0.0)
 from tenth_round_misstatement_factors import build_balance_events
 # build_balance_events does not retain AP, so deduplicate the PIT rows directly.
 for c in ["ACT_PUBTIME","END_DATE","END_DATE_REP"]: raw[c]=pd.to_datetime(raw[c],errors="coerce")
 raw["END_DATE"]=raw.END_DATE.dt.normalize(); raw["END_DATE_REP"]=raw.END_DATE_REP.dt.normalize()
 raw=raw.loc[raw.MERGED_FLAG.astype("string").eq("1") & raw.IS_CURRENT_PERIOD.fillna(False).astype(bool) & raw.END_DATE.eq(raw.END_DATE_REP)].dropna(subset=["SECURITY_ID","ACT_PUBTIME","END_DATE"])
 from quarterly_f_score import REPORT_QUARTERS
 raw["QUARTER_INDEX"]=raw.END_DATE.dt.year.astype("int64")*4+raw.REPORT_TYPE.map(REPORT_QUARTERS).astype("int64"); raw["SECURITY_ID"]=pd.to_numeric(raw.SECURITY_ID).astype("int64")
 raw=raw.sort_values(["SECURITY_ID","QUARTER_INDEX","ACT_PUBTIME","ID"]).drop_duplicates(["SECURITY_ID","QUARTER_INDEX"],keep="first")[["SECURITY_ID","QUARTER_INDEX","AP","ACT_PUBTIME"]].rename(columns={"ACT_PUBTIME":"AP_TIME"})
 d=d.merge(raw,on=["SECURITY_ID","QUARTER_INDEX"],how="left",validate="one_to_one")
 assets=d.T_ASSETS; short=pd.to_numeric(d.ST_BORR,errors="coerce").fillna(0)+pd.to_numeric(d.NCL_WITHIN_1_Y,errors="coerce").fillna(0); cash=pd.to_numeric(d.CASH_C_EQUIV,errors="coerce").fillna(0); cfo=d.TTM_N_CF_OPERATE_A
 operating=pd.to_numeric(d.AR,errors="coerce").fillna(0)+pd.to_numeric(d.INVENTORIES,errors="coerce").fillna(0)-pd.to_numeric(d.AP,errors="coerce").fillna(0)
 gap=-ratio(operating.clip(lower=0)+short-cash-cfo,assets)
 coverage=ratio(cash+cfo+0.8*pd.to_numeric(d.AR,errors="coerce").fillna(0)+0.5*pd.to_numeric(d.INVENTORIES,errors="coerce").fillna(0),short+0.01*assets)
 cycle=ratio(pd.to_numeric(d.AR,errors="coerce").fillna(0),d.TTM_REVENUE)+ratio(pd.to_numeric(d.INVENTORIES,errors="coerce").fillna(0)-pd.to_numeric(d.AP,errors="coerce").fillna(0),d.TTM_COGS)
 interaction=-(cycle.clip(-5,5)*ratio(short,assets)).clip(-20,20)
 rawv={"r62_low_operating_cycle_funding_gap":gap,"r62_liquid_short_debt_coverage":coverage,"r62_low_cycle_debt_interaction":interaction}; q=d.QUARTER_INDEX; zs=[]
 for v in rawv.values():
  c=v.groupby(q,sort=False).transform("median"); mad=(v-c).abs().groupby(q,sort=False).transform("median"); zs.append(((v-c)/(1.4826*mad).where(mad.gt(1e-12))).clip(-8,8))
 rawv["r62_maturity_quality_equal3"]=pd.concat(zs,axis=1).mean(axis=1,skipna=False)
 time=pd.concat([pd.to_datetime(d.BALANCE_EVENT_TIME),pd.to_datetime(d.FLOW_EVENT_TIME),pd.to_datetime(d.AP_TIME)],axis=1).max(axis=1); out=[]
 for f,v in rawv.items():
  x=d[["SECURITY_ID","QUARTER_INDEX"]].copy(); x["EVENT_TIME"]=time; x["factor"]=f; x["value"]=pd.to_numeric(v,errors="coerce").clip(-50,50); out.append(x.dropna(subset=["EVENT_TIME","value"]))
 return pd.concat(out,ignore_index=True)
def generate(panel,pit,out):
 dates=pd.read_parquet(panel,columns=["TRADE_DATE"]).TRADE_DATE; cal=pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values(); workflow.CANDIDATE_COLUMNS=FACTORS; wide=workflow.prepare_wide_events(events(pit),cal); chunks=[]; cov=[]
 for y in sorted(set(cal.year)):
  filt=[("TRADE_DATE",">=",pd.Timestamp(y,1,1)),("TRADE_DATE","<=",pd.Timestamp(y,12,31))]; keys=_normalize_panel(pd.read_parquet(panel,columns=KEYS,filters=filt)); m=workflow._map_sparse(keys,wide); chunks.append(m); cov.extend({"year":y,"factor":f,"missing_rate":float(m[f].isna().mean())} for f in FACTORS); print(y,len(m),flush=True)
 out.mkdir(parents=True,exist_ok=True); pd.concat(chunks).sort_values(KEYS).to_parquet(out/"round62_sparse_before_fill.parquet",index=False,compression="zstd"); pd.DataFrame(cov).to_csv(out/"round62_coverage.csv",index=False,encoding="utf-8-sig")
def fill(sparse,ic,output):
 assert set(FACTORS)<=set(pd.read_csv(ic).factor.astype(str)); d=pd.read_parquet(sparse,columns=KEYS+FACTORS); before=d[FACTORS].isna().mean(); med=d.groupby("TRADE_DATE",sort=False)[FACTORS].transform("median"); d[FACTORS]=d[FACTORS].fillna(med); whole=d[FACTORS].isna().sum(); d[FACTORS]=d[FACTORS].fillna(0); d.to_parquet(output,index=False,compression="zstd"); pd.DataFrame({"factor":FACTORS,"missing_rate_before_fill":[before[f] for f in FACTORS],"whole_day_neutral_rows":[whole[f] for f in FACTORS]}).to_csv(output.with_suffix(".fill_report.csv"),index=False,encoding="utf-8-sig")
def main():
 p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd",required=True); g=s.add_parser("generate-sparse"); g.add_argument("--panel",type=Path,required=True); g.add_argument("--pit-dir",type=Path,required=True); g.add_argument("--output-dir",type=Path,required=True); f=s.add_parser("fill-after-test"); f.add_argument("--sparse",type=Path,required=True); f.add_argument("--sparse-ic",type=Path,required=True); f.add_argument("--output",type=Path,required=True); a=p.parse_args(); generate(a.panel.resolve(),a.pit_dir.resolve(),a.output_dir.resolve()) if a.cmd=="generate-sparse" else fill(a.sparse.resolve(),a.sparse_ic.resolve(),a.output.resolve())
if __name__=="__main__": main()
