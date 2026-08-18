"""Round 64: unsupervised cross-accounting relationship anomalies.

Quarter/industry cross-sectional accounting regressions only; no return label.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
import eighth_round_literature_factors as workflow
from dense_literature_profitability_factors import _merge_flow_tables
from event_financial_factor_search import KEYS,_normalize_panel
from fifteenth_round_temporal_financial_factors import _read_inputs
from quarterly_f_score import COMMON_COLUMNS,build_standalone_quarterly_metric

FACTORS=["r64_low_ar_relation_anomaly","r64_low_inventory_relation_anomaly","r64_positive_cfo_relation_residual","r64_low_joint_accounting_anomaly"]
def ratio(a,b):
 b=pd.to_numeric(b,errors="coerce"); return pd.to_numeric(a,errors="coerce").div(b.abs().where(b.abs().gt(1e-12))).replace([np.inf,-np.inf],np.nan)
def residual(y,x,groups,min_n=30):
 out=pd.Series(np.nan,index=y.index,dtype=float)
 frame=pd.concat([y.rename("y"),x],axis=1)
 for _,idx in frame.groupby(groups,sort=False).groups.items():
  g=frame.loc[idx].dropna();
  if len(g)<min_n: continue
  X=np.column_stack([np.ones(len(g)),g[x.columns].to_numpy(float)]); yy=g.y.to_numpy(float); beta,*_=np.linalg.lstsq(X,yy,rcond=None); out.loc[g.index]=yy-X@beta
 return out
def robust_z(v,groups):
 c=v.groupby(groups,sort=False).transform("median"); mad=(v-c).abs().groupby(groups,sort=False).transform("median"); return ((v-c)/(1.4826*mad).where(mad.gt(1e-12))).replace([np.inf,-np.inf],np.nan).clip(-8,8)
def events(pit):
 flows,bal=_read_inputs(pit); inc=pd.read_parquet(pit/"new_pit_income",columns=COMMON_COLUMNS+["N_INCOME_ATTR_P","REVENUE","COGS"],engine="pyarrow")
 for f in ["N_INCOME_ATTR_P","REVENUE","COGS"]: flows[f]=build_standalone_quarterly_metric(inc,f,name="income PIT")
 q=_merge_flow_tables(flows,["N_INCOME_ATTR_P","REVENUE","COGS","N_CF_OPERATE_A"],ttm=False); d=bal.merge(q,on=["SECURITY_ID","QUARTER_INDEX"],how="inner",validate="one_to_one").sort_values(["SECURITY_ID","QUARTER_INDEX"])
 lag=d[["SECURITY_ID","QUARTER_INDEX","T_ASSETS","AR","INVENTORIES"]].copy(); lag.QUARTER_INDEX+=4; lag=lag.rename(columns={"T_ASSETS":"L4_ASSETS","AR":"L4_AR","INVENTORIES":"L4_INV"}); d=d.merge(lag,on=["SECURITY_ID","QUARTER_INDEX"],how="inner",validate="one_to_one")
 g=[d.QUARTER_INDEX,d.INDUSTRY_CATEGORY.astype("string").fillna("UNKNOWN")]; sales_growth=ratio(d.REVENUE-d.groupby("SECURITY_ID",sort=False).REVENUE.shift(4),d.L4_ASSETS); cogs_growth=ratio(d.COGS-d.groupby("SECURITY_ID",sort=False).COGS.shift(4),d.L4_ASSETS)
 ar_change=ratio(d.AR-d.L4_AR,d.L4_ASSETS); inv_change=ratio(d.INVENTORIES-d.L4_INV,d.L4_ASSETS); cfoa=ratio(d.N_CF_OPERATE_A,d.L4_ASSETS); nia=ratio(d.N_INCOME_ATTR_P,d.L4_ASSETS); reva=ratio(d.REVENUE,d.L4_ASSETS)
 ar_res=residual(ar_change,pd.DataFrame({"sales_growth":sales_growth,"reva":reva}),g); inv_res=residual(inv_change,pd.DataFrame({"cogs_growth":cogs_growth,"reva":reva}),g); cfo_res=residual(cfoa,pd.DataFrame({"nia":nia,"reva":reva}),g)
 za=robust_z(ar_res,g); zi=robust_z(inv_res,g); zc=robust_z(cfo_res,g); comp=pd.concat([za,zi,zc],axis=1)
 raw={"r64_low_ar_relation_anomaly":-za.abs(),"r64_low_inventory_relation_anomaly":-zi.abs(),"r64_positive_cfo_relation_residual":zc,"r64_low_joint_accounting_anomaly":-np.sqrt(comp.pow(2).mean(axis=1,skipna=False))}; time=pd.concat([pd.to_datetime(d.BALANCE_EVENT_TIME),pd.to_datetime(d.FLOW_EVENT_TIME)],axis=1).max(axis=1); out=[]
 for f,v in raw.items():
  x=d[["SECURITY_ID","QUARTER_INDEX"]].copy(); x["EVENT_TIME"]=time; x["factor"]=f; x["value"]=pd.to_numeric(v,errors="coerce"); out.append(x.dropna(subset=["EVENT_TIME","value"]))
 return pd.concat(out,ignore_index=True)
def generate(panel,pit,out):
 dates=pd.read_parquet(panel,columns=["TRADE_DATE"]).TRADE_DATE; cal=pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values(); workflow.CANDIDATE_COLUMNS=FACTORS; wide=workflow.prepare_wide_events(events(pit),cal); chunks=[];cov=[]
 for y in sorted(set(cal.year)):
  filt=[("TRADE_DATE",">=",pd.Timestamp(y,1,1)),("TRADE_DATE","<=",pd.Timestamp(y,12,31))]; keys=_normalize_panel(pd.read_parquet(panel,columns=KEYS,filters=filt)); m=workflow._map_sparse(keys,wide);chunks.append(m);cov.extend({"year":y,"factor":f,"missing_rate":float(m[f].isna().mean())} for f in FACTORS);print(y,len(m),flush=True)
 out.mkdir(parents=True,exist_ok=True);pd.concat(chunks).sort_values(KEYS).to_parquet(out/"round64_sparse_before_fill.parquet",index=False,compression="zstd");pd.DataFrame(cov).to_csv(out/"round64_coverage.csv",index=False,encoding="utf-8-sig")
def fill(sparse,ic,output):
 assert set(FACTORS)<=set(pd.read_csv(ic).factor.astype(str));d=pd.read_parquet(sparse,columns=KEYS+FACTORS);before=d[FACTORS].isna().mean();med=d.groupby("TRADE_DATE",sort=False)[FACTORS].transform("median");d[FACTORS]=d[FACTORS].fillna(med);whole=d[FACTORS].isna().sum();d[FACTORS]=d[FACTORS].fillna(0);d.to_parquet(output,index=False,compression="zstd");pd.DataFrame({"factor":FACTORS,"missing_rate_before_fill":[before[f] for f in FACTORS],"whole_day_neutral_rows":[whole[f] for f in FACTORS]}).to_csv(output.with_suffix(".fill_report.csv"),index=False,encoding="utf-8-sig")
def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest="cmd",required=True);g=s.add_parser("generate-sparse");g.add_argument("--panel",type=Path,required=True);g.add_argument("--pit-dir",type=Path,required=True);g.add_argument("--output-dir",type=Path,required=True);f=s.add_parser("fill-after-test");f.add_argument("--sparse",type=Path,required=True);f.add_argument("--sparse-ic",type=Path,required=True);f.add_argument("--output",type=Path,required=True);a=p.parse_args();generate(a.panel.resolve(),a.pit_dir.resolve(),a.output_dir.resolve()) if a.cmd=="generate-sparse" else fill(a.sparse.resolve(),a.sparse_ic.resolve(),a.output.resolve())
if __name__=="__main__":main()
