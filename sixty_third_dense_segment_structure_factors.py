"""Round 63: PIT main-business segment concentration and diversification."""
from __future__ import annotations
import argparse,glob
from pathlib import Path
import numpy as np,pandas as pd
import eighth_round_literature_factors as workflow
from event_financial_factor_search import KEYS,_normalize_panel
from pead_sue_factor import assign_available_trade_date
from twenty_third_alternative_event_factors import security_mapping

FACTORS=["r63_low_product_revenue_hhi","r63_low_region_revenue_hhi","r63_product_revenue_breadth","r63_segment_diversification_equal3"]
def events(data_root,pit_root):
 cols=["PARTY_ID","ACT_PUBTIME","END_DATE","MERGED_FLAG","CLASSIF_CD","IS_LATEST","ITEM_PARENT_NO","ITEM_NO","REVENUE_PCTGE","REVENUE_IS_PCTGE"]
 d=pd.concat([pd.read_parquet(f,columns=cols) for f in glob.glob(str(data_root/"fdmt_main_oper_n"/"*.parquet"))],ignore_index=True); pair,_=security_mapping(pit_root); d=d.merge(pair[["PARTY_ID","SECURITY_ID"]].drop_duplicates(),on="PARTY_ID",how="inner")
 d["ACT_PUBTIME"]=pd.to_datetime(d.ACT_PUBTIME,errors="coerce"); d["END_DATE"]=pd.to_datetime(d.END_DATE,errors="coerce").dt.normalize(); d=d[d.MERGED_FLAG.astype(str).isin(["1","True","true"]) & d.IS_LATEST.fillna(0).astype(int).eq(1)].dropna(subset=["ACT_PUBTIME","END_DATE","CLASSIF_CD"])
 # Top-level rows only, avoiding parent/child double counting.
 d=d[d.ITEM_PARENT_NO.isna()].copy(); pct=pd.to_numeric(d.REVENUE_IS_PCTGE,errors="coerce").fillna(pd.to_numeric(d.REVENUE_PCTGE,errors="coerce")); d["SHARE"]=(pct/100).where(pct.between(0,110)); key=["SECURITY_ID","END_DATE","CLASSIF_CD","ACT_PUBTIME"]; g=d.dropna(subset=["SHARE"]).groupby(key,sort=False).agg(HHI=("SHARE",lambda x:float(np.square(x).sum())),BREADTH=("SHARE",lambda x:float((x>=.01).sum())),SHARE_SUM=("SHARE","sum")).reset_index(); g=g[g.SHARE_SUM.between(.70,1.30)]
 latest=g.sort_values(key).drop_duplicates(["SECURITY_ID","END_DATE","CLASSIF_CD"],keep="last"); wide=latest.pivot(index=["SECURITY_ID","END_DATE"],columns="CLASSIF_CD",values=["HHI","BREADTH","ACT_PUBTIME"]); wide.columns=[f"{a}_{int(b)}" for a,b in wide.columns]; wide=wide.reset_index(); times=[c for c in wide if c.startswith("ACT_PUBTIME_")]; wide["EVENT_TIME"]=wide[times].max(axis=1); wide["QUARTER_INDEX"]=wide.END_DATE.dt.year.astype("int64")*4+wide.END_DATE.dt.quarter
 # Dictionary convention: 1=industry/product, 2=product, 3=region. Product uses 2 with 1 fallback.
 ph=wide.get("HHI_2",wide.get("HHI_1")); pb=wide.get("BREADTH_2",wide.get("BREADTH_1")); rh=wide.get("HHI_3"); raw={"r63_low_product_revenue_hhi":-ph,"r63_low_region_revenue_hhi":-rh,"r63_product_revenue_breadth":pb}
 q=wide.QUARTER_INDEX; zs=[]
 for v in raw.values():
  c=v.groupby(q,sort=False).transform("median"); mad=(v-c).abs().groupby(q,sort=False).transform("median"); zs.append(((v-c)/(1.4826*mad).where(mad.gt(1e-12))).clip(-8,8))
 comp=pd.concat(zs,axis=1); raw["r63_segment_diversification_equal3"]=comp.mean(axis=1,skipna=True).where(comp.notna().sum(axis=1).ge(2)); out=[]
 for f,v in raw.items():
  x=wide[["SECURITY_ID","QUARTER_INDEX","EVENT_TIME"]].copy(); x["factor"]=f; x["value"]=pd.to_numeric(v,errors="coerce"); out.append(x.dropna(subset=["EVENT_TIME","value"]))
 return pd.concat(out,ignore_index=True)
def generate(panel,data,pit,out):
 dates=pd.read_parquet(panel,columns=["TRADE_DATE"]).TRADE_DATE; cal=pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values(); workflow.CANDIDATE_COLUMNS=FACTORS; wide=workflow.prepare_wide_events(events(data,pit),cal); chunks=[]; cov=[]
 for y in sorted(set(cal.year)):
  filt=[("TRADE_DATE",">=",pd.Timestamp(y,1,1)),("TRADE_DATE","<=",pd.Timestamp(y,12,31))]; keys=_normalize_panel(pd.read_parquet(panel,columns=KEYS,filters=filt)); m=workflow._map_sparse(keys,wide); chunks.append(m); cov.extend({"year":y,"factor":f,"missing_rate":float(m[f].isna().mean())} for f in FACTORS); print(y,len(m),flush=True)
 out.mkdir(parents=True,exist_ok=True); pd.concat(chunks).sort_values(KEYS).to_parquet(out/"round63_sparse_before_fill.parquet",index=False,compression="zstd"); pd.DataFrame(cov).to_csv(out/"round63_coverage.csv",index=False,encoding="utf-8-sig")
def fill(sparse,ic,output):
 tested=set(pd.read_csv(ic).factor.astype(str)); factors=[f for f in FACTORS if f in tested]; d=pd.read_parquet(sparse,columns=KEYS+factors); before=d[factors].isna().mean(); med=d.groupby("TRADE_DATE",sort=False)[factors].transform("median"); d[factors]=d[factors].fillna(med); whole=d[factors].isna().sum(); d[factors]=d[factors].fillna(0); d.to_parquet(output,index=False,compression="zstd"); pd.DataFrame({"factor":factors,"missing_rate_before_fill":[before[f] for f in factors],"whole_day_neutral_rows":[whole[f] for f in factors]}).to_csv(output.with_suffix(".fill_report.csv"),index=False,encoding="utf-8-sig")
def main():
 p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd",required=True); g=s.add_parser("generate-sparse"); g.add_argument("--panel",type=Path,required=True); g.add_argument("--data-root",type=Path,required=True); g.add_argument("--pit-root",type=Path,required=True); g.add_argument("--output-dir",type=Path,required=True); f=s.add_parser("fill-after-test"); f.add_argument("--sparse",type=Path,required=True); f.add_argument("--sparse-ic",type=Path,required=True); f.add_argument("--output",type=Path,required=True); a=p.parse_args(); generate(a.panel.resolve(),a.data_root.resolve(),a.pit_root.resolve(),a.output_dir.resolve()) if a.cmd=="generate-sparse" else fill(a.sparse.resolve(),a.sparse_ic.resolve(),a.output.resolve())
if __name__=="__main__":main()
