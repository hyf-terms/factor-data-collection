"""Round 61: dense book-tax difference and tax-payment quality factors."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import eighth_round_literature_factors as workflow
from dense_literature_profitability_factors import _merge_flow_tables
from event_financial_factor_search import KEYS, _normalize_panel
from fifteenth_round_temporal_financial_factors import _read_inputs as _unused_read
from pead_sue_factor import assign_available_trade_date
from quarterly_f_score import COMMON_COLUMNS, FINANCIAL_INDUSTRIES, REPORT_QUARTERS, REPORT_TYPES, build_standalone_quarterly_metric

FACTORS=[
 "r61_low_absolute_book_tax_gap_ttm","r61_low_signed_book_tax_gap_ttm",
 "r61_tax_expense_cash_alignment_ttm","r61_low_effective_tax_rate_deviation",
 "r61_tax_quality_equal3",
]

def ratio(a,b):
 b=pd.to_numeric(b,errors="coerce"); return pd.to_numeric(a,errors="coerce").div(b.abs().where(b.abs().gt(1e-12))).replace([np.inf,-np.inf],np.nan)

def balance(pit):
 cols=COMMON_COLUMNS+["INDUSTRY_CATEGORY","T_ASSETS"]
 d=pd.read_parquet(pit/"new_pit_balance",columns=list(dict.fromkeys(cols)),engine="pyarrow")
 for c in ["ACT_PUBTIME","END_DATE","END_DATE_REP"]: d[c]=pd.to_datetime(d[c],errors="coerce")
 d["END_DATE"]=d.END_DATE.dt.normalize(); d["END_DATE_REP"]=d.END_DATE_REP.dt.normalize(); d["T_ASSETS"]=pd.to_numeric(d.T_ASSETS,errors="coerce")
 m=(d.MERGED_FLAG.astype("string").eq("1") & d.REPORT_TYPE.astype("string").isin(REPORT_TYPES) & d.IS_CURRENT_PERIOD.fillna(False).astype(bool) & d.END_DATE.eq(d.END_DATE_REP) & ~d.INDUSTRY_CATEGORY.isin(FINANCIAL_INDUSTRIES))
 d=d.loc[m].dropna(subset=["SECURITY_ID","ACT_PUBTIME","END_DATE","T_ASSETS"]); d["SECURITY_ID"]=pd.to_numeric(d.SECURITY_ID).astype("int64"); d["QUARTER_INDEX"]=d.END_DATE.dt.year.astype("int64")*4+d.REPORT_TYPE.map(REPORT_QUARTERS).astype("int8")
 return d.sort_values(["SECURITY_ID","QUARTER_INDEX","ACT_PUBTIME","ID"]).drop_duplicates(["SECURITY_ID","QUARTER_INDEX"],keep="first")[["SECURITY_ID","QUARTER_INDEX","ACT_PUBTIME","T_ASSETS"]]

def events(pit:Path):
 inc=pd.read_parquet(pit/"new_pit_income",columns=COMMON_COLUMNS+["T_PROFIT","INCOME_TAX"],engine="pyarrow")
 cf=pd.read_parquet(pit/"new_pit_cashflow",columns=COMMON_COLUMNS+["C_PAID_FOR_TAXES"],engine="pyarrow")
 flows={}
 for field in ["T_PROFIT","INCOME_TAX"]: flows[field]=build_standalone_quarterly_metric(inc,field,name="income PIT")
 flows["C_PAID_FOR_TAXES"]=build_standalone_quarterly_metric(cf,"C_PAID_FOR_TAXES",name="cashflow PIT")
 ttm=_merge_flow_tables(flows,["T_PROFIT","INCOME_TAX","C_PAID_FOR_TAXES"],ttm=True)
 bal=balance(pit); lag=bal.copy(); lag["QUARTER_INDEX"]+=4; lag=lag.rename(columns={"ACT_PUBTIME":"L4_TIME","T_ASSETS":"L4_ASSETS"})
 d=ttm.merge(bal,on=["SECURITY_ID","QUARTER_INDEX"],how="inner").merge(lag,on=["SECURITY_ID","QUARTER_INDEX"],how="inner")
 pretax=d.TTM_T_PROFIT; expense=d.TTM_INCOME_TAX; cash=d.TTM_C_PAID_FOR_TAXES; assets=d.L4_ASSETS
 implied_taxable=cash/0.25; gap=ratio(pretax-implied_taxable,assets)
 cash_alignment=-ratio((expense-cash).abs(),assets)
 etr=expense.div(pretax.where(pretax.gt(0))).where(expense.ge(0)).clip(0,1)
 d["ETR"]=etr
 d=d.sort_values(["SECURITY_ID","QUARTER_INDEX"])
 hist=d.groupby("SECURITY_ID",sort=False).ETR.transform(lambda s:s.shift(1).rolling(12,min_periods=6).median())
 etr_quality=-(etr-hist).abs()
 raw={"r61_low_absolute_book_tax_gap_ttm":-gap.abs(),"r61_low_signed_book_tax_gap_ttm":-gap,"r61_tax_expense_cash_alignment_ttm":cash_alignment,"r61_low_effective_tax_rate_deviation":etr_quality}
 quarter=d.QUARTER_INDEX; zs={}
 for k,v in raw.items():
  center=v.groupby(quarter,sort=False).transform("median"); mad=(v-center).abs().groupby(quarter,sort=False).transform("median"); zs[k]=((v-center)/(1.4826*mad).where(mad.gt(1e-12))).clip(-8,8)
 raw["r61_tax_quality_equal3"]=pd.concat([zs["r61_low_absolute_book_tax_gap_ttm"],zs["r61_tax_expense_cash_alignment_ttm"],zs["r61_low_effective_tax_rate_deviation"]],axis=1).mean(axis=1,skipna=True).where(pd.concat([zs["r61_low_absolute_book_tax_gap_ttm"],zs["r61_tax_expense_cash_alignment_ttm"],zs["r61_low_effective_tax_rate_deviation"]],axis=1).notna().sum(axis=1).ge(2))
 time=pd.concat([pd.to_datetime(d.FLOW_EVENT_TIME),pd.to_datetime(d.ACT_PUBTIME),pd.to_datetime(d.L4_TIME)],axis=1).max(axis=1)
 out=[]
 for f,v in raw.items():
  x=d[["SECURITY_ID","QUARTER_INDEX"]].copy(); x["EVENT_TIME"]=time; x["factor"]=f; x["value"]=pd.to_numeric(v,errors="coerce").clip(-50,50); out.append(x.dropna(subset=["EVENT_TIME","value"]))
 return pd.concat(out,ignore_index=True)

def generate(panel,pit,outdir):
 dates=pd.read_parquet(panel,columns=["TRADE_DATE"])["TRADE_DATE"]; cal=pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values(); ev=events(pit); workflow.CANDIDATE_COLUMNS=FACTORS; wide=workflow.prepare_wide_events(ev,cal); chunks=[]; cov=[]
 for y in sorted(set(cal.year)):
  filters=[("TRADE_DATE",">=",pd.Timestamp(y,1,1)),("TRADE_DATE","<=",pd.Timestamp(y,12,31))]; keys=_normalize_panel(pd.read_parquet(panel,columns=KEYS,filters=filters)); m=workflow._map_sparse(keys,wide); chunks.append(m); cov.extend({"year":y,"factor":f,"missing_rate":float(m[f].isna().mean())} for f in FACTORS); print(y,len(m),flush=True)
 outdir.mkdir(parents=True,exist_ok=True); pd.concat(chunks).sort_values(KEYS).to_parquet(outdir/"round61_sparse_before_fill.parquet",index=False,compression="zstd"); pd.DataFrame(cov).to_csv(outdir/"round61_coverage.csv",index=False,encoding="utf-8-sig")

def fill(sparse,ic,output):
 tested=set(pd.read_csv(ic).factor.astype(str)); assert set(FACTORS)<=tested
 d=pd.read_parquet(sparse,columns=KEYS+FACTORS); before=d[FACTORS].isna().mean(); med=d.groupby("TRADE_DATE",sort=False)[FACTORS].transform("median"); d[FACTORS]=d[FACTORS].fillna(med); whole=d[FACTORS].isna().sum(); d[FACTORS]=d[FACTORS].fillna(0); output.parent.mkdir(parents=True,exist_ok=True); d.to_parquet(output,index=False,compression="zstd"); pd.DataFrame({"factor":FACTORS,"missing_rate_before_fill":[before[f] for f in FACTORS],"whole_day_neutral_rows":[whole[f] for f in FACTORS]}).to_csv(output.with_suffix(".fill_report.csv"),index=False,encoding="utf-8-sig")

def main():
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True); g=sub.add_parser("generate-sparse"); g.add_argument("--panel",type=Path,required=True); g.add_argument("--pit-dir",type=Path,required=True); g.add_argument("--output-dir",type=Path,required=True); f=sub.add_parser("fill-after-test"); f.add_argument("--sparse",type=Path,required=True); f.add_argument("--sparse-ic",type=Path,required=True); f.add_argument("--output",type=Path,required=True); a=p.parse_args(); generate(a.panel.resolve(),a.pit_dir.resolve(),a.output_dir.resolve()) if a.cmd=="generate-sparse" else fill(a.sparse.resolve(),a.sparse_ic.resolve(),a.output.resolve())
if __name__=="__main__": main()
