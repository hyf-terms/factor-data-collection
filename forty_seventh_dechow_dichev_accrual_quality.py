"""Round 47: PIT Dechow-Dichev working-capital accrual quality.

For each firm, working-capital accruals are explained by previous, current and
next-quarter operating cash flow.  At disclosure t, the newest fitted accrual
observation is t-1, whose lead CFO is the already disclosed CFO at t.  Thus no
future information is used.  Negative rolling residual volatility is the
quality signal. Fixed 8- and 12-observation windows are tested.
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


FACTORS = ["r47_dd_accrual_quality_8q", "r47_dd_accrual_quality_12q"]
BALANCE_FIELDS = ["INDUSTRY_CATEGORY", "T_ASSETS", "T_CA", "CASH_C_EQUIV", "T_CL", "ST_BORR", "NCL_WITHIN_1_Y"]


def balance_events(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw[COMMON_COLUMNS + BALANCE_FIELDS].copy()
    for c in ["ACT_PUBTIME", "END_DATE", "END_DATE_REP"]: d[c] = pd.to_datetime(d[c], errors="coerce")
    d["END_DATE"] = d["END_DATE"].dt.normalize(); d["END_DATE_REP"] = d["END_DATE_REP"].dt.normalize()
    for c in BALANCE_FIELDS[1:]: d[c] = pd.to_numeric(d[c], errors="coerce")
    d[["CASH_C_EQUIV", "ST_BORR", "NCL_WITHIN_1_Y"]] = d[["CASH_C_EQUIV", "ST_BORR", "NCL_WITHIN_1_Y"]].fillna(0.0)
    mask = (d["MERGED_FLAG"].astype("string").eq("1") & d["REPORT_TYPE"].astype("string").isin(REPORT_TYPES)
            & d["IS_CURRENT_PERIOD"].fillna(False).astype(bool) & d["END_DATE"].eq(d["END_DATE_REP"])
            & ~d["INDUSTRY_CATEGORY"].isin(FINANCIAL_INDUSTRIES))
    d = d.loc[mask].dropna(subset=["SECURITY_ID", "ACT_PUBTIME", "END_DATE", "T_ASSETS", "T_CA", "T_CL"])
    d["SECURITY_ID"] = pd.to_numeric(d["SECURITY_ID"]).astype("int64")
    d["FISCAL_YEAR"] = d["END_DATE"].dt.year.astype("int16")
    d["FISCAL_QUARTER"] = d["REPORT_TYPE"].map(REPORT_QUARTERS).astype("int8")
    d["QUARTER_INDEX"] = d["FISCAL_YEAR"].astype("int64") * 4 + d["FISCAL_QUARTER"]
    return d.sort_values(["SECURITY_ID", "QUARTER_INDEX", "ACT_PUBTIME", "ID"]).drop_duplicates(["SECURITY_ID", "QUARTER_INDEX"], keep="first")


def factor_events(pit_dir: Path) -> pd.DataFrame:
    cash = pd.read_parquet(pit_dir / "new_pit_cashflow", columns=COMMON_COLUMNS + ["N_CF_OPERATE_A"], engine="pyarrow")
    cash["N_CF_OPERATE_A"] = pd.to_numeric(cash["N_CF_OPERATE_A"], errors="coerce")
    cfo = build_standalone_quarterly_metric(cash, "N_CF_OPERATE_A", name="cashflow PIT")
    cfo = _merge_flow_tables({"N_CF_OPERATE_A": cfo}, ["N_CF_OPERATE_A"], ttm=False)
    raw_b = pd.read_parquet(pit_dir / "new_pit_balance", columns=COMMON_COLUMNS + BALANCE_FIELDS, engine="pyarrow")
    bal = balance_events(raw_b)
    d = bal.merge(cfo[["SECURITY_ID", "QUARTER_INDEX", "N_CF_OPERATE_A", "FLOW_EVENT_TIME"]], on=["SECURITY_ID", "QUARTER_INDEX"], how="inner", validate="one_to_one")
    d = d.sort_values(["SECURITY_ID", "QUARTER_INDEX"])
    g = d.groupby("SECURITY_ID", sort=False)
    avg_assets = (d["T_ASSETS"] + g["T_ASSETS"].shift(1)) / 2.0
    noncash_wc = d["T_CA"] - d["CASH_C_EQUIV"] - d["T_CL"] + d["ST_BORR"] + d["NCL_WITHIN_1_Y"]
    d["tca"] = (noncash_wc - noncash_wc.groupby(d["SECURITY_ID"], sort=False).shift(1)) / avg_assets
    d["cfo"] = d["N_CF_OPERATE_A"] / avg_assets
    d["event_time"] = pd.concat([pd.to_datetime(d["ACT_PUBTIME"]), pd.to_datetime(d["FLOW_EVENT_TIME"])], axis=1).max(axis=1)
    rows = []
    for sid, group in d.groupby("SECURITY_ID", sort=False):
        q = group.sort_values("QUARTER_INDEX").copy()
        q["cfo_lag"] = q["cfo"].shift(1); q["cfo_lead"] = q["cfo"].shift(-1)
        for end in range(len(q)):
            # At end t, regression observations end at t-1; CFO_t supplies lead CFO.
            for window, factor in [(8, FACTORS[0]), (12, FACTORS[1])]:
                hist = q.iloc[max(0, end-window):end].dropna(subset=["tca", "cfo_lag", "cfo", "cfo_lead"])
                if len(hist) < max(6, window // 2): continue
                x = np.column_stack([np.ones(len(hist)), hist[["cfo_lag", "cfo", "cfo_lead"]].to_numpy(float)])
                y = hist["tca"].to_numpy(float)
                beta, *_ = np.linalg.lstsq(x, y, rcond=None)
                resid = y - x @ beta
                rows.append({"SECURITY_ID": int(sid), "QUARTER_INDEX": int(q.iloc[end]["QUARTER_INDEX"]), "EVENT_TIME": q.iloc[end]["event_time"], "factor": factor, "value": -float(np.std(resid, ddof=1))})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--panel", type=Path, required=True); p.add_argument("--pit-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--coverage", type=Path, required=True); args=p.parse_args()
    dates=pd.read_parquet(args.panel,columns=["TRADE_DATE"])["TRADE_DATE"]; cal=pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    raw=factor_events(args.pit_dir); pieces=[]
    for factor, group in raw.groupby("factor",sort=False):
        a=assign_available_trade_date(group,cal).sort_values(["SECURITY_ID","AVAILABLE_DATE","EVENT_TIME","QUARTER_INDEX"])
        newest=a.groupby("SECURITY_ID",sort=False)["QUARTER_INDEX"].cummax(); a=a.loc[a["QUARTER_INDEX"].eq(newest)].drop_duplicates(["SECURITY_ID","AVAILABLE_DATE"],keep="last")
        pieces.append(a[["SECURITY_ID","AVAILABLE_DATE","factor","value"]])
    wide=pd.concat(pieces).pivot_table(index=["SECURITY_ID","AVAILABLE_DATE"],columns="factor",values="value",aggfunc="last").reset_index().sort_values(["SECURITY_ID","AVAILABLE_DATE"])
    wide[FACTORS]=wide.groupby("SECURITY_ID",sort=False)[FACTORS].ffill(); chunks=[]; coverage=[]
    for year in sorted(set(cal.year)):
        filters=[("TRADE_DATE",">=",pd.Timestamp(year,1,1)),("TRADE_DATE","<=",pd.Timestamp(year,12,31))]; panel=_normalize_panel(pd.read_parquet(args.panel,columns=KEYS,filters=filters))
        mapped=pd.merge_asof(panel.sort_values(["TRADE_DATE","SECURITY_ID"]),wide.sort_values(["AVAILABLE_DATE","SECURITY_ID"]),by="SECURITY_ID",left_on="TRADE_DATE",right_on="AVAILABLE_DATE",direction="backward")[KEYS+FACTORS]
        mapped[FACTORS]=mapped[FACTORS].astype("float32"); chunks.append(mapped)
        for factor in FACTORS: coverage.append({"year":year,"factor":factor,"missing_rate":float(mapped[factor].isna().mean())})
        print(year,f"rows={len(mapped):,}")
    args.output.parent.mkdir(parents=True,exist_ok=True); pd.concat(chunks).sort_values(KEYS).to_parquet(args.output,index=False,compression="zstd"); pd.DataFrame(coverage).to_csv(args.coverage,index=False,encoding="utf-8-sig")


if __name__ == "__main__": main()
