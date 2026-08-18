"""Rounds 65--69: five structurally distinct PIT financial factor families.

Families: debt maturity, complete statement revisions, audit text/opinion,
subsidiary financial structure, and contract execution.  No label, return,
percentile rank, Q1 restriction, or fixed 60-day window is used.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eighth_round_literature_factors as workflow
from event_financial_factor_search import KEYS, _normalize_panel
from sixty_third_dense_segment_structure_factors import security_mapping
from sixtieth_dense_disclosure_quality_factors import build_quarterly_states


FACTORS = [
    "r65_low_near_term_debt_share", "r65_long_debt_share",
    "r65_low_one_year_maturity_pressure", "r65_debt_maturity_equal3",
    "r66_low_statement_version_count", "r66_low_revision_span",
    "r66_low_revision_breadth", "r66_revision_process_equal3",
    "r67_clean_financial_audit", "r67_clean_internal_control_audit",
    "r67_auditor_continuity", "r67_low_kam_breadth",
    "r67_low_estimation_kam_share", "r67_audit_quality_equal5",
    "r68_profitable_subsidiary_share", "r68_low_subsidiary_loss_intensity",
    "r68_low_subsidiary_roa_dispersion", "r68_low_subsidiary_revenue_hhi",
    "r68_subsidiary_quality_equal4",
    "r69_contract_liability_yoy_release", "r69_contract_liability_growth",
    "r69_major_contract_revenue_intensity", "r69_low_related_contract_share",
]


def ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    return x.div(y.abs().where(y.abs().gt(1e-9))).replace([np.inf, -np.inf], np.nan)


def robust_z(values: pd.Series, group: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    center = x.groupby(group, sort=False).transform("median")
    mad = (x - center).abs().groupby(group, sort=False).transform("median")
    return ((x - center) / (1.4826 * mad).where(mad.gt(1e-12))).clip(-8, 8)


def bit_true(values: pd.Series) -> pd.Series:
    """Decode MySQL BIT, boolean, and numeric latest flags consistently."""
    return values.map(
        lambda value: value in (True, 1, "1", b"\x01", "\x01")
        if not pd.isna(value) else False
    )


def read_table(root: Path, table: str, columns: list[str]) -> pd.DataFrame:
    files = glob.glob(str(root / table / "*.parquet"))
    if not files:
        raise FileNotFoundError(root / table)
    return pd.concat([pd.read_parquet(file, columns=columns) for file in files], ignore_index=True)


def attach_security(data: pd.DataFrame, pit_root: Path) -> pd.DataFrame:
    mapping, _ = security_mapping(pit_root)
    mapping = mapping[["PARTY_ID", "SECURITY_ID"]].drop_duplicates("PARTY_ID")
    return data.merge(mapping, on="PARTY_ID", how="inner", validate="many_to_one")


def as_events(data: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    pieces = []
    for factor, source in mapping.items():
        x = data[["SECURITY_ID", "QUARTER_INDEX", "EVENT_TIME"]].copy()
        x["factor"] = factor
        x["value"] = pd.to_numeric(data[source], errors="coerce")
        pieces.append(x.dropna(subset=["EVENT_TIME", "value"]))
    return pd.concat(pieces, ignore_index=True)


def debt_events(root: Path, pit: Path) -> pd.DataFrame:
    b = read_table(root, "fdmt_borrowing", ["PARTY_ID", "ACT_PUBTIME", "END_DATE", "BORR_TYPE", "ITEM_NAME", "BORR_AMOU"])
    b = attach_security(b, pit)
    b["EVENT_TIME"] = pd.to_datetime(b.ACT_PUBTIME, errors="coerce")
    b["END_DATE"] = pd.to_datetime(b.END_DATE, errors="coerce").dt.normalize()
    b["BORR_AMOU"] = pd.to_numeric(b.BORR_AMOU, errors="coerce")
    # Use disclosed totals only; summing total plus detail rows double-counts debt.
    total = b.ITEM_NAME.astype("string").str.contains("合计", na=False)
    b = b[total].sort_values("EVENT_TIME").drop_duplicates(["SECURITY_ID", "END_DATE", "BORR_TYPE"], keep="last")
    b["kind"] = np.where(b.BORR_TYPE.astype("string").str.contains("短期", na=False), "short", "long")
    w = b.pivot_table(index=["SECURITY_ID", "END_DATE"], columns="kind", values="BORR_AMOU", aggfunc="last").reset_index()
    t = b.groupby(["SECURITY_ID", "END_DATE"], sort=False).EVENT_TIME.max().rename("EVENT_TIME").reset_index()
    w = w.merge(t, on=["SECURITY_ID", "END_DATE"], how="left")
    n = read_table(root, "fdmt_ncl_oney", ["PARTY_ID", "ACT_PUBTIME", "END_DATE", "ITEM_NAME", "ITEM_VALUE"])
    n = attach_security(n, pit); n["EVENT_TIME_N"] = pd.to_datetime(n.ACT_PUBTIME, errors="coerce"); n["END_DATE"] = pd.to_datetime(n.END_DATE, errors="coerce").dt.normalize(); n["ITEM_VALUE"] = pd.to_numeric(n.ITEM_VALUE, errors="coerce")
    nt = n.ITEM_NAME.astype("string").str.contains("合计", na=False)
    n = n[nt].sort_values("EVENT_TIME_N").drop_duplicates(["SECURITY_ID", "END_DATE"], keep="last")
    w = w.merge(n[["SECURITY_ID", "END_DATE", "EVENT_TIME_N", "ITEM_VALUE"]], on=["SECURITY_ID", "END_DATE"], how="outer")
    w["EVENT_TIME"] = w[["EVENT_TIME", "EVENT_TIME_N"]].max(axis=1); w["QUARTER_INDEX"] = w.END_DATE.dt.year * 4 + w.END_DATE.dt.quarter
    total_debt = w.get("short", 0).fillna(0) + w.get("long", 0).fillna(0)
    w["near"] = -ratio(w.get("short", 0).fillna(0) + w.ITEM_VALUE.fillna(0), total_debt + w.ITEM_VALUE.fillna(0))
    w["long_share"] = ratio(w.get("long", 0), total_debt)
    w["pressure"] = -ratio(w.ITEM_VALUE, total_debt)
    z = pd.concat([robust_z(w[c], w.QUARTER_INDEX) for c in ["near", "long_share", "pressure"]], axis=1)
    w["equal"] = z.mean(axis=1, skipna=True).where(z.notna().sum(axis=1).ge(2))
    return as_events(w, {"r65_low_near_term_debt_share":"near", "r65_long_debt_share":"long_share", "r65_low_one_year_maturity_pressure":"pressure", "r65_debt_maturity_equal3":"equal"})


def revision_events(pit: Path) -> pd.DataFrame:
    s = build_quarterly_states(pit)
    versions = s[[c for c in s if c.endswith("_VERSION_COUNT")]]
    spans = s[[c for c in s if c.endswith("_REVISION_SPAN")]]
    magnitudes = s[[c for c in s if c.endswith("_REVISION_MAGNITUDE")]]
    s["vcount"] = -versions.mean(axis=1, skipna=True)
    s["span"] = -spans.mean(axis=1, skipna=True)
    s["breadth"] = -magnitudes.gt(1e-8).sum(axis=1).astype(float)
    z = pd.concat([robust_z(s[c], s.QUARTER_INDEX) for c in ["vcount", "span", "breadth"]], axis=1)
    s["equal"] = z.mean(axis=1, skipna=True).where(z.notna().sum(axis=1).ge(2))
    return as_events(s, {"r66_low_statement_version_count":"vcount", "r66_low_revision_span":"span", "r66_low_revision_breadth":"breadth", "r66_revision_process_equal3":"equal"})


def audit_events(root: Path, pit: Path) -> pd.DataFrame:
    a = read_table(root, "fdmt_adt_opn_n", ["PARTY_ID", "PUBLISH_DATE", "END_DATE", "DM_OPN_TYPE", "DM_AUDIT_AGENCY", "IC_OPN_TYPE"])
    a = attach_security(a, pit); a["EVENT_TIME"] = pd.to_datetime(a.PUBLISH_DATE, errors="coerce"); a["END_DATE"] = pd.to_datetime(a.END_DATE, errors="coerce").dt.normalize(); a = a.sort_values("EVENT_TIME").drop_duplicates(["SECURITY_ID", "END_DATE"], keep="last")
    a["clean"] = a.DM_OPN_TYPE.astype("string").eq("1").astype(float); a["ic_clean"] = a.IC_OPN_TYPE.astype("string").eq("1").astype(float)
    a["continuity"] = a.groupby("SECURITY_ID", sort=False).DM_AUDIT_AGENCY.transform(lambda x: x.astype("string").eq(x.astype("string").shift()).astype(float))
    k = read_table(root, "fdmt_main_adt_matters", ["PARTY_ID", "PUBLISH_DATE", "END_DATE", "AUDIT_PROJECT", "AUDIT_MATTERS"])
    k = attach_security(k, pit); k["PUBLISH_DATE"] = pd.to_datetime(k.PUBLISH_DATE, errors="coerce"); k["END_DATE"] = pd.to_datetime(k.END_DATE, errors="coerce").dt.normalize()
    risky = "减值|坏账|公允价值|估值|预计负债|持续经营|商誉|存货跌价"
    k["RISKY"] = (k.AUDIT_PROJECT.astype("string") + " " + k.AUDIT_MATTERS.astype("string")).str.contains(risky, regex=True, na=False).astype(float)
    kg = k.groupby(["SECURITY_ID", "END_DATE"], sort=False).agg(KAM_COUNT=("AUDIT_PROJECT", "size"), RISK_COUNT=("RISKY", "sum"), KAM_TIME=("PUBLISH_DATE", "max")).reset_index()
    a = a.merge(kg, on=["SECURITY_ID", "END_DATE"], how="left"); a["EVENT_TIME"] = a[["EVENT_TIME", "KAM_TIME"]].max(axis=1); a["kam"] = -a.KAM_COUNT; a["risk_share"] = -ratio(a.RISK_COUNT, a.KAM_COUNT); a["QUARTER_INDEX"] = a.END_DATE.dt.year * 4 + a.END_DATE.dt.quarter
    comps = ["clean", "ic_clean", "continuity", "kam", "risk_share"]
    z = pd.concat([robust_z(a[c], a.QUARTER_INDEX) for c in comps], axis=1); a["equal"] = z.mean(axis=1, skipna=True).where(z.notna().sum(axis=1).ge(3))
    return as_events(a, {"r67_clean_financial_audit":"clean", "r67_clean_internal_control_audit":"ic_clean", "r67_auditor_continuity":"continuity", "r67_low_kam_breadth":"kam", "r67_low_estimation_kam_share":"risk_share", "r67_audit_quality_equal5":"equal"})


def subsidiary_events(root: Path, pit: Path) -> pd.DataFrame:
    d = read_table(root, "fdmt_sub_fin_pit", ["PARTY_ID", "ACT_PUBTIME", "END_DATE", "IS_NEW", "COM_ID", "ITEM_NAME", "VALUE", "INVOLVED_REL"])
    d = attach_security(d, pit); d["EVENT_TIME"] = pd.to_datetime(d.ACT_PUBTIME, errors="coerce"); d["END_DATE"] = pd.to_datetime(d.END_DATE, errors="coerce").dt.normalize(); d["VALUE"] = pd.to_numeric(d.VALUE, errors="coerce")
    d = d[bit_true(d.IS_NEW) & d.INVOLVED_REL.astype("string").str.contains("子公司|控股|孙公司", regex=True, na=False)]
    d = d.sort_values("EVENT_TIME").drop_duplicates(["SECURITY_ID", "END_DATE", "COM_ID", "ITEM_NAME"], keep="last")
    w = d.pivot_table(index=["SECURITY_ID", "END_DATE", "COM_ID"], columns="ITEM_NAME", values="VALUE", aggfunc="last").reset_index()
    times = d.groupby(["SECURITY_ID", "END_DATE"], sort=False).EVENT_TIME.max().rename("EVENT_TIME")
    def col(name: str) -> pd.Series: return pd.to_numeric(w[name], errors="coerce") if name in w else pd.Series(np.nan, index=w.index)
    w["NI"] = col("净利润"); w["ASSET"] = col("总资产"); w["REV"] = col("营业收入").fillna(col("主营业务收入")); w["ROA"] = ratio(w.NI, w.ASSET)
    rows=[]
    for key,g in w.groupby(["SECURITY_ID","END_DATE"],sort=False):
        rev=g.REV.clip(lower=0); share=rev/rev.sum() if rev.sum()>0 else pd.Series(np.nan,index=g.index)
        rows.append({"SECURITY_ID":key[0],"END_DATE":key[1],"profitable":float(g.NI.gt(0).mean()) if g.NI.notna().any() else np.nan,"loss":-float((-g.NI.clip(upper=0)).sum()/(g.ASSET.abs().sum()+1e-9)) if g.NI.notna().any() and g.ASSET.notna().any() else np.nan,"disp":-float(g.ROA.std()) if g.ROA.notna().sum()>=2 else np.nan,"hhi":-float((share**2).sum()) if share.notna().any() else np.nan})
    q=pd.DataFrame(rows).merge(times.reset_index(),on=["SECURITY_ID","END_DATE"],how="left"); q["QUARTER_INDEX"]=q.END_DATE.dt.year*4+q.END_DATE.dt.quarter
    comps=["profitable","loss","disp","hhi"]; z=pd.concat([robust_z(q[c],q.QUARTER_INDEX) for c in comps],axis=1);q["equal"]=z.mean(axis=1,skipna=True).where(z.notna().sum(axis=1).ge(2))
    return as_events(q,{"r68_profitable_subsidiary_share":"profitable","r68_low_subsidiary_loss_intensity":"loss","r68_low_subsidiary_roa_dispersion":"disp","r68_low_subsidiary_revenue_hhi":"hhi","r68_subsidiary_quality_equal4":"equal"})


def contract_events(root: Path, pit: Path) -> pd.DataFrame:
    d=read_table(root,"fdmt_con_lab_na",["PARTY_ID","ACT_PUBTIME","END_DATE","IS_NEW","TOT_CON_LIA"]);d=attach_security(d,pit);d["EVENT_TIME"]=pd.to_datetime(d.ACT_PUBTIME,errors="coerce");d["END_DATE"]=pd.to_datetime(d.END_DATE,errors="coerce").dt.normalize();d["TOT_CON_LIA"]=pd.to_numeric(d.TOT_CON_LIA,errors="coerce");d=d[bit_true(d.IS_NEW)].sort_values("EVENT_TIME").drop_duplicates(["SECURITY_ID","END_DATE"],keep="last");d["QUARTER_INDEX"]=d.END_DATE.dt.year*4+d.END_DATE.dt.quarter;d=d.sort_values(["SECURITY_ID","QUARTER_INDEX"]);lag=d.groupby("SECURITY_ID").TOT_CON_LIA.shift(4);d["release"]=-ratio(d.TOT_CON_LIA-lag,lag);d["growth"] = ratio(d.TOT_CON_LIA-lag,lag)
    m=read_table(root,"equ_major_contract_pit",["PARTY_ID","PUBLISH_DATE","BIDDING_AMOUNT","IS_RELATED","IS_RELA_TRANS"]);m=attach_security(m,pit);m["EVENT_TIME"]=pd.to_datetime(m.PUBLISH_DATE,errors="coerce");m["BIDDING_AMOUNT"]=pd.to_numeric(m.BIDDING_AMOUNT,errors="coerce");m["related"]=(m.IS_RELATED.fillna(0).astype(float).gt(0)|m.IS_RELA_TRANS.fillna(0).astype(float).gt(0)).astype(float);m=m.groupby(["SECURITY_ID","EVENT_TIME"],sort=False).agg(intensity=("BIDDING_AMOUNT","sum"),related=("related","mean")).reset_index();m["QUARTER_INDEX"]=m.EVENT_TIME.dt.year*400+m.EVENT_TIME.dt.dayofyear;m["related"]=-m.related
    base=as_events(d,{"r69_contract_liability_yoy_release":"release","r69_contract_liability_growth":"growth"}); ev=as_events(m,{"r69_major_contract_revenue_intensity":"intensity","r69_low_related_contract_share":"related"})
    # Composite remains sparse unless at least two independently observed components coexist.
    return pd.concat([base,ev],ignore_index=True)


def build_events(root: Path,pit: Path)->pd.DataFrame:
    return pd.concat([debt_events(root,pit),revision_events(pit),audit_events(root,pit),subsidiary_events(root,pit),contract_events(root,pit)],ignore_index=True)


def generate(panel:Path,root:Path,pit:Path,out:Path)->None:
    dates=pd.read_parquet(panel,columns=["TRADE_DATE"]).TRADE_DATE;cal=pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values();events=build_events(root,pit); workflow.CANDIDATE_COLUMNS=FACTORS
    wide=workflow.prepare_wide_events(events,cal); chunks=[];coverage=[]
    for year in sorted(set(cal.year)):
        filt=[("TRADE_DATE",">=",pd.Timestamp(year,1,1)),("TRADE_DATE","<=",pd.Timestamp(year,12,31))];keys=_normalize_panel(pd.read_parquet(panel,columns=KEYS,filters=filt));mapped=workflow._map_sparse(keys,wide);chunks.append(mapped)
        coverage.extend({"year":year,"factor":f,"missing_rate":float(mapped[f].isna().mean()),"observed_days":int(mapped.loc[mapped[f].notna(),"TRADE_DATE"].nunique())} for f in FACTORS);print(year,len(mapped),flush=True)
    out.mkdir(parents=True,exist_ok=True);pd.concat(chunks).sort_values(KEYS).to_parquet(out/"round65_69_sparse_before_fill.parquet",index=False,compression="zstd");pd.DataFrame(coverage).to_csv(out/"round65_69_coverage.csv",index=False,encoding="utf-8-sig");events.to_parquet(out/"round65_69_event_audit.parquet",index=False,compression="zstd");(out/"metadata.json").write_text(json.dumps({"factors":FACTORS,"uses_rank":False,"uses_label":False,"q1_or_60d_restriction":False},ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--panel",type=Path,required=True);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--pit-root",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();generate(a.panel.resolve(),a.data_root.resolve(),a.pit_root.resolve(),a.output_dir.resolve())
if __name__=="__main__":main()
