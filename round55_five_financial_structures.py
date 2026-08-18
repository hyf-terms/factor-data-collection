"""Round 55: five literature-motivated, non-profit financial structures.

The script never reads labels or an existing successful factor.  It maps the
latest actually disclosed observation to the daily universe and leaves natural
missing values intact for the required sparse diagnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from event_financial_factor_search import KEYS, _normalize_panel
from pead_sue_factor import assign_available_trade_date
from twenty_sixth_asset_quality_detail_factors import attach_assets, read_balance_assets
from twenty_third_alternative_event_factors import event_time, map_party, read_partitioned, security_mapping


FAMILIES = {
    "related_party": [
        "r55_low_related_receivable_assets",
        "r55_related_net_funding_assets",
        "r55_related_receivable_yoy_reduction",
        "r55_low_related_bad_debt_rate",
    ],
    "guarantee": [
        "r55_low_total_guarantee_net_assets",
        "r55_low_external_guarantee_assets",
        "r55_low_related_guarantee_assets",
        "r55_guarantee_yoy_reduction",
    ],
    "inventory": [
        "r55_low_inventory_provision_rate",
        "r55_inventory_provision_yoy_reduction",
        "r55_low_inventory_component_hhi",
        "r55_low_inventory_top_component_share",
    ],
    "trader": [
        "r55_low_customer_top5_concentration",
        "r55_customer_concentration_yoy_reduction",
        "r55_low_supplier_top5_concentration",
        "r55_supplier_concentration_yoy_reduction",
    ],
    "subsidy": [
        "r55_low_government_grant_assets",
        "r55_low_government_grant_nr_share",
        "r55_government_grant_yoy_reduction",
    ],
}


def ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    den = pd.to_numeric(b, errors="coerce")
    return pd.to_numeric(a, errors="coerce").div(den.where(den.abs().gt(1.0))).replace([np.inf, -np.inf], np.nan)


def common_clean(d: pd.DataFrame, time_col: str) -> pd.DataFrame:
    x = d.copy()
    x["EVENT_TIME"] = event_time(x, time_col, date_only=(time_col == "PUBLISH_DATE"))
    x["END_DATE"] = pd.to_datetime(x["END_DATE"], errors="coerce").dt.normalize()
    if "END_DATE_REP" in x:
        x["END_DATE_REP"] = pd.to_datetime(x["END_DATE_REP"], errors="coerce").dt.normalize()
        x = x.loc[x["END_DATE"].eq(x["END_DATE_REP"])]
    if "MERGED_FLAG" in x:
        x = x.loc[x["MERGED_FLAG"].astype(str).eq("1")]
    return x.dropna(subset=["SECURITY_ID", "EVENT_TIME", "END_DATE"])


def annual_lag(d: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    lag = d[["SECURITY_ID", "END_DATE", *columns]].copy()
    lag["END_DATE"] = lag["END_DATE"] + pd.DateOffset(years=1)
    return lag.rename(columns={c: f"L1_{c}" for c in columns})


def related_events(root: Path, pair: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    cols = ["PARTY_ID", "TICKER_SYMBOL", "ACT_PUBTIME", "END_DATE", "END_DATE_REP", "IS_NEW",
            "ADJUSTED_FLAG", "REPORT_TYPE", "MERGED_FLAG", "TYPE", "ITEM_NAME", "B_BALANCE", "BAD_DEBT_RES"]
    x = map_party(read_partitioned(root, "fdmt_related_rec_pay", cols), pair)
    x = common_clean(x, "ACT_PUBTIME")
    x = x.loc[x["ADJUSTED_FLAG"].astype(str).eq("期末余额")]
    for c in ["B_BALANCE", "BAD_DEBT_RES"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    rec_names = {"应收账款", "其他应收款", "预付款项", "预付账款", "合同资产", "应收票据", "应收款项融资", "长期应收款"}
    pay_names = {"应付账款", "其他应付款", "合同负债", "应付票据", "预收账款", "预收款项", "长期应付款"}
    x["REC"] = x["B_BALANCE"].where(x["TYPE"].astype(str).eq("1") & x["ITEM_NAME"].isin(rec_names), 0.0)
    x["PAY"] = x["B_BALANCE"].where(x["TYPE"].astype(str).eq("2") & x["ITEM_NAME"].isin(pay_names), 0.0)
    x["REC_BAD"] = x["BAD_DEBT_RES"].where(x["TYPE"].astype(str).eq("1") & x["ITEM_NAME"].isin(rec_names), 0.0)
    keys = ["SECURITY_ID", "END_DATE", "EVENT_TIME"]
    x = x.groupby(keys, as_index=False).agg(REC=("REC", "sum"), PAY=("PAY", "sum"), REC_BAD=("REC_BAD", "sum"))
    x = x.sort_values(keys).drop_duplicates(["SECURITY_ID", "END_DATE"], keep="first")
    x = attach_assets(x, assets)
    x = x.merge(annual_lag(x, ["REC", "T_ASSETS"]), on=["SECURITY_ID", "END_DATE"], how="left", validate="one_to_one")
    x[FAMILIES["related_party"][0]] = -ratio(x["REC"], x["T_ASSETS"]).clip(-5, 5)
    x[FAMILIES["related_party"][1]] = ratio(x["PAY"] - x["REC"], x["T_ASSETS"]).clip(-5, 5)
    x[FAMILIES["related_party"][2]] = -ratio(x["REC"] - x["L1_REC"], x["L1_T_ASSETS"]).clip(-5, 5)
    x[FAMILIES["related_party"][3]] = -ratio(x["REC_BAD"], x["REC"].abs()).clip(-2, 2)
    return x[["SECURITY_ID", "END_DATE", "EVENT_TIME", *FAMILIES["related_party"]]]


def guarantee_events(root: Path, pair: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    cols = ["PARTY_ID", "PUBLISH_DATE", "END_DATE", "GUARANTEE_TOTAL_AMOUNT", "EXTERNAL_GUARANTEE_BALANCE",
            "GUARANTEE_TOTAL_AMOUNT_NAR", "GUARANTEE_AMOUNT_RELATED"]
    x = read_partitioned(root, "equ_accumulated_guarantee", cols).merge(
        pair[["PARTY_ID", "SECURITY_ID"]].drop_duplicates("PARTY_ID"), on="PARTY_ID", how="left", validate="many_to_one")
    x = common_clean(x, "PUBLISH_DATE")
    for c in cols[3:]: x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.sort_values(["SECURITY_ID", "END_DATE", "EVENT_TIME"]).drop_duplicates(["SECURITY_ID", "END_DATE"], keep="last")
    x = attach_assets(x, assets)
    x = x.merge(annual_lag(x, ["GUARANTEE_TOTAL_AMOUNT", "T_ASSETS"]), on=["SECURITY_ID", "END_DATE"], how="left", validate="one_to_one")
    nar = x["GUARANTEE_TOTAL_AMOUNT_NAR"] / 100.0
    x[FAMILIES["guarantee"][0]] = -nar.where(nar.notna(), ratio(x["GUARANTEE_TOTAL_AMOUNT"], x["T_ASSETS"])).clip(-10, 10)
    x[FAMILIES["guarantee"][1]] = -ratio(x["EXTERNAL_GUARANTEE_BALANCE"], x["T_ASSETS"]).clip(-10, 10)
    x[FAMILIES["guarantee"][2]] = -ratio(x["GUARANTEE_AMOUNT_RELATED"], x["T_ASSETS"]).clip(-10, 10)
    x[FAMILIES["guarantee"][3]] = -ratio(x["GUARANTEE_TOTAL_AMOUNT"] - x["L1_GUARANTEE_TOTAL_AMOUNT"], x["L1_T_ASSETS"]).clip(-10, 10)
    return x[["SECURITY_ID", "END_DATE", "EVENT_TIME", *FAMILIES["guarantee"]]]


def inventory_events(root: Path, pair: pd.DataFrame) -> pd.DataFrame:
    cols = ["PARTY_ID", "TICKER_SYMBOL", "ACT_PUBTIME", "END_DATE", "END_DATE_REP", "IS_NEW",
            "ADJUSTED_FLAG", "REPORT_TYPE", "MERGED_FLAG", "ITEM_ID", "B_BALANCE", "PROVISION", "B_VALUE"]
    x = common_clean(map_party(read_partitioned(root, "fdmt_bs_inventory", cols), pair), "ACT_PUBTIME")
    x = x.loc[x["ADJUSTED_FLAG"].astype(str).eq("期末余额")]
    for c in ["B_BALANCE", "PROVISION", "B_VALUE"]: x[c] = pd.to_numeric(x[c], errors="coerce")
    x["ITEM_ID"] = x["ITEM_ID"].astype(str)
    keys = ["SECURITY_ID", "END_DATE", "EVENT_TIME"]
    total = x.loc[x["ITEM_ID"].eq("101700")].groupby(keys, as_index=False).agg(TOTAL=("B_BALANCE", "last"), PROVISION=("PROVISION", "last"))
    comp = x.loc[~x["ITEM_ID"].isin(["101700", "101797", "101798", "101799"]) & x["B_BALANCE"].gt(0)].copy()
    comp["SQ"] = comp["B_BALANCE"] ** 2
    structure = comp.groupby(keys, as_index=False).agg(COMP_SUM=("B_BALANCE", "sum"), COMP_SQ=("SQ", "sum"), TOP=("B_BALANCE", "max"))
    x = total.merge(structure, on=keys, how="left")
    x = x.sort_values(keys).drop_duplicates(["SECURITY_ID", "END_DATE"], keep="first")
    x = x.merge(annual_lag(x, ["PROVISION", "TOTAL"]), on=["SECURITY_ID", "END_DATE"], how="left", validate="one_to_one")
    x[FAMILIES["inventory"][0]] = -ratio(x["PROVISION"], x["TOTAL"].abs()).clip(-2, 2)
    x[FAMILIES["inventory"][1]] = -ratio(x["PROVISION"] - x["L1_PROVISION"], x["L1_TOTAL"].abs()).clip(-2, 2)
    x[FAMILIES["inventory"][2]] = -ratio(x["COMP_SQ"], x["COMP_SUM"] ** 2).clip(0, 1)
    x[FAMILIES["inventory"][3]] = -ratio(x["TOP"], x["COMP_SUM"]).clip(0, 1)
    return x[["SECURITY_ID", "END_DATE", "EVENT_TIME", *FAMILIES["inventory"]]]


def trader_events(root: Path, pair: pd.DataFrame) -> pd.DataFrame:
    cols = ["PARTY_ID", "TICKER_SYMBOL", "PUBLISH_DATE", "END_DATE", "END_DATE_REP", "TRADER_TYPE_CD", "TRADER_RANK", "RATIO"]
    x = common_clean(map_party(read_partitioned(root, "fdmt_trader", cols), pair), "PUBLISH_DATE")
    x["RATIO"] = pd.to_numeric(x["RATIO"], errors="coerce")
    # Vendor percentages can be represented as 0-1 or 0-100; normalize by report sum when necessary.
    keys = ["SECURITY_ID", "END_DATE", "EVENT_TIME", "TRADER_TYPE_CD"]
    x = x.loc[pd.to_numeric(x["TRADER_RANK"], errors="coerce").le(5) & x["RATIO"].ge(0)]
    agg = x.groupby(keys, as_index=False).agg(CONC=("RATIO", "sum"), TOP=("RATIO", "max"))
    agg["CONC"] = agg["CONC"].where(agg["CONC"].le(1.5), agg["CONC"] / 100.0).clip(0, 1)
    wide = agg.pivot_table(index=["SECURITY_ID", "END_DATE", "EVENT_TIME"], columns="TRADER_TYPE_CD", values="CONC", aggfunc="last").reset_index()
    wide = wide.rename(columns={1: "CUSTOMER", 2: "SUPPLIER", "1": "CUSTOMER", "2": "SUPPLIER"})
    for c in ["CUSTOMER", "SUPPLIER"]:
        if c not in wide: wide[c] = np.nan
    wide = wide.sort_values(["SECURITY_ID", "END_DATE", "EVENT_TIME"]).drop_duplicates(["SECURITY_ID", "END_DATE"], keep="last")
    wide = wide.merge(annual_lag(wide, ["CUSTOMER", "SUPPLIER"]), on=["SECURITY_ID", "END_DATE"], how="left", validate="one_to_one")
    wide[FAMILIES["trader"][0]] = -wide["CUSTOMER"]
    wide[FAMILIES["trader"][1]] = -(wide["CUSTOMER"] - wide["L1_CUSTOMER"])
    wide[FAMILIES["trader"][2]] = -wide["SUPPLIER"]
    wide[FAMILIES["trader"][3]] = -(wide["SUPPLIER"] - wide["L1_SUPPLIER"])
    return wide[["SECURITY_ID", "END_DATE", "EVENT_TIME", *FAMILIES["trader"]]]


def subsidy_events(root: Path, pair: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    cols = ["PARTY_ID", "TICKER_SYMBOL", "SCANNED_TIME", "PUBLISH_DATE", "END_DATE", "END_DATE_REP",
            "REPORT_TYPE", "MERGED_FLAG", "ADJUSTED_FLAG", "GOV_GRANTS", "NR_PROFIT_LOSS_SUBTOTAL", "NR_PROFIT_LOSS_TOTAL"]
    x = common_clean(map_party(read_partitioned(root, "fdmt_nr_profit_loss", cols), pair), "PUBLISH_DATE")
    for c in ["GOV_GRANTS", "NR_PROFIT_LOSS_SUBTOTAL", "NR_PROFIT_LOSS_TOTAL"]: x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.sort_values(["SECURITY_ID", "END_DATE", "EVENT_TIME"]).drop_duplicates(["SECURITY_ID", "END_DATE"], keep="first")
    x = attach_assets(x, assets)
    x = x.merge(annual_lag(x, ["GOV_GRANTS", "T_ASSETS"]), on=["SECURITY_ID", "END_DATE"], how="left", validate="one_to_one")
    nr = x["NR_PROFIT_LOSS_TOTAL"].where(x["NR_PROFIT_LOSS_TOTAL"].notna(), x["NR_PROFIT_LOSS_SUBTOTAL"])
    x[FAMILIES["subsidy"][0]] = -ratio(x["GOV_GRANTS"], x["T_ASSETS"]).clip(-5, 5)
    x[FAMILIES["subsidy"][1]] = -ratio(x["GOV_GRANTS"].abs(), nr.abs()).clip(0, 5)
    x[FAMILIES["subsidy"][2]] = -ratio(x["GOV_GRANTS"] - x["L1_GOV_GRANTS"], x["L1_T_ASSETS"]).clip(-5, 5)
    return x[["SECURITY_ID", "END_DATE", "EVENT_TIME", *FAMILIES["subsidy"]]]


def map_daily(events: pd.DataFrame, factors: list[str], panel_path: Path, output: Path, coverage_path: Path) -> None:
    events = events.dropna(subset=["SECURITY_ID"]).copy()
    events["SECURITY_ID"] = pd.to_numeric(events["SECURITY_ID"], errors="raise").astype("int64")
    dates = pd.read_parquet(panel_path, columns=["TRADE_DATE"])["TRADE_DATE"]
    calendar = pd.DatetimeIndex(pd.to_datetime(dates).unique()).normalize().sort_values()
    available = assign_available_trade_date(events, calendar).sort_values(["SECURITY_ID", "AVAILABLE_DATE", "EVENT_TIME", "END_DATE"])
    newest = available.groupby("SECURITY_ID", sort=False)["END_DATE"].cummax()
    available = available.loc[available["END_DATE"].eq(newest)].drop_duplicates(["SECURITY_ID", "AVAILABLE_DATE"], keep="last")
    available = available[["SECURITY_ID", "AVAILABLE_DATE", *factors]].sort_values(["SECURITY_ID", "AVAILABLE_DATE"])
    available[factors] = available.groupby("SECURITY_ID", sort=False)[factors].ffill()
    chunks, coverage = [], []
    for year in sorted(set(calendar.year)):
        panel = _normalize_panel(pd.read_parquet(panel_path, columns=KEYS, filters=[("TRADE_DATE", ">=", pd.Timestamp(year, 1, 1)), ("TRADE_DATE", "<=", pd.Timestamp(year, 12, 31))]))
        mapped = pd.merge_asof(panel.sort_values(["TRADE_DATE", "SECURITY_ID"]), available.sort_values(["AVAILABLE_DATE", "SECURITY_ID"]), by="SECURITY_ID", left_on="TRADE_DATE", right_on="AVAILABLE_DATE", direction="backward")
        mapped = mapped[KEYS + factors]
        mapped[factors] = mapped[factors].astype("float32")
        chunks.append(mapped)
        coverage.extend({"year": year, "factor": f, "missing_rate": float(mapped[f].isna().mean()), "nonmissing": int(mapped[f].notna().sum())} for f in factors)
        print(output.stem, year, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(chunks).sort_values(KEYS).to_parquet(output, index=False, compression="zstd")
    pd.DataFrame(coverage).to_csv(coverage_path, index=False, encoding="utf-8-sig")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--pit-root", type=Path, required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--families", nargs="+", choices=sorted(FAMILIES), default=list(FAMILIES))
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    pair, _ = security_mapping(a.pit_root)
    assets = read_balance_assets(a.pit_root)
    builders = {
        "related_party": lambda: related_events(a.source_root, pair, assets),
        "guarantee": lambda: guarantee_events(a.source_root, pair, assets),
        "inventory": lambda: inventory_events(a.source_root, pair),
        "trader": lambda: trader_events(a.source_root, pair),
        "subsidy": lambda: subsidy_events(a.source_root, pair, assets),
    }
    for family in a.families:
        build = builders[family]
        events = build()
        events.to_parquet(a.output_dir / f"{family}_events.parquet", index=False, compression="zstd")
        map_daily(events, FAMILIES[family], a.panel, a.output_dir / f"{family}_sparse.parquet", a.output_dir / f"{family}_coverage.csv")


if __name__ == "__main__":
    main()
