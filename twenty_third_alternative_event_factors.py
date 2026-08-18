"""Build independent event factors from the alternative DataYes bundle.

No label or existing successful factor is read.  Raw event observations are
saved before economically justified time decay.  Dense values use zero for
"no active news", not cross-sectional imputation or percentile ranks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pead_sue_factor import assign_available_trade_date


KEYS = ["TRADE_DATE", "SECURITY_ID"]


def read_partitioned(root: Path, table: str, columns: list[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in sorted((root / table).glob("*.parquet")):
        names = pq.read_schema(path).names
        if all(column in names for column in columns):
            pieces.append(pd.read_parquet(path, columns=columns))
    if not pieces:
        return pd.DataFrame(columns=columns)
    return pd.concat(pieces, ignore_index=True)


def security_mapping(pit_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    for path in sorted((pit_root / "new_pit_balance").rglob("*.parquet")):
        names = pq.read_schema(path).names
        wanted = ["SECURITY_ID", "PARTY_ID", "TICKER_SYMBOL"]
        if all(column in names for column in wanted):
            pieces.append(pd.read_parquet(path, columns=wanted))
    mapping = pd.concat(pieces, ignore_index=True).dropna()
    mapping["TICKER_SYMBOL"] = mapping["TICKER_SYMBOL"].astype(str).str.zfill(6)
    mapping = mapping.drop_duplicates()
    pair = mapping.drop_duplicates(["PARTY_ID", "TICKER_SYMBOL"], keep="last")
    ticker = mapping[["TICKER_SYMBOL", "SECURITY_ID"]].drop_duplicates()
    counts = ticker.groupby("TICKER_SYMBOL")["SECURITY_ID"].transform("nunique")
    ticker = ticker.loc[counts.eq(1)].drop_duplicates("TICKER_SYMBOL")
    return pair, ticker


def map_party(data: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["TICKER_SYMBOL"] = result["TICKER_SYMBOL"].astype(str).str.zfill(6)
    result = result.merge(
        pair, on=["PARTY_ID", "TICKER_SYMBOL"], how="left", validate="many_to_one"
    )
    return result.dropna(subset=["SECURITY_ID"]).assign(
        SECURITY_ID=lambda x: x["SECURITY_ID"].astype("int64")
    )


def event_time(data: pd.DataFrame, column: str, date_only: bool = False) -> pd.Series:
    values = pd.to_datetime(data[column], errors="coerce")
    if date_only:
        values = values.dt.normalize() + pd.Timedelta(1439, unit="m")
    return values


def prepare_available(
    data: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    value_columns: list[str],
) -> pd.DataFrame:
    clean = data.dropna(subset=["SECURITY_ID", "EVENT_TIME"]).copy()
    clean["SECURITY_ID"] = pd.to_numeric(
        clean["SECURITY_ID"], errors="raise"
    ).astype("int64")
    available = assign_available_trade_date(clean, calendar)
    available = available.rename(columns={"AVAILABLE_DATE": "TRADE_DATE"})
    available = available.sort_values(["TRADE_DATE", "SECURITY_ID", "EVENT_TIME"])
    return (
        available.groupby(KEYS, as_index=False)[value_columns]
        .last()
        .sort_values(KEYS)
        .reset_index(drop=True)
    )


def dense_decay(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    source_columns: list[str],
    specifications: list[tuple[str, str, int]],
) -> pd.DataFrame:
    right = events.rename(columns={"TRADE_DATE": "EVENT_DATE"}).copy()
    right = right.sort_values(["EVENT_DATE", "SECURITY_ID"])
    left = panel.sort_values(["TRADE_DATE", "SECURITY_ID"])
    merged = pd.merge_asof(
        left,
        right,
        left_on="TRADE_DATE",
        right_on="EVENT_DATE",
        by="SECURITY_ID",
        direction="backward",
        allow_exact_matches=True,
    )
    age = (merged["TRADE_DATE"] - merged["EVENT_DATE"]).dt.days
    output = merged[KEYS].copy()
    for output_name, source_name, half_life in specifications:
        active = age.ge(0) & age.le(half_life * 6)
        decay = np.exp2(-age.astype("float64") / half_life)
        output[output_name] = (
            pd.to_numeric(merged[source_name], errors="coerce")
            .mul(decay).where(active, 0.0).fillna(0.0).astype("float32")
        )
    return output.sort_values(KEYS).reset_index(drop=True)


def aggregate_last(data: pd.DataFrame, values: list[str]) -> pd.DataFrame:
    data = data.sort_values(["EVENT_TIME", "SECURITY_ID"])
    return data.groupby(["SECURITY_ID", "EVENT_TIME"], as_index=False)[values].last()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alternative-root", type=Path, required=True)
    parser.add_argument("--pit-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel, columns=KEYS)
    panel["TRADE_DATE"] = pd.to_datetime(panel["TRADE_DATE"]).dt.normalize()
    panel["SECURITY_ID"] = panel["SECURITY_ID"].astype("int64")
    panel = panel.drop_duplicates(KEYS).sort_values(KEYS).reset_index(drop=True)
    calendar = pd.DatetimeIndex(panel["TRADE_DATE"].unique()).sort_values()
    pair, ticker = security_mapping(args.pit_root)

    sparse_families: list[pd.DataFrame] = []
    dense_families: list[pd.DataFrame] = []
    diagnostics: dict[str, object] = {}

    # Analyst report-level earnings revisions.
    analyst = read_partitioned(
        args.alternative_root, "rr_profit_adjust_v2",
        ["SEC_CODE", "THIS_WRITE_DATE", "PROFIT_ADJUST_PER", "EPS_ADJUST_PER", "INCOME_ADJUST_PER"],
    )
    analyst["TICKER_SYMBOL"] = analyst["SEC_CODE"].astype(str).str.zfill(6)
    analyst = analyst.merge(ticker, on="TICKER_SYMBOL", how="left", validate="many_to_one")
    analyst["EVENT_TIME"] = event_time(analyst, "THIS_WRITE_DATE", date_only=True)
    for column in ["PROFIT_ADJUST_PER", "EPS_ADJUST_PER", "INCOME_ADJUST_PER"]:
        analyst[column] = pd.to_numeric(analyst[column], errors="coerce").clip(-2, 2)
    analyst["REVISION_BREADTH"] = np.sign(analyst["PROFIT_ADJUST_PER"])
    analyst = analyst.groupby(["SECURITY_ID", "EVENT_TIME"], as_index=False).agg(
        PROFIT_REVISION=("PROFIT_ADJUST_PER", "median"),
        EPS_REVISION=("EPS_ADJUST_PER", "median"),
        INCOME_REVISION=("INCOME_ADJUST_PER", "median"),
        REVISION_BREADTH=("REVISION_BREADTH", "mean"),
    )
    analyst_avail = prepare_available(
        analyst, calendar,
        ["PROFIT_REVISION", "EPS_REVISION", "INCOME_REVISION", "REVISION_BREADTH"],
    )
    sparse_families.append(analyst_avail)
    dense_families.append(dense_decay(panel, analyst_avail,
        ["PROFIT_REVISION", "EPS_REVISION", "INCOME_REVISION", "REVISION_BREADTH"], [
            ("r23_analyst_profit_revision_hl20", "PROFIT_REVISION", 20),
            ("r23_analyst_profit_revision_hl60", "PROFIT_REVISION", 60),
            ("r23_analyst_eps_revision_hl20", "EPS_REVISION", 20),
            ("r23_analyst_revision_breadth_hl20", "REVISION_BREADTH", 20),
        ]))

    # Management performance guidance.
    guidance = read_partitioned(
        args.alternative_root, "fdmt_ef_v2",
        ["PARTY_ID", "TICKER_SYMBOL", "ACT_PUBTIME", "N_INCAP_CHGR_LL", "N_INCAP_CHGR_UPL",
         "N_INCOME_CHGR_LL", "N_INCOME_CHGR_UPL"],
    )
    guidance = map_party(guidance, pair)
    guidance["EVENT_TIME"] = event_time(guidance, "ACT_PUBTIME")
    attr = guidance[["N_INCAP_CHGR_LL", "N_INCAP_CHGR_UPL"]].mean(axis=1)
    total = guidance[["N_INCOME_CHGR_LL", "N_INCOME_CHGR_UPL"]].mean(axis=1)
    guidance["GUIDANCE_YOY"] = attr.fillna(total).div(100).clip(-5, 5)
    guidance = aggregate_last(guidance, ["GUIDANCE_YOY"])
    guidance_avail = prepare_available(guidance, calendar, ["GUIDANCE_YOY"])
    sparse_families.append(guidance_avail)
    dense_families.append(dense_decay(panel, guidance_avail, ["GUIDANCE_YOY"], [
        ("r23_guidance_profit_yoy_hl20", "GUIDANCE_YOY", 20),
        ("r23_guidance_profit_yoy_hl60", "GUIDANCE_YOY", 60),
    ]))

    # Express-report earnings growth.
    express = read_partitioned(
        args.alternative_root, "fdmt_ee",
        ["PARTY_ID", "TICKER_SYMBOL", "ACT_PUBTIME", "N_INCOME_ATTR_P_YOY", "N_INCOME_CUT_YOY"],
    )
    express = map_party(express, pair)
    express["EVENT_TIME"] = event_time(express, "ACT_PUBTIME")
    express["EXPRESS_YOY"] = pd.to_numeric(express["N_INCOME_ATTR_P_YOY"], errors="coerce").div(100).clip(-5, 5)
    express["EXPRESS_CUT_YOY"] = pd.to_numeric(express["N_INCOME_CUT_YOY"], errors="coerce").div(100).clip(-5, 5)
    express = aggregate_last(express, ["EXPRESS_YOY", "EXPRESS_CUT_YOY"])
    express_avail = prepare_available(express, calendar, ["EXPRESS_YOY", "EXPRESS_CUT_YOY"])
    sparse_families.append(express_avail)
    dense_families.append(dense_decay(panel, express_avail, ["EXPRESS_YOY", "EXPRESS_CUT_YOY"], [
        ("r23_express_profit_yoy_hl20", "EXPRESS_YOY", 20),
        ("r23_express_cut_profit_yoy_hl20", "EXPRESS_CUT_YOY", 20),
    ]))

    # Audit opinion and key-audit-matter complexity.
    audit = read_partitioned(
        args.alternative_root, "fdmt_adt_opn_n",
        ["PARTY_ID", "TICKER_SYMBOL", "PUBLISH_DATE", "DM_OPN_TYPE", "IC_OPN_TYPE"],
    )
    audit = map_party(audit, pair)
    audit["EVENT_TIME"] = event_time(audit, "PUBLISH_DATE", date_only=True)
    audit["AUDIT_CLEAN"] = np.where(audit["DM_OPN_TYPE"].astype(str).eq("1"), 1.0, -1.0)
    audit["IC_AUDIT_CLEAN"] = np.where(audit["IC_OPN_TYPE"].astype(str).eq("1"), 1.0, -1.0)
    audit = aggregate_last(audit, ["AUDIT_CLEAN", "IC_AUDIT_CLEAN"])
    audit_avail = prepare_available(audit, calendar, ["AUDIT_CLEAN", "IC_AUDIT_CLEAN"])
    sparse_families.append(audit_avail)
    dense_families.append(dense_decay(panel, audit_avail, ["AUDIT_CLEAN", "IC_AUDIT_CLEAN"], [
        ("r23_audit_opinion_quality_hl120", "AUDIT_CLEAN", 120),
        ("r23_internal_control_audit_hl120", "IC_AUDIT_CLEAN", 120),
    ]))

    matters = read_partitioned(
        args.alternative_root, "fdmt_main_adt_matters",
        ["PARTY_ID", "TICKER_SYMBOL", "PUBLISH_DATE", "AUDIT_PROJECT"],
    )
    matters = map_party(matters, pair)
    matters["EVENT_TIME"] = event_time(matters, "PUBLISH_DATE", date_only=True)
    matters = matters.groupby(["SECURITY_ID", "EVENT_TIME"], as_index=False).agg(
        KAM_COMPLEXITY=("AUDIT_PROJECT", "count")
    )
    matters["KAM_COMPLEXITY"] = -np.log1p(matters["KAM_COMPLEXITY"])
    matters_avail = prepare_available(matters, calendar, ["KAM_COMPLEXITY"])
    sparse_families.append(matters_avail)
    dense_families.append(dense_decay(panel, matters_avail, ["KAM_COMPLEXITY"], [
        ("r23_low_key_audit_complexity_hl120", "KAM_COMPLEXITY", 120),
    ]))

    # Major contracts, shareholder plans, and buybacks.
    contracts = read_partitioned(
        args.alternative_root, "equ_major_contract_pit",
        ["PARTY_ID", "TICKER_SYMBOL", "PUBLISH_DATE", "BIDDING_AMOUNT", "CONTR_TOTAL_UPL", "CONTR_TOTAL_LOL"],
    )
    contracts = map_party(contracts, pair)
    contracts["EVENT_TIME"] = event_time(contracts, "PUBLISH_DATE", date_only=True)
    ratio = pd.to_numeric(contracts["BIDDING_AMOUNT"], errors="coerce").abs()
    contracts["CONTRACT_SCALE"] = np.log1p(ratio.clip(0, 1000))
    contracts = contracts.groupby(["SECURITY_ID", "EVENT_TIME"], as_index=False).agg(
        CONTRACT_SCALE=("CONTRACT_SCALE", "max")
    )
    contract_avail = prepare_available(contracts, calendar, ["CONTRACT_SCALE"])
    sparse_families.append(contract_avail)
    dense_families.append(dense_decay(panel, contract_avail, ["CONTRACT_SCALE"], [
        ("r23_major_contract_scale_hl20", "CONTRACT_SCALE", 20),
        ("r23_major_contract_scale_hl60", "CONTRACT_SCALE", 60),
    ]))

    holders = read_partitioned(
        args.alternative_root, "equ_change_plan",
        ["SECURITY_ID", "FIRST_PUBLISH_DATE", "PUBLISH_DATE", "CHANGE_DIR", "RATIO_UPL", "RATIO_LL"],
    )
    holders["EVENT_TIME"] = event_time(holders, "FIRST_PUBLISH_DATE", date_only=True)
    fallback = event_time(holders, "PUBLISH_DATE", date_only=True)
    holders["EVENT_TIME"] = holders["EVENT_TIME"].fillna(fallback)
    direction = np.where(pd.to_numeric(holders["CHANGE_DIR"], errors="coerce").eq(1), 1.0, -1.0)
    magnitude = holders[["RATIO_UPL", "RATIO_LL"]].mean(axis=1).abs().fillna(1.0).div(100)
    holders["HOLDER_PLAN"] = direction * np.log1p(magnitude.clip(0, 1) * 100)
    holders = holders.groupby(["SECURITY_ID", "EVENT_TIME"], as_index=False).agg(
        HOLDER_PLAN=("HOLDER_PLAN", "sum")
    )
    holder_avail = prepare_available(holders, calendar, ["HOLDER_PLAN"])
    sparse_families.append(holder_avail)
    dense_families.append(dense_decay(panel, holder_avail, ["HOLDER_PLAN"], [
        ("r23_holder_change_plan_hl20", "HOLDER_PLAN", 20),
        ("r23_holder_change_plan_hl60", "HOLDER_PLAN", 60),
    ]))

    buyback = read_partitioned(
        args.alternative_root, "equ_share_buy_back",
        ["SECURITY_ID", "PRE_PUB_DATE", "PUBLISH_DATE", "VALUE_UPL", "VALUE_LL"],
    )
    buyback["EVENT_TIME"] = event_time(buyback, "PRE_PUB_DATE", date_only=True)
    buyback["EVENT_TIME"] = buyback["EVENT_TIME"].fillna(event_time(buyback, "PUBLISH_DATE", date_only=True))
    amount = buyback[["VALUE_UPL", "VALUE_LL"]].mean(axis=1).abs()
    buyback["BUYBACK_PLAN"] = np.log1p(amount.clip(lower=0))
    buyback = buyback.groupby(["SECURITY_ID", "EVENT_TIME"], as_index=False).agg(
        BUYBACK_PLAN=("BUYBACK_PLAN", "max")
    )
    buyback_avail = prepare_available(buyback, calendar, ["BUYBACK_PLAN"])
    sparse_families.append(buyback_avail)
    dense_families.append(dense_decay(panel, buyback_avail, ["BUYBACK_PLAN"], [
        ("r23_buyback_plan_hl20", "BUYBACK_PLAN", 20),
        ("r23_buyback_plan_hl60", "BUYBACK_PLAN", 60),
    ]))

    sparse = sparse_families[0]
    for family in sparse_families[1:]:
        sparse = sparse.merge(family, on=KEYS, how="outer", validate="one_to_one")
    sparse = sparse.sort_values(KEYS).reset_index(drop=True)
    sparse.to_parquet(output / "round23_sparse_events.parquet", index=False)

    dense = dense_families[0]
    for family in dense_families[1:]:
        dense = dense.merge(family, on=KEYS, how="inner", validate="one_to_one")
    factor_columns = [column for column in dense.columns if column not in KEYS]
    dense.to_parquet(output / "round23_dense_event_factors.parquet", index=False)
    diagnostics.update({
        "panel_rows": len(panel), "sparse_rows": len(sparse),
        "factor_columns": factor_columns,
        "sparse_non_null": {column: int(sparse[column].notna().sum()) for column in sparse.columns if column not in KEYS},
        "dense_missing": {column: int(dense[column].isna().sum()) for column in factor_columns},
    })
    (output / "metadata.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
