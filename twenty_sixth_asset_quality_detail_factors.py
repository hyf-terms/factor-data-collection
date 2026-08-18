"""Build dense asset-quality factors from accounting-note details."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from twenty_fourth_event_accumulation_factors import ewm_state
from twenty_third_alternative_event_factors import (
    KEYS, event_time, map_party, prepare_available, read_partitioned,
    security_mapping,
)


def read_balance_assets(root: Path) -> pd.DataFrame:
    columns = [
        "SECURITY_ID", "ACT_PUBTIME", "END_DATE", "END_DATE_REP",
        "MERGED_FLAG", "REPORT_TYPE", "T_ASSETS",
    ]
    pieces = []
    for path in sorted((root / "new_pit_balance").rglob("*.parquet")):
        names = pq.read_schema(path).names
        if all(column in names for column in columns):
            pieces.append(pd.read_parquet(path, columns=columns))
    data = pd.concat(pieces, ignore_index=True)
    data["EVENT_TIME"] = pd.to_datetime(data["ACT_PUBTIME"], errors="coerce")
    data["END_DATE"] = pd.to_datetime(data["END_DATE"], errors="coerce").dt.normalize()
    data["END_DATE_REP"] = pd.to_datetime(data["END_DATE_REP"], errors="coerce").dt.normalize()
    data["T_ASSETS"] = pd.to_numeric(data["T_ASSETS"], errors="coerce")
    data = data.loc[
        data["MERGED_FLAG"].astype(str).eq("1")
        & data["END_DATE"].eq(data["END_DATE_REP"])
    ].dropna(subset=["SECURITY_ID", "EVENT_TIME", "END_DATE", "T_ASSETS"])
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    return (
        data.sort_values(["SECURITY_ID", "END_DATE", "EVENT_TIME"])
        .drop_duplicates(["SECURITY_ID", "END_DATE"], keep="first")
        [["SECURITY_ID", "END_DATE", "EVENT_TIME", "T_ASSETS"]]
    )


def attach_assets(events: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    merged = events.merge(
        assets, on=["SECURITY_ID", "END_DATE"], how="left",
        suffixes=("", "_ASSET"), validate="many_to_one",
    )
    merged["EVENT_TIME"] = merged[["EVENT_TIME", "EVENT_TIME_ASSET"]].max(axis=1)
    return merged.drop(columns="EVENT_TIME_ASSET")


def aging_metrics(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    data = data.copy()
    data["EVENT_TIME"] = event_time(data, "ACT_PUBTIME")
    data["END_DATE"] = pd.to_datetime(data["END_DATE"], errors="coerce").dt.normalize()
    data["B_BALANCE"] = pd.to_numeric(data["B_BALANCE"], errors="coerce")
    data["BAD_DEBT_RESERVE"] = pd.to_numeric(data["BAD_DEBT_RESERVE"], errors="coerce")
    data = data.loc[data["MERGED_FLAG"].astype(str).eq("1")].dropna(
        subset=["SECURITY_ID", "EVENT_TIME", "END_DATE"]
    )
    keys = ["SECURITY_ID", "END_DATE", "EVENT_TIME"]
    total = data.loc[data["ITEM_NAME"].astype(str).str.contains("账龄法合计", na=False)]
    total = total.groupby(keys, as_index=False).agg(
        TOTAL_BALANCE=("B_BALANCE", "max"), TOTAL_RESERVE=("BAD_DEBT_RESERVE", "max")
    )
    names = data["ITEM_NAME"].astype(str)
    fine = data.loc[names.isin(["3-4年", "4-5年", "5年以上"])]
    fine = fine.groupby(keys, as_index=False)["B_BALANCE"].sum().rename(columns={"B_BALANCE": "LONG_FINE"})
    broad = data.loc[names.eq("3年以上")].groupby(keys, as_index=False)["B_BALANCE"].max().rename(columns={"B_BALANCE": "LONG_BROAD"})
    medium = data.loc[names.isin(["3-5年", "5年以上"])]
    medium = medium.groupby(keys, as_index=False)["B_BALANCE"].sum().rename(columns={"B_BALANCE": "LONG_MEDIUM"})
    result = total.merge(fine, on=keys, how="left").merge(broad, on=keys, how="left").merge(medium, on=keys, how="left")
    result["LONG_BALANCE"] = result[["LONG_FINE", "LONG_BROAD", "LONG_MEDIUM"]].max(axis=1).fillna(0)
    denominator = result["TOTAL_BALANCE"].abs().clip(lower=1.0)
    result[f"{prefix}_LONG_SHARE"] = (result["LONG_BALANCE"] / denominator).clip(0, 2)
    result[f"{prefix}_BAD_DEBT_RATE"] = (result["TOTAL_RESERVE"] / denominator).clip(0, 2)
    result[f"{prefix}_UNCOVERED_LONG"] = (
        (result["LONG_BALANCE"] - result["TOTAL_RESERVE"]).clip(lower=0) / denominator
    ).clip(0, 2)
    return result[keys + [f"{prefix}_LONG_SHARE", f"{prefix}_BAD_DEBT_RATE", f"{prefix}_UNCOVERED_LONG"]]


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
    panel = panel.drop_duplicates(KEYS)
    calendar = pd.DatetimeIndex(panel["TRADE_DATE"].unique()).sort_values()
    pair, _ = security_mapping(args.pit_root)
    assets = read_balance_assets(args.pit_root)

    aging_columns = [
        "PARTY_ID", "TICKER_SYMBOL", "ACT_PUBTIME", "END_DATE", "MERGED_FLAG",
        "ITEM_NAME", "B_BALANCE", "BAD_DEBT_RESERVE",
    ]
    ar = map_party(read_partitioned(args.alternative_root, "fdmt_acc_rec_age", aging_columns), pair)
    other = map_party(read_partitioned(args.alternative_root, "fdmt_oth_rec_age", aging_columns), pair)
    ar_events = aging_metrics(ar, "AR")
    other_events = aging_metrics(other, "OTHER_AR")

    reserves = read_partitioned(
        args.alternative_root, "fdmt_ass_imp_pre",
        ["PARTY_ID", "TICKER_SYMBOL", "ACT_PUBTIME", "END_DATE", "MERGED_FLAG",
         "TO_NOTES_PRE", "BAD_DEBT_TOT", "S_PRICE_DROP", "GOODWILL_DEV"],
    )
    reserves = map_party(reserves, pair)
    reserves["EVENT_TIME"] = event_time(reserves, "ACT_PUBTIME")
    reserves["END_DATE"] = pd.to_datetime(reserves["END_DATE"], errors="coerce").dt.normalize()
    reserves = reserves.loc[reserves["MERGED_FLAG"].astype(str).eq("1")]
    reserves = attach_assets(reserves, assets)
    for column in ["TO_NOTES_PRE", "BAD_DEBT_TOT", "S_PRICE_DROP", "GOODWILL_DEV"]:
        reserves[column] = pd.to_numeric(reserves[column], errors="coerce")
    base = reserves["T_ASSETS"].abs().clip(lower=1.0)
    reserves["TOTAL_RESERVE_ASSETS"] = -(reserves["TO_NOTES_PRE"] / base).clip(-2, 2)
    reserves["BAD_DEBT_RESERVE_ASSETS"] = -(reserves["BAD_DEBT_TOT"] / base).clip(-2, 2)
    reserves["INVENTORY_RESERVE_ASSETS"] = -(reserves["S_PRICE_DROP"] / base).clip(-2, 2)
    reserves["GOODWILL_RESERVE_ASSETS"] = -(reserves["GOODWILL_DEV"] / base).clip(-2, 2)
    reserve_values = ["TOTAL_RESERVE_ASSETS", "BAD_DEBT_RESERVE_ASSETS", "INVENTORY_RESERVE_ASSETS", "GOODWILL_RESERVE_ASSETS"]
    reserves = reserves.groupby(["SECURITY_ID", "EVENT_TIME"], as_index=False)[reserve_values].last()

    losses = read_partitioned(
        args.alternative_root, "fdmt_ass_imp_lossv2",
        ["PARTY_ID", "TICKER_SYMBOL", "ACT_PUBTIME", "END_DATE", "MERGED_FLAG", "ASS_DEV_T", "IFP_LO", "BP_LO"],
    )
    losses = map_party(losses, pair)
    losses["EVENT_TIME"] = event_time(losses, "ACT_PUBTIME")
    losses["END_DATE"] = pd.to_datetime(losses["END_DATE"], errors="coerce").dt.normalize()
    losses = losses.loc[losses["MERGED_FLAG"].astype(str).eq("1")]
    losses = attach_assets(losses, assets)
    base = losses["T_ASSETS"].abs().clip(lower=1.0)
    losses["LOW_IMPAIRMENT_LOSS_ASSETS"] = -(pd.to_numeric(losses["ASS_DEV_T"], errors="coerce") / base).clip(-2, 2)
    losses = losses.groupby(["SECURITY_ID", "EVENT_TIME"], as_index=False)[["LOW_IMPAIRMENT_LOSS_ASSETS"]].last()

    contracts = read_partitioned(
        args.alternative_root, "fdmt_con_lab_na",
        ["PARTY_ID", "TICKER_SYMBOL", "ACT_PUBTIME", "END_DATE", "MERGED_FLAG", "TOT_CON_LIA"],
    )
    contracts = map_party(contracts, pair)
    contracts["EVENT_TIME"] = event_time(contracts, "ACT_PUBTIME")
    contracts["END_DATE"] = pd.to_datetime(contracts["END_DATE"], errors="coerce").dt.normalize()
    contracts = contracts.loc[contracts["MERGED_FLAG"].astype(str).eq("1")]
    contracts = attach_assets(contracts, assets)
    contracts["CONTRACT_LIABILITY_ASSETS"] = (
        pd.to_numeric(contracts["TOT_CON_LIA"], errors="coerce")
        / contracts["T_ASSETS"].abs().clip(lower=1.0)
    ).clip(-2, 2)
    contracts = contracts.sort_values(["SECURITY_ID", "END_DATE", "EVENT_TIME"])
    contracts["CONTRACT_LIABILITY_CHANGE"] = contracts.groupby("SECURITY_ID")["CONTRACT_LIABILITY_ASSETS"].diff(4)
    contracts = contracts.groupby(["SECURITY_ID", "EVENT_TIME"], as_index=False)[["CONTRACT_LIABILITY_ASSETS", "CONTRACT_LIABILITY_CHANGE"]].last()

    sparse_families = []
    for frame, values in [
        (ar_events, ["AR_LONG_SHARE", "AR_BAD_DEBT_RATE", "AR_UNCOVERED_LONG"]),
        (other_events, ["OTHER_AR_LONG_SHARE", "OTHER_AR_BAD_DEBT_RATE", "OTHER_AR_UNCOVERED_LONG"]),
        (reserves, reserve_values),
        (losses, ["LOW_IMPAIRMENT_LOSS_ASSETS"]),
        (contracts, ["CONTRACT_LIABILITY_ASSETS", "CONTRACT_LIABILITY_CHANGE"]),
    ]:
        sparse_families.append(prepare_available(frame, calendar, values))
    sparse = sparse_families[0]
    for family in sparse_families[1:]:
        sparse = sparse.merge(family, on=KEYS, how="outer", validate="one_to_one")
    sparse = sparse.sort_values(KEYS).reset_index(drop=True)
    sparse.to_parquet(output / "round26_sparse_asset_quality.parquet", index=False)

    data = panel.merge(sparse, on=KEYS, how="left", validate="one_to_one")
    data = data.sort_values(["SECURITY_ID", "TRADE_DATE"]).reset_index(drop=True)
    specifications = [
        ("r26_low_ar_long_age_hl120", "AR_LONG_SHARE", 120, -1),
        ("r26_ar_reserve_coverage_hl120", "AR_BAD_DEBT_RATE", 120, 1),
        ("r26_low_ar_uncovered_long_hl120", "AR_UNCOVERED_LONG", 120, -1),
        ("r26_low_other_ar_long_age_hl120", "OTHER_AR_LONG_SHARE", 120, -1),
        ("r26_low_other_ar_uncovered_hl120", "OTHER_AR_UNCOVERED_LONG", 120, -1),
        ("r26_low_total_reserve_assets_hl120", "TOTAL_RESERVE_ASSETS", 120, 1),
        ("r26_low_bad_debt_reserve_assets_hl120", "BAD_DEBT_RESERVE_ASSETS", 120, 1),
        ("r26_low_inventory_reserve_assets_hl120", "INVENTORY_RESERVE_ASSETS", 120, 1),
        ("r26_low_goodwill_reserve_assets_hl120", "GOODWILL_RESERVE_ASSETS", 120, 1),
        ("r26_low_impairment_loss_assets_hl120", "LOW_IMPAIRMENT_LOSS_ASSETS", 120, 1),
        ("r26_contract_liability_assets_hl120", "CONTRACT_LIABILITY_ASSETS", 120, 1),
        ("r26_contract_liability_change_hl120", "CONTRACT_LIABILITY_CHANGE", 120, 1),
    ]
    result = data[KEYS].copy()
    for output_name, source, half_life, sign in specifications:
        result[output_name] = sign * ewm_state(data, source, half_life)
    result = result.sort_values(KEYS).reset_index(drop=True)
    result.to_parquet(output / "round26_asset_quality_factors.parquet", index=False)
    metadata = {
        "sparse_rows": len(sparse),
        "factor_columns": [name for name, _, _, _ in specifications],
        "sparse_non_null": {column: int(sparse[column].notna().sum()) for column in sparse.columns if column not in KEYS},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
