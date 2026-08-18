"""Build market-cap-scaled buyback and shareholder trading event factors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from twenty_third_alternative_event_factors import KEYS, prepare_available, read_partitioned
from twenty_fourth_event_accumulation_factors import ewm_state


def read_market(root: Path) -> pd.DataFrame:
    pieces = [
        pd.read_parquet(path, columns=["TRADE_DATE", "SECURITY_ID", "MARKET_VALUE_A"])
        for path in sorted((root / "market_daily").rglob("*.parquet"))
    ]
    data = pd.concat(pieces, ignore_index=True)
    data["TRADE_DATE"] = pd.to_datetime(data["TRADE_DATE"]).dt.normalize()
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    data["MARKET_VALUE_A"] = pd.to_numeric(data["MARKET_VALUE_A"], errors="coerce")
    return data.dropna().drop_duplicates(KEYS, keep="last")


def attach_market(events: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    left = events.sort_values(["EVENT_TIME", "SECURITY_ID"])
    right = market.rename(columns={"TRADE_DATE": "MARKET_DATE"}).sort_values(
        ["MARKET_DATE", "SECURITY_ID"]
    )
    return pd.merge_asof(
        left, right, left_on="EVENT_TIME", right_on="MARKET_DATE",
        by="SECURITY_ID", direction="backward", allow_exact_matches=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alternative-root", type=Path, required=True)
    parser.add_argument("--market-root", type=Path, required=True)
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
    market = read_market(args.market_root)

    buyback = read_partitioned(
        args.alternative_root, "equ_share_buy_back",
        ["ID", "SECURITY_ID", "PRE_PUB_DATE", "PUBLISH_DATE", "VALUE_UPL", "VALUE_LL", "BUY_BACK_VALUE"],
    )
    for column in ["PRE_PUB_DATE", "PUBLISH_DATE"]:
        buyback[column] = pd.to_datetime(buyback[column], errors="coerce").dt.normalize()
    for column in ["VALUE_UPL", "VALUE_LL", "BUY_BACK_VALUE"]:
        buyback[column] = pd.to_numeric(buyback[column], errors="coerce")
    buyback["SECURITY_ID"] = pd.to_numeric(buyback["SECURITY_ID"], errors="coerce")

    plans = buyback.dropna(subset=["SECURITY_ID", "PRE_PUB_DATE"]).copy()
    plans = plans.sort_values(["ID", "PRE_PUB_DATE", "PUBLISH_DATE"]).drop_duplicates("ID", keep="first")
    plans["EVENT_TIME"] = plans["PRE_PUB_DATE"]
    plans["PLAN_VALUE"] = plans[["VALUE_UPL", "VALUE_LL"]].mean(axis=1).fillna(
        plans["VALUE_UPL"].fillna(plans["VALUE_LL"])
    )
    plans["SECURITY_ID"] = plans["SECURITY_ID"].astype("int64")
    plans = attach_market(plans, market)
    plans["BUYBACK_PLAN_SCALE"] = (plans["PLAN_VALUE"] / plans["MARKET_VALUE_A"]).clip(0, 0.5)

    actual = buyback.dropna(subset=["SECURITY_ID", "PUBLISH_DATE", "BUY_BACK_VALUE"]).copy()
    actual["SECURITY_ID"] = actual["SECURITY_ID"].astype("int64")
    actual = actual.sort_values(["ID", "PUBLISH_DATE"])
    actual["IMPLEMENTED_INCREMENT"] = actual.groupby("ID")["BUY_BACK_VALUE"].diff()
    actual["IMPLEMENTED_INCREMENT"] = actual["IMPLEMENTED_INCREMENT"].fillna(actual["BUY_BACK_VALUE"])
    actual["IMPLEMENTED_INCREMENT"] = actual["IMPLEMENTED_INCREMENT"].clip(lower=0)
    actual["EVENT_TIME"] = actual["PUBLISH_DATE"]
    actual = attach_market(actual, market)
    actual["BUYBACK_IMPLEMENT_SCALE"] = (
        actual["IMPLEMENTED_INCREMENT"] / actual["MARKET_VALUE_A"]
    ).clip(0, 0.5)

    holders = read_partitioned(
        args.alternative_root, "equ_change_plan",
        ["ID", "SECURITY_ID", "FIRST_PUBLISH_DATE", "CHANGE_DIR", "RATIO_UPL", "RATIO_LL", "SH_NAME_TYPE"],
    )
    holders["EVENT_TIME"] = pd.to_datetime(
        holders["FIRST_PUBLISH_DATE"], errors="coerce"
    ).dt.normalize()
    holders["SECURITY_ID"] = pd.to_numeric(holders["SECURITY_ID"], errors="coerce")
    holders = holders.dropna(subset=["SECURITY_ID", "EVENT_TIME"]).copy()
    holders["SECURITY_ID"] = holders["SECURITY_ID"].astype("int64")
    holders = holders.sort_values(["ID", "EVENT_TIME"]).drop_duplicates("ID", keep="first")
    magnitude = holders[["RATIO_UPL", "RATIO_LL"]].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1).abs().div(100).clip(0, 0.5)
    direction = np.where(pd.to_numeric(holders["CHANGE_DIR"], errors="coerce").eq(1), 1.0, -1.0)
    holders["HOLDER_SIGNED_SCALE"] = magnitude * direction
    holder_type = pd.to_numeric(holders["SH_NAME_TYPE"], errors="coerce")
    holders["INSIDER_SIGNED_SCALE"] = holders["HOLDER_SIGNED_SCALE"].where(holder_type.isin([1, 2]))

    plans = plans.groupby(["SECURITY_ID", "EVENT_TIME"], as_index=False)[["BUYBACK_PLAN_SCALE"]].sum(min_count=1)
    actual = actual.groupby(["SECURITY_ID", "EVENT_TIME"], as_index=False)[["BUYBACK_IMPLEMENT_SCALE"]].sum(min_count=1)
    holders = holders.groupby(["SECURITY_ID", "EVENT_TIME"], as_index=False)[
        ["HOLDER_SIGNED_SCALE", "INSIDER_SIGNED_SCALE"]
    ].sum(min_count=1)
    plan_events = prepare_available(plans, calendar, ["BUYBACK_PLAN_SCALE"])
    actual_events = prepare_available(actual, calendar, ["BUYBACK_IMPLEMENT_SCALE"])
    holder_events = prepare_available(holders, calendar, ["HOLDER_SIGNED_SCALE", "INSIDER_SIGNED_SCALE"])
    sparse = plan_events.merge(actual_events, on=KEYS, how="outer").merge(
        holder_events, on=KEYS, how="outer"
    ).sort_values(KEYS)
    sparse.to_parquet(output / "round28_sparse_capital_allocation.parquet", index=False)
    data = panel.merge(sparse, on=KEYS, how="left", validate="one_to_one")
    data = data.sort_values(["SECURITY_ID", "TRADE_DATE"]).reset_index(drop=True)
    data["NET_CAPITAL_ALLOCATION"] = (
        data["BUYBACK_PLAN_SCALE"].fillna(0)
        + data["BUYBACK_IMPLEMENT_SCALE"].fillna(0)
        + data["HOLDER_SIGNED_SCALE"].fillna(0)
    )
    specifications = [
        ("r28_buyback_plan_scale_hl20", "BUYBACK_PLAN_SCALE", 20),
        ("r28_buyback_plan_scale_hl60", "BUYBACK_PLAN_SCALE", 60),
        ("r28_buyback_implement_scale_hl20", "BUYBACK_IMPLEMENT_SCALE", 20),
        ("r28_buyback_implement_scale_hl60", "BUYBACK_IMPLEMENT_SCALE", 60),
        ("r28_holder_signed_scale_hl20", "HOLDER_SIGNED_SCALE", 20),
        ("r28_holder_signed_scale_hl60", "HOLDER_SIGNED_SCALE", 60),
        ("r28_insider_signed_scale_hl60", "INSIDER_SIGNED_SCALE", 60),
        ("r28_net_capital_allocation_hl20", "NET_CAPITAL_ALLOCATION", 20),
        ("r28_net_capital_allocation_hl60", "NET_CAPITAL_ALLOCATION", 60),
    ]
    result = data[KEYS].copy()
    for name, source, half_life in specifications:
        result[name] = ewm_state(data, source, half_life)
        print(name, flush=True)
    result.to_parquet(output / "round28_capital_allocation_factors.parquet", index=False)
    metadata = {
        "buyback_plans": int(plans["BUYBACK_PLAN_SCALE"].notna().sum()),
        "buyback_implementation_updates": int(actual["BUYBACK_IMPLEMENT_SCALE"].notna().sum()),
        "holder_plans": int(holders["HOLDER_SIGNED_SCALE"].notna().sum()),
        "factor_columns": [name for name, _, _ in specifications],
        "uses_rank": False,
        "uses_label": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
