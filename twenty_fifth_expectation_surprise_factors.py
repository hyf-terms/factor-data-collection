"""Build guidance/express surprises relative to prior analyst expectations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from twenty_fourth_event_accumulation_factors import ewm_state
from twenty_third_alternative_event_factors import (
    KEYS,
    event_time,
    map_party,
    prepare_available,
    read_partitioned,
    security_mapping,
)


def prior_consensus(events: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    left = events.sort_values(["EVENT_TIME", "SECURITY_ID", "FORE_YEAR"])
    right = consensus.sort_values(["CONSENSUS_TIME", "SECURITY_ID", "FORE_YEAR"])
    return pd.merge_asof(
        left, right,
        left_on="EVENT_TIME", right_on="CONSENSUS_TIME",
        by=["SECURITY_ID", "FORE_YEAR"], direction="backward",
        allow_exact_matches=True,
    )


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
    pair, ticker = security_mapping(args.pit_root)

    analyst = read_partitioned(
        args.alternative_root, "rr_profit_adjust_v2",
        ["SEC_CODE", "FORE_YEAR", "THIS_WRITE_DATE", "THIS_PROFIT"],
    )
    analyst["TICKER_SYMBOL"] = analyst["SEC_CODE"].astype(str).str.zfill(6)
    analyst = analyst.merge(ticker, on="TICKER_SYMBOL", how="left", validate="many_to_one")
    analyst["FORE_YEAR"] = pd.to_numeric(analyst["FORE_YEAR"], errors="coerce")
    analyst["CONSENSUS_TIME"] = event_time(analyst, "THIS_WRITE_DATE", date_only=True)
    analyst["PROFIT_FORECAST"] = pd.to_numeric(analyst["THIS_PROFIT"], errors="coerce") * 10_000
    analyst = analyst.dropna(subset=["SECURITY_ID", "FORE_YEAR", "CONSENSUS_TIME", "PROFIT_FORECAST"])
    analyst["SECURITY_ID"] = analyst["SECURITY_ID"].astype("int64")
    analyst["FORE_YEAR"] = analyst["FORE_YEAR"].astype("int64")
    consensus = analyst.groupby(
        ["SECURITY_ID", "FORE_YEAR", "CONSENSUS_TIME"], as_index=False
    ).agg(
        CONSENSUS_PROFIT=("PROFIT_FORECAST", "median"),
        CONSENSUS_STD=("PROFIT_FORECAST", "std"),
        ANALYST_COUNT=("PROFIT_FORECAST", "count"),
    )

    guidance = read_partitioned(
        args.alternative_root, "fdmt_ef_v2",
        ["PARTY_ID", "TICKER_SYMBOL", "ACT_PUBTIME", "END_DATE",
         "EXPN_INCAP_LL", "EXPN_INCAP_UPL", "EXPN_INCOME_LL", "EXPN_INCOME_UPL"],
    )
    guidance = map_party(guidance, pair)
    guidance["EVENT_TIME"] = event_time(guidance, "ACT_PUBTIME")
    guidance["END_DATE"] = pd.to_datetime(guidance["END_DATE"], errors="coerce")
    guidance = guidance.loc[
        guidance["END_DATE"].dt.month.eq(12) & guidance["END_DATE"].dt.day.eq(31)
    ].copy()
    guidance["FORE_YEAR"] = guidance["END_DATE"].dt.year.astype("int64")
    attr_mid = guidance[["EXPN_INCAP_LL", "EXPN_INCAP_UPL"]].mean(axis=1)
    total_mid = guidance[["EXPN_INCOME_LL", "EXPN_INCOME_UPL"]].mean(axis=1)
    guidance["GUIDANCE_PROFIT"] = attr_mid.fillna(total_mid)
    guidance = guidance.dropna(subset=["GUIDANCE_PROFIT", "EVENT_TIME"])
    guidance = guidance.sort_values(["SECURITY_ID", "FORE_YEAR", "EVENT_TIME"])
    guidance["PRIOR_GUIDANCE"] = guidance.groupby(["SECURITY_ID", "FORE_YEAR"])["GUIDANCE_PROFIT"].shift()
    guidance["GUIDANCE_REVISION"] = (
        (guidance["GUIDANCE_PROFIT"] - guidance["PRIOR_GUIDANCE"])
        / guidance["PRIOR_GUIDANCE"].abs().clip(lower=1.0)
    ).clip(-3, 3)
    guidance_with_consensus = prior_consensus(guidance, consensus)
    guidance_with_consensus["GUIDANCE_SURPRISE"] = (
        (guidance_with_consensus["GUIDANCE_PROFIT"] - guidance_with_consensus["CONSENSUS_PROFIT"])
        / guidance_with_consensus["CONSENSUS_PROFIT"].abs().clip(lower=1.0)
    ).clip(-3, 3)
    dispersion = (
        guidance_with_consensus["CONSENSUS_STD"]
        / guidance_with_consensus["CONSENSUS_PROFIT"].abs().clip(lower=1.0)
    ).fillna(1.0).clip(0.05, 2.0)
    coverage = np.sqrt(guidance_with_consensus["ANALYST_COUNT"].fillna(0).clip(0, 25) / 5)
    guidance_with_consensus["GUIDANCE_SURPRISE_CONF"] = (
        guidance_with_consensus["GUIDANCE_SURPRISE"] * coverage / dispersion
    ).clip(-10, 10)
    guidance_events = guidance_with_consensus.groupby(
        ["SECURITY_ID", "EVENT_TIME"], as_index=False
    )[["GUIDANCE_SURPRISE", "GUIDANCE_SURPRISE_CONF", "GUIDANCE_REVISION"]].last()
    guidance_avail = prepare_available(
        guidance_events, calendar,
        ["GUIDANCE_SURPRISE", "GUIDANCE_SURPRISE_CONF", "GUIDANCE_REVISION"],
    )

    express = read_partitioned(
        args.alternative_root, "fdmt_ee",
        ["PARTY_ID", "TICKER_SYMBOL", "ACT_PUBTIME", "END_DATE", "N_INCOME_ATTR_P"],
    )
    express = map_party(express, pair)
    express["EVENT_TIME"] = event_time(express, "ACT_PUBTIME")
    express["END_DATE"] = pd.to_datetime(express["END_DATE"], errors="coerce")
    express = express.loc[
        express["END_DATE"].dt.month.eq(12) & express["END_DATE"].dt.day.eq(31)
    ].copy()
    express["FORE_YEAR"] = express["END_DATE"].dt.year.astype("int64")
    express["EXPRESS_PROFIT"] = pd.to_numeric(express["N_INCOME_ATTR_P"], errors="coerce")
    express = express.dropna(subset=["EXPRESS_PROFIT", "EVENT_TIME"])
    express_consensus = prior_consensus(express, consensus)
    express_consensus["EXPRESS_SURPRISE"] = (
        (express_consensus["EXPRESS_PROFIT"] - express_consensus["CONSENSUS_PROFIT"])
        / express_consensus["CONSENSUS_PROFIT"].abs().clip(lower=1.0)
    ).clip(-3, 3)
    express_dispersion = (
        express_consensus["CONSENSUS_STD"]
        / express_consensus["CONSENSUS_PROFIT"].abs().clip(lower=1.0)
    ).fillna(1.0).clip(0.05, 2.0)
    express_coverage = np.sqrt(express_consensus["ANALYST_COUNT"].fillna(0).clip(0, 25) / 5)
    express_consensus["EXPRESS_SURPRISE_CONF"] = (
        express_consensus["EXPRESS_SURPRISE"] * express_coverage / express_dispersion
    ).clip(-10, 10)

    guidance_history = guidance[["SECURITY_ID", "FORE_YEAR", "EVENT_TIME", "GUIDANCE_PROFIT"]].rename(
        columns={"EVENT_TIME": "GUIDANCE_TIME"}
    ).sort_values(["GUIDANCE_TIME", "SECURITY_ID", "FORE_YEAR"])
    express_sorted = express_consensus.sort_values(["EVENT_TIME", "SECURITY_ID", "FORE_YEAR"])
    express_increment = pd.merge_asof(
        express_sorted, guidance_history,
        left_on="EVENT_TIME", right_on="GUIDANCE_TIME",
        by=["SECURITY_ID", "FORE_YEAR"], direction="backward", allow_exact_matches=True,
    )
    express_increment["EXPRESS_VS_GUIDANCE"] = (
        (express_increment["EXPRESS_PROFIT"] - express_increment["GUIDANCE_PROFIT"])
        / express_increment["GUIDANCE_PROFIT"].abs().clip(lower=1.0)
    ).clip(-3, 3)
    express_events = express_increment.groupby(
        ["SECURITY_ID", "EVENT_TIME"], as_index=False
    )[["EXPRESS_SURPRISE", "EXPRESS_SURPRISE_CONF", "EXPRESS_VS_GUIDANCE"]].last()
    express_avail = prepare_available(
        express_events, calendar,
        ["EXPRESS_SURPRISE", "EXPRESS_SURPRISE_CONF", "EXPRESS_VS_GUIDANCE"],
    )

    sparse = guidance_avail.merge(express_avail, on=KEYS, how="outer", validate="one_to_one")
    sparse = sparse.sort_values(KEYS).reset_index(drop=True)
    sparse.to_parquet(output / "round25_sparse_expectation_surprises.parquet", index=False)
    data = panel.merge(sparse, on=KEYS, how="left", validate="one_to_one")
    data = data.sort_values(["SECURITY_ID", "TRADE_DATE"]).reset_index(drop=True)
    specifications = [
        ("r25_guidance_surprise_hl5", "GUIDANCE_SURPRISE", 5),
        ("r25_guidance_surprise_hl10", "GUIDANCE_SURPRISE", 10),
        ("r25_guidance_surprise_hl20", "GUIDANCE_SURPRISE", 20),
        ("r25_guidance_surprise_conf_hl10", "GUIDANCE_SURPRISE_CONF", 10),
        ("r25_guidance_revision_hl10", "GUIDANCE_REVISION", 10),
        ("r25_express_surprise_hl5", "EXPRESS_SURPRISE", 5),
        ("r25_express_surprise_hl10", "EXPRESS_SURPRISE", 10),
        ("r25_express_surprise_hl20", "EXPRESS_SURPRISE", 20),
        ("r25_express_surprise_conf_hl10", "EXPRESS_SURPRISE_CONF", 10),
        ("r25_express_vs_guidance_hl10", "EXPRESS_VS_GUIDANCE", 10),
    ]
    result = data[KEYS].copy()
    for output_name, source_name, half_life in specifications:
        result[output_name] = ewm_state(data, source_name, half_life)
    result = result.sort_values(KEYS).reset_index(drop=True)
    result.to_parquet(output / "round25_expectation_surprise_factors.parquet", index=False)
    metadata = {
        "consensus_observations": len(consensus),
        "guidance_surprises": int(guidance_events["GUIDANCE_SURPRISE"].notna().sum()),
        "express_surprises": int(express_events["EXPRESS_SURPRISE"].notna().sum()),
        "factor_columns": [name for name, _, _ in specifications],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
