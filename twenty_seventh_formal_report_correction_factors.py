"""Build formal-report corrections to previously published express/guidance values.

The signal uses only information available at the formal report timestamp.  It
does not read labels, successful factors, or percentile ranks.  Event values
are saved separately so they can be tested sparsely before zero-news decay is
used to form the strict full-universe version.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from twenty_third_alternative_event_factors import (
    KEYS,
    event_time,
    map_party,
    prepare_available,
    read_partitioned,
    security_mapping,
)
from twenty_fourth_event_accumulation_factors import ewm_state


def read_formal_income(pit_root: Path) -> pd.DataFrame:
    columns = [
        "SECURITY_ID", "ACT_PUBTIME", "END_DATE", "END_DATE_REP",
        "REPORT_TYPE", "FISCAL_PERIOD", "MERGED_FLAG", "IS_CURRENT_PERIOD",
        "N_INCOME_ATTR_P",
    ]
    pieces: list[pd.DataFrame] = []
    for path in sorted((pit_root / "new_pit_income").rglob("*.parquet")):
        if all(column in pq.read_schema(path).names for column in columns):
            pieces.append(pd.read_parquet(path, columns=columns))
    if not pieces:
        raise FileNotFoundError("new_pit_income parquet files were not found")
    data = pd.concat(pieces, ignore_index=True)
    data["EVENT_TIME"] = event_time(data, "ACT_PUBTIME")
    data["END_DATE"] = pd.to_datetime(data["END_DATE"], errors="coerce").dt.normalize()
    data["END_DATE_REP"] = pd.to_datetime(data["END_DATE_REP"], errors="coerce").dt.normalize()
    data["ACTUAL_PROFIT"] = pd.to_numeric(data["N_INCOME_ATTR_P"], errors="coerce")
    data = data.loc[
        data["IS_CURRENT_PERIOD"].fillna(False)
        & data["END_DATE"].eq(data["END_DATE_REP"])
        & data["MERGED_FLAG"].astype(str).eq("1")
    ].dropna(subset=["SECURITY_ID", "EVENT_TIME", "END_DATE", "ACTUAL_PROFIT"])
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    # The first formal value is the new information.  Later restatements are a
    # different economic event and are intentionally excluded in this round.
    return (
        data.sort_values(["SECURITY_ID", "END_DATE", "EVENT_TIME"])
        .drop_duplicates(["SECURITY_ID", "END_DATE"], keep="first")
        .reset_index(drop=True)
    )


def latest_prior(
    actual: pd.DataFrame,
    prior: pd.DataFrame,
    time_column: str,
    value_column: str,
) -> pd.DataFrame:
    left = actual.sort_values(["EVENT_TIME", "SECURITY_ID", "END_DATE"])
    right = prior.sort_values([time_column, "SECURITY_ID", "END_DATE"])
    return pd.merge_asof(
        left,
        right[["SECURITY_ID", "END_DATE", time_column, value_column]],
        left_on="EVENT_TIME",
        right_on=time_column,
        by=["SECURITY_ID", "END_DATE"],
        direction="backward",
        allow_exact_matches=False,
    )


def symmetric_change(actual: pd.Series, prior: pd.Series) -> pd.Series:
    numerator = 2.0 * (actual - prior)
    denominator = actual.abs() + prior.abs()
    return numerator.div(denominator.where(denominator.gt(1.0))).clip(-2.0, 2.0)


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
    actual = read_formal_income(args.pit_root)

    express = read_partitioned(
        args.alternative_root,
        "fdmt_ee",
        ["PARTY_ID", "TICKER_SYMBOL", "ACT_PUBTIME", "END_DATE", "N_INCOME_ATTR_P"],
    )
    express = map_party(express, pair)
    express["EXPRESS_TIME"] = event_time(express, "ACT_PUBTIME")
    express["END_DATE"] = pd.to_datetime(express["END_DATE"], errors="coerce").dt.normalize()
    express["EXPRESS_PROFIT"] = pd.to_numeric(express["N_INCOME_ATTR_P"], errors="coerce")
    express = express.dropna(subset=["EXPRESS_TIME", "END_DATE", "EXPRESS_PROFIT"])

    guidance = read_partitioned(
        args.alternative_root,
        "fdmt_ef_v2",
        [
            "PARTY_ID", "TICKER_SYMBOL", "ACT_PUBTIME", "END_DATE",
            "EXPN_INCAP_LL", "EXPN_INCAP_UPL", "EXPN_INCOME_LL", "EXPN_INCOME_UPL",
        ],
    )
    guidance = map_party(guidance, pair)
    guidance["GUIDANCE_TIME"] = event_time(guidance, "ACT_PUBTIME")
    guidance["END_DATE"] = pd.to_datetime(guidance["END_DATE"], errors="coerce").dt.normalize()
    attr_mid = guidance[["EXPN_INCAP_LL", "EXPN_INCAP_UPL"]].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1)
    total_mid = guidance[["EXPN_INCOME_LL", "EXPN_INCOME_UPL"]].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1)
    guidance["GUIDANCE_PROFIT"] = attr_mid.fillna(total_mid)
    guidance = guidance.dropna(subset=["GUIDANCE_TIME", "END_DATE", "GUIDANCE_PROFIT"])

    with_express = latest_prior(actual, express, "EXPRESS_TIME", "EXPRESS_PROFIT")
    with_both = latest_prior(with_express, guidance, "GUIDANCE_TIME", "GUIDANCE_PROFIT")
    with_both["FORMAL_VS_EXPRESS"] = symmetric_change(
        with_both["ACTUAL_PROFIT"], with_both["EXPRESS_PROFIT"]
    )
    with_both["FORMAL_VS_GUIDANCE"] = symmetric_change(
        with_both["ACTUAL_PROFIT"], with_both["GUIDANCE_PROFIT"]
    )
    with_both["FORMAL_CORRECTION_MEAN"] = with_both[
        ["FORMAL_VS_EXPRESS", "FORMAL_VS_GUIDANCE"]
    ].mean(axis=1)
    with_both["PRIOR_ACCURACY"] = -with_both[
        ["FORMAL_VS_EXPRESS", "FORMAL_VS_GUIDANCE"]
    ].abs().mean(axis=1)
    with_both["IS_ANNUAL"] = with_both["REPORT_TYPE"].eq("A")
    with_both["ANNUAL_FORMAL_CORRECTION"] = with_both["FORMAL_CORRECTION_MEAN"].where(
        with_both["IS_ANNUAL"]
    )

    values = [
        "FORMAL_VS_EXPRESS", "FORMAL_VS_GUIDANCE", "FORMAL_CORRECTION_MEAN",
        "PRIOR_ACCURACY", "ANNUAL_FORMAL_CORRECTION",
    ]
    event_rows = with_both.groupby(["SECURITY_ID", "EVENT_TIME"], as_index=False)[values].last()
    sparse = prepare_available(event_rows, calendar, values)
    sparse.to_parquet(output / "round27_sparse_formal_corrections.parquet", index=False)

    data = panel.merge(sparse, on=KEYS, how="left", validate="one_to_one")
    data = data.sort_values(["SECURITY_ID", "TRADE_DATE"]).reset_index(drop=True)
    specifications = [
        ("r27_formal_vs_express_hl5", "FORMAL_VS_EXPRESS", 5),
        ("r27_formal_vs_express_hl10", "FORMAL_VS_EXPRESS", 10),
        ("r27_formal_vs_guidance_hl5", "FORMAL_VS_GUIDANCE", 5),
        ("r27_formal_vs_guidance_hl10", "FORMAL_VS_GUIDANCE", 10),
        ("r27_formal_correction_mean_hl5", "FORMAL_CORRECTION_MEAN", 5),
        ("r27_formal_correction_mean_hl10", "FORMAL_CORRECTION_MEAN", 10),
        ("r27_prior_accuracy_hl20", "PRIOR_ACCURACY", 20),
        ("r27_annual_formal_correction_hl10", "ANNUAL_FORMAL_CORRECTION", 10),
        ("r27_annual_formal_correction_hl20", "ANNUAL_FORMAL_CORRECTION", 20),
    ]
    result = data[KEYS].copy()
    for output_name, source_name, half_life in specifications:
        result[output_name] = ewm_state(data, source_name, half_life)
        print(output_name, flush=True)
    result.to_parquet(output / "round27_formal_report_correction_factors.parquet", index=False)
    metadata = {
        "formal_reports": len(actual),
        "formal_with_express": int(with_both["EXPRESS_PROFIT"].notna().sum()),
        "formal_with_guidance": int(with_both["GUIDANCE_PROFIT"].notna().sum()),
        "sparse_events": len(sparse),
        "factor_columns": [name for name, _, _ in specifications],
        "uses_rank": False,
        "uses_label": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
