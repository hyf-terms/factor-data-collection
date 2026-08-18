"""Build analyst rating and target-price revision event factors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from twenty_third_alternative_event_factors import KEYS, prepare_available, read_partitioned, security_mapping
from twenty_fourth_event_accumulation_factors import ewm_state


def direction_from_mark(mark: pd.Series) -> pd.Series:
    values = pd.to_numeric(mark, errors="coerce")
    return pd.Series(np.select([values.eq(1), values.eq(2)], [1.0, -1.0], default=0.0), index=mark.index)


def symmetric_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    denominator = current.abs() + previous.abs()
    return (2 * (current - previous)).div(denominator.where(denominator.gt(1e-12))).clip(-2, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-root", type=Path, required=True)
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
    _, ticker = security_mapping(args.pit_root)

    rating = read_partitioned(
        args.selected_root, "rr_rating_adjust_v2",
        ["SEC_CODE", "THIS_WRITE_DATE", "THIS_RATING", "LAST_RATING", "RATING_ADJUST_MARK", "ORG_ID"],
    )
    rating["TICKER_SYMBOL"] = rating["SEC_CODE"].astype(str).str.extract(r"(\d{6})", expand=False)
    rating = rating.merge(ticker, on="TICKER_SYMBOL", how="left", validate="many_to_one")
    rating["EVENT_TIME"] = pd.to_datetime(rating["THIS_WRITE_DATE"], errors="coerce").dt.normalize()
    rating["THIS_RATING"] = pd.to_numeric(rating["THIS_RATING"], errors="coerce")
    rating["LAST_RATING"] = pd.to_numeric(rating["LAST_RATING"], errors="coerce")
    rating["RATING_DIRECTION"] = direction_from_mark(rating["RATING_ADJUST_MARK"])
    rating["RATING_CHANGE"] = ((rating["THIS_RATING"] - rating["LAST_RATING"]) / 2).clip(-3, 3)
    rating["RATING_LEVEL"] = ((rating["THIS_RATING"] - 5) / 2).clip(-2, 1)
    rating = rating.dropna(subset=["SECURITY_ID", "EVENT_TIME"]).copy()
    rating["SECURITY_ID"] = rating["SECURITY_ID"].astype("int64")
    rating_daily = rating.groupby(["SECURITY_ID", "EVENT_TIME"], as_index=False).agg(
        RATING_BREADTH=("RATING_DIRECTION", "mean"),
        RATING_CHANGE=("RATING_CHANGE", "mean"),
        RATING_LEVEL=("RATING_LEVEL", "mean"),
        RATING_REPORTS=("ORG_ID", "nunique"),
    )
    rating_daily["RATING_CONFIRMATION"] = rating_daily["RATING_BREADTH"] * np.sqrt(
        rating_daily["RATING_REPORTS"].clip(lower=1)
    )

    target = read_partitioned(
        args.selected_root, "rr_tar_price_adjust2",
        ["SEC_CODE", "THIS_WRITE_DATE", "THIS_TAR_PRICE", "LAST_TAR_PRICE", "CURRENT_PRICE", "TAR_PRICE_MARK", "ORG_ID"],
    )
    target["TICKER_SYMBOL"] = target["SEC_CODE"].astype(str).str.extract(r"(\d{6})", expand=False)
    target = target.merge(ticker, on="TICKER_SYMBOL", how="left", validate="many_to_one")
    target["EVENT_TIME"] = pd.to_datetime(target["THIS_WRITE_DATE"], errors="coerce").dt.normalize()
    for column in ["THIS_TAR_PRICE", "LAST_TAR_PRICE", "CURRENT_PRICE"]:
        target[column] = pd.to_numeric(target[column], errors="coerce")
    target["TARGET_DIRECTION"] = direction_from_mark(target["TAR_PRICE_MARK"])
    target["TARGET_REVISION"] = symmetric_change(target["THIS_TAR_PRICE"], target["LAST_TAR_PRICE"])
    target["TARGET_UPSIDE"] = (
        target["THIS_TAR_PRICE"] / target["CURRENT_PRICE"].where(target["CURRENT_PRICE"].gt(0)) - 1
    ).clip(-1, 3)
    target = target.dropna(subset=["SECURITY_ID", "EVENT_TIME"]).copy()
    target["SECURITY_ID"] = target["SECURITY_ID"].astype("int64")
    target_daily = target.groupby(["SECURITY_ID", "EVENT_TIME"], as_index=False).agg(
        TARGET_BREADTH=("TARGET_DIRECTION", "mean"),
        TARGET_REVISION=("TARGET_REVISION", "mean"),
        TARGET_UPSIDE=("TARGET_UPSIDE", "mean"),
        TARGET_REPORTS=("ORG_ID", "nunique"),
    )
    target_daily["TARGET_CONFIRMATION"] = target_daily["TARGET_REVISION"] * np.sqrt(
        target_daily["TARGET_REPORTS"].clip(lower=1)
    )

    events = rating_daily.merge(target_daily, on=["SECURITY_ID", "EVENT_TIME"], how="outer")
    events["OPINION_CHANGE"] = events[["RATING_CHANGE", "TARGET_REVISION"]].mean(axis=1)
    events["OPINION_LEVEL"] = events[["RATING_LEVEL", "TARGET_UPSIDE"]].mean(axis=1)
    source_columns = [
        "RATING_BREADTH", "RATING_CHANGE", "RATING_LEVEL", "RATING_CONFIRMATION",
        "TARGET_BREADTH", "TARGET_REVISION", "TARGET_UPSIDE", "TARGET_CONFIRMATION",
        "OPINION_CHANGE", "OPINION_LEVEL",
    ]
    sparse = prepare_available(events, calendar, source_columns)
    sparse.to_parquet(output / "round32_sparse_rating_target_events.parquet", index=False)
    dense = panel.merge(sparse, on=KEYS, how="left", validate="one_to_one")
    dense = dense.sort_values(["SECURITY_ID", "TRADE_DATE"]).reset_index(drop=True)
    specifications = [
        ("r32_rating_breadth_hl10", "RATING_BREADTH", 10),
        ("r32_rating_breadth_hl20", "RATING_BREADTH", 20),
        ("r32_rating_change_hl20", "RATING_CHANGE", 20),
        ("r32_rating_level_hl20", "RATING_LEVEL", 20),
        ("r32_rating_confirmation_hl20", "RATING_CONFIRMATION", 20),
        ("r32_target_breadth_hl20", "TARGET_BREADTH", 20),
        ("r32_target_revision_hl10", "TARGET_REVISION", 10),
        ("r32_target_revision_hl20", "TARGET_REVISION", 20),
        ("r32_target_upside_hl20", "TARGET_UPSIDE", 20),
        ("r32_target_confirmation_hl20", "TARGET_CONFIRMATION", 20),
        ("r32_opinion_change_hl20", "OPINION_CHANGE", 20),
        ("r32_opinion_level_hl20", "OPINION_LEVEL", 20),
    ]
    result = dense[KEYS].copy()
    for name, source, half_life in specifications:
        result[name] = ewm_state(dense, source, half_life)
    result.to_parquet(output / "round32_rating_target_factors.parquet", index=False)
    metadata = {
        "rating_reports": len(rating), "target_reports": len(target),
        "event_security_dates": len(sparse),
        "factor_columns": [name for name, _, _ in specifications],
        "uses_rank": False, "uses_label": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
