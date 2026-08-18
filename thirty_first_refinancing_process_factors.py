"""Build refinancing-review process and termination event factors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from twenty_eighth_capital_allocation_factors import attach_market, read_market
from twenty_third_alternative_event_factors import KEYS, prepare_available, read_partitioned, security_mapping
from twenty_fourth_event_accumulation_factors import ewm_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alternative-root", type=Path, required=True)
    parser.add_argument("--pit-root", type=Path, required=True)
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
    _, ticker = security_mapping(args.pit_root)
    market = read_market(args.market_root)

    data = read_partitioned(
        args.alternative_root, "equ_refin_proj_dyna",
        [
            "ID", "TICKER_SYMBOL", "PUBLISH_DATE", "ACCEPT_DATE",
            "LATEST_INQUIRY_DATE", "LATEST_REPLY_DATE", "AUDIT_DATE",
            "SUB_REG_DATE", "REG_RESULT_DATE", "END_DATE", "AUDIT_STATUS",
            "FIN_TYPE", "EXP_FIN_AMOUNT",
        ],
    )
    data["TICKER_SYMBOL"] = data["TICKER_SYMBOL"].astype(str).str.zfill(6)
    data = data.merge(ticker, on="TICKER_SYMBOL", how="left", validate="many_to_one")
    data = data.dropna(subset=["SECURITY_ID"]).copy()
    data["SECURITY_ID"] = data["SECURITY_ID"].astype("int64")
    for column in [
        "PUBLISH_DATE", "ACCEPT_DATE", "LATEST_INQUIRY_DATE", "LATEST_REPLY_DATE",
        "AUDIT_DATE", "SUB_REG_DATE", "REG_RESULT_DATE", "END_DATE",
    ]:
        data[column] = pd.to_datetime(data[column], errors="coerce").dt.normalize()
    data["FIN_AMOUNT_YUAN"] = pd.to_numeric(data["EXP_FIN_AMOUNT"], errors="coerce") * 1e8
    # Attach the announcement-date market cap as the fixed project denominator.
    base = data.rename(columns={"PUBLISH_DATE": "EVENT_TIME"})
    base = attach_market(base, market)
    data = data.merge(
        base[["ID", "MARKET_VALUE_A"]].drop_duplicates("ID"), on="ID", how="left", validate="one_to_one"
    )
    data["FIN_SCALE"] = (data["FIN_AMOUNT_YUAN"] / data["MARKET_VALUE_A"]).clip(0, 1)
    accept = data["ACCEPT_DATE"]
    inquiry_delay = (data["LATEST_REPLY_DATE"] - data["LATEST_INQUIRY_DATE"]).dt.days
    review_duration = (data["AUDIT_DATE"] - accept).dt.days

    stage_specs = [
        ("ACCEPT_DATE", "REFIN_ACCEPT", -1.0),
        ("LATEST_INQUIRY_DATE", "REFIN_INQUIRY", -0.5),
        ("LATEST_REPLY_DATE", "REFIN_REPLY", -0.5),
        ("AUDIT_DATE", "REFIN_AUDIT", -1.0),
        ("SUB_REG_DATE", "REFIN_SUBMIT_REG", -1.0),
        ("REG_RESULT_DATE", "REFIN_REG_RESULT", -1.0),
        ("END_DATE", "REFIN_TERMINATION", 1.0),
    ]
    pieces: list[pd.DataFrame] = []
    for date_column, value_column, sign in stage_specs:
        piece = data[["SECURITY_ID", date_column, "FIN_SCALE"]].rename(columns={date_column: "EVENT_TIME"})
        piece[value_column] = sign * piece["FIN_SCALE"]
        pieces.append(piece[["SECURITY_ID", "EVENT_TIME", value_column]])
    # Review-friction measures are known when the corresponding reply/audit is published.
    friction = data[["SECURITY_ID", "LATEST_REPLY_DATE", "AUDIT_DATE", "FIN_SCALE"]].copy()
    friction["INQUIRY_DELAY"] = -np.log1p(inquiry_delay.clip(lower=0)).div(np.log(91)).clip(-2, 0)
    friction["REVIEW_DURATION"] = -np.log1p(review_duration.clip(lower=0)).div(np.log(366)).clip(-2, 0)
    inquiry_piece = friction[["SECURITY_ID", "LATEST_REPLY_DATE", "INQUIRY_DELAY"]].rename(
        columns={"LATEST_REPLY_DATE": "EVENT_TIME"}
    )
    audit_piece = friction[["SECURITY_ID", "AUDIT_DATE", "REVIEW_DURATION"]].rename(
        columns={"AUDIT_DATE": "EVENT_TIME"}
    )
    pieces.extend([inquiry_piece, audit_piece])

    events = pd.concat(pieces, ignore_index=True, sort=False)
    value_columns = [name for _, name, _ in stage_specs] + ["INQUIRY_DELAY", "REVIEW_DURATION"]
    events = events.groupby(["SECURITY_ID", "EVENT_TIME"], as_index=False)[value_columns].sum(min_count=1)
    sparse = prepare_available(events, calendar, value_columns)
    sparse.to_parquet(output / "round31_sparse_refinancing_events.parquet", index=False)
    dense = panel.merge(sparse, on=KEYS, how="left", validate="one_to_one")
    dense = dense.sort_values(["SECURITY_ID", "TRADE_DATE"]).reset_index(drop=True)
    dense["REFIN_PROGRESS"] = dense[
        ["REFIN_ACCEPT", "REFIN_AUDIT", "REFIN_SUBMIT_REG", "REFIN_REG_RESULT"]
    ].sum(axis=1, min_count=1)
    specifications = [
        ("r31_refin_accept_scale_hl20", "REFIN_ACCEPT", 20),
        ("r31_refin_accept_scale_hl60", "REFIN_ACCEPT", 60),
        ("r31_refin_progress_hl20", "REFIN_PROGRESS", 20),
        ("r31_refin_progress_hl60", "REFIN_PROGRESS", 60),
        ("r31_refin_termination_hl20", "REFIN_TERMINATION", 20),
        ("r31_refin_termination_hl60", "REFIN_TERMINATION", 60),
        ("r31_inquiry_delay_hl20", "INQUIRY_DELAY", 20),
        ("r31_review_duration_hl60", "REVIEW_DURATION", 60),
    ]
    result = dense[KEYS].copy()
    for name, source, half_life in specifications:
        result[name] = ewm_state(dense, source, half_life)
    result.to_parquet(output / "round31_refinancing_process_factors.parquet", index=False)
    metadata = {
        "projects": len(data), "event_security_dates": len(sparse),
        "factor_columns": [name for name, _, _ in specifications],
        "uses_rank": False, "uses_label": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
