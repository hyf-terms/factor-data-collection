"""Round 49: PIT impairment-structure events, independent of profit signals.

The source is the detailed asset-impairment note.  Reported losses are stored
as negative values, so a larger (less negative) assets-scaled value represents
better impairment quality.  We separate credit-related from non-credit asset
impairments and also form same-fiscal-quarter improvements.  Construction uses
neither profit fields, labels, ranks nor existing factors.  Output is sparse by
design and must be diagnosed before any event decay/densification is attempted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from twenty_sixth_asset_quality_detail_factors import attach_assets, read_balance_assets
from twenty_third_alternative_event_factors import (
    KEYS, event_time, map_party, prepare_available, read_partitioned,
    security_mapping,
)


CREDIT_FIELDS = [
    "BD_LO", "IFP_LO", "FAAFS_LO", "HTMI_LO", "LTEI_LO", "IRE_LO",
    "C_LO", "PBA_LO", "LAP_LO", "CA_LO", "HSA_LO", "RFA_LO",
]
NONCREDIT_FIELDS = ["FA_LO", "CM_LO", "OAGA_LO", "IA_LO", "BP_LO", "OA_LO"]
FACTORS = [
    "r49_total_impairment_quality_event",
    "r49_credit_impairment_quality_event",
    "r49_noncredit_impairment_quality_event",
    "r49_low_impairment_breadth_event",
    "r49_total_impairment_yoy_improvement",
    "r49_credit_impairment_yoy_improvement",
    "r49_noncredit_impairment_yoy_improvement",
]


def build_events(alternative_root: Path, pit_root: Path) -> pd.DataFrame:
    pair, _ = security_mapping(pit_root)
    assets = read_balance_assets(pit_root)
    columns = [
        "PARTY_ID", "TICKER_SYMBOL", "ACT_PUBTIME", "END_DATE",
        "END_DATE_REP", "MERGED_FLAG", "REPORT_TYPE", "ADJUSTED_FLAG",
        "ASS_DEV_T", *CREDIT_FIELDS, *NONCREDIT_FIELDS,
    ]
    data = map_party(
        read_partitioned(alternative_root, "fdmt_ass_imp_lossv2", columns), pair
    )
    data["EVENT_TIME"] = event_time(data, "ACT_PUBTIME")
    data["END_DATE"] = pd.to_datetime(data["END_DATE"], errors="coerce").dt.normalize()
    data["END_DATE_REP"] = pd.to_datetime(data["END_DATE_REP"], errors="coerce").dt.normalize()
    data = data.loc[
        data["MERGED_FLAG"].astype(str).eq("1")
        & data["END_DATE"].eq(data["END_DATE_REP"])
    ].dropna(subset=["SECURITY_ID", "EVENT_TIME", "END_DATE", "ASS_DEV_T"])
    for field in ["ASS_DEV_T", *CREDIT_FIELDS, *NONCREDIT_FIELDS]:
        data[field] = pd.to_numeric(data[field], errors="coerce")
    # A missing detailed line in a disclosed impairment note means no amount
    # was reported for that category, not that the report itself is missing.
    data[[*CREDIT_FIELDS, *NONCREDIT_FIELDS]] = data[[*CREDIT_FIELDS, *NONCREDIT_FIELDS]].fillna(0.0)
    data = data.sort_values(["SECURITY_ID", "END_DATE", "EVENT_TIME"]).drop_duplicates(
        ["SECURITY_ID", "END_DATE"], keep="first"
    )
    data = attach_assets(data, assets)
    scale = data["T_ASSETS"].abs().where(data["T_ASSETS"].abs().gt(1.0))
    data["credit_loss"] = data[CREDIT_FIELDS].sum(axis=1)
    # Total minus credit is more robust to taxonomy changes than summing only
    # named non-credit fields; the latter are retained for breadth diagnostics.
    data["noncredit_loss"] = data["ASS_DEV_T"] - data["credit_loss"]
    data[FACTORS[0]] = data["ASS_DEV_T"] / scale
    data[FACTORS[1]] = data["credit_loss"] / scale
    data[FACTORS[2]] = data["noncredit_loss"] / scale
    loss_lines = data[[*CREDIT_FIELDS, *NONCREDIT_FIELDS]].lt(0).sum(axis=1)
    data[FACTORS[3]] = -loss_lines.astype("float64")
    data["quarter"] = data["END_DATE"].dt.quarter
    data["fiscal_year"] = data["END_DATE"].dt.year
    data["quarter_index"] = data["fiscal_year"] * 4 + data["quarter"]
    lag = data[["SECURITY_ID", "quarter_index", *FACTORS[:3]]].copy()
    lag["quarter_index"] += 4
    lag = lag.rename(columns={factor: f"lag4_{factor}" for factor in FACTORS[:3]})
    data = data.merge(lag, on=["SECURITY_ID", "quarter_index"], how="left", validate="one_to_one")
    for level, improvement in zip(FACTORS[:3], FACTORS[4:]):
        data[improvement] = data[level] - data[f"lag4_{level}"]
    return data[["SECURITY_ID", "EVENT_TIME", "END_DATE", *FACTORS]]


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
    events = build_events(args.alternative_root, args.pit_root)
    sparse = prepare_available(events, calendar, FACTORS)
    result = panel.merge(sparse, on=KEYS, how="left", validate="one_to_one").sort_values(KEYS)
    result[FACTORS] = result[FACTORS].replace([np.inf, -np.inf], np.nan).astype("float32")
    result.to_parquet(output / "round49_impairment_structure_sparse.parquet", index=False, compression="zstd")
    metadata = {
        "event_rows": int(len(events)),
        "event_firms": int(events["SECURITY_ID"].nunique()),
        "factor_columns": FACTORS,
        "panel_non_null": {factor: int(result[factor].notna().sum()) for factor in FACTORS},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
