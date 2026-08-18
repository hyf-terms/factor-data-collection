"""Accumulate round-23 event information without labels or rank transforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["TRADE_DATE", "SECURITY_ID"]


def add_event_difference(sparse: pd.DataFrame, source: str, output: str) -> None:
    observed = sparse.loc[sparse[source].notna(), KEYS + [source]].copy()
    observed = observed.sort_values(["SECURITY_ID", "TRADE_DATE"])
    observed[output] = observed.groupby("SECURITY_ID")[source].diff()
    sparse[output] = sparse[KEYS].merge(
        observed[KEYS + [output]], on=KEYS, how="left", validate="one_to_one"
    )[output].to_numpy()


def ewm_state(data: pd.DataFrame, source: str, half_life: int) -> pd.Series:
    values = pd.to_numeric(data[source], errors="coerce").fillna(0.0)
    state = (
        values.groupby(data["SECURITY_ID"], sort=False)
        .ewm(halflife=half_life, adjust=False)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return state.astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--sparse-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel, columns=KEYS)
    panel["TRADE_DATE"] = pd.to_datetime(panel["TRADE_DATE"]).dt.normalize()
    panel["SECURITY_ID"] = panel["SECURITY_ID"].astype("int64")
    panel = panel.drop_duplicates(KEYS)
    sparse = pd.read_parquet(args.sparse_events)
    sparse["TRADE_DATE"] = pd.to_datetime(sparse["TRADE_DATE"]).dt.normalize()
    sparse["SECURITY_ID"] = sparse["SECURITY_ID"].astype("int64")

    sparse["ANALYST_CONFIRMATION"] = sparse[["PROFIT_REVISION", "EPS_REVISION"]].mean(axis=1)
    sparse["ANALYST_MARGIN_REVISION"] = sparse["PROFIT_REVISION"] - sparse["INCOME_REVISION"]
    add_event_difference(sparse, "GUIDANCE_YOY", "GUIDANCE_INNOVATION")
    add_event_difference(sparse, "EXPRESS_YOY", "EXPRESS_INNOVATION")
    add_event_difference(sparse, "AUDIT_CLEAN", "AUDIT_TRANSITION")
    sparse["CONTRACT_EVENT"] = sparse["CONTRACT_SCALE"].notna().astype("float64")
    sparse["BUYBACK_EVENT"] = sparse["BUYBACK_PLAN"].notna().astype("float64")
    sparse["HOLDER_EVENT"] = sparse["HOLDER_PLAN"].notna().astype("float64")

    data = panel.merge(sparse, on=KEYS, how="left", validate="one_to_one")
    data = data.sort_values(["SECURITY_ID", "TRADE_DATE"]).reset_index(drop=True)
    specifications = [
        ("r24_analyst_profit_accum_hl10", "PROFIT_REVISION", 10),
        ("r24_analyst_profit_accum_hl20", "PROFIT_REVISION", 20),
        ("r24_analyst_profit_accum_hl60", "PROFIT_REVISION", 60),
        ("r24_analyst_breadth_accum_hl20", "REVISION_BREADTH", 20),
        ("r24_analyst_confirmation_hl20", "ANALYST_CONFIRMATION", 20),
        ("r24_analyst_margin_revision_hl20", "ANALYST_MARGIN_REVISION", 20),
        ("r24_guidance_accum_hl10", "GUIDANCE_YOY", 10),
        ("r24_guidance_accum_hl20", "GUIDANCE_YOY", 20),
        ("r24_guidance_accum_hl60", "GUIDANCE_YOY", 60),
        ("r24_guidance_innovation_hl20", "GUIDANCE_INNOVATION", 20),
        ("r24_express_accum_hl10", "EXPRESS_YOY", 10),
        ("r24_express_accum_hl20", "EXPRESS_YOY", 20),
        ("r24_express_innovation_hl20", "EXPRESS_INNOVATION", 20),
        ("r24_audit_transition_hl120", "AUDIT_TRANSITION", 120),
        ("r24_contract_frequency_hl20", "CONTRACT_EVENT", 20),
        ("r24_contract_frequency_hl60", "CONTRACT_EVENT", 60),
        ("r24_holder_event_frequency_hl20", "HOLDER_EVENT", 20),
        ("r24_buyback_frequency_hl20", "BUYBACK_EVENT", 20),
        ("r24_buyback_frequency_hl60", "BUYBACK_EVENT", 60),
    ]
    result = data[KEYS].copy()
    for output_name, source_name, half_life in specifications:
        result[output_name] = ewm_state(data, source_name, half_life)
        print(output_name, flush=True)
    result = result.sort_values(KEYS).reset_index(drop=True)
    factor_columns = [name for name, _, _ in specifications]
    result.to_parquet(output / "round24_event_accumulation_factors.parquet", index=False)
    metadata = {
        "rows": len(result), "factor_columns": factor_columns,
        "missing": {column: int(result[column].isna().sum()) for column in factor_columns},
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
