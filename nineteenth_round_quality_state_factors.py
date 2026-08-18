"""Round 19: pre-specified profitability-quality state factors.

Only independently constructed fundamental components are inputs.  Existing
composite/best factors and their neutralized residuals are never read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from round9_13_ensemble_optimization import KEYS, _robust_z


DENSE_COMPONENTS = [
    "dense_lit_q_gp_assets",
    "dense_lit_q_op_assets",
    "dense_lit_qroa",
    "dense_lit_qcfoa",
    "dense_lit_q_low_accruals",
    "dense_lit_q_gross_margin",
    "dense_lit_ttm_gp_assets",
    "dense_lit_q_low_asset_growth",
]
TEMPORAL_COMPONENTS = ["r15_gp_streak_sue4", "r15_sue_breadth4"]


def _weighted(z: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-12:
        raise ValueError(f"weights must sum to one, got {total}")
    return sum(weight * z[column] for column, weight in weights.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel, columns=KEYS)
    dense = pd.read_parquet(args.dense, columns=KEYS + DENSE_COMPONENTS)
    temporal = pd.read_parquet(args.temporal, columns=KEYS + TEMPORAL_COMPONENTS)
    for frame in (panel, dense, temporal):
        frame["TRADE_DATE"] = pd.to_datetime(frame["TRADE_DATE"]).dt.normalize()
    data = panel.merge(dense, on=KEYS, how="left", validate="one_to_one").merge(
        temporal, on=KEYS, how="left", validate="one_to_one"
    )
    components = DENSE_COMPONENTS + TEMPORAL_COMPONENTS
    data[components] = data[components].fillna(
        data.groupby("TRADE_DATE", sort=False)[components].transform("median")
    ).fillna(0.0)
    z = _robust_z(data, components)

    definitions = {
        "r19_profit_confirmation": {
            "dense_lit_q_gp_assets": 0.30,
            "dense_lit_q_op_assets": 0.20,
            "r15_gp_streak_sue4": 0.20,
            "r15_sue_breadth4": 0.30,
        },
        "r19_operating_cash_confirmation": {
            "dense_lit_q_op_assets": 0.40,
            "dense_lit_qcfoa": 0.30,
            "r15_sue_breadth4": 0.30,
        },
        "r19_accrual_adjusted_state": {
            "dense_lit_q_op_assets": 0.40,
            "dense_lit_q_low_accruals": 0.25,
            "r15_sue_breadth4": 0.35,
        },
        "r19_margin_growth_state": {
            "dense_lit_q_gross_margin": 0.30,
            "r15_gp_streak_sue4": 0.35,
            "r15_sue_breadth4": 0.35,
        },
        "r19_multi_horizon_profit": {
            "dense_lit_q_gp_assets": 0.35,
            "dense_lit_ttm_gp_assets": 0.25,
            "r15_gp_streak_sue4": 0.20,
            "r15_sue_breadth4": 0.20,
        },
        "r19_quality_level_equal4": {
            "dense_lit_q_gp_assets": 0.25,
            "dense_lit_q_op_assets": 0.25,
            "dense_lit_qroa": 0.25,
            "dense_lit_qcfoa": 0.25,
        },
        "r19_cash_profit_state": {
            "dense_lit_qcfoa": 0.35,
            "dense_lit_q_gp_assets": 0.25,
            "r15_sue_breadth4": 0.25,
            "r15_gp_streak_sue4": 0.15,
        },
        "r19_conservative_growth_quality": {
            "dense_lit_q_gp_assets": 0.35,
            "dense_lit_q_low_accruals": 0.25,
            "dense_lit_q_low_asset_growth": 0.15,
            "r15_sue_breadth4": 0.25,
        },
    }
    result = data[KEYS].copy()
    for factor, weights in definitions.items():
        result[factor] = _weighted(z, weights)
    factor_columns = list(definitions)
    result[factor_columns] = result[factor_columns].astype("float32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False, compression="zstd")
    print(f"saved {len(result):,} rows and {len(factor_columns)} quality-state factors")


if __name__ == "__main__":
    main()
