"""Round 43: fixed profitability-level and persistence confirmation.

Six directly constructed accounting signals are robustly standardized and
equally weighted: quarterly and TTM gross/operating profitability, four-
quarter gross-profit persistence/SUE, and four-quarter earnings-surprise
breadth.  No existing composite, neutral residual, or label is read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from round9_13_ensemble_optimization import KEYS, _robust_z


DENSE = [
    "dense_lit_q_gp_assets", "dense_lit_ttm_gp_assets",
    "dense_lit_q_op_assets", "dense_lit_ttm_op_assets",
]
TEMPORAL = ["r15_gp_streak_sue4", "r15_sue_breadth4"]
FACTOR = "r43_profitability_persistence_confirmation_equal6"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = pd.read_parquet(args.panel, columns=KEYS).drop_duplicates(KEYS)
    dense = pd.read_parquet(args.dense, columns=KEYS + DENSE).drop_duplicates(KEYS)
    temporal = pd.read_parquet(args.temporal, columns=KEYS + TEMPORAL).drop_duplicates(KEYS)
    for frame in (panel, dense, temporal):
        frame["TRADE_DATE"] = pd.to_datetime(frame["TRADE_DATE"]).dt.normalize()
        frame["SECURITY_ID"] = pd.to_numeric(frame["SECURITY_ID"]).astype("int64")
    data = panel.merge(dense, on=KEYS, how="left", validate="one_to_one").merge(
        temporal, on=KEYS, how="left", validate="one_to_one"
    ).sort_values(["SECURITY_ID", "TRADE_DATE"])
    columns = DENSE + TEMPORAL
    data[columns] = data.groupby("SECURITY_ID", sort=False)[columns].ffill()
    medians = data.groupby("TRADE_DATE", sort=False)[columns].transform("median")
    data[columns] = data[columns].fillna(medians).fillna(0.0)
    z = _robust_z(data, columns)
    out = data[KEYS].copy()
    out[FACTOR] = z[columns].mean(axis=1).astype("float32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(KEYS).to_parquet(args.output, index=False, compression="zstd")
    print(f"saved {len(out):,} rows to {args.output}")


if __name__ == "__main__":
    main()
