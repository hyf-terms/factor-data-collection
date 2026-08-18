"""Build dense, no-rank composites from sparse-tested round-15 signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from round9_13_ensemble_optimization import KEYS, _robust_z


COMPONENTS = [
    "r15_gp_streak_sue",
    "r15_sue_breadth",
    "r15_ni_streak_sue",
    "r15_op_streak_sue",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_parquet(args.input, columns=KEYS + COMPONENTS)
    data["TRADE_DATE"] = pd.to_datetime(data["TRADE_DATE"]).dt.normalize()
    z = _robust_z(data, COMPONENTS)
    result = data[KEYS + COMPONENTS].copy()
    result["r15_gp_breadth_equal"] = 0.50 * z["r15_gp_streak_sue"] + 0.50 * z["r15_sue_breadth"]
    result["r15_gp_breadth_7030"] = 0.70 * z["r15_gp_streak_sue"] + 0.30 * z["r15_sue_breadth"]
    result["r15_gp_breadth_3070"] = 0.30 * z["r15_gp_streak_sue"] + 0.70 * z["r15_sue_breadth"]
    result["r15_profit_streak_equal"] = z[
        ["r15_gp_streak_sue", "r15_ni_streak_sue", "r15_op_streak_sue"]
    ].mean(axis=1)
    result["r15_temporal_breadth_profit"] = (
        0.40 * z["r15_gp_streak_sue"]
        + 0.25 * z["r15_sue_breadth"]
        + 0.175 * z["r15_ni_streak_sue"]
        + 0.175 * z["r15_op_streak_sue"]
    )
    factor_columns = [column for column in result if column not in KEYS]
    result[factor_columns] = result[factor_columns].astype("float32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False, compression="zstd")
    print(f"saved {len(result):,} rows, {len(factor_columns)} factors: {args.output}")


if __name__ == "__main__":
    main()
