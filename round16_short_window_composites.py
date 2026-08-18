"""Dense composites for sparse-tested 4/6-quarter temporal signals."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from round9_13_ensemble_optimization import KEYS, _robust_z


COMPONENTS = [
    "r15_gp_streak_sue4",
    "r15_gp_streak_sue6",
    "r15_sue_breadth4",
    "r15_sue_breadth6",
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
    result["r16_short4_gp_breadth_equal"] = (
        0.50 * z["r15_gp_streak_sue4"] + 0.50 * z["r15_sue_breadth4"]
    )
    result["r16_short4_gp_breadth_7030"] = (
        0.70 * z["r15_gp_streak_sue4"] + 0.30 * z["r15_sue_breadth4"]
    )
    result["r16_short4_gp_breadth_3070"] = (
        0.30 * z["r15_gp_streak_sue4"] + 0.70 * z["r15_sue_breadth4"]
    )
    result["r16_short46_breadth_equal"] = (
        0.25 * z["r15_gp_streak_sue4"]
        + 0.25 * z["r15_gp_streak_sue6"]
        + 0.25 * z["r15_sue_breadth4"]
        + 0.25 * z["r15_sue_breadth6"]
    )
    factors = [column for column in result if column not in KEYS]
    result[factors] = result[factors].astype("float32")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False, compression="zstd")
    print(f"saved {len(result):,} rows and {len(factors)} factors")


if __name__ == "__main__":
    main()
